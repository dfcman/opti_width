"""============================================================================
roll_sl_optimize.py - 롤 슬리팅(Slitting) 최적화 모듈

[개요]
대형 원단 롤을 작은 폭의 롤들로 자를 때, 손실(trim loss)을 최소화하면서
주문 수량을 만족시키는 최적의 절단 패턴을 찾는 최적화 알고리즘입니다.

[사용 알고리즘]
- Column Generation (열 생성법): 대규모 문제에서 효율적인 패턴 탐색
- Dynamic Programming (동적 계획법): 서브문제에서 유망 패턴 도출
- Mixed Integer Programming (혼합 정수 계획법): 최종 해 도출

[주요 특징]
- 단일폭과 복합폭(2~3개 롤 조합) 모두 지원
- 소폭 롤 제한, 패턴 복잡도 페널티 등 실제 생산 제약 반영
============================================================================"""

import pandas as pd
from ortools.linear_solver import pywraplp

# ============================================================================
# 전역 상수 정의 (Global Constants)
# ============================================================================

# --- 생산량 관련 페널티 ---
OVER_PROD_PENALTY = 2000.0      # 주문량 초과 생산에 대한 페널티 (과잉 생산 억제)
UNDER_PROD_PENALTY = 10000.0    # 주문량 미달 생산에 대한 페널티 (미달 생산 강하게 억제)

# --- Column Generation 파라미터 ---
PATTERN_VALUE_THRESHOLD = 1.0 + 1e-6  # 새로운 패턴이 유의미하다고 판단하는 기준값 (reduced cost)
CG_MAX_ITERATIONS = 200               # Column Generation 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 25          # 목적 함수 개선이 없을 때 조기 중단 기준 반복 횟수
CG_SUBPROBLEM_TOP_N = 1               # 각 반복에서 서브문제로부터 추가할 상위 N개 패턴

# --- 문제 크기 및 시간 제한 ---
SMALL_PROBLEM_THRESHOLD = 10          # 이 값 이하면 모든 패턴 열거 (Brute-force)
FINAL_MIP_TIME_LIMIT_MS = 60000       # 최종 MIP 풀이 시간 제한 (60초)

# --- 복합폭(Composite) 관련 설정 ---
COMPOSITE_MIN_MULTIPLIER = 2          # 복합폭 최소 롤 개수 (예: 500*2 = 1000mm)
COMPOSITE_MAX_MULTIPLIER = 3          # 복합폭 최대 롤 개수 (예: 500*3 = 1500mm)
COMPOSITE_USAGE_PENALTY = 3000.0      # 복합폭 사용 페널티 (단폭 대비 생산성 저하 반영)
COMPOSITE_BASE_CANDIDATES = 20        # 복합폭 생성 시 고려할 기본 롤 후보 개수
COMPOSITE_GENERATION_LIMIT = 2000     # 생성 가능한 복합폭 종류의 최대 개수

# --- 패턴 복잡도 및 소폭 제한 ---
PATTERN_COMPLEXITY_PENALTY = 2000.0   # 패턴 복잡도(아이템 개수) 페널티
SMALL_WIDTH_LIMIT = 480               # 소폭 판정 기준 (mm)
MAX_SMALL_WIDTH_PER_PATTERN = 2       # 한 패턴 내 최대 소폭 롤 수
OVER_PROD_WEIGHT_CAP = 6.0            # 소량 주문 초과 페널티 가중치 상한
MIXED_COMPOSITE_PENALTY = 500.0       # 서로 다른 규격 조합 복합롤 추가 페널티



class RollSLOptimize:
    """
    롤 슬리팅 최적화 클래스
    
    대형 원단 롤을 작은 폭의 롤들로 절단할 때, 손실(trim loss)을 최소화하면서
    주문 수량을 만족시키는 최적의 절단 패턴을 찾는 최적화 알고리즘을 구현합니다.
    
    [주요 기능]
    - 단일폭/복합폭 아이템 자동 생성
    - Column Generation 기반 효율적 패턴 탐색
    - 다양한 제약조건(소폭 제한, 패턴 복잡도 등) 반영
    
    [최적화 목표]
    minimize: trim_loss + over_prod_penalty + under_prod_penalty 
              + complexity_penalty + composite_penalty + mixed_penalty
    """

    def __init__(
        self,
        df_spec_pre,
        max_width=1000,
        min_width=0,
        max_pieces=8,
        b_wgt=0,
        sl_trim=0,
        min_sl_width=0,
        max_sl_width=0,
        composite_min=COMPOSITE_MIN_MULTIPLIER,
        composite_max=COMPOSITE_MAX_MULTIPLIER,
        composite_penalty=COMPOSITE_USAGE_PENALTY,
        pattern_complexity_penalty=PATTERN_COMPLEXITY_PENALTY,
        small_width_limit=SMALL_WIDTH_LIMIT,
        max_small_width_per_pattern=MAX_SMALL_WIDTH_PER_PATTERN,
        lot_no=None
    ):
        """
        최적화 객체 초기화
        
        Args:
            df_spec_pre (pd.DataFrame): 주문 데이터 DataFrame
                필수 컬럼: group_order_no, 지폭(또는 width), 주문수량(또는 주문롤수/order_roll_cnt)
            max_width (int): 허용 복합폭 최대값 (mm)
            min_width (int): 허용 복합폭 최소값 (mm)
            max_pieces (int): 한 패턴에 포함 가능한 최대 아이템(청크) 수
            b_wgt (float): 평량 (참고용)
            sl_trim (int): 슬리터 트림 폭 (mm) - 복합폭 계산 시 추가됨
            min_sl_width (int): 슬리터 최소 폭 (mm)
            max_sl_width (int): 슬리터 최대 폭 (mm)
            composite_min (int): 복합폭 구성 최소 롤 개수
            composite_max (int): 복합폭 구성 최대 롤 개수
            composite_penalty (float): 복합폭 사용 페널티
            pattern_complexity_penalty (float): 패턴 복잡도 페널티
            small_width_limit (int): 소폭 판정 기준 (mm)
            max_small_width_per_pattern (int): 패턴당 최대 소폭 롤 수
            lot_no (str): Lot 번호
        
        Raises:
            KeyError: 필수 컬럼이 없을 경우
        """
        # --- 입력 파라미터 저장 ---
        self.df_spec_pre = df_spec_pre.copy()
        self.max_width = int(max_width)      # 패턴의 최대 허용 폭
        self.min_width = int(min_width)      # 패턴의 최소 요구 폭
        self.max_pieces = int(max_pieces)    # 패턴당 최대 아이템 수
        self.b_wgt = float(b_wgt)            # 평량 (참고용)
        self.sl_trim = sl_trim               # 슬리터 트림
        self.min_sl_width = min_sl_width     # 슬리터 최소 폭
        self.max_sl_width = max_sl_width     # 슬리터 최대 폭
        self.composite_min = max(2, int(composite_min))  # 복합폭 최소 롤수 (최소 2)
        self.composite_max = max(self.composite_min, int(composite_max))
        self.composite_penalty = float(composite_penalty)
        self.pattern_complexity_penalty = float(pattern_complexity_penalty)
        self.small_width_limit = int(small_width_limit)
        self.max_small_width_per_pattern = int(max_small_width_per_pattern)
        self.lot_no = lot_no

        # --- 주문수량 컬럼 탐색 ---
        demand_col_candidates = ['주문수량', '주문롤수', 'order_roll_cnt']
        self.demand_column = next((c for c in demand_col_candidates if c in self.df_spec_pre.columns), None)
        if not self.demand_column:
            raise KeyError("df_spec_pre에 주문 수량 컬럼이 필요합니다. (주문수량 / 주문롤수 / order_roll_cnt 중 하나)")

        # --- 지폭 컬럼 탐색 ---
        width_col_candidates = ['지폭', 'width']
        self.width_column = next((c for c in width_col_candidates if c in self.df_spec_pre.columns), None)
        if not self.width_column:
            raise KeyError("df_spec_pre에 지폭(width) 컬럼이 필요합니다. (지폭 / width 중 하나)")

        # --- 데이터 타입 변환 및 결측치 처리 ---
        self.df_spec_pre[self.demand_column] = pd.to_numeric(
            self.df_spec_pre[self.demand_column],
            errors='coerce'
        ).fillna(0)
        self.df_spec_pre[self.width_column] = pd.to_numeric(
            self.df_spec_pre[self.width_column],
            errors='coerce'
        ).fillna(0)

        # --- 주문별 수요량 및 폭 정보 구성 ---
        # demands: {group_order_no: 총 주문 수량}
        self.demands = self.df_spec_pre.groupby('group_order_no')[self.demand_column].sum().to_dict()
        # base_item_widths: {group_order_no: 지폭}
        self.base_item_widths = self.df_spec_pre.set_index('group_order_no')[self.width_column].to_dict()

        # --- 패턴 저장소 초기화 ---
        self.patterns = []       # 생성된 패턴 리스트 [{아이템명: 개수}, ...]
        self.pattern_keys = set()  # 중복 방지용 패턴 키 집합

        # --- 아이템 정보 저장소 초기화 ---
        self.item_info = {}        # {아이템명: 폭} - 단폭/복합폭 모두 포함
        self.item_composition = {} # {아이템명: {기본아이템: 개수}} - 아이템 구성 정보
        self.item_piece_count = {} # {아이템명: 피스수} - 복합폭의 경우 2 이상
        self.base_items = []       # 기본 아이템(단폭) 리스트
        self.composite_items = []  # 복합폭 아이템 리스트

        # --- 아이템(단폭/복합폭) 생성 ---
        self._prepare_items()

        # --- 전체 아이템 리스트 및 최대 수요량 ---
        self.items = list(self.item_info.keys())  # 모든 아이템(단폭+복합폭)
        self.max_demand = max(self.demands.values()) if self.demands else 1

    def _prepare_items(self):
        """
        기본 아이템(단폭)과 복합폭 아이템을 생성하여 등록합니다.
        
        [처리 순서]
        1. 기본 아이템(단폭) 등록: 각 주문의 지폭을 개별 아이템으로 등록
        2. 순수 복합폭 생성: 동일 규격 N개 조합 (예: 500*2, 500*3)
        3. 혼합 복합폭 생성: 서로 다른 규격 조합 (예: 500+600)
        
        수정 대상:
            self.item_info, self.item_composition, self.item_piece_count,
            self.base_items, self.composite_items
        """
        # --- 1단계: 기본 아이템(단폭) 등록 ---
        for item, width in self.base_item_widths.items():
            if width <= 0:
                continue
            self.item_info[item] = width                 # 아이템명 -> 폭
            self.item_composition[item] = {item: 1}      # 구성: 자기 자신 1개
            self.item_piece_count[item] = 1              # 피스 수: 1
            self.base_items.append(item)

        # --- 2단계: 순수 복합폭 생성 (동일 규격 N개 조합) ---
        # 예: 500mm 규격을 2개 조합 -> 1000mm 복합폭 + sl_trim
        for base_item, base_width in self.base_item_widths.items():
            if base_width <= 0:
                continue
            for num_repeats in range(self.composite_min, self.composite_max + 1):
                composite_width = base_width * num_repeats
                composite_w_with_trim = composite_width + self.sl_trim
                
                # 슬리터 허용 범위 내인 경우에만 등록
                if self.min_sl_width <= composite_w_with_trim <= self.max_sl_width:
                    composition = {base_item: num_repeats}
                    name = self._make_composite_name(composition)
                    if name not in self.item_info:
                        self._register_composite_item(name, composite_w_with_trim, composition, num_repeats)

        # --- 3단계: 혼합 복합폭 생성 (서로 다른 규격 조합) ---
        max_combo_pieces = min(self.max_pieces, self.composite_max)
        min_combo_pieces = min(max_combo_pieces, self.composite_min)
        if min_combo_pieces > max_combo_pieces:
            return

        # 수요량이 많고 폭이 큰 순서로 후보 선정
        base_candidates = sorted(
            self.base_items,
            key=lambda key: (-self.demands.get(key, 0), -self.item_info[key])
        )[:COMPOSITE_BASE_CANDIDATES]
        # 폭 기준 오름차순 정렬 (backtracking 효율화)
        base_candidates = sorted(base_candidates, key=lambda key: self.item_info[key])

        seen_compositions = set()  # 중복 조합 방지
        composite_cap = COMPOSITE_GENERATION_LIMIT  # 생성 개수 제한

        def backtrack(start_idx, composition, total_width, total_pieces):
            """
            Backtracking으로 혼합 복합폭 조합을 탐색합니다.
            
            Args:
                start_idx: 탐색 시작 인덱스 (중복 조합 방지)
                composition: 현재 조합 {아이템: 개수}
                total_width: 현재 총 폭
                total_pieces: 현재 총 피스 수
            
            Returns:
                True: 생성 제한 도달로 탐색 중단
                False: 계속 탐색
            """
            nonlocal composite_cap
            
            composite_w_with_trim = total_width + self.sl_trim

            # 유효한 복합폭이면 등록
            if total_pieces >= min_combo_pieces and self.min_sl_width <= composite_w_with_trim <= self.max_sl_width:
                key = tuple(sorted(composition.items()))
                if key not in seen_compositions:
                    seen_compositions.add(key)
                    composition_snapshot = dict(composition)
                    name = self._make_composite_name(composition_snapshot)
                    if name not in self.item_info:
                        self._register_composite_item(name, composite_w_with_trim, composition_snapshot, total_pieces)
                        composite_cap -= 1
                        if composite_cap <= 0:
                            return True  # 생성 제한 도달
            
            # 최대 피스 수 도달 시 더 이상 추가 불가
            if total_pieces >= max_combo_pieces:
                return False

            # 각 후보 아이템을 추가하며 재귀 탐색
            for idx in range(start_idx, len(base_candidates)):
                base_item = base_candidates[idx]
                width = self.item_info[base_item]
                if width <= 0:
                    continue
                
                # 최대 슬리터 폭 초과 시 스킵 (가지치기)
                if (total_width + width + self.sl_trim) > self.max_sl_width:
                    continue
                
                composition[base_item] = composition.get(base_item, 0) + 1
                should_stop = backtrack(idx, composition, total_width + width, total_pieces + 1)
                composition[base_item] -= 1
                if composition[base_item] == 0:
                    del composition[base_item]
                if should_stop:
                    return True
            return False

        backtrack(0, {}, 0, 0)

    def _clear_patterns(self):
        """패턴 저장소를 초기화합니다."""
        self.patterns = []
        self.pattern_keys = set()

    def _rebuild_pattern_cache(self):
        """패턴 중복 확인용 캐시를 재구축합니다."""
        self.pattern_keys = {frozenset(p.items()) for p in self.patterns}

    def _small_units_for_item(self, item_name):
        """
        아이템이 기여하는 소폭 롤 수를 반환합니다.
        
        단폭 아이템만 소폭 제한 대상이며, 복합폭은 무시됩니다.
        소폭 판정 기준: self.small_width_limit (기본 480mm)
        
        Args:
            item_name: 아이템 이름
        
        Returns:
            int: 소폭 롤 수 (0 또는 qty)
        """
        piece_count = self.item_piece_count.get(item_name, 0)
        if piece_count != 1:  # 복합폭은 제외
            return 0
        composition = self.item_composition.get(item_name, {})
        if len(composition) != 1:
            return 0
        base_item, qty = next(iter(composition.items()))
        base_width = self.base_item_widths.get(base_item, 0)
        if 0 < base_width <= self.small_width_limit:
            return qty
        return 0

    def _count_small_width_units(self, pattern):
        """패턴 내부의 소폭(기준 이하) 롤 수를 계산합니다."""
        return sum(self._small_units_for_item(item_name) * count for item_name, count in pattern.items())

    def _is_mixed_composite(self, item_name):
        """
        아이템이 혼합 복합폭(서로 다른 규격 조합)인지 확인합니다.
        
        Args:
            item_name: 아이템 이름
        
        Returns:
            bool: 혼합 복합폭 여부
        """
        composition = self.item_composition.get(item_name, {})
        if not composition:
            return False
        if len(composition) == 1:  # 단일 규격으로만 구성
            return False
        return True

    def _count_mixed_composites(self, pattern):
        """
        패턴 내 혼합 복합폭 아이템의 총 개수를 계산합니다.
        
        Args:
            pattern: 패턴 dict {아이템명: 개수}
        
        Returns:
            int: 혼합 복합폭 총 개수
        """
        return sum(
            count for item_name, count in pattern.items()
            if self._is_mixed_composite(item_name)
        )

    def _add_pattern(self, pattern):
        """
        유효한 패턴을 패턴 리스트에 추가합니다.
        
        Args:
            pattern: 패턴 dict {아이템명: 개수}
        
        Returns:
            bool: 추가 성공 여부 (중복/소폭 초과 시 False)
        """
        key = frozenset(pattern.items())
        if key in self.pattern_keys:  # 중복 패턴 제외
            return False
        if self._count_small_width_units(pattern) > self.max_small_width_per_pattern:  # 소폭 제한 초과
            return False
        self.patterns.append(dict(pattern))
        self.pattern_keys.add(key)
        return True

    def _count_pattern_pieces(self, pattern):
        """
        패턴의 총 피스(롤) 수를 계산합니다.
        복합폭의 경우 구성 피스 수가 반영됩니다.
        """
        return sum(self.item_piece_count[item] * count for item, count in pattern.items())

    def _count_pattern_composite_units(self, pattern):
        """
        패턴 내 복합폭 단위 수를 계산합니다.
        복합폭 1개 = (piece_count - 1) 단위
        예: 500*2 복합폭 = 1 단위, 500*3 = 2 단위
        """
        return sum(max(0, self.item_piece_count[item] - 1) * count for item, count in pattern.items())

    def _effective_demand(self, item):
        """
        아이템의 실제 수요량을 계산합니다.
        복합폭의 경우 구성 아이템들의 수요를 합산합니다.
        """
        composition = self.item_composition[item]
        return sum(self.demands.get(base_item, 0) * qty for base_item, qty in composition.items())

    def _format_width(self, value):
        """폭 값을 정수 또는 소수점 2자리로 포맷합니다."""
        return int(value) if abs(value - int(value)) < 1e-6 else round(value, 2)

    def _format_item_label(self, item_name):
        """
        아이템의 레이블을 포맷합니다.
        
        단폭: "500"
        순수 복합폭: "1000(500*2)"
        혼합 복합폭: "1100(500*1+600*1)"
        
        Args:
            item_name: 아이템 이름
        
        Returns:
            str: 포맷된 레이블
        """
        composition = self.item_composition[item_name]
        item_width = self.item_info[item_name]
        if len(composition) == 1:
            base_item, qty = next(iter(composition.items()))
            base_width = self.base_item_widths.get(base_item, 0)
            if qty <= 1:  # 단폭
                return str(self._format_width(item_width))
            return f"{self._format_width(item_width)}({self._format_width(base_width)}*{qty})"  # 순수 복합폭
        # 혼합 복합폭
        parts = []
        for base_item, qty in sorted(composition.items()):
            base_width = self.base_item_widths.get(base_item, 0)
            parts.append(f"{self._format_width(base_width)}*{qty}")
        return f"{self._format_width(item_width)}(" + '+'.join(parts) + ")"

    def _register_composite_item(self, name, width, composition, piece_count):
        """
        복합폭 아이템을 등록합니다.
        
        Args:
            name: 아이템 이름
            width: 총 폭 (sl_trim 포함)
            composition: 구성 정보 {기본아이템: 개수}
            piece_count: 피스 수
        """
        self.item_info[name] = width
        self.item_composition[name] = dict(composition)
        self.item_piece_count[name] = piece_count
        self.composite_items.append(name)

    def _make_composite_name(self, composition):
        """
        복합폭 구성 정보로부터 고유한 이름을 생성합니다.
        
        순수 복합폭: "{item}__x{qty}" (예: "ORDER001__x2")
        혼합 복합폭: "mix__{item1}x{qty1}__{item2}x{qty2}"
        
        Args:
            composition: 구성 정보 {아이템: 개수}
        
        Returns:
            str: 생성된 이름
        """
        items = sorted(composition.items())
        if len(items) == 1:
            item, qty = items[0]
            return f"{item}__x{qty}"
        parts = [f"{item}x{qty}" for item, qty in items]
        return f"mix__{'__'.join(parts)}"

    def _generate_initial_patterns(self):
        """
        휴리스틱 기반으로 초기 패턴을 생성합니다.
        
        [생성 전략]
        1. 수요순/폭순 정렬 후 Greedy 방식으로 패턴 구성
        2. 단일 아이템 반복 패턴 생성
        3. 미커버 아이템 보완 패턴 생성
        
        대규모 문제에서 Column Generation의 시작점 역할을 합니다.
        
        수정 대상:
            self.patterns: 생성된 패턴들이 추가됨
        """
        self._clear_patterns()
        if not self.items:
            return

        # --- 정렬 기준별 아이템 리스트 준비 ---
        sorted_by_demand = sorted(
            self.items,
            key=lambda i: (self._effective_demand(i), self.item_info[i]),
            reverse=True
        )
        sorted_by_width_desc = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda i: self.item_info[i])

        heuristics = [sorted_by_demand, sorted_by_width_desc, sorted_by_width_asc]

        for sorted_items in heuristics:
            for item in sorted_items:
                item_width = self.item_info[item]
                if item_width <= 0 or item_width > self.max_width:
                    continue
                item_small_units = self._small_units_for_item(item)
                if item_small_units > self.max_small_width_per_pattern:
                    continue

                pattern = {item: 1}
                current_width = item_width
                current_pieces = 1
                current_small_units = item_small_units

                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    best_fit_item = None
                    for candidate in sorted_items:
                        candidate_width = self.item_info[candidate]
                        if candidate_width <= 0:
                            continue
                        if candidate_width > remaining_width:
                            continue
                        candidate_small = self._small_units_for_item(candidate)
                        if current_small_units + candidate_small > self.max_small_width_per_pattern:
                            continue
                        best_fit_item = candidate
                        break
                    if not best_fit_item:
                        break
                    pattern[best_fit_item] = pattern.get(best_fit_item, 0) + 1
                    current_width += self.item_info[best_fit_item]
                    current_pieces += 1
                    current_small_units += self._small_units_for_item(best_fit_item)

                while current_width < self.min_width and current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    add_item = None
                    for candidate in sorted_by_width_desc:
                        candidate_width = self.item_info[candidate]
                        if candidate_width <= 0:
                            continue
                        if candidate_width > remaining_width:
                            continue
                        candidate_small = self._small_units_for_item(candidate)
                        if current_small_units + candidate_small > self.max_small_width_per_pattern:
                            continue
                        add_item = candidate
                        break
                    if not add_item:
                        break
                    pattern[add_item] = pattern.get(add_item, 0) + 1
                    current_width += self.item_info[add_item]
                    current_pieces += 1
                    current_small_units += self._small_units_for_item(add_item)

                if current_width >= self.min_width and current_pieces <= self.max_pieces:
                    self._add_pattern(pattern)

        for item in self.items:
            item_width = self.item_info[item]
            if item_width <= 0:
                continue

            max_repeat_width = int(self.max_width // item_width) if item_width > 0 else 0
            num_items = min(max_repeat_width, self.max_pieces)
            while num_items > 0:
                total_width = item_width * num_items
                small_units = self._small_units_for_item(item) * num_items
                if small_units > self.max_small_width_per_pattern:
                    num_items -= 1
                    continue
                if total_width >= self.min_width and num_items <= self.max_pieces:
                    if self._add_pattern({item: num_items}):
                        break
                num_items -= 1

        covered_items = {name for pattern in self.patterns for name in pattern}
        uncovered_items = set(self.items) - covered_items

        for item in uncovered_items:
            item_width = self.item_info[item]
            if item_width <= 0 or item_width > self.max_width:
                    continue
            item_small_units = self._small_units_for_item(item)
            if item_small_units > self.max_small_width_per_pattern:
                continue

            pattern = {item: 1}
            current_width = item_width
            current_pieces = 1
            current_small_units = item_small_units

            while current_width < self.min_width and current_pieces < self.max_pieces:
                remaining_width = self.max_width - current_width
                addition = None
                for candidate in sorted_by_width_desc:
                    candidate_width = self.item_info[candidate]
                    if candidate_width <= 0:
                        continue
                    if candidate_width > remaining_width:
                        continue
                    candidate_small = self._small_units_for_item(candidate)
                    if current_small_units + candidate_small > self.max_small_width_per_pattern:
                        continue
                    addition = candidate
                    break
                if not addition:
                    break
                pattern[addition] = pattern.get(addition, 0) + 1
                current_width += self.item_info[addition]
                current_pieces += 1
                current_small_units += self._small_units_for_item(addition)

            if current_width >= self.min_width and current_pieces <= self.max_pieces:
                self._add_pattern(pattern)

    def _generate_all_patterns(self):
        """
        소규모 문제용 - 모든 가능한 패턴을 열거합니다 (Brute-force).
        
        Backtracking을 사용하여 min_width/max_width, max_pieces, 
        소폭 제한을 만족하는 모든 패턴을 생성합니다.
        
        SMALL_PROBLEM_THRESHOLD(기본 10) 이하의 아이템 수에서만 사용됩니다.
        
        수정 대상:
            self.patterns: 생성된 모든 유효 패턴들
        """
        all_patterns = []
        seen_patterns = set()
        item_list = list(self.items)

        def add_pattern(pattern):
            """유효한 패턴을 리스트에 추가"""
            if not pattern:
                return
            key = frozenset(pattern.items())
            if key in seen_patterns:
                return
            total_width = sum(self.item_info[item] * count for item, count in pattern.items())
            total_pieces = sum(pattern.values())
            # 모든 제약조건 확인
            if (
                self.min_width <= total_width <= self.max_width
                and total_pieces <= self.max_pieces
                and self._count_small_width_units(pattern) <= self.max_small_width_per_pattern
            ):
                all_patterns.append(dict(pattern))
                seen_patterns.add(key)

        def backtrack(start_index, current_pattern, current_width, current_pieces, current_small_units):
            """Backtracking으로 모든 조합 탐색"""
            add_pattern(current_pattern)
            if current_pieces >= self.max_pieces or start_index >= len(item_list):
                return

            for i in range(start_index, len(item_list)):
                item = item_list[i]
                item_width = self.item_info[item]
                if item_width <= 0:
                    continue
                if current_width + item_width > self.max_width:  # 폭 초과
                    continue
                if current_pieces + 1 > self.max_pieces:  # 피스 초과
                    continue
                item_small = self._small_units_for_item(item)
                if current_small_units + item_small > self.max_small_width_per_pattern:  # 소폭 초과
                    continue
                current_pattern[item] = current_pattern.get(item, 0) + 1
                backtrack(i, current_pattern, current_width + item_width, current_pieces + 1, current_small_units + item_small)
                current_pattern[item] -= 1
                if current_pattern[item] == 0:
                    del current_pattern[item]

        backtrack(0, {}, 0, 0, 0)
        self.patterns = all_patterns
        self._rebuild_pattern_cache()

    def _solve_master_problem(self, is_final_mip=False):
        """
        마스터 문제를 풀이합니다 (LP 또는 MIP).
        
        [결정 변수]
        - x[j]: 패턴 j의 사용 횟수
        - over_prod_vars[item]: 아이템별 초과 생산량
        - under_prod_vars[item]: 아이템별 미달 생산량
        
        [제약조건]
        - 생산량 + 미달량 = 수요량 + 초과량 (각 아이템별)
        
        [목적함수]
        minimize: trim_loss + over_penalty + under_penalty 
                  + complexity_penalty + composite_penalty + mixed_penalty
        
        Args:
            is_final_mip: True이면 정수해(MIP), False이면 연속해(LP)
        
        Returns:
            dict: {
                'objective': 목적함수 값,
                'pattern_counts': {패턴인덱스: 사용횟수},
                'over_production': {아이템: 초과량},
                'under_production': {아이템: 미달량},
                'duals': {아이템: 쌍대값} (LP만)
            }
            또는 None (해 없음)
        """
        # --- Solver 설정 ---
        solver_name = 'SCIP' if is_final_mip else 'GLOP'  # MIP: SCIP, LP: GLOP
        solver = pywraplp.Solver.CreateSolver(solver_name)
        if not solver:
            return None
        if is_final_mip and hasattr(solver, 'SetTimeLimit'):
            solver.SetTimeLimit(FINAL_MIP_TIME_LIMIT_MS)

        # --- 결정 변수 생성 ---
        # x[j]: 패턴 j의 사용 횟수 (MIP: 정수, LP: 실수)
        x = {
            j: (solver.IntVar if is_final_mip else solver.NumVar)(0, solver.infinity(), f'P_{j}')
            for j in range(len(self.patterns))
        }
        # 초과/미달 생산량 변수
        over_prod_vars = {item: solver.NumVar(0, solver.infinity(), f'Over_{item}') for item in self.demands}
        under_prod_vars = {
            item: solver.NumVar(0, max(0, self.demands[item]), f'Under_{item}') for item in self.demands
        }

        # --- 제약조건: 생산량 + 미달 = 수요 + 초과 ---
        constraints = {}
        for item, demand in self.demands.items():
            # 패턴들의 총 생산량 계산
            production_expr = solver.Sum(
                sum(self.item_composition[item_name].get(item, 0) * count for item_name, count in self.patterns[j].items()) * x[j]
                for j in range(len(self.patterns))
            )
            constraints[item] = solver.Add(
                production_expr + under_prod_vars[item] == demand + over_prod_vars[item],
                f'demand_{item}'
            )

        # --- 패턴별 메트릭 계산 ---
        # 트림 손실 (max_width - 사용폭)
        pattern_trim = {
            j: self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())
            for j, pattern in enumerate(self.patterns)
        }
        # 패턴별 피스 수
        pattern_pieces = {
            j: sum(pattern.values())
            for j, pattern in enumerate(self.patterns)
        }
        # 패턴별 복합폭 단위 수
        pattern_composite_units = {
            j: self._count_pattern_composite_units(pattern)
            for j, pattern in enumerate(self.patterns)
        }
        # 패턴별 혼합 복합폭 수
        pattern_mixed_counts = {
            j: self._count_mixed_composites(pattern)
            for j, pattern in enumerate(self.patterns)
        }

        # --- 목적함수 구성 ---
        # 1. 트림 손실
        total_trim_loss = solver.Sum(pattern_trim[j] * x[j] for j in range(len(self.patterns)))
        
        # 2. 초과 생산 페널티 (소량 주문일수록 가중치 높음)
        over_prod_terms = []
        base_demand = max(self.max_demand, 1)
        for item in self.demands:
            demand = max(self.demands[item], 1)
            weight = min(OVER_PROD_WEIGHT_CAP, base_demand / demand)
            over_prod_terms.append(OVER_PROD_PENALTY * weight * over_prod_vars[item])
        total_over_penalty = solver.Sum(over_prod_terms) if over_prod_terms else solver.Sum([])
        
        # 3. 미달 생산 페널티
        total_under_penalty = solver.Sum(UNDER_PROD_PENALTY * under_prod_vars[item] for item in self.demands)
        
        # 4. 패턴 복잡도 페널티
        total_complexity_penalty = solver.Sum(
            self.pattern_complexity_penalty * pattern_pieces[j] * x[j] for j in range(len(self.patterns))
        )
        
        # 5. 복합폭 사용 페널티
        total_composite_penalty = solver.Sum(
            self.composite_penalty * pattern_composite_units[j] * x[j] for j in range(len(self.patterns))
        )
        
        # 6. 혼합 복합폭 페널티
        total_mixed_penalty = solver.Sum(
            MIXED_COMPOSITE_PENALTY * pattern_mixed_counts[j] * x[j] for j in range(len(self.patterns))
        )

        # --- 최소화 목표 ---
        solver.Minimize(total_trim_loss + total_over_penalty + total_under_penalty +
                        total_complexity_penalty + total_composite_penalty + total_mixed_penalty)

        # --- 풀이 ---
        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None

        # --- 결과 반환 ---
        solution = {
            'objective': solver.Objective().Value(),
            'pattern_counts': {j: x[j].solution_value() for j in range(len(self.patterns))},
            'over_production': {item: over_prod_vars[item].solution_value() for item in self.demands},
            'under_production': {item: under_prod_vars[item].solution_value() for item in self.demands},
        }
        if not is_final_mip:  # LP의 경우 쌍대값 반환 (Column Generation용)
            solution['duals'] = {item: constraints[item].dual_value() for item in self.demands}
        return solution

    def _solve_subproblem(self, duals):
        """
        Column Generation의 서브문제를 풀이합니다.
        
        마스터 문제의 쌍대 변수(dual values)를 사용하여
        최적화에 도움이 되는 새로운 패턴을 탐색합니다.
        
        [알고리즘]
        Dynamic Programming을 사용한 Knapsack 문제
        - 상태: (pieces, width)
        - 값: 쌍대값 기반 패턴 가치
        
        Args:
            duals: {아이템: 쌍대값} - 마스터 문제의 제약조건 쌍대값
        
        Returns:
            list: 상위 N개 유망 패턴 [{'pattern': dict, 'value': float, ...}, ...]
        """
        width_limit = self.max_width
        piece_limit = self.max_pieces
        
        # --- 아이템별 가치 계산 ---
        item_details = []
        for item in self.items:
            item_width = self.item_info[item]
            item_pieces = 1 
            if item_width <= 0 or item_pieces > piece_limit:
                continue
            composition = self.item_composition[item]
            # 쌍대값 기반 아이템 가치 계산
            item_value = sum(duals.get(base, 0) * qty for base, qty in composition.items())
            if self._is_mixed_composite(item):  # 혼합 복합폭 페널티
                item_value -= MIXED_COMPOSITE_PENALTY * 0.05
            if item_value <= 0:  # 가치 없는 아이템 제외
                continue
            item_details.append((item, item_width, item_pieces, item_value))
        if not item_details:
            return []

        # --- DP 테이블 초기화 ---
        # dp_value[pieces][width]: 해당 상태의 최대 가치
        dp_value = [[float('-inf')] * (width_limit + 1) for _ in range(piece_limit + 1)]
        # dp_parent[pieces][width]: 경로 추적용 (prev_pieces, prev_width, item_name)
        dp_parent = [[None] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_value[0][0] = 0.0  # 초기 상태

        # --- DP 전이 ---
        for pieces in range(piece_limit + 1):
            for width in range(width_limit + 1):
                current_value = dp_value[pieces][width]
                if current_value == float('-inf'):
                    continue
                for item_name, item_width, item_pieces, item_value in item_details:
                    next_pieces = pieces + item_pieces
                    next_width = width + item_width
                    if next_pieces > piece_limit or next_width > width_limit:
                        continue
                    new_value = current_value + item_value
                    # 더 좋은 값이면 업데이트
                    if new_value > dp_value[next_pieces][next_width] + 1e-9:
                        dp_value[next_pieces][next_width] = new_value
                        dp_parent[next_pieces][next_width] = (pieces, width, item_name)

        # --- 유망 패턴 추출 ---
        candidate_patterns = []
        seen_patterns = set()
        for pieces in range(1, piece_limit + 1):
            for width in range(self.min_width, width_limit + 1):
                value = dp_value[pieces][width]
                if value <= PATTERN_VALUE_THRESHOLD:  # 임계값 이하 제외
                    continue
                parent = dp_parent[pieces][width]
                if not parent:
                    continue
                    
                # 경로 역추적으로 패턴 재구성
                pattern = {}
                cur_pieces, cur_width = pieces, width
                while cur_pieces > 0:
                    parent_info = dp_parent[cur_pieces][cur_width]
                    if not parent_info:
                        pattern = None
                        break
                    prev_pieces, prev_width, item_name = parent_info
                    pattern[item_name] = pattern.get(item_name, 0) + 1
                    cur_pieces, cur_width = prev_pieces, prev_width
                if not pattern:
                    continue
                
                # 제약조건 검증
                total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                if total_width < self.min_width or total_width > self.max_width:
                    continue
                if self._count_small_width_units(pattern) > self.max_small_width_per_pattern:
                    continue
                key = frozenset(pattern.items())
                if key in seen_patterns:
                    continue
                    
                # 페널티 적용 후 최종 가치 계산
                total_pieces = sum(pattern.values())
                composite_units = self._count_pattern_composite_units(pattern)
                adjusted_value = value - (self.pattern_complexity_penalty * total_pieces) - (self.composite_penalty * composite_units)
                if adjusted_value <= PATTERN_VALUE_THRESHOLD:
                    continue
                    
                seen_patterns.add(key)
                candidate_patterns.append({
                    'pattern': pattern,
                    'value': adjusted_value,
                    'width': total_width,
                    'pieces': total_pieces
                })

        # 가치 내림차순 정렬 후 상위 N개 반환
        candidate_patterns.sort(key=lambda x: x['value'], reverse=True)
        return candidate_patterns[:CG_SUBPROBLEM_TOP_N]

    def run_optimize(self, start_prod_seq=0):
        """
        메인 최적화 실행 함수입니다.
        
        [처리 흐름]
        1. 패턴 생성 (문제 크기에 따라 방식 선택)
           - 소규모: 모든 패턴 열거 (Brute-force)
           - 대규모: 휴리스틱 초기 패턴 생성
        2. 패턴 유효성 검증 및 필터링
        3. Column Generation 반복 (대규모 문제만)
           - LP 마스터 문제 풀이
           - 서브문제로 새 패턴 탐색
           - 유망 패턴 추가
        4. 최종 MIP 풀이
        5. 결과 포맷팅
        
        Args:
            start_prod_seq: 시작 생산 순서 번호
        
        Returns:
            dict: 최적화 결과
                - pattern_result: 패턴 요약 DataFrame
                - pattern_details_for_db: DB 저장용 상세
                - pattern_roll_details_for_db: 롤별 상세
                - fulfillment_summary: 주문 충족 현황 DataFrame
                - composite_usage: 복합폭 사용 정보
                - last_prod_seq: 마지막 생산 순서
            또는 {\"error\": \"에러 메시지\"} (실패 시)
        """
        # --- 1단계: 패턴 생성 (문제 크기에 따라 방식 선택) ---
        if len(self.base_items) <= SMALL_PROBLEM_THRESHOLD:
            # 소규모 문제: 모든 패턴 열거
            self._generate_all_patterns()
        else:
            # 대규모 문제: 휴리스틱 초기 패턴 생성
            self._generate_initial_patterns()

        if not self.patterns:
             return {"error": "초기 유효 패턴을 생성하지 못했습니다."}

        if not self.patterns:
            return {"error": "유효한 패턴을 생성하지 못했습니다."}

        # --- 2단계: 패턴 유효성 검증 및 필터링 ---
        self.patterns = [
            pattern for pattern in self.patterns
            if self.min_width <= sum(self.item_info[item] * count for item, count in pattern.items()) <= self.max_width
            and sum(pattern.values()) <= self.max_pieces
            and self._count_small_width_units(pattern) <= self.max_small_width_per_pattern
        ]
        if not self.patterns:
            return {"error": f"{self.min_width}mm 이상을 충족하는 패턴이 없습니다."}

        self._rebuild_pattern_cache()

        # --- 3단계: Column Generation (대규모 문제만) ---
        if len(self.base_items) > SMALL_PROBLEM_THRESHOLD:
            best_objective = None
            stagnation_count = 0  # 개선 없는 반복 카운터
            
            for iteration in range(CG_MAX_ITERATIONS):
                # LP 완화 문제 풀이
                master_solution = self._solve_master_problem(is_final_mip=False)
                if not master_solution:
                    break
                    
                # 목적함수 개선 여부 확인
                current_objective = master_solution['objective']
                if best_objective is None or current_objective < best_objective - 1e-6:
                    best_objective = current_objective
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # 서브문제로 새 패턴 탐색
                candidate_patterns = self._solve_subproblem(master_solution.get('duals', {}))
                added_pattern = False
                for candidate in candidate_patterns:
                    if self._add_pattern(candidate['pattern']):
                        added_pattern = True
                        
                # 종료 조건: 새 패턴 없음 또는 개선 정체
                if not added_pattern:
                    break
                if stagnation_count >= CG_NO_IMPROVEMENT_LIMIT:
                    break
                    
            self._rebuild_pattern_cache()

        # --- 4단계: 최종 MIP 풀이 ---
        final_solution = self._solve_master_problem(is_final_mip=True)
        if not final_solution:
            return {"error": f"최종 해를 찾지 못했습니다. {self.min_width}mm 이상을 충족하는 주문이 부족합니다."}

        # --- 5단계: 결과 포맷팅 ---
        return self._format_results(final_solution, start_prod_seq)

    def _format_results(self, final_solution, start_prod_seq=0):
        """
        최적화 결과를 DB/출력용 형태로 변환합니다.
        
        [처리 내용]
        - 패턴별 레이블 생성
        - DB 저장용 상세 정보 구성
        - 주문 충족 현황 요약 생성
        - 복합폭 사용 정보 추출
        
        Args:
            final_solution: _solve_master_problem의 MIP 결과
            start_prod_seq: 시작 생산 순서 번호
        
        Returns:
            dict: {
                \"pattern_result\": DataFrame - 패턴 요약
                \"pattern_details_for_db\": list - DB 저장용 패턴 상세
                \"pattern_roll_details_for_db\": list - DB 저장용 롤별 상세
                \"fulfillment_summary\": DataFrame - 주문 충족 현황
                \"composite_usage\": list - 복합폭 사용 정보
                \"last_prod_seq\": int - 마지막 생산 순서
            }
        """
        # --- 결과 저장소 초기화 ---
        result_patterns = []           # 패턴 요약 (출력용)
        pattern_details_for_db = []    # DB 저장용 패턴 상세
        pattern_roll_details_for_db = []  # DB 저장용 롤별 상세
        composite_usage = []           # 복합폭 사용 정보

        production_counts = {item: 0 for item in self.demands}  # 아이템별 생산량 추적
        prod_seq = start_prod_seq

        # Extract common properties from the first row of the dataframe
        first_row = self.df_spec_pre.iloc[0]
        # Helper for safe int conversion
        def safe_int(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        common_props = {
            'diameter': safe_int(first_row.get('dia', 0)),
            'color': first_row.get('color', ''),
            'luster': safe_int(first_row.get('luster', 0)),
            'p_lot': self.lot_no,
            'core': safe_int(first_row.get('core', 0)),
            'order_pattern': first_row.get('order_pattern', '')
        }

        for j, count in final_solution['pattern_counts'].items():
            if count < 0.99:
                continue
            pattern_dict = self.patterns[j]
            roll_count = int(round(count))
            prod_seq += 1

            sorted_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)

            pattern_labels = []
            total_width = 0
            widths_for_db = []
            group_nos_for_db = []
            composite_meta_for_db = []
            roll_seq_counter = 0

            for item_name, num in sorted_items:
                item_width = self.item_info[item_name]
                composition = self.item_composition[item_name]
                total_width += item_width * num

                pattern_labels.extend([self._format_item_label(item_name)] * num)
                widths_for_db.extend([item_width] * num)
                primary_group = next(iter(composition.keys()))
                group_nos_for_db.extend([primary_group] * num)
                composite_meta_for_db.extend([dict(composition)] * num)

                composite_units = max(0, self.item_piece_count[item_name] - 1) * num
                if composite_units > 0:
                    composite_usage.append({
                        'prod_seq': prod_seq,
                        'item': item_name,
                        'composite_width': item_width,
                        'components': dict(composition),
                        'count': roll_count * num
                    })

                for base_item, qty in composition.items():
                    production_counts[base_item] += roll_count * num * qty

                for _ in range(num):
                    roll_seq_counter += 1
                    expanded_widths = []
                    expanded_groups = []
                    for base_item, qty in composition.items():
                        base_width = self.base_item_widths.get(base_item, 0)
                        if base_width <= 0:
                            base_width = item_width / max(1, self.item_piece_count[item_name])
                        expanded_widths.extend([base_width] * qty)
                        expanded_groups.extend([base_item] * qty)

                    pattern_roll_details_for_db.append({
                        'rollwidth': item_width,
                        'widths': (expanded_widths + [0] * 7)[:7],
                        'group_nos': (expanded_groups + [''] * 7)[:7],
                        'count': roll_count,
                        'prod_seq': prod_seq,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'R',
                        **common_props
                    })

            loss = self.max_width - total_width

            result_patterns.append({
                'pattern': ', '.join(pattern_labels),
                'pattern_width': total_width,
                'loss_per_roll': loss,
                'count': roll_count,
                'prod_seq': prod_seq,
                'rs_gubun': 'R',
                **common_props
            })

            pattern_details_for_db.append({
                'widths': (widths_for_db + [0] * 8)[:8],
                'group_nos': (group_nos_for_db + [''] * 8)[:8],
                'count': roll_count,
                'prod_seq': prod_seq,
                'composite_map': composite_meta_for_db,
                'rs_gubun': 'R',
                **common_props
            })
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['pattern', 'pattern_width', 'count', 'loss_per_roll']]

        df_demand = pd.DataFrame.from_dict(self.demands, orient='index', columns=['요구롤수'])
        df_demand.index.name = 'group_order_no'

        df_production = pd.DataFrame.from_dict(production_counts, orient='index', columns=['생산롤수'])
        df_production.index.name = 'group_order_no'

        df_summary = df_demand.join(df_production, how='outer').fillna(0)
        df_summary['과부족량'] = df_summary['생산롤수'] - df_summary['요구롤수']

        info_cols = ['group_order_no', self.width_column, '롤길이', '품질', '출고예정수', 'quality_grade', 'export_yn']
        available_info_cols = [c for c in info_cols if c in self.df_spec_pre.columns]
        group_info_df = self.df_spec_pre[available_info_cols].drop_duplicates(subset=['group_order_no'])

        fulfillment_summary = pd.merge(group_info_df, df_summary.reset_index(), on='group_order_no', how='right')
        rename_map = {
            self.width_column: '가로',
            '롤길이': '세로'
        }
        fulfillment_summary = fulfillment_summary.rename(
            columns={k: v for k, v in rename_map.items() if k in fulfillment_summary.columns}
        )

        final_cols = ['group_order_no', '가로', '세로', '출고예정수', '품질', '요구롤수', '생산롤수', '과부족량']
        final_cols = [c for c in final_cols if c in fulfillment_summary.columns]
        if final_cols:
            fulfillment_summary = fulfillment_summary[final_cols]

        return {
            "pattern_result": df_patterns.sort_values('count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "composite_usage": composite_usage,
            "last_prod_seq": prod_seq
        }

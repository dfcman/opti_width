"""
롤지 복합폭(CA) 최적화 모듈 (roll_optimize_ca.py)

대형 원단 롤을 복합폭 단위로 절단할 때, 손실(trim loss)을 최소화하면서
주문 수량을 충족시키는 최적의 절단 패턴을 찾는 알고리즘을 구현합니다.

[주요 알고리즘]
- Column Generation: 대규모 문제에서 효율적인 패턴 탐색
- Dynamic Programming: 서브문제(Knapsack) 해결
- Mixed Integer Programming (MIP): 최종 정수해 도출

[특징]
- 복합폭(여러 롤을 합친 폭) 자동 생성 및 최적화
- Gurobi 우선, OR-Tools(SCIP) 폴백 지원
- 패턴 종류 수 최소화 페널티 적용 (셋업 비용 고려)

[핵심 제약조건]
- 최소/최대 슬리터 폭 (min_cm_width, max_cm_width)
- 최대 슬리터 수 (max_sl_count)
- 소폭 롤 개수 제한 (MAX_SMALL_WIDTH_PER_PATTERN)
"""
import pandas as pd
import logging
from ortools.linear_solver import pywraplp
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# 전역 상수 정의
# ============================================================

# --- 생산량 페널티 ---
OVER_PROD_PENALTY  = 500000.0  # 주문량 초과 생산에 대한 페널티  500000
UNDER_PROD_PENALTY = 1000000.0  # 주문량 미달 생산에 대한 페널티


# --- 복합폭 및 패턴 관련 페널티 ---
COMPOSITE_USAGE_PENALTY = 0  # 복합폭 사용 페널티 (복합폭 생성 최적화이므로 0)
PATTERN_COUNT_PENALTY = 5000.0  # 패턴 종류 개수 페널티 (셋업 비용)
COMPOSITE_BASE_CANDIDATES = 20  # 복합폭 생성 시 고려할 기본 롤 후보 개수
COMPOSITE_GENERATION_LIMIT = 2000  # 생성 가능한 복합폭 종류의 최대 개수

# --- Column Generation(열 생성) 파라미터 ---
PATTERN_VALUE_THRESHOLD = 1.0 + 1e-6  # 새로운 패턴이 유의미하다고 판단하는 기준값
CG_MAX_ITERATIONS = 200  # 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 25  # 목적 함수 값 개선이 없을 때 조기 중단을 위한 반복 횟수
CG_SUBPROBLEM_TOP_N = 1  # 각 반복에서 서브문제로부터 가져올 상위 N개 패턴

# --- 문제 크기 및 시간 제한 ---
SMALL_PROBLEM_THRESHOLD = 10  # 이 값 이하이면 모든 가능한 패턴을 열거
FINAL_MIP_TIME_LIMIT_MS = 180000  # 최종 MIP 풀이 시간 제한 (60초)


# --- 소폭 제한 ---
SMALL_WIDTH_LIMIT = 480  # 소폭 판정 기준(mm)
MAX_SMALL_WIDTH_PER_PATTERN = 2  # 한 패턴에서 허용되는 소폭 롤 수

# --- 기타 페널티 ---
OVER_PROD_WEIGHT_CAP = 6.0  # 소량 주문에 대한 초과 페널티 가중치 상한
MIXED_COMPOSITE_PENALTY = 50.0  # 서로 다른 규격 조합 복합롤에 대한 추가 페널티

# --- 복합롤 롤길이 제약 ---
ALLOW_DIFF_LENGTH_COMPOSITE = 'N'  # 'Y': 롤길이가 달라도 복합롤 생성 가능, 'N': 같은 롤길이끼리만 복합롤 생성

# --- 2단계 최적화(Two-Stage) 파라미터 ---
TWO_STAGE_TOLERANCE = 0.05  # Stage 1 비용 대비 허용 오차 (10%)


class RollOptimizeCa:
    """
    롤지 복합폭(CA) 최적화 클래스
    
    복합폭 단위로 절단 패턴을 구성하여 주문을 충족시키는 최적화를 수행합니다.
    
    [주요 기능]
    - 단폭/순수복합폭/혼합복합폭 아이템 자동 생성
    - Column Generation 기반 효율적 패턴 탐색
    - Gurobi/OR-Tools 이중 솔버 지원
    - 패턴 종류 수 최소화 (Big-M 제약)
    
    [최적화 목표]
    minimize: trim_loss + over_prod_penalty + under_prod_penalty 
              + pattern_count_penalty + composite_penalty + mixed_penalty
    """

    def __init__(
        self,
        db=None,
        plant=None,
        pm_no=None,
        schedule_unit=None,
        lot_no=None,
        version=None,
        paper_type=None,
        b_wgt=0,
        color=None,
        p_type=None,
        p_wgt=0,
        p_color=None,
        p_machine=None,
        df_spec_pre=None,
        coating_yn='N',
        min_width=0,
        max_width=1000,
        max_pieces=8,
        min_cm_width=0,
        max_cm_width=0,
        max_sl_count=5,
        ww_trim_size_sheet=0,
        ww_trim_size=0,
        num_threads=4
    ):
        """
        최적화 객체 초기화
        
        Args:
            db: 데이터베이스 연결 객체
            plant: 공장 코드 (2000, 3000, 5000, 8000)
            pm_no: PM 번호
            schedule_unit: 스케줄 단위
            lot_no: Lot 번호
            version: 버전
            paper_type: 지종
            b_wgt: 평량 (g/m²)
            color: 색상
            p_type: 생산 지종
            p_wgt: 생산 평량
            p_color: 생산 색상
            p_machine: 생산 머신
            df_spec_pre: 주문 데이터 DataFrame (필수 컬럼: group_order_no, 지폭/width, 주문수량/주문롤수)
            coating_yn: 코팅 여부 ('Y'/'N')
            min_width: 패턴 최소 폭 (mm)
            max_width: 패턴 최대 폭 (mm)
            max_pieces: 패턴당 최대 피스 수
            min_cm_width: 복합폭 최소 폭 (mm)
            max_cm_width: 복합폭 최대 폭 (mm)
            max_sl_count: 복합폭당 최대 슬리터 수
            ww_trim_size_sheet: 
            ww_trim_size: 슬리터 트림 사이즈 (mm)
            num_threads: 솔버 스레드 수
        
        Raises:
            KeyError: 필수 컬럼(주문수량, 지폭)이 없는 경우
        """
        self.db = db
        self.plant = plant
        self.pm_no = pm_no
        self.schedule_unit = schedule_unit
        self.lot_no = lot_no
        self.version = version
        self.df_spec_pre = df_spec_pre.copy() if df_spec_pre is not None else pd.DataFrame()
        
        # 공통 속성
        self.paper_type = paper_type
        self.b_wgt = float(b_wgt) if b_wgt else 0
        self.color = color
        self.p_type = p_type
        self.p_wgt = float(p_wgt) if p_wgt else 0
        self.p_color = p_color
        self.p_machine = p_machine
        self.coating_yn = coating_yn
        self.num_threads = num_threads
        
        # 폭/피스 제약
        self.max_width = int(max_width) if max_width else 0
        self.min_width = int(min_width) if min_width else 0
        self.max_pieces = int(max_pieces) if max_pieces else 8
        
        # 복합폭(CM) 제약 - ww_trim_size를 sl_trim으로 사용
        self.min_cm_width = int(min_cm_width) if min_cm_width else 0
        self.max_cm_width = int(max_cm_width) if max_cm_width else 0
        self.max_sl_count = int(max_sl_count) if max_sl_count else 5
        self.ww_trim_size_sheet = int(ww_trim_size_sheet) if ww_trim_size_sheet else 0
        self.sl_trim = int(ww_trim_size) if ww_trim_size else 0
        self.min_sl_width = self.min_cm_width
        self.max_sl_width = self.max_cm_width
        
        # 복합폭 설정 - 상수 직접 사용
        self.composite_min = 1
        self.composite_max = self.max_sl_count
        self.composite_penalty = COMPOSITE_USAGE_PENALTY
        self.pattern_count_penalty = PATTERN_COUNT_PENALTY
        self.small_width_limit = SMALL_WIDTH_LIMIT
        self.max_small_width_per_pattern = MAX_SMALL_WIDTH_PER_PATTERN
        
        # 롤 길이 정보 저장 (df_spec_pre에서 추출)
        # std_length: 패턴(원지)의 표준 길이 (예: 24180mm)
        # roll_length: 오더 롤 길이 (예: 6000mm)
        if 'std_length' in self.df_spec_pre.columns:
            self.std_length = int(pd.to_numeric(self.df_spec_pre['std_length'].iloc[0], errors='coerce') or 0)
        else:
            self.std_length = 0
        
        self.pattern_length = self.std_length if self.std_length > 0 else 0

        demand_col_candidates = ['주문수량', '주문롤수', 'order_roll_cnt']
        self.demand_column = next((c for c in demand_col_candidates if c in self.df_spec_pre.columns), None)
        if not self.demand_column:
            raise KeyError("df_spec_pre에 주문 수량 컬럼이 필요합니다. (주문수량 / 주문롤수 / order_roll_cnt 중 하나)")

        width_col_candidates = ['지폭', 'width']
        self.width_column = next((c for c in width_col_candidates if c in self.df_spec_pre.columns), None)
        if not self.width_column:
            raise KeyError("df_spec_pre에 지폭(width) 컬럼이 필요합니다. (지폭 / width 중 하나)")

        self.df_spec_pre[self.demand_column] = pd.to_numeric(
            self.df_spec_pre[self.demand_column],
            errors='coerce'
        ).fillna(0)
        self.df_spec_pre[self.width_column] = pd.to_numeric(
            self.df_spec_pre[self.width_column],
            errors='coerce'
        ).fillna(0)

        self.demands = self.df_spec_pre.groupby('group_order_no')[self.demand_column].sum().to_dict()
        self.base_item_widths = self.df_spec_pre.set_index('group_order_no')[self.width_column].to_dict()
        
        # 아이템별 rolls_per_pattern 계산 (각 group_order_no별로 std_length / roll_length)
        self.item_rolls_per_pattern = {}
        # 아이템별 롤길이 저장 (복합롤 생성 시 롤길이 체크용)
        self.item_roll_lengths = {}
        for _, row in self.df_spec_pre.drop_duplicates(subset=['group_order_no']).iterrows():
            group_no = row['group_order_no']
            item_std_length = int(pd.to_numeric(row.get('std_length', 0), errors='coerce') or 0)
            item_roll_length = int(pd.to_numeric(row.get('롤길이', 0), errors='coerce') or 0)
            self.item_roll_lengths[group_no] = item_roll_length  # 롤길이 저장
            if item_std_length > 0 and item_roll_length > 0:
                self.item_rolls_per_pattern[group_no] = item_std_length // item_roll_length
            else:
                self.item_rolls_per_pattern[group_no] = 1

        self.patterns = []
        self.pattern_keys = set()

        self.item_info = {}
        self.item_composition = {}
        self.item_piece_count = {}
        self.base_items = []
        self.composite_items = []

        self._prepare_items()

        self.items = list(self.item_info.keys())
        self.max_demand = max(self.demands.values()) if self.demands else 1

    def _prepare_items(self):
        """
        기본 아이템(단폭)과 복합폭 아이템을 생성하여 등록합니다.
        
        [처리 순서]
        1. 단폭(1폭) 아이템 등록 - min_cm_width~max_cm_width 범위 내
        2. 순수 복합폭 생성 - 동일 규격 N개 조합 (예: 500*3)
        3. 혼합 복합폭 생성 - 서로 다른 규격 조합 (예: 500*2+600*1)
        
        수정 대상:
            self.item_info: 아이템별 폭 정보
            self.item_composition: 아이템별 구성 정보
            self.item_piece_count: 아이템별 피스 수
            self.base_items: 단폭 아이템 리스트
            self.composite_items: 복합폭 아이템 리스트
        """
        # --- 복합폭 생성용 후보 (모든 지폭 포함, 범위 체크 없음) ---
        all_base_for_composite = {}
        for item, width in self.base_item_widths.items():
            if width <= 0:
                continue
            all_base_for_composite[item] = width  # 원본 지폭 (trim 미포함)
        
        # 1폭(단폭)으로 패턴에 직접 사용 가능한 것 (범위 체크)
        for item, width in self.base_item_widths.items():
            if width <= 0:
                continue
            width_with_trim = width + self.sl_trim
            
            # 1폭도 min_cm_width~max_cm_width 범위 내에 있어야 패턴에 직접 사용 가능
            if self.min_sl_width <= width_with_trim <= self.max_sl_width:
                self.item_info[item] = width_with_trim  # trim 추가된 폭 저장
                self.item_composition[item] = {item: 1}
                self.item_piece_count[item] = 1
                self.base_items.append(item)

        # Add pure composite items (동일 규격 반복)
        # 순수 복합폭은 동일 규격 반복이므로 롤길이 체크 불필요
        for base_item, base_width in all_base_for_composite.items():
            for num_repeats in range(self.composite_min, self.composite_max + 1):
                composite_width = base_width * num_repeats
                composite_w_with_trim = composite_width + self.sl_trim
                
                if self.min_sl_width <= composite_w_with_trim <= self.max_sl_width:
                    composition = {base_item: num_repeats}
                    name = self._make_composite_name(composition)
                    if name not in self.item_info:
                        self._register_composite_item(name, composite_w_with_trim, composition, num_repeats)

        # 혼합 복합폭 생성을 위한 후보 (복합폭 생성용 모든 지폭 사용)
        # max_combo_pieces: 복합롤 1개에 들어가는 규격(지폭) 개수 제한 = max_sl_count
        # (max_pieces는 패턴에 들어가는 복합롤 개수이므로 여기서는 사용하지 않음)
        max_combo_pieces = self.composite_max  # = max_sl_count
        min_combo_pieces = self.composite_min
        if min_combo_pieces > max_combo_pieces:
            return

        # 복합폭 생성용 후보: 모든 지폭 사용 (수요순 정렬)
        base_candidates = sorted(
            all_base_for_composite.keys(),
            key=lambda key: (-self.demands.get(key, 0), -all_base_for_composite[key])
        )[:COMPOSITE_BASE_CANDIDATES]
        base_candidates = sorted(base_candidates, key=lambda key: all_base_for_composite[key])

        seen_compositions = set()
        composite_cap = COMPOSITE_GENERATION_LIMIT

        def backtrack(start_idx, composition, total_width, total_pieces, current_roll_length):
            """혼합 복합폭 생성을 위한 백트래킹 함수
            
            Args:
                start_idx: 탐색 시작 인덱스
                composition: 현재 구성 {item: count}
                total_width: 현재까지의 총 폭
                total_pieces: 현재까지의 총 피스 수
                current_roll_length: 현재 복합폭의 롤길이 (첫 아이템의 롤길이)
            """
            nonlocal composite_cap
            
            composite_w_with_trim = total_width + self.sl_trim

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
                            return True
            
            if total_pieces >= max_combo_pieces:
                return False

            for idx in range(start_idx, len(base_candidates)):
                base_item = base_candidates[idx]
                # 복합폭 생성용 후보에서 원본 지폭 사용 (trim 미포함)
                width = all_base_for_composite.get(base_item, 0)
                if width <= 0:
                    continue
                
                if (total_width + width + self.sl_trim) > self.max_sl_width:
                    continue
                
                # 롤길이가 다른 경우 복합롤 생성 제한 체크
                candidate_roll_length = self.item_roll_lengths.get(base_item, 0)
                if ALLOW_DIFF_LENGTH_COMPOSITE == 'N' and current_roll_length is not None:
                    if candidate_roll_length != current_roll_length:
                        continue  # 롤길이가 다르면 해당 아이템 skip
                
                # 첫 아이템인 경우 현재 롤길이 설정
                new_roll_length = current_roll_length if current_roll_length is not None else candidate_roll_length
                
                composition[base_item] = composition.get(base_item, 0) + 1
                should_stop = backtrack(idx, composition, total_width + width, total_pieces + 1, new_roll_length)
                composition[base_item] -= 1
                if composition[base_item] == 0:
                    del composition[base_item]
                if should_stop:
                    return True
            return False

        backtrack(0, {}, 0, 0, None)
        
        # === 생성된 복합롤 정보 로깅 ===
        logging.info(f"\n{'='*60}")
        logging.info(f"[복합롤 생성 결과] 단폭: {len(self.base_items)}개, 복합폭: {len(self.composite_items)}개")
        logging.info(f"{'='*60}")
        
        # 단폭(1폭) 아이템 출력
        if self.base_items:
            logging.info("[단폭 아이템 (패턴에 직접 사용 가능)]")
            for item in self.base_items:
                width = self.item_info[item]
                roll_length = self.item_roll_lengths.get(item, 0)
                demand = self.demands.get(item, 0)
                logging.info(f"  - {item}: 폭={width}mm, 롤길이={roll_length}mm, 수요={demand}")
        
        # 복합폭 아이템 출력 (순수 복합폭 / 혼합 복합폭 구분)
        if self.composite_items:
            pure_composites = []  # 동일 규격 반복
            mixed_composites = []  # 서로 다른 규격 조합
            
            for item in self.composite_items:
                composition = self.item_composition[item]
                if len(composition) == 1:
                    pure_composites.append(item)
                else:
                    mixed_composites.append(item)
            
            if pure_composites:
                logging.info(f"\n[순수 복합폭 (동일 규격 반복)] - {len(pure_composites)}개")
                for item in pure_composites[:20]:  # 최대 20개만 출력
                    width = self.item_info[item]
                    label = self._format_item_label(item)
                    logging.info(f"  - {label}")
                if len(pure_composites) > 20:
                    logging.info(f"  ... 외 {len(pure_composites) - 20}개")
            
            if mixed_composites:
                logging.info(f"\n[혼합 복합폭 (서로 다른 규격 조합)] - {len(mixed_composites)}개")
                for item in mixed_composites[:30]:  # 최대 30개만 출력
                    width = self.item_info[item]
                    label = self._format_item_label(item)
                    composition = self.item_composition[item]
                    # 롤길이 정보 추가
                    roll_lengths = [self.item_roll_lengths.get(base, 0) for base in composition.keys()]
                    unique_lengths = list(set(roll_lengths))
                    len_info = f"롤길이={unique_lengths}" if len(unique_lengths) > 1 else f"롤길이={unique_lengths[0]}mm"
                    logging.info(f"  - {label} ({len_info})")
                if len(mixed_composites) > 30:
                    logging.info(f"  ... 외 {len(mixed_composites) - 30}개")
        
        logging.info(f"{'='*60}\n")

    def _clear_patterns(self):
        """패턴 저장소를 초기화합니다."""
        self.patterns = []
        self.pattern_keys = set()

    def _rebuild_pattern_cache(self):
        """패턴 중복 검사용 캐시를 재구축합니다."""
        self.pattern_keys = {frozenset(p.items()) for p in self.patterns}

    def _small_units_for_item(self, item_name):
        """
        아이템이 포함하는 소폭 롤 개수를 반환합니다.
        
        단폭(1폭) 아이템만 소폭 제한 대상이며, 복합폭은 0을 반환합니다.
        
        Args:
            item_name: 아이템 이름
        
        Returns:
            int: 소폭 롤 개수 (0 또는 1)
        """
        piece_count = self.item_piece_count.get(item_name, 0)
        if piece_count != 1:
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
        """
        패턴 내부의 소폭(기준 이하) 롤 수를 계산합니다.
        
        Args:
            pattern: {아이템명: 개수} 딕셔너리
        
        Returns:
            int: 패턴 내 소폭 롤 총 개수
        """
        return sum(self._small_units_for_item(item_name) * count for item_name, count in pattern.items())

    def _is_mixed_composite(self, item_name):
        """
        아이템이 혼합 복합폭(서로 다른 규격 조합)인지 판정합니다.
        
        Args:
            item_name: 아이템 이름
        
        Returns:
            bool: 혼합 복합폭이면 True
        """
        composition = self.item_composition.get(item_name, {})
        if not composition:
            return False
        if len(composition) == 1:
            return False
        return True

    def _count_mixed_composites(self, pattern):
        """
        패턴 내 혼합 복합폭 아이템 개수를 계산합니다.
        
        Args:
            pattern: {아이템명: 개수} 딕셔너리
        
        Returns:
            int: 혼합 복합폭 아이템 총 개수
        """
        return sum(
            count for item_name, count in pattern.items()
            if self._is_mixed_composite(item_name)
        )

    def _add_pattern(self, pattern):
        """
        유효한 패턴을 패턴 리스트에 추가합니다.
        
        중복 패턴과 소폭 제한 초과 패턴은 추가되지 않습니다.
        
        Args:
            pattern: {아이템명: 개수} 딕셔너리
        
        Returns:
            bool: 추가 성공 여부
        """
        key = frozenset(pattern.items())
        if key in self.pattern_keys:
            return False
        if self._count_small_width_units(pattern) > self.max_small_width_per_pattern:
            return False
        self.patterns.append(dict(pattern))
        self.pattern_keys.add(key)
        return True

    def _count_pattern_pieces(self, pattern):
        """
        패턴의 총 피스 수를 계산합니다.
        
        Args:
            pattern: {아이템명: 개수} 딕셔너리
        
        Returns:
            int: 총 피스 수
        """
        return sum(self.item_piece_count[item] * count for item, count in pattern.items())

    def _count_pattern_composite_units(self, pattern):
        """
        패턴의 복합폭 단위 수를 계산합니다.
        
        복합폭 단위 = (아이템 피스 수 - 1) * 개수
        
        Args:
            pattern: {아이템명: 개수} 딕셔너리
        
        Returns:
            int: 복합폭 단위 총 수
        """
        return sum(max(0, self.item_piece_count[item] - 1) * count for item, count in pattern.items())

    def _count_pattern_item_count(self, pattern):
        """
        패턴 내 복합롤(아이템) 개수를 계산합니다.
        
        패턴에 포함된 아이템의 종류 수 (한 패턴에 복합롤이 몇 개 들어가는지)
        예: {A: 2, B: 1} -> 2개 (아이템 종류가 2개)
        
        Args:
            pattern: {아이템명: 개수} 딕셔너리
        
        Returns:
            int: 패턴 내 아이템(복합롤) 종류 수
        """
        return len(pattern)

    def _format_width(self, value):
        """
        폭 값을 포맷팅합니다 (정수면 정수로, 아니면 소수 2자리).
        
        Args:
            value: 폭 값
        
        Returns:
            int or float: 포맷된 폭 값
        """
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
            if qty <= 1:
                return str(self._format_width(item_width))
            return f"{self._format_width(item_width)}({self._format_width(base_width)}*{qty})"
        parts = []
        for base_item, qty in sorted(composition.items()):
            base_width = self.base_item_widths.get(base_item, 0)
            parts.append(f"{self._format_width(base_width)}*{qty}")
        return f"{self._format_width(item_width)}(" + '+'.join(parts) + ")"

    def _register_composite_item(self, name, width, composition, piece_count):
        """
        복합폭 아이템을 등록합니다.
        
        Args:
            name: 복합폭 이름
            width: 총 폭 (트림 포함)
            composition: 구성 딕셔너리 {base_item: count}
            piece_count: 총 피스 수
        """
        self.item_info[name] = width
        self.item_composition[name] = dict(composition)
        self.item_piece_count[name] = piece_count
        self.composite_items.append(name)

    def _make_composite_name(self, composition):
        """
        복합폭 구성에서 고유 이름을 생성합니다.
        
        순수 복합폭: "{item}__x{qty}"
        혼합 복합폭: "mix__{item1}x{qty1}__{item2}x{qty2}..."
        
        Args:
            composition: 구성 딕셔너리 {base_item: count}
        
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
        sorted_by_demand = sorted(self.items, key=lambda i: (self._effective_demand(i), self.item_info[i]), reverse=True)
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
            add_pattern(current_pattern)
            if current_pieces >= self.max_pieces or start_index >= len(item_list):
                return

            for i in range(start_index, len(item_list)):
                item = item_list[i]
                item_width = self.item_info[item]
                if item_width <= 0:
                    continue
                if current_width + item_width > self.max_width:
                    continue
                if current_pieces + 1 > self.max_pieces:
                    continue
                item_small = self._small_units_for_item(item)
                if current_small_units + item_small > self.max_small_width_per_pattern:
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
        - y[j]: 패턴 j의 사용 여부 (Binary, MIP만)
        - over_prod_vars[item]: 아이템별 초과 생산량
        - under_prod_vars[item]: 아이템별 미달 생산량
        
        [제약조건]
        - 생산량 + 미달량 = 수요량 + 초과량 (각 아이템별)
        - x[j] <= M * y[j] (Big-M, MIP만)
        
        [목적함수]
        minimize: trim_loss + over_penalty + under_penalty 
                  + pattern_count_penalty + composite_penalty + mixed_penalty
        
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
        # ============================================================
        # 1. [Final MIP] Gurobi 직접 솔버 시도
        # ============================================================
        if is_final_mip:
            try:
                logging.info(f"[Final MIP] 총 {len(self.patterns)}개의 패턴 생성됨")
                logging.info("Trying Gurobi Direct Solver RollOptimizeCA (gurobipy)...")
                model = gp.Model("RollOptimizationCA")
                model.setParam("OutputFlag", 0)
                model.setParam("LogToConsole", 0)
                model.setParam("TimeLimit", FINAL_MIP_TIME_LIMIT_MS / 1000.0)
                model.setParam("MIPFocus", 0)

                # Variables: x (Pattern Counts)
                x = {}
                for j in range(len(self.patterns)):
                    x[j] = model.addVar(vtype=GRB.INTEGER, name=f'P_{j}')
                
                # Variables: y (Binary - 패턴 사용 여부)
                y = {}
                for j in range(len(self.patterns)):
                    y[j] = model.addVar(vtype=GRB.BINARY, name=f'Y_{j}')
                
                # Variables: Over/Under Production
                over_prod_vars = {}
                under_prod_vars = {}
                for item, demand in self.demands.items():
                    over_prod_vars[item] = model.addVar(vtype=GRB.CONTINUOUS, name=f'Over_{item}')
                    under_prod_vars[item] = model.addVar(lb=0, ub=max(0, demand), vtype=GRB.CONTINUOUS, name=f'Under_{item}')

                # Constraints: Demand (아이템별 rolls_per_pattern 적용)
                for item, demand in self.demands.items():
                    item_rpp = self.item_rolls_per_pattern.get(item, 1)
                    production_expr = gp.quicksum(
                        sum(self.item_composition[item_name].get(item, 0) * count 
                            for item_name, count in self.patterns[j].items()) * x[j] * item_rpp
                        for j in range(len(self.patterns))
                    )
                    model.addConstr(
                        production_expr + under_prod_vars[item] == demand + over_prod_vars[item],
                        name=f'demand_{item}'
                    )

                # Constraint: x[j] <= M * y[j] (패턴 사용 시 y[j]=1)
                M = sum(self.demands.values()) + 1  # Big-M 값
                for j in range(len(self.patterns)):
                    model.addConstr(x[j] <= M * y[j], name=f'link_{j}')

                # Pre-calculate pattern metrics
                pattern_trim = {
                    j: self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())
                    for j, pattern in enumerate(self.patterns)
                }
                pattern_composite_units = {
                    j: self._count_pattern_composite_units(pattern)
                    for j, pattern in enumerate(self.patterns)
                }
                pattern_mixed_counts = {
                    j: self._count_mixed_composites(pattern)
                    for j, pattern in enumerate(self.patterns)
                }
                # 패턴 내 복합롤(아이템) 개수 계산 (기준 초과 시 페널티 부여용)
                pattern_item_counts = {
                    j: self._count_pattern_item_count(pattern)
                    for j, pattern in enumerate(self.patterns)
                }

                # Objective: Minimize total cost
                total_trim_loss = gp.quicksum(pattern_trim[j] * x[j] for j in range(len(self.patterns)))
                
                # Over production penalty with weight
                over_prod_terms = []
                base_demand = max(self.max_demand, 1)
                for item in self.demands:
                    demand = max(self.demands[item], 1)
                    weight = min(OVER_PROD_WEIGHT_CAP, base_demand / demand)
                    over_prod_terms.append(OVER_PROD_PENALTY * weight * over_prod_vars[item])
                total_over_penalty = gp.quicksum(over_prod_terms) if over_prod_terms else 0
                
                total_under_penalty = gp.quicksum(UNDER_PROD_PENALTY * under_prod_vars[item] for item in self.demands)
                
                # 패턴 종류 개수 페널티 (사용된 패턴 수만큼)
                total_pattern_count_penalty = gp.quicksum(
                    self.pattern_count_penalty * y[j] 
                    for j in range(len(self.patterns))
                )
                total_composite_penalty = gp.quicksum(
                    self.composite_penalty * pattern_composite_units[j] * x[j] 
                    for j in range(len(self.patterns))
                )
                total_mixed_penalty = gp.quicksum(
                    MIXED_COMPOSITE_PENALTY * pattern_mixed_counts[j] * x[j] 
                    for j in range(len(self.patterns))
                )

                model.setObjective(
                    total_trim_loss + total_over_penalty + total_under_penalty +
                    total_pattern_count_penalty + total_composite_penalty + total_mixed_penalty,
                    GRB.MINIMIZE
                )

                model.optimize()

                if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                    status_msg = "Optimal" if model.Status == GRB.OPTIMAL else "Feasible (TimeLimit)"
                    logging.info(f"Using solver: GUROBI for Final MIP (Success: {status_msg}, Obj={model.ObjVal})")
                    solution = {
                        'objective': model.ObjVal,
                        'pattern_counts': {j: x[j].X for j in range(len(self.patterns))},
                        'over_production': {item: over_prod_vars[item].X for item in self.demands},
                        'under_production': {item: under_prod_vars[item].X for item in self.demands},
                    }
                    return solution
                else:
                    logging.warning(f"Gurobi failed (Status={model.Status}). Fallback to SCIP.")

            except Exception as e:
                logging.warning(f"Gurobi execution failed: {e}. Fallback to SCIP.")

        # ============================================================
        # 2. [Fallback/Default] OR-Tools Solver (SCIP or GLOP)
        # ============================================================
        solver_name = 'SCIP' if is_final_mip else 'GLOP'
        solver = pywraplp.Solver.CreateSolver(solver_name)
        if not solver:
            return None
        if is_final_mip and hasattr(solver, 'SetTimeLimit'):
            solver.SetTimeLimit(FINAL_MIP_TIME_LIMIT_MS)

        x = {
            j: (solver.IntVar if is_final_mip else solver.NumVar)(0, solver.infinity(), f'P_{j}')
            for j in range(len(self.patterns))
        }
        
        # Binary variables for pattern usage (only in final MIP)
        y = {}
        if is_final_mip:
            for j in range(len(self.patterns)):
                y[j] = solver.IntVar(0, 1, f'Y_{j}')
        
        over_prod_vars = {item: solver.NumVar(0, solver.infinity(), f'Over_{item}') for item in self.demands}
        under_prod_vars = {
            item: solver.NumVar(0, max(0, self.demands[item]), f'Under_{item}') for item in self.demands
        }

        constraints = {}
        for item, demand in self.demands.items():
            item_rpp = self.item_rolls_per_pattern.get(item, 1)
            production_expr = solver.Sum(
                sum(self.item_composition[item_name].get(item, 0) * count for item_name, count in self.patterns[j].items()) * x[j] * item_rpp
                for j in range(len(self.patterns))
            )
            constraints[item] = solver.Add(
                production_expr + under_prod_vars[item] == demand + over_prod_vars[item],
                f'demand_{item}'
            )

        # Constraint: x[j] <= M * y[j] (only in final MIP)
        if is_final_mip:
            M = sum(self.demands.values()) + 1
            for j in range(len(self.patterns)):
                solver.Add(x[j] <= M * y[j], f'link_{j}')

        pattern_trim = {
            j: self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())
            for j, pattern in enumerate(self.patterns)
        }
        pattern_composite_units = {
            j: self._count_pattern_composite_units(pattern)
            for j, pattern in enumerate(self.patterns)
        }
        pattern_mixed_counts = {
            j: self._count_mixed_composites(pattern)
            for j, pattern in enumerate(self.patterns)
        }
        # 패턴 내 복합롤(아이템) 개수 계산 (기준 초과 시 페널티 부여용)
        pattern_item_counts = {
            j: self._count_pattern_item_count(pattern)
            for j, pattern in enumerate(self.patterns)
        }

        total_trim_loss = solver.Sum(pattern_trim[j] * x[j] for j in range(len(self.patterns)))
        over_prod_terms = []
        base_demand = max(self.max_demand, 1)
        for item in self.demands:
            demand = max(self.demands[item], 1)
            weight = min(OVER_PROD_WEIGHT_CAP, base_demand / demand)
            over_prod_terms.append(OVER_PROD_PENALTY * weight * over_prod_vars[item])
        total_over_penalty = solver.Sum(over_prod_terms) if over_prod_terms else solver.Sum([])
        total_under_penalty = solver.Sum(UNDER_PROD_PENALTY * under_prod_vars[item] for item in self.demands)
        
        # 패턴 종류 개수 페널티 (only in final MIP)
        if is_final_mip:
            total_pattern_count_penalty = solver.Sum(
                self.pattern_count_penalty * y[j] for j in range(len(self.patterns))
            )
        else:
            total_pattern_count_penalty = solver.Sum([])  # LP relaxation에서는 0
        
        total_composite_penalty = solver.Sum(
            self.composite_penalty * pattern_composite_units[j] * x[j] for j in range(len(self.patterns))
        )
        total_mixed_penalty = solver.Sum(
            MIXED_COMPOSITE_PENALTY * pattern_mixed_counts[j] * x[j] for j in range(len(self.patterns))
        )
        
        # # 패턴 내 복합롤 개수가 기준(MAX_COMPOSITE_WITHOUT_PENALTY) 초과 시 페널티
        # total_extra_composite_penalty = solver.Sum(
        #     EXTRA_COMPOSITE_PENALTY * max(0, pattern_item_counts[j] - MAX_COMPOSITE_WITHOUT_PENALTY) * x[j]
        #     for j in range(len(self.patterns))
        # )

        solver.Minimize(total_trim_loss + total_over_penalty + total_under_penalty +
                        total_pattern_count_penalty + total_composite_penalty + total_mixed_penalty)

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None

        solution = {
            'objective': solver.Objective().Value(),
            'pattern_counts': {j: x[j].solution_value() for j in range(len(self.patterns))},
            'over_production': {item: over_prod_vars[item].solution_value() for item in self.demands},
            'under_production': {item: under_prod_vars[item].solution_value() for item in self.demands},
        }
        if not is_final_mip:
            solution['duals'] = {item: constraints[item].dual_value() for item in self.demands}
        return solution

    def solve_two_stage(self, tolerance=None):
        """
        2단계 최적화(2-Stage Optimization)를 수행합니다.
        
        [Stage 1] 수율+손실 최적화
        - Over/Under Penalty + Trim Loss + 복합폭 페널티 최소화
        - 이론적 최적 수율 및 손실 확보
        
        [Stage 2] 패턴 개수 최소화
        - Stage 1의 최적 비용에 Tolerance(허용 오차)를 적용하여 제약조건으로 추가
        - 패턴 종류 개수(y의 합)를 직접 최소화
        
        Args:
            tolerance: Stage 1 비용 대비 허용 오차 (기본 TWO_STAGE_TOLERANCE 상수 사용)
        
        Returns:
            dict: {
                'objective': 목적함수 값,
                'pattern_counts': {패턴인덱스: 사용횟수},
                'over_production': {아이템: 초과량},
                'under_production': {아이템: 미달량},
            }
            또는 None (해 없음)
        """
        # tolerance가 None이면 상수 사용
        if tolerance is None:
            tolerance = TWO_STAGE_TOLERANCE
        
        try:
            logging.info(f"\n{'='*60}")
            logging.info("[2-Stage Optimization] 시작")
            logging.info(f"총 {len(self.patterns)}개의 패턴에 대해 2단계 최적화 수행")
            logging.info(f"Tolerance: {tolerance*100:.1f}%")
            logging.info(f"{'='*60}")
            
            model = gp.Model("RollOptimizationCA_TwoStage")
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)
            model.setParam("TimeLimit", FINAL_MIP_TIME_LIMIT_MS / 1000.0)
            
            # ============================================================
            # 변수 선언
            # ============================================================
            # Variables: x (Pattern Counts)
            x = {}
            for j in range(len(self.patterns)):
                x[j] = model.addVar(vtype=GRB.INTEGER, name=f'P_{j}')
            
            # Variables: y (Binary - 패턴 사용 여부)
            y = {}
            for j in range(len(self.patterns)):
                y[j] = model.addVar(vtype=GRB.BINARY, name=f'Y_{j}')
            
            # Variables: Over/Under Production
            over_prod_vars = {}
            under_prod_vars = {}
            for item, demand in self.demands.items():
                over_prod_vars[item] = model.addVar(vtype=GRB.CONTINUOUS, name=f'Over_{item}')
                under_prod_vars[item] = model.addVar(lb=0, ub=max(0, demand), vtype=GRB.CONTINUOUS, name=f'Under_{item}')
            
            model.update()
            
            # ============================================================
            # 제약조건: 수요 충족
            # ============================================================
            for item, demand in self.demands.items():
                item_rpp = self.item_rolls_per_pattern.get(item, 1)
                production_expr = gp.quicksum(
                    sum(self.item_composition[item_name].get(item, 0) * count 
                        for item_name, count in self.patterns[j].items()) * x[j] * item_rpp
                    for j in range(len(self.patterns))
                )
                model.addConstr(
                    production_expr + under_prod_vars[item] == demand + over_prod_vars[item],
                    name=f'demand_{item}'
                )
            
            # Constraint: x[j] <= M * y[j] (패턴 사용 시 y[j]=1)
            M = sum(self.demands.values()) + 1
            for j in range(len(self.patterns)):
                model.addConstr(x[j] <= M * y[j], name=f'link_{j}')
            
            # ============================================================
            # 패턴 메트릭 사전 계산
            # ============================================================
            pattern_trim = {
                j: self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())
                for j, pattern in enumerate(self.patterns)
            }
            pattern_composite_units = {
                j: self._count_pattern_composite_units(pattern)
                for j, pattern in enumerate(self.patterns)
            }
            pattern_mixed_counts = {
                j: self._count_mixed_composites(pattern)
                for j, pattern in enumerate(self.patterns)
            }
            
            # ============================================================
            # [Stage 1] 수율 최적화 (Over/Under Penalty만)
            # ============================================================
            logging.info("\n[Stage 1] 수율 최적화 시작 (Over/Under Penalty만 최소화)")
            
            # Over production penalty with weight
            over_prod_terms = []
            base_demand = max(self.max_demand, 1)
            for item in self.demands:
                demand = max(self.demands[item], 1)
                weight = min(OVER_PROD_WEIGHT_CAP, base_demand / demand)
                over_prod_terms.append(OVER_PROD_PENALTY * weight * over_prod_vars[item])
            total_over_penalty = gp.quicksum(over_prod_terms) if over_prod_terms else 0
            
            total_under_penalty = gp.quicksum(UNDER_PROD_PENALTY * under_prod_vars[item] for item in self.demands)

             # Trim Loss
            total_trim_loss = gp.quicksum(pattern_trim[j] * x[j] for j in range(len(self.patterns)))
            
            
            # 복합폭 사용 페널티
            total_composite_penalty = gp.quicksum(
                self.composite_penalty * pattern_composite_units[j] * x[j] 
                for j in range(len(self.patterns))
            )
            
            # 혼합 복합폭 페널티
            total_mixed_penalty = gp.quicksum(
                MIXED_COMPOSITE_PENALTY * pattern_mixed_counts[j] * x[j] 
                for j in range(len(self.patterns))
            )
            
            # Stage 1 목적함수: Over/Under Penalty만
            stage1_objective = total_over_penalty + total_under_penalty + total_trim_loss + total_composite_penalty + total_mixed_penalty
            
            model.setObjective(stage1_objective, GRB.MINIMIZE)
            model.optimize()
            
            if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and not (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                logging.warning(f"[Stage 1] 실패 (Status={model.Status}). 기존 방식으로 폴백.")
                return self._solve_master_problem(is_final_mip=True)
            
            stage1_cost = model.ObjVal
            logging.info(f"[Stage 1] 완료 - 최적 비용: {stage1_cost:.2f}")
            
            # Stage 1 결과 저장 (수율 정보)
            stage1_over_prod = sum(over_prod_vars[item].X for item in self.demands)
            stage1_under_prod = sum(under_prod_vars[item].X for item in self.demands)
            logging.info(f"[Stage 1] 초과 생산 합계: {stage1_over_prod:.1f}, 미달 생산 합계: {stage1_under_prod:.1f}")
            
            # ============================================================
            # [Stage 2] 패턴 최적화 (수율 제약 하에 Trim Loss + 패턴 페널티 최소화)
            # ============================================================
            logging.info(f"\n[Stage 2] 패턴 최적화 시작 (수율 제약: Cost <= {stage1_cost:.2f} * {1 + tolerance})")
            
            # 수율 제약 추가: Stage 1 비용 + Tolerance 이내로 유지
            cutoff_cost = stage1_cost * (1.0 + tolerance)
            model.addConstr(stage1_objective <= cutoff_cost, "Efficiency_Constraint")
            
            # Stage 2 목적함수: 패턴 개수(y의 합) 최소화
            # 수율 제약이 걸려 있으므로 수율은 유지하면서 패턴 개수만 줄임
            stage2_objective = gp.quicksum(y[j] for j in range(len(self.patterns)))
            
            model.setObjective(stage2_objective, GRB.MINIMIZE)
            # model.setParam('MIPGap', 0.02)  # 패턴 개수 최소화에서 약간의 갭 허용
            model.optimize()
            
            if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and not (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                logging.warning(f"[Stage 2] 실패 (Status={model.Status}). Stage 1 결과 사용.")
                # Stage 1 결과로 다시 최적화
                model.setObjective(stage1_objective, GRB.MINIMIZE)
                model.optimize()
            
            status_msg = "Optimal" if model.Status == GRB.OPTIMAL else "Feasible"
            logging.info(f"[Stage 2] 완료 - 상태: {status_msg}, 목적함수: {model.ObjVal:.2f}")
            
            # 최종 결과 출력
            used_patterns = sum(1 for j in range(len(self.patterns)) if x[j].X > 0.5)
            total_trim = sum(pattern_trim[j] * x[j].X for j in range(len(self.patterns)))
            logging.info(f"[2-Stage 결과] 사용 패턴 수: {used_patterns}, 총 Trim Loss: {total_trim:.0f}")
            logging.info(f"{'='*60}\n")
            
            solution = {
                'objective': model.ObjVal,
                'pattern_counts': {j: x[j].X for j in range(len(self.patterns))},
                'over_production': {item: over_prod_vars[item].X for item in self.demands},
                'under_production': {item: under_prod_vars[item].X for item in self.demands},
            }
            return solution
            
        except Exception as e:
            logging.warning(f"[2-Stage Optimization] 실패: {e}. 기존 방식으로 폴백.")
            return self._solve_master_problem(is_final_mip=True)

    def _select_best_solution(self, solution1, solution2):
        """
        두 솔루션을 비교하여 더 좋은 해를 선택합니다.
        
        [비교 우선순위]
        1. 미달 생산 합계 (낮을수록 좋음) - 가장 중요
        2. 패턴 개수 (적을수록 좋음)
        3. Trim Loss 합계 (낮을수록 좋음)
        
        Args:
            solution1: 방식 1 (단일 MIP) 솔루션
            solution2: 방식 2 (2-Stage) 솔루션
        
        Returns:
            더 좋은 솔루션 또는 None
        """
        # 둘 다 실패한 경우
        if not solution1 and not solution2:
            logging.warning("[솔루션 비교] 두 방식 모두 해를 찾지 못함")
            return None
        
        # 하나만 성공한 경우
        if not solution1:
            logging.info("[솔루션 비교] 방식 1 실패 → 방식 2 (2-Stage) 선택")
            return solution2
        if not solution2:
            logging.info("[솔루션 비교] 방식 2 실패 → 방식 1 (단일 MIP) 선택")
            return solution1
        
        # 둘 다 성공한 경우 - 비교
        def calc_metrics(solution, name):
            """솔루션 메트릭 계산"""
            under_prod = sum(solution.get('under_production', {}).values())
            over_prod = sum(solution.get('over_production', {}).values())
            pattern_count = sum(1 for cnt in solution.get('pattern_counts', {}).values() if cnt > 0.5)
            
            # Trim Loss 계산
            trim_loss = 0
            for j, cnt in solution.get('pattern_counts', {}).items():
                if cnt > 0.5:
                    pattern = self.patterns[j]
                    pattern_width = sum(self.item_info[item] * count for item, count in pattern.items())
                    trim_loss += (self.max_width - pattern_width) * cnt
            
            return {
                'name': name,
                'under_prod': under_prod,
                'over_prod': over_prod,
                'pattern_count': pattern_count,
                'trim_loss': trim_loss
            }
        
        m1 = calc_metrics(solution1, "단일 MIP")
        m2 = calc_metrics(solution2, "2-Stage")
        
        # 비교 결과 로깅
        logging.info(f"\n{'='*60}")
        logging.info("[솔루션 비교 결과]")
        logging.info(f"{'='*60}")
        logging.info(f"{'항목':<15} {'단일 MIP':>15} {'2-Stage':>15}")
        logging.info(f"{'-'*45}")
        logging.info(f"{'미달 생산':<15} {m1['under_prod']:>15.1f} {m2['under_prod']:>15.1f}")
        logging.info(f"{'초과 생산':<15} {m1['over_prod']:>15.1f} {m2['over_prod']:>15.1f}")
        logging.info(f"{'패턴 개수':<15} {m1['pattern_count']:>15} {m2['pattern_count']:>15}")
        logging.info(f"{'Trim Loss':<15} {m1['trim_loss']:>15.0f} {m2['trim_loss']:>15.0f}")
        logging.info(f"{'='*60}")
        
        # 비교 로직 (우선순위: 미달생산 → 패턴개수 → Trim Loss)
        # 1. 미달 생산 비교 (낮을수록 좋음)
        if abs(m1['under_prod'] - m2['under_prod']) > 0.01:
            if m1['under_prod'] < m2['under_prod']:
                logging.info(f"[선택] 단일 MIP (미달 생산 더 적음: {m1['under_prod']:.1f} < {m2['under_prod']:.1f})")
                return solution1
            else:
                logging.info(f"[선택] 2-Stage (미달 생산 더 적음: {m2['under_prod']:.1f} < {m1['under_prod']:.1f})")
                return solution2
        
        # 2. 패턴 개수 비교 (적을수록 좋음)
        if m1['pattern_count'] != m2['pattern_count']:
            if m1['pattern_count'] < m2['pattern_count']:
                logging.info(f"[선택] 단일 MIP (패턴 개수 더 적음: {m1['pattern_count']} < {m2['pattern_count']})")
                return solution1
            else:
                logging.info(f"[선택] 2-Stage (패턴 개수 더 적음: {m2['pattern_count']} < {m1['pattern_count']})")
                return solution2
        
        # 3. Trim Loss 비교 (낮을수록 좋음)
        if m1['trim_loss'] <= m2['trim_loss']:
            logging.info(f"[선택] 단일 MIP (Trim Loss: {m1['trim_loss']:.0f} <= {m2['trim_loss']:.0f})")
            return solution1
        else:
            logging.info(f"[선택] 2-Stage (Trim Loss: {m2['trim_loss']:.0f} < {m1['trim_loss']:.0f})")
            return solution2

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

        dp_value = [[float('-inf')] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_parent = [[None] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_value[0][0] = 0.0

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
                    if new_value > dp_value[next_pieces][next_width] + 1e-9:
                        dp_value[next_pieces][next_width] = new_value
                        dp_parent[next_pieces][next_width] = (pieces, width, item_name)

        candidate_patterns = []
        seen_patterns = set()
        for pieces in range(1, piece_limit + 1):
            for width in range(self.min_width, width_limit + 1):
                value = dp_value[pieces][width]
                if value <= PATTERN_VALUE_THRESHOLD:
                    continue
                parent = dp_parent[pieces][width]
                if not parent:
                    continue
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
                
                total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                if total_width < self.min_width or total_width > self.max_width:
                    continue
                if self._count_small_width_units(pattern) > self.max_small_width_per_pattern:
                    continue
                key = frozenset(pattern.items())
                if key in seen_patterns:
                    continue
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
                - pattern_roll_cut_details_for_db: 절단별 상세
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
                master_solution = self._solve_master_problem(is_final_mip=False)
                if not master_solution:
                    break
                current_objective = master_solution['objective']
                if best_objective is None or current_objective < best_objective - 1e-6:
                    best_objective = current_objective
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                candidate_patterns = self._solve_subproblem(master_solution.get('duals', {}))
                added_pattern = False
                for candidate in candidate_patterns:
                    if self._add_pattern(candidate['pattern']):
                        added_pattern = True
                if not added_pattern:
                    break
                if stagnation_count >= CG_NO_IMPROVEMENT_LIMIT:
                    break
            self._rebuild_pattern_cache()

        # --- 4단계: 두 방식 비교 후 최적해 선택 ---
        # 방식 1: 기존 단일 MIP
        # 방식 2: 2-Stage Optimization
        # 두 방식의 결과를 비교하여 더 좋은 해를 선택
        
        logging.info("\n[솔루션 비교] 두 방식 실행 및 비교 시작")
        
        # 방식 1: 기존 단일 MIP
        solution_single = self._solve_master_problem(is_final_mip=True)
        
        # 방식 2: 2-Stage Optimization
        solution_two_stage = self.solve_two_stage()
        
        # 결과 비교 및 선택
        final_solution = self._select_best_solution(solution_single, solution_two_stage)
        
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
        - 절단별 상세 정보 생성 (복합폭 → 개별 지폭 분리)
        
        Args:
            final_solution: _solve_master_problem의 MIP 결과
            start_prod_seq: 시작 생산 순서 번호
        
        Returns:
            dict: {
                "pattern_result": DataFrame - 패턴 요약
                "pattern_details_for_db": list - DB 저장용 패턴 상세
                "pattern_roll_details_for_db": list - DB 저장용 롤별 상세
                "pattern_roll_cut_details_for_db": list - 절단별 상세
                "fulfillment_summary": DataFrame - 주문 충족 현황
                "composite_usage": list - 복합폭 사용 정보
                "last_prod_seq": int - 마지막 생산 순서
            }
        """
        # --- 결과 저장소 초기화 ---
        result_patterns = []           # 패턴 요약 (출력용)
        pattern_details_for_db = []    # DB 저장용 패턴 상세
        pattern_roll_details_for_db = []  # DB 저장용 롤별 상세
        pattern_roll_cut_details_for_db = []  # 절단별 상세 (복합폭 분리)
        composite_usage = []           # 복합폭 사용 정보

        production_counts = {item: 0 for item in self.demands}
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
                    base_item_rpp = self.item_rolls_per_pattern.get(base_item, 1)
                    production_counts[base_item] += roll_count * num * qty * base_item_rpp

                for _ in range(num):
                    roll_seq_counter += 1
                    expanded_widths = []
                    expanded_groups = []
                    expanded_rs_gubuns = []
                    for base_item, qty in composition.items():
                        base_width = self.base_item_widths.get(base_item, 0)
                        if base_width <= 0:
                            base_width = item_width / max(1, self.item_piece_count[item_name])
                        expanded_widths.extend([base_width] * qty)
                        expanded_groups.extend([base_item] * qty)
                        expanded_rs_gubuns.extend('R' * qty)

                    # roll별 trim_loss 계산 (rollwidth - sum(widths))
                    roll_widths_list = (expanded_widths + [0] * 7)[:7]
                    roll_trim_loss = item_width - sum(roll_widths_list)
                    
                    pattern_roll_details_for_db.append({
                        'rollwidth': item_width,
                        'roll_widths': roll_widths_list,
                        'widths': roll_widths_list,
                        'group_nos': (expanded_groups + [''] * 7)[:7],
                        'rs_gubuns': (expanded_rs_gubuns + [''] * 7)[:7],
                        'count': roll_count,
                        'prod_seq': prod_seq,
                        'roll_seq': roll_seq_counter,
                        'pattern_length': self.pattern_length,
                        'loss_per_roll': roll_trim_loss,  # rollwidth - sum(widths)
                        'rs_gubun': 'W',
                        'sc_trim': self.ww_trim_size_sheet,
                        'sl_trim': self.sl_trim if self.coating_yn == 'Y' else 0, 
                        **common_props
                    })

            loss = self.max_width - total_width

            result_patterns.append({
                'pattern': ', '.join(pattern_labels),
                'pattern_width': total_width,
                'loss_per_roll': loss,
                'count': roll_count,
                'prod_seq': prod_seq,
                'rs_gubun': 'W',
                **common_props
            })

            pattern_details_for_db.append({
                'widths': (widths_for_db + [0] * 8)[:8],
                'group_nos': (group_nos_for_db + [''] * 8)[:8],
                'count': roll_count,
                'prod_seq': prod_seq,
                'composite_map': composite_meta_for_db,
                'pattern_length': self.pattern_length,  # [New] 롤 길이 추가
                'rs_gubun': 'W',
                **common_props
            })
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns['std_length'] = self.std_length
            df_patterns = df_patterns[['pattern', 'pattern_width', 'count', 'loss_per_roll', 'std_length']]

        df_demand = pd.DataFrame.from_dict(self.demands, orient='index', columns=['주문수량'])
        df_demand.index.name = 'group_order_no'

        df_production = pd.DataFrame.from_dict(production_counts, orient='index', columns=['생산롤수'])
        df_production.index.name = 'group_order_no'

        df_summary = df_demand.join(df_production, how='outer').fillna(0)
        df_summary['과부족(롤)'] = df_summary['생산롤수'] - df_summary['주문수량']

        info_cols = ['group_order_no', self.width_column, '롤길이', '등급']
        available_info_cols = [c for c in info_cols if c in self.df_spec_pre.columns]
        group_info_df = self.df_spec_pre[available_info_cols].drop_duplicates(subset=['group_order_no'])

        fulfillment_summary = pd.merge(group_info_df, df_summary.reset_index(), on='group_order_no', how='right')
        rename_map = {
            self.width_column: '지폭',
        }
        fulfillment_summary = fulfillment_summary.rename(
            columns={k: v for k, v in rename_map.items() if k in fulfillment_summary.columns}
        )

        final_cols = ['group_order_no', '지폭', '롤길이', '등급', '주문수량', '생산롤수', '과부족(롤)']
        final_cols = [c for c in final_cols if c in fulfillment_summary.columns]
        if final_cols:
            fulfillment_summary = fulfillment_summary[final_cols]

        # [New] pattern_roll_cut_details_for_db 생성 (execute.py 호환성)
        # 복합폭(예: 1245=405*3)을 개별 지폭(405, 405, 405)으로 분리
        global_cut_seq = 0
        for roll_detail in pattern_roll_details_for_db:
            widths = roll_detail.get('widths', [])
            group_nos = roll_detail.get('group_nos', [])
            
            # widths에서 0이 아닌 값만 추출하여 개별 row 생성
            cut_seq_in_roll = 0
            for i, w in enumerate(widths):
                if w > 0:
                    global_cut_seq += 1
                    cut_seq_in_roll += 1
                    g_no = group_nos[i] if i < len(group_nos) else ''
                    pattern_roll_cut_details_for_db.append({
                        'prod_seq': roll_detail['prod_seq'],
                        'unit_no': roll_detail['prod_seq'],
                        'seq': global_cut_seq,
                        'roll_seq': roll_detail['roll_seq'],
                        'cut_seq': cut_seq_in_roll,
                        'rs_gubun': 'W',
                        'width': w,  # 개별 지폭
                        'group_no': g_no,
                        'weight': 0,
                        'pattern_length': roll_detail.get('pattern_length', self.pattern_length),
                        'count': roll_detail['count'],
                        'p_lot': roll_detail.get('p_lot'),
                        'diameter': roll_detail.get('diameter'),
                        'core': roll_detail.get('core'),
                        'color': roll_detail.get('color'),
                        'luster': roll_detail.get('luster')
                    })

        return {
            "pattern_result": df_patterns.sort_values('count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,  # [New] 추가
            "fulfillment_summary": fulfillment_summary,
            "composite_usage": composite_usage,
            "last_prod_seq": prod_seq
        }

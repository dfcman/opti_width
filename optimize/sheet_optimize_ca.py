"""
===============================================================================
SheetOptimizeCa Module - 쉬트지 복합폭(Composite Width) 최적화 모듈
===============================================================================

[모듈 개요]
이 모듈은 쉬트지(Sheet) 오더에 대해 "복합폭(Composite Width)" 개념을 적용하여
최적의 절단 패턴을 찾습니다. 일반적인 쉬트 최적화(sheet_optimize.py)와 달리,
여러 개의 기본 지폭을 묶어서 하나의 "복합 아이템"으로 처리합니다.

[주요 특징]
1. **복합폭(Composite Width) 개념**:
   - 동일하거나 서로 다른 지폭들을 조합하여 하나의 슬리터 칼(Slitter Knife)로
     한 번에 절단할 수 있는 "복합 아이템"을 생성합니다.
   - 예: 710mm x 2 = 1420mm 복합폭, 또는 710mm + 850mm = 1560mm 복합폭

2. **미터(m) 기반 수요 계산**:
   - 쉬트지 주문은 톤(ton) 단위로 들어오지만, 내부적으로는 필요한 총 길이(m)로
     변환하여 계산합니다.
   - 평량(b_wgt), 가로(지폭), 세로(장당 길이)를 기반으로 변환합니다.

3. **패턴 구조**:
   - 각 패턴은 여러 복합 아이템들의 조합 + 해당 패턴의 롤 길이로 구성됩니다.
   - 패턴 = {composition: {아이템명: 개수}, length: 롤 길이, loss_per_roll: 손실}

[최적화 알고리즘]
- 이 모듈 자체는 패턴 생성 로직을 포함하지 않으며, 외부에서 `self.patterns`에
  미리 패턴을 주입한 후 `run_optimize()`를 호출해야 합니다.
- 최종 해는 `_solve_master_problem_ilp()`를 통해 정수계획법(MIP)으로 구합니다.

[클래스 구조]
- SheetOptimizeCa: 메인 최적화 클래스
  - __init__: 주문 데이터 로드 및 초기화
  - _prepare_items: 복합폭 아이템 생성
  - _calculate_demand_meters: 주문량을 미터 단위로 변환
  - run_optimize: 최적화 실행 (패턴이 미리 설정되어 있어야 함)
  - _format_results: 결과 포맷팅
  - _build_pattern_details: DB 저장용 상세 정보 생성
  - _build_fulfillment_summary: 주문 이행 요약 생성

===============================================================================
"""

import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import math
import random
import time
import logging
import gurobipy as gp
from gurobipy import GRB
import configparser
import os


# 페널티 값 (롤 수 기반)
OVER_PROD_PENALTY  = 100000.0     # 과잉 생산에 대한 페널티 (1롤당)
UNDER_PROD_PENALTY = 50000.0   # 부족 생산에 대한 페널티 (1롤당, 부족은 과잉보다 더 큰 페널티)
PATTERN_COUNT_PENALTY = 100000.0  # 패턴 종류 수에 대한 페널티 (생산 효율을 위해 패턴 수 줄임)
DISALLOWED_SINGLE_BASE_WIDTHS = {}  # 단일 사용을 금지할 주문 폭 집합
SINGLE_STRIP_PENALTY = 50000.0  # 패턴 내 단폭 아이템(x1) 개수에 대한 페널티 (복합폭 x2 이상 사용 유도)
PATTERN_COMPLEXITY_PENALTY = 1.0  # 복잡도 페널티 (한 패턴에 여러 규격 섞지 않도록)
MIXED_SHEET_LENGTH_PENALTY = 50000.0  # 패턴 내 다른 세로 길이 조합에 대한 페널티

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 1      # 패턴에 포함될 수 있는 최소 폭(piece)의 수
SMALL_PROBLEM_THRESHOLD = 8     # 전체 탐색을 수행할 최대 주문 지폭 종류 수
SOLVER_TIME_LIMIT_MS = 180000    # 최종 MIP 솔버의 최대 실행 시간 (밀리초)
CG_MAX_ITERATIONS = 1000         # 열 생성(Column Generation) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 50    # 개선 없는 경우, 열 생성 조기 종료 조건
CG_SUBPROBLEM_TOP_N = 10         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴
# 나이프 로드 제약: 패턴 생산 횟수는 k1*a + k2*b 형태여야 함 (a,b>=0 정수)
# KNIFE_LOAD_K1 = 1
# KNIFE_LOAD_K2 = 1



# TRIM_PENALTY = 0          # 트림(loss) 면적(mm^2)당 페널티. 폐기물 비용.
# ITEM_SINGLE_STRIP_PENALTIES = {}

class SheetOptimizeCa:
    """
    쉬트지 복합폭(Composite Width) 최적화 클래스.
    
    이 클래스는 여러 지폭을 조합한 '복합 아이템'을 기반으로 최적 절단 패턴을 찾습니다.
    일반 쉬트 최적화와 달리, 복합폭 개념을 통해 슬리터 칼 배치를 더 유연하게 처리합니다.
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
            b_wgt=None,
            color=None,
            p_type=None,
            p_wgt=None,
            p_color=None,
            p_machine=None,
            coating_yn=None,
            df_spec_pre=None,
            min_width=None,
            max_width=None,
            max_pieces=None,
            min_sheet_roll_length=None,
            max_sheet_roll_length=None,
            std_roll_cnt=None,
            sheet_trim=None,
            min_sc_width=None,
            max_sc_width=None,
            min_cm_width=None,
            max_cm_width=None,
            max_sl_count=None,
            ww_trim_size=None,
            ww_trim_size_sheet=None,
            num_threads=4,
            double_cutter='N' # [New] 복합폭 생성 옵션 (Y: 이종규격 허용, N: 동일규격만 허용)
    ):
        """
        SheetOptimizeCa 생성자.
        """
        # 저장
        self.df_orders = df_spec_pre.copy()
        
        self.double_cutter = double_cutter # [New]
        self.max_width = max_width
        self.min_width = min_width
        self.max_pieces = max_pieces
        self.b_wgt = b_wgt
        self.min_sheet_roll_length = min_sheet_roll_length
        self.max_sheet_roll_length = max_sheet_roll_length
        self.sheet_trim = sheet_trim
        self.min_sc_width = min_sc_width
        self.max_sc_width = max_sc_width
        
        self.color = color
        self.paper_type = paper_type
        self.p_type = p_type
        self.p_wgt = p_wgt
        self.p_color = p_color
        self.p_machine = p_machine
        self.coating_yn = coating_yn
        self.min_cm_width = min_cm_width
        self.max_cm_width = max_cm_width
        self.max_sl_count = max_sl_count
        self.std_roll_cnt = std_roll_cnt
        self.ww_trim_size = ww_trim_size
        self.ww_trim_size_sheet = ww_trim_size_sheet
        
        self.db = db
        self.lot_no = lot_no
        self.version = version
        self.num_threads = num_threads

        # 주문 데이터의 '가로' 컬럼을 '지폭'으로 복사 (내부 처리용)
        self.df_orders['지폭'] = self.df_orders['가로']
        
        self.original_max_width = max_width
        self.min_pieces = MIN_PIECES_PER_PATTERN

        # [Constraint] User Request: Disallowed Combinations (List of tuples)
        # 예: [(635, 636), (648, 636)] -> 635와 636 혼합 금지
        self.disallowed_combinations = [(635, 636)]

        # 주문량을 미터 및 롤 수 단위로 변환하여 수요 계산
        self.df_orders, self.demands_in_meters, self.order_sheet_lengths, self.demands_in_rolls = self._calculate_demand_meters(df_spec_pre)
        self.order_widths = list(self.demands_in_meters.keys())  # 고유 주문 지폭 목록

        # 지폭별 주문 톤수 요약 정보 생성 (리포팅용)
        width_summary = {}
        tons_per_width = self.df_orders.groupby('지폭')['주문톤'].sum()
        for width, required_meters in self.demands_in_meters.items():
            order_tons = tons_per_width.get(width, 0)
            width_summary[width] = {'order_tons': order_tons}
        self.width_summary = width_summary

        # 복합 아이템 생성: 여러 지폭을 조합하여 슬리터 칼로 한 번에 자를 수 있는 단위
        self.items, self.item_info, self.item_composition = self._prepare_items(min_sc_width, max_sc_width)

        logging.info(f"--- 패턴 제약조건: 최소 {self.min_pieces}폭, 최대 {self.max_pieces}폭 ---")

        # 패턴 저장소 초기화 (외부에서 패턴을 주입해야 함)
        self.patterns = []

    def _is_pattern_valid(self, current_pattern):
        """
        패턴의 유효성을 검사합니다. 특히 금지된 조합이 포함되어 있는지 확인합니다.
        
        Args:
            current_pattern (dict): {item_name: count, ...}
            
        Returns:
            bool: True if valid, False if invalid (contains disallowed combination)
        """
        if not self.disallowed_combinations:
            return True
            
        # 1. 현재 패턴에 포함된 모든 '기본 지폭' 수집 (Composite Item 내부 포함)
        all_base_widths = set()
        for it_name in current_pattern:
            if it_name in self.item_composition:
                 all_base_widths.update(self.item_composition[it_name].keys())
        
        # 2. 금지된 조합 확인
        for width1, width2 in self.disallowed_combinations:
            if width1 in all_base_widths and width2 in all_base_widths:
                return False
                
        return True

    def _prepare_items(self, min_sc_width, max_sc_width):
        """
        복합 아이템(Composite Items)을 생성합니다.
        
        복합 아이템이란 하나 이상의 기본 지폭을 조합하여 슬리터 칼로
        한 번에 절단할 수 있는 단위를 말합니다.
        
        생성되는 아이템 유형:
        1. 단일 지폭 복합: "710x2" → 710mm 2개 = 1420mm + trim
        2. 혼합 지폭 복합: "710x1+850x1" → 710mm 1개 + 850mm 1개 = 1560mm + trim (double_cutter='Y'일 때만)
        
        [Mod] double_cutter='N'일 때는 (지폭, 세로) 튜플을 키로 사용하여
        세로가 다른 주문을 별도 아이템으로 분리합니다.
        
        Args:
            min_sc_width (int): 슬리터 칼 최소 폭 제약 (이 값 이상이어야 유효)
            max_sc_width (int): 슬리터 칼 최대 폭 제약 (이 값 이하여야 유효)
        
        Returns:
            tuple: (items, item_info, item_composition)
                - items: 복합 아이템 이름 목록 (예: ["710x1", "710x2", "710x1+850x1"])
                - item_info: {아이템명: 총 폭(mm)} 매핑
                - item_composition: {아이템명: {기본키: 개수}} 구성 정보
                  - double_cutter='N': 기본키 = (지폭, 세로) 튜플
                  - double_cutter='Y': 기본키 = 지폭
        """
        from itertools import combinations_with_replacement
        
        items = []
        item_info = {}  # item_name -> 총 폭 (mm)
        item_composition = {}  # composite_item_name -> {original_key: count}
        
        # 하나의 복합 아이템에 포함될 수 있는 최대 기본 지폭 개수
        # [Mod] max_sl_count가 있으면 사용, 없으면 기본값 4
        max_pieces_in_composite = self.max_sl_count if self.max_sl_count and self.max_sl_count > 0 else 4 

        # ============================================================
        # Step 1: 단일 지폭 복합 아이템 생성 (같은 지폭 N개 조합)
        # 예: 710mm x 1 = 710mm, 710mm x 2 = 1420mm, ...
        # ============================================================
        
        # [Mod] Trim 계산 로직: Coating Y이면 (sheet_trim + ww_trim_size_sheet), 아니면 sheet_trim
        if self.coating_yn == 'Y':
             effective_trim = (self.sheet_trim or 0) + (self.ww_trim_size_sheet or 0)
        else:
             effective_trim = self.sheet_trim

        # [Mod] double_cutter='N'일 때는 order_widths가 (지폭, 세로) 튜플임
        for key in self.order_widths:
            # 키에서 지폭 추출
            if isinstance(key, tuple):
                width, sheet_length = key
            else:
                width = key
                sheet_length = self.order_sheet_lengths.get(key, 0)
            
            for i in range(1, max_pieces_in_composite + 1):
                # 복합폭 계산: (기본 지폭 × 개수) + 트림 손실
                base_width = width * i + effective_trim
                
                # [Mod] Same Spec Items (Step 1) check:
                # User Request (Implicit): 939x1 (939mm) should be valid if it meets slitter min (500mm),
                # even if min_cm (1000mm) is higher.
                # So we ONLY check min_sc_width here, and skip min_cm_width unless strict enforcement is needed.
                # Assuming min_cm_width is primarily for "Mixed" composites stability.
                pass
                
                # [Original Code removed]
                # if self.min_cm_width is not None and self.max_cm_width is not None:
                #    if not (self.min_cm_width <= base_width <= self.max_cm_width):
                #        continue

                # 슬리터 칼 제약조건 체크
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                # [Mod] 아이템 명명: double_cutter='N'일 때는 세로 포함
                if isinstance(key, tuple):
                    item_name = f"{width}_{sheet_length}x{i}"
                else:
                    item_name = f"{width}x{i}"
                    
                if base_width <= self.original_max_width:
                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = base_width
                        item_composition[item_name] = {key: i}

        # ============================================================
        # Step 2: 혼합 지폭 복합 아이템 생성 (다른 지폭 조합)
        # 예: 710mm + 850mm = 1560mm
        # [Mod] double_cutter='Y'일 때만 허용
        # [Mod] 세로 길이가 달라도 조합 가능하도록 수정
        # ============================================================
        if self.double_cutter == 'Y':
            # 모든 지폭(order_widths) 키를 대상으로 혼합 조합 생성
            all_keys = list(self.order_widths)
            
            for i in range(2, max_pieces_in_composite + 1):
                # 중복 조합(combinations_with_replacement) 생성
                for combo in combinations_with_replacement(all_keys, i):
                    # 단일 지폭만으로 구성된 조합은 Step 1에서 이미 처리됨
                    if len(set(combo)) == 1:
                        continue

                    # 복합폭 계산: 모든 지폭 합계 + 트림 손실
                    # [Mod] 키가 튜플이면 첫 번째 요소(지폭)만 합산
                    combo_widths = [k[0] if isinstance(k, tuple) else k for k in combo]
                    base_width = sum(combo_widths) + effective_trim
                    
                    # [New] CM (Composite) 폭 제약 확인
                    if self.min_cm_width is not None and self.max_cm_width is not None:
                        if not (self.min_cm_width <= base_width <= self.max_cm_width):
                            continue

                    # 슬리터 칼 제약조건 체크
                    if not (min_sc_width <= base_width <= max_sc_width):
                        continue

                    if base_width <= self.original_max_width:
                        # 조합 구성 카운팅 (예: ((710,545), (710,545), (850,545)) → {(710,545): 2, (850,545): 1})
                        comp_counts = Counter(combo)
                        # 아이템 명명: 지폭 내림차순 정렬된 "지폭_세로x개수" 조합
                        if isinstance(combo[0], tuple):
                            # 튜플 키: 지폭_세로x개수+지폭_세로x개수 형식
                            item_name = "+".join([f"{k[0]}_{k[1]}x{c}" for k, c in sorted(comp_counts.items(), key=lambda x: x[0][0] if isinstance(x[0], tuple) else x[0], reverse=True)])
                        else:
                            item_name = "+".join([f"{w}x{c}" for w, c in sorted(comp_counts.items(), key=lambda x: x[0], reverse=True)])

                        if item_name not in items:
                            items.append(item_name)
                            item_info[item_name] = base_width
                            item_composition[item_name] = dict(comp_counts)

        return items, item_info, item_composition

    def _calculate_demand_meters(self, df_orders):
        """
        쉬트지 주문량(톤)을 필요 생산 길이(미터)로 변환합니다.
        
        변환 공식:
        1. 장당 무게(g) = 평량(g/m²) × 가로(m) × 세로(m)
        2. 필요 장수 = 주문톤(ton) × 1,000,000(g/ton) ÷ 장당무게(g)
        3. 필요 길이(m) = 필요 장수 × 세로(m)
        
        [Mod] double_cutter='N'일 때는 지폭+세로 조합을 키로 사용하여
        세로가 다른 주문을 별도로 분리합니다.
        
        Args:
            df_orders (pd.DataFrame): 주문 데이터 (가로, 세로, 주문톤 컬럼 필수)
        
        Returns:
            tuple: (df_copy, demand_meters, order_sheet_lengths)
                - df_copy: meters 컬럼이 추가된 주문 DataFrame
                - demand_meters: {키: 필요미터} 딕셔너리
                  - double_cutter='N': 키 = (지폭, 세로) 튜플
                  - double_cutter='Y': 키 = 지폭
                - order_sheet_lengths: {키: 세로길이} 딕셔너리
        """
        df_copy = df_orders.copy()

        def calculate_meters(row):
            """개별 주문 행에 대한 필요 생산 길이(m)를 계산합니다."""
            width_mm = row.get('지폭', row.get('width', 0))    # 지폭 (mm)
            length_mm = row.get('세로', row.get('length', 0))   # 장당 세로 길이 (mm)
            order_ton = row.get('주문톤', row.get('order_ton_cnt', 0))  # 주문량 (톤)

            # 유효성 검사: 0 이하 값이 있으면 계산 불가
            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0 or order_ton <= 0:
                return 0

            # 장당 무게 계산 (그램)
            # 평량(g/m²) × 가로(mm→m) × 세로(mm→m)
            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1000000
            if sheet_weight_g <= 0:
                return 0

            # 필요 장수 = 주문톤(g으로 변환) ÷ 장당무게(g)
            total_sheets_needed = (order_ton * 1000000) / sheet_weight_g
            # 필요 미터 = 장수 × 세로(mm→m)
            total_meters_needed = total_sheets_needed * (length_mm / 1000)
            return total_meters_needed

        # 컬럼명 소문자 변환 (DB에서 대문자로 넘어오는 경우 대응)
        df_copy.columns = [c.lower() for c in df_copy.columns]

        # 영문 컬럼명을 한글로 매핑
        rename_map = {
            'width': '지폭', 
            'length': '세로', 
            'order_ton_cnt': '주문톤',
            '가로': '지폭' # 가로가 들어올 경우도 대비
        }
        df_copy = df_copy.rename(columns=rename_map)
            
        # 각 주문 행에 대해 필요 미터 계산
        df_copy['meters'] = df_copy.apply(calculate_meters, axis=1)
        
        # [Debug] double_cutter 값 확인
        logging.info(f"[DEBUG] double_cutter value: '{self.double_cutter}' (type: {type(self.double_cutter).__name__})")
        
        # [Mod] 세로가 다른 주문은 반드시 별도 패턴으로 분리해야 함
        # (롤 길이가 세로 길이에 따라 달라지기 때문)
        # 따라서 항상 (지폭, 세로) 튜플을 키로 사용
        df_copy['demand_key'] = list(zip(df_copy['지폭'].astype(int), df_copy['세로'].astype(int)))
        demand_meters = df_copy.groupby('demand_key')['meters'].sum().to_dict()
        order_sheet_lengths = df_copy.groupby('demand_key')['세로'].first().to_dict()
        
        # [New] 롤 수 기반 수요 계산 (표준 롤 길이로 나눔)
        std_roll_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2
        demand_rolls = {key: meters / std_roll_length for key, meters in demand_meters.items()}
        
        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        logging.info(f"--- (지폭, 세로)별 필요 총 길이 (표준롤길이: {std_roll_length}m) ---")
        logging.info("-" * 50)
        for key, meters in demand_meters.items():
            rolls = demand_rolls[key]
            logging.info(f"  {key}: {meters:.2f}m ({rolls:.2f}롤)")

        return df_copy, demand_meters, order_sheet_lengths, demand_rolls

    def _generate_all_patterns(self):
        """
        작은 문제에 대해 모든 가능한 패턴을 생성합니다 (Brute-force).
        
        주문 지폭 종류가 SMALL_PROBLEM_THRESHOLD 이하인 경우에 사용됩니다.
        재귀적으로 모든 유효한 복합폭 아이템 조합을 생성합니다.
        
        CA 버전 특징:
        - 패턴은 {composition: {...}, length: N, loss_per_roll: M} 형태로 저장
        - length는 min/max_sheet_roll_length 범위 내에서 결정
        """
        all_patterns = []
        seen_patterns = set()
        item_list = list(self.items)

        def find_combinations_recursive(start_index, current_pattern, current_width, current_pieces):
            """재귀적으로 유효한 패턴 조합을 탐색합니다."""
            # 유효한 패턴인지 확인
            if self.min_width <= current_width <= self.max_width and self.min_pieces <= current_pieces <= self.max_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in seen_patterns:
                    # [Constraint] User Request: Check disallowed combinations
                    if not self._is_pattern_valid(current_pattern):
                         seen_patterns.add(pattern_key) 
                    else:
                        # CA 버전: 패턴 길이는 min/max 범위의 중간값 사용
                        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2
                        loss_per_roll = self.max_width - current_width
                        
                        all_patterns.append({
                            'composition': current_pattern.copy(),
                            'length': pattern_length,
                            'loss_per_roll': loss_per_roll
                        })
                        seen_patterns.add(pattern_key)

            # 종료 조건
            if current_pieces >= self.max_pieces or start_index >= len(item_list):
                return

            # 현재 아이템을 포함하지 않고 다음으로 넘어감
            find_combinations_recursive(start_index + 1, current_pattern, current_width, current_pieces)

            # 현재 아이템을 포함하여 재귀 호출
            item = item_list[start_index]
            item_width = self.item_info[item]
            if current_width + item_width <= self.max_width:
                current_pattern[item] = current_pattern.get(item, 0) + 1
                find_combinations_recursive(start_index, current_pattern, current_width + item_width, current_pieces + 1)
                current_pattern[item] -= 1
                if current_pattern[item] == 0:
                    del current_pattern[item]

        find_combinations_recursive(0, {}, 0, 0)
        self.patterns = all_patterns
        logging.info(f"--- 전체 탐색으로 {len(self.patterns)}개의 패턴 생성됨 ---")

    def _generate_initial_patterns_db(self):
        """th_pattern_tot_sheet 테이블의 사용자 편집 패턴 데이터를 활용하여 초기 패턴을 생성합니다 (CA 버전)."""
        if not self.db or not self.lot_no:
            logging.info("--- DB 정보가 없어 기존 패턴을 불러올 수 없습니다. ---")
            return

        logging.info("\n--- DB(th_pattern_tot_sheet)에서 사용자 편집 패턴을 불러와 초기 패턴을 생성합니다. ---")
        # CA용 패턴 가져오기 메서드 호출
        db_patterns_list = self.db.get_sheet_ca_patterns_from_db(self.lot_no)

        if not db_patterns_list:
            logging.info("--- DB에 저장된 사용자 편집 패턴이 없거나, 현재 오더와 일치하는 패턴이 없습니다. ---")
            return

        logging.info(f"--- 현재 생성된 유효 아이템 목록 (총 {len(self.items)}개): {self.items[:20]} ... ---")
        
        initial_patterns_from_db = []
        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2

        for pattern_item_list in db_patterns_list:
            pattern_dict = dict(Counter(pattern_item_list))
            
            # DB 패턴 아이템 복구 및 검증
            all_items_valid = True
            
            for item_name in pattern_dict.keys():
                if item_name in self.items:
                    continue
                
                # 아이템 복구 시도
                is_recovered = False
                try:
                    # 복합폭 파싱 (예: "710x1+850x1" 또는 "710x2")
                    sub_items = item_name.split('+')
                    total_width = 0
                    composition = {}
                    
                    valid_sub_items = True
                    for sub in sub_items:
                        # "710x2" 형식 파싱
                        if 'x' not in sub:
                            valid_sub_items = False
                            break
                        w_str, c_str = sub.split('x')
                        w = int(w_str)
                        c = int(c_str)
                        
                        # 지폭이 현재 주문에 존재하는지 확인
                        if w not in self.order_widths:
                            valid_sub_items = False
                            break
                        
                        total_width += w * c
                        composition[w] = composition.get(w, 0) + c
                    
                    if valid_sub_items:
                        # 트림 포함하여 총 폭 계산 (단, 710x1+850x1은 트림이 한 번만 포함됨)
                        # [Mod] Trim 계산 로직 적용
                        if self.coating_yn == 'Y':
                             effective_trim = (self.ww_trim_size or 0) + (self.ww_trim_size_sheet or 0)
                        else:
                             effective_trim = self.sheet_trim
                        
                        composite_width = total_width + effective_trim
                        
                        # 슬리터 칼 제약 및 최대 폭 제약 확인
                        # [Mod] CM (Composite) 폭 제약 추가 확인
                        is_cm_valid = True
                        if self.min_cm_width is not None and self.max_cm_width is not None:
                            if not (self.min_cm_width <= composite_width <= self.max_cm_width):
                                is_cm_valid = False
                        
                        if is_cm_valid and self.min_sc_width <= composite_width <= self.max_sc_width and composite_width <= self.original_max_width:
                             # 아이템 등록
                             self.items.append(item_name)
                             self.item_info[item_name] = composite_width
                             self.item_composition[item_name] = composition
                             is_recovered = True
                             logging.info(f"    -> [Recover] DB 패턴 아이템 복구: {item_name} (폭: {composite_width}mm)")
                except Exception as e:
                    logging.warning(f"    - [Error] DB 아이템 {item_name} 파싱/복구 실패: {e}")
                
                if not is_recovered:
                    all_items_valid = False
                    break
            
            if all_items_valid:
                # 패턴 전체 유효성 (총 너비 등) 확인
                current_total_width = sum(self.item_info[name] * count for name, count in pattern_dict.items())
                current_total_pieces = sum(pattern_dict.values()) 
                 
                if self.min_width <= current_total_width <= self.max_width and self.min_pieces <= current_total_pieces <= self.max_pieces:
                    # 패턴 구조체 생성
                    loss_per_roll = self.max_width - current_total_width
                    new_pat = {
                        'composition': pattern_dict,
                        'length': pattern_length,
                        'loss_per_roll': loss_per_roll
                    }
                    initial_patterns_from_db.append(new_pat)

        if initial_patterns_from_db:
            seen_patterns = {frozenset(p['composition'].items()) for p in self.patterns}
            added_count = 0
            for pat in initial_patterns_from_db:
                key = frozenset(pat['composition'].items())
                if key not in seen_patterns:
                    self.patterns.append(pat)
                    seen_patterns.add(key)
                    added_count += 1
            logging.info(f"--- DB에서 {added_count}개의 사용자 편집 패턴을 추가했습니다. ---")

    def _generate_initial_patterns(self):
        """
        휴리스틱을 사용하여 초기 패턴을 생성합니다.
        
        다양한 정렬 전략(수요순, 너비순, 랜덤)을 사용하여 First-Fit 방식으로
        초기 패턴 집합을 생성합니다. Column Generation의 시작점으로 사용됩니다.
        """
        seen_patterns = set()
        
        # 1. 다양한 휴리스틱을 위한 정렬된 아이템 리스트 생성
        sorted_by_demand = sorted(
            self.items,
            key=lambda i: self.demands_in_meters.get(list(self.item_composition[i].keys())[0], 0),
            reverse=True
        )
        sorted_by_demand_asc = sorted(
            self.items,
            key=lambda i: self.demands_in_meters.get(list(self.item_composition[i].keys())[0], 0),
            reverse=False
        )
        sorted_by_width_desc = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda i: self.item_info[i], reverse=False)

        # 2. Random Shuffles
        random.seed(42)
        random_shuffles = []
        for _ in range(5):
            items_copy = list(self.items)
            random.shuffle(items_copy)
            random_shuffles.append(items_copy)

        heuristics = [
            sorted_by_demand, 
            sorted_by_width_desc, 
            sorted_by_width_asc, 
            sorted_by_demand_asc
        ] + random_shuffles

        # CA 버전: 패턴 길이 기본값
        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2

        # 3. 각 휴리스틱에 대해 First-Fit과 유사한 패턴 생성
        for sorted_items in heuristics:
            for item in sorted_items:
                item_width = self.item_info[item]
                
                current_pattern = {item: 1}
                current_width = item_width
                current_pieces = 1

                # max_pieces에 도달할 때까지 아이템 추가
                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    
                    # [Constraint] User Request: Check disallowed combinations via helper
                    def is_valid_combination(candidate_item, current_pat):
                        # 임시 패턴 구성
                        temp_pat = current_pat.copy()
                        temp_pat[candidate_item] = temp_pat.get(candidate_item, 0) + 1
                        return self._is_pattern_valid(temp_pat)

                    # 남은 공간에 맞는 가장 큰 아이템을 찾음 (First-Fit) + 금지 조합 확인
                    best_fit_item = next((i for i in sorted_items if self.item_info[i] <= remaining_width and is_valid_combination(i, current_pattern)), None)
                    
                    if not best_fit_item:
                        break 

                    current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
                    current_width += self.item_info[best_fit_item]
                    current_pieces += 1

                # 최종 유효성 검사 후 패턴 추가
                if self.min_width <= current_width and self.min_pieces <= current_pieces:
                    pattern_key = frozenset(current_pattern.items())
                    if pattern_key not in seen_patterns:
                        loss_per_roll = self.max_width - current_width
                        self.patterns.append({
                            'composition': current_pattern.copy(),
                            'length': pattern_length,
                            'loss_per_roll': loss_per_roll
                        })
                        seen_patterns.add(pattern_key)

        # 4. 모든 복합폭에 대해 '순수 품목 패턴' 생성
        for item in sorted_by_width_asc:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0:
                continue

            num_items = min(int(self.max_width / item_width), self.max_pieces)
            
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                
                if self.min_width <= total_width and self.min_pieces <= num_items:
                    pattern_key = frozenset(new_pattern.items())
                    if pattern_key not in seen_patterns:
                        loss_per_roll = self.max_width - total_width
                        self.patterns.append({
                            'composition': new_pattern.copy(),
                            'length': pattern_length,
                            'loss_per_roll': loss_per_roll
                        })
                        seen_patterns.add(pattern_key)
                        break
                
                num_items -= 1

        logging.info(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")

    def _solve_master_problem_ilp(self, is_final_mip=False):
        """
        마스터 문제(Master Problem)를 선형계획법(LP) 또는 정수계획법(MIP)으로 해결합니다.
        
        CA 버전 특징:
        - 수요 단위: 미터(m) 기반 (demands_in_meters 사용)
        - 생산량 계산: 복합폭 내 기본 지폭 분해 후 패턴 길이 곱함
        
        Args:
            is_final_mip (bool): True이면 정수해, False이면 LP 완화해 + Dual Value
        
        Returns:
            dict: 최적화 결과 (pattern_counts, over_production, under_production, duals)
        """
        # 1. [Final MIP] Try Gurobi Direct Solver
        if is_final_mip:
            try:
                logging.info("Trying Gurobi Direct Solver SheetOptimizeCA (gurobipy)...")
                model = gp.Model("SheetOptimizationCA")
                model.setParam("OutputFlag", 0)
                model.setParam("LogToConsole", 0)
                if hasattr(self, 'num_threads'):
                    model.setParam("Threads", self.num_threads)
                
                model.setParam("TimeLimit", SOLVER_TIME_LIMIT_MS / 1000.0)
                model.setParam("MIPFocus", 0) 

                # Variables: x (Pattern Counts)
                x = {}
                for j in range(len(self.patterns)):
                    x[j] = model.addVar(vtype=GRB.INTEGER, name=f'P_{j}')
                
                # Variables: Over/Under Production
                # [Mod] 롤 수 기반 과/부족 변수 (미터 대신 롤 수 단위)
                over_prod_vars = {}
                for w in self.demands_in_rolls:
                    over_prod_vars[w] = model.addVar(vtype=GRB.CONTINUOUS, name=f'Over_{w}')
                
                under_prod_vars = {}
                for width, required_rolls in self.demands_in_rolls.items():
                    allowed_under = max(0.1, required_rolls * 0.1)  # 10% tolerance in rolls
                    under_prod_vars[width] = model.addVar(lb=0, ub=allowed_under, vtype=GRB.CONTINUOUS, name=f'Under_{width}')

                # Variables: Pattern Usage (Binary) for Count Penalty
                y = {}
                for j in range(len(self.patterns)):
                    y[j] = model.addVar(vtype=GRB.BINARY, name=f'Use_{j}')
                
                # Big-M Constraints: x[j] <= M * y[j]
                M = 1000
                for j in range(len(self.patterns)):
                    model.addConstr(x[j] <= M * y[j], name=f'Link_{j}')

                # [Mod] Constraints: Demand (롤 수 기반)
                # 생산 롤 수 + 부족 롤 수 = 필요 롤 수 + 과잉 롤 수
                for width, required_rolls in self.demands_in_rolls.items():
                    # 각 패턴에서 해당 지폭의 생산 롤 수 계산
                    # 패턴 j의 x[j] = 패턴 j 선택 롤 수
                    # 패턴 j가 해당 지폭을 생산하면 해당 지폭의 롤 수에 기여
                    total_width_prod_expr = gp.quicksum(
                        x[j] * sum(
                            count * self.item_composition[item_name].get(width, 0)
                            for item_name, count in self.patterns[j]['composition'].items()
                        )
                        for j in range(len(self.patterns))
                    )
                    model.addConstr(total_width_prod_expr + under_prod_vars[width] == required_rolls + over_prod_vars[width], name=f'demand_{width}')

                # [Mod] Objective: 롤 수 기반 페널티
                total_rolls = gp.quicksum(x[j] for j in range(len(self.patterns)))
                total_over_prod_penalty = gp.quicksum(OVER_PROD_PENALTY * var for var in over_prod_vars.values())
                total_under_prod_penalty = gp.quicksum(UNDER_PROD_PENALTY * var for var in under_prod_vars.values())
                
                total_complexity_penalty = gp.quicksum(
                    PATTERN_COMPLEXITY_PENALTY * max(0, len(self.patterns[j]['composition']) - 1) * x[j]
                    for j in range(len(self.patterns))
                )
                
                total_pattern_count_penalty = gp.quicksum(
                    PATTERN_COUNT_PENALTY * y[j] for j in range(len(self.patterns))
                )
                
                # [New] 단폭(x1) 아이템 사용 페널티
                # 패턴 내 단폭(x1) 아이템 개수를 세어 페널티 부여
                # 예: "636x1" -> 단폭 1개 -> 페널티 1 * SINGLE_STRIP_PENALTY
                # 예: "636x2" -> 단폭 0개 -> 페널티 없음
                # 예: "1031_670x1+530_780x1" -> 혼합 복합롤 -> 페널티 제외
                def count_single_strips(pattern_composition):
                    """패턴 내 단폭(x1) 아이템의 총 개수를 반환 (혼합 복합롤 제외)"""
                    single_count = 0
                    for item_name, item_count in pattern_composition.items():
                        # '+' 가 포함되어 있으면 혼합 복합롤이므로 단폭 페널티 제외
                        if '+' in item_name:
                            continue
                        # 단일 아이템이 x1으로 끝나면 단폭으로 카운트
                        if item_name.endswith('x1'):
                            single_count += item_count
                    return single_count
                
                total_single_strip_penalty = gp.quicksum(
                    SINGLE_STRIP_PENALTY * count_single_strips(self.patterns[j]['composition']) * x[j]
                    for j in range(len(self.patterns))
                )
                
                # [New] 다른 세로 길이 조합 페널티
                # 패턴 내 아이템들의 세로 길이가 다르면 페널티 부여
                def count_mixed_sheet_lengths(pattern_composition):
                    """패턴 내 서로 다른 세로 길이 종류 수를 반환 (1이면 동일, 2이상이면 혼합)"""
                    sheet_lengths = set()
                    for item_name, item_count in pattern_composition.items():
                        if item_count > 0:
                            # 아이템명에서 세로 길이 추출
                            # 형식: "788_545x2" 또는 "788x2"
                            sub_items = item_name.split('+')
                            for sub in sub_items:
                                if '_' in sub:
                                    # "788_545x2" 형식
                                    parts = sub.split('_')
                                    sheet_length = int(parts[1].split('x')[0])
                                    sheet_lengths.add(sheet_length)
                                else:
                                    # "788x2" 형식 (세로 정보 없음)
                                    sheet_lengths.add(0)
                    # 다른 세로 길이 종류 수 (1이면 동일, 2이상이면 혼합)
                    return max(0, len(sheet_lengths) - 1)
                
                total_mixed_sheet_length_penalty = gp.quicksum(
                    MIXED_SHEET_LENGTH_PENALTY * count_mixed_sheet_lengths(self.patterns[j]['composition']) * x[j]
                    for j in range(len(self.patterns))
                )
                
                model.setObjective(
                    total_rolls + total_over_prod_penalty + total_under_prod_penalty +
                    total_complexity_penalty +
                    total_pattern_count_penalty +
                    total_single_strip_penalty +
                    total_mixed_sheet_length_penalty,
                    GRB.MINIMIZE
                )

                model.optimize()

                if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                    status_msg = "Optimal" if model.Status == GRB.OPTIMAL else "Feasible (TimeLimit)"
                    logging.info(f"Using solver: GUROBI for Final MIP (Success: {status_msg}, Obj={model.ObjVal})")
                    
                    # [Debug] Objective 구성 요소 출력
                    logging.info("--- [DEBUG] Objective 구성 요소 분석 ---")
                    
                    # 1. 총 롤 수
                    total_rolls_val = sum(x[j].X for j in range(len(self.patterns)))
                    logging.info(f"  1. 총 롤 수: {total_rolls_val:.2f}")
                    
                    # 2. 패턴별 선택 롤 수
                    logging.info("  2. 패턴별 선택 롤 수:")
                    for j in range(len(self.patterns)):
                        if x[j].X > 0.01:
                            logging.info(f"     P{j}: {x[j].X:.2f}롤 | {self.patterns[j]['composition']}")
                    
                    # 3. 과잉/부족 생산
                    over_val = sum(over_prod_vars[w].X for w in over_prod_vars)
                    under_val = sum(under_prod_vars[w].X for w in under_prod_vars)
                    logging.info(f"  3. 과잉 생산 (롤 수): {over_val:.4f} × {OVER_PROD_PENALTY} = {over_val * OVER_PROD_PENALTY:.2f}")
                    logging.info(f"  4. 부족 생산 (롤 수): {under_val:.4f} × {UNDER_PROD_PENALTY} = {under_val * UNDER_PROD_PENALTY:.2f}")
                    for w in over_prod_vars:
                        if over_prod_vars[w].X > 0.001 or under_prod_vars[w].X > 0.001:
                            logging.info(f"     {w}: 과잉 {over_prod_vars[w].X:.4f}롤, 부족 {under_prod_vars[w].X:.4f}롤")
                    
                    # 4. 패턴 종류 수 페널티
                    pattern_count_val = sum(1 for j in range(len(self.patterns)) if x[j].X > 0.01)
                    logging.info(f"  5. 패턴 종류 수: {pattern_count_val}개 × {PATTERN_COUNT_PENALTY} = {pattern_count_val * PATTERN_COUNT_PENALTY:.2f}")
                    
                    # 5. 복잡도 페널티
                    complexity_val = sum(
                        PATTERN_COMPLEXITY_PENALTY * max(0, len(self.patterns[j]['composition']) - 1) * x[j].X
                        for j in range(len(self.patterns))
                    )
                    logging.info(f"  6. 복잡도 페널티: {complexity_val:.2f}")
                    
                    # 6. 단폭 페널티
                    single_strip_val = sum(
                        SINGLE_STRIP_PENALTY * count_single_strips(self.patterns[j]['composition']) * x[j].X
                        for j in range(len(self.patterns))
                    )
                    logging.info(f"  7. 단폭(x1) 페널티: {single_strip_val:.2f}")
                    
                    # 7. 다른 세로 혼합 페널티
                    mixed_length_val = sum(
                        MIXED_SHEET_LENGTH_PENALTY * count_mixed_sheet_lengths(self.patterns[j]['composition']) * x[j].X
                        for j in range(len(self.patterns))
                    )
                    logging.info(f"  8. 다른 세로 혼합 페널티: {mixed_length_val:.2f}")
                    
                    # 합계
                    calculated_obj = (total_rolls_val + 
                                     over_val * OVER_PROD_PENALTY + 
                                     under_val * UNDER_PROD_PENALTY + 
                                     pattern_count_val * PATTERN_COUNT_PENALTY +
                                     complexity_val + single_strip_val + mixed_length_val)
                    logging.info(f"  ===== 계산된 Objective: {calculated_obj:.2f} (Gurobi: {model.ObjVal:.2f}) =====")
                    
                    solution = {
                        'objective': model.ObjVal,
                        'pattern_counts': {j: x[j].X for j in range(len(self.patterns))},
                        'over_production': {w: over_prod_vars[w].X for w in over_prod_vars},
                        'under_production': {w: under_prod_vars[w].X for w in under_prod_vars}
                    }
                    return solution
                else:
                    logging.warning(f"Gurobi failed (Status={model.Status}). Fallback to SCIP.")

            except Exception as e:
                logging.warning(f"Gurobi execution failed: {e}. Fallback to SCIP.")

        # 2. [Fallback/Default] OR-Tools Solver (SCIP or GLOP)
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        
        if hasattr(solver, 'SetNumThreads'):
            solver.SetNumThreads(self.num_threads)

        if is_final_mip:
            solver.SetTimeLimit(SOLVER_TIME_LIMIT_MS)

        # 변수 정의: 각 패턴의 사용 횟수
        x = {}
        for j in range(len(self.patterns)):
            if is_final_mip:
                x[j] = solver.IntVar(0, solver.infinity(), f'P_{j}')
            else:
                x[j] = solver.NumVar(0, solver.infinity(), f'P_{j}')

        # 과생산/부족생산 변수
        over_prod_vars = {w: solver.NumVar(0, solver.infinity(), f'Over_{w}') for w in self.demands_in_meters}
        under_prod_vars = {}
        for width, required_meters in self.demands_in_meters.items():
            allowed_under = max(1, math.ceil(required_meters * 0.1))  # 최대 10% 부족 허용
            under_prod_vars[width] = solver.NumVar(0, allowed_under, f'Under_{width}')

        # 제약조건: 생산량 + 부족량 = 수요량 + 과생산량
        constraints = {}
        for width, required_meters in self.demands_in_meters.items():
            # 각 패턴이 해당 지폭에 대해 생산하는 미터 계산
            production_for_width = solver.Sum(
                x[j] * sum(
                    self.item_composition[item_name].get(width, 0) * count
                    for item_name, count in self.patterns[j]['composition'].items()
                ) * self.patterns[j]['length']
                for j in range(len(self.patterns))
            )
            constraints[width] = solver.Add(
                production_for_width + under_prod_vars[width] == required_meters + over_prod_vars[width],
                f'demand_{width}'
            )

        # 목적함수: 총 롤 수 + 페널티 최소화
        total_rolls = solver.Sum(x.values())
        total_over_prod_penalty = solver.Sum(OVER_PROD_PENALTY * var for var in over_prod_vars.values())
        total_under_prod_penalty = solver.Sum(UNDER_PROD_PENALTY * var for var in under_prod_vars.values())
        # [Mod] Complexity Penalty Refinement:
        # Group items by their underlying base widths.
        # e.g., '788x1' and '788x2' should both count towards the '788mm' base width group.
        # This prevents penalizing mixtures of x1/x2 variants of the same width.
        
        pattern_complexity_vars = []
        for j, pat in enumerate(self.patterns):
             distinct_base_widths = set()
             for item_name in pat['composition']:
                 if item_name in self.item_composition:
                     # Add all original base widths (e.g., 788, 710) for this item
                     distinct_base_widths.update(self.item_composition[item_name].keys())
             
             # Penalty based on number of distinct base width types - 1
             complexity_score = max(0, len(distinct_base_widths) - 1)
             pattern_complexity_vars.append(complexity_score)

        total_complexity_penalty = solver.Sum(
            PATTERN_COMPLEXITY_PENALTY * pattern_complexity_vars[j] * x[j]
            for j in range(len(self.patterns))
        )
        
        total_pattern_count_penalty = 0
        if is_final_mip:
            # [Constraint] User Request: Minimize total number of patterns used
            # 패턴 사용 여부 변수 (Binary)
            y = {j: solver.IntVar(0, 1, f'Use_{j}') for j in range(len(self.patterns))}
            
            # Big-M 제약조건: x[j] <= M * y[j]
            # M은 충분히 큰 수 (예: 1000, 롤 수가 1000개를 넘지 않는다고 가정)
            M = 1000
            for j in range(len(self.patterns)):
                solver.Add(x[j] <= M * y[j])
                
            total_pattern_count_penalty = solver.Sum(
                PATTERN_COUNT_PENALTY * y[j] for j in range(len(self.patterns))
            )

        # [New] 단폭(x1) 아이템 사용 페널티 (OR-Tools용)
        def count_single_strips_ortools(pattern_composition):
            """패턴 내 단폭(x1) 아이템의 총 개수를 반환 (혼합 복합롤 제외)"""
            single_count = 0
            for item_name, item_count in pattern_composition.items():
                # '+' 가 포함되어 있으면 혼합 복합롤이므로 단폭 페널티 제외
                if '+' in item_name:
                    continue
                # 단일 아이템이 x1으로 끝나면 단폭으로 카운트
                if item_name.endswith('x1'):
                    single_count += item_count
            return single_count
        
        total_single_strip_penalty = solver.Sum(
            SINGLE_STRIP_PENALTY * count_single_strips_ortools(self.patterns[j]['composition']) * x[j]
            for j in range(len(self.patterns))
        )

        solver.Minimize(
            total_rolls + total_over_prod_penalty + total_under_prod_penalty + 
            total_complexity_penalty +
            total_pattern_count_penalty +
            total_single_strip_penalty
        )
        
        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            solution = {
                'objective': solver.Objective().Value(),
                'pattern_counts': {j: var.solution_value() for j, var in x.items()},
                'over_production': {w: var.solution_value() for w, var in over_prod_vars.items()},
                'under_production': {w: var.solution_value() for w, var in under_prod_vars.items()}
            }
            # [Debug] Log Penalty Values (OR-Tools 버전 호환성을 위해 변수 값 직접 계산)
            logging.info(f"[DEBUG] Solver Objective: {solver.Objective().Value()}")
            
            # 변수 값 직접 합산 (구버전 OR-Tools 호환)
            debug_total_rolls = sum(var.solution_value() for var in x.values())
            debug_over_prod = sum(OVER_PROD_PENALTY * var.solution_value() for var in over_prod_vars.values())
            debug_under_prod = sum(UNDER_PROD_PENALTY * var.solution_value() for var in under_prod_vars.values())
            debug_complexity = sum(PATTERN_COMPLEXITY_PENALTY * pattern_complexity_vars[j] * x[j].solution_value() for j in range(len(self.patterns)))
            
            logging.info(f"[DEBUG] Total Rolls: {debug_total_rolls}")
            logging.info(f"[DEBUG] OverProd Penalty: {debug_over_prod}")
            logging.info(f"[DEBUG] UnderProd Penalty: {debug_under_prod}")
            logging.info(f"[DEBUG] Complexity Penalty: {debug_complexity}")
            if is_final_mip:
                debug_pattern_count = sum(PATTERN_COUNT_PENALTY * y[j].solution_value() for j in range(len(self.patterns)))
                logging.info(f"[DEBUG] Pattern Count Penalty: {debug_pattern_count}")
            
            if not is_final_mip:
                solution['duals'] = {w: constraints[w].dual_value() for w in self.demands_in_meters}
            return solution
        return None

    def _solve_subproblem_dp(self, duals):
        """
        서브 문제(Sub-problem)를 동적 계획법(DP)으로 해결합니다.
        
        Master Problem에서 얻은 Dual Value를 활용하여 Reduced Cost가 음수인
        새로운 유망 패턴 후보를 탐색합니다.
        
        Args:
            duals (dict): {지폭: dual_value} 딕셔너리
        
        Returns:
            list: 새로운 패턴 후보 리스트
        """
        width_limit = self.max_width
        piece_limit = self.max_pieces
        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2

        # item_details: (item_name, item_width, item_value)
        item_details = []
        for item_name in self.items:
            item_width = self.item_info[item_name]
            # CA 버전: 미터 기반 가치 계산 (dual * 패턴 길이)
            item_value = sum(
                count * duals.get(width, 0) * pattern_length
                for width, count in self.item_composition[item_name].items()
            )
            
            if item_value <= 0:
                continue
            item_details.append((item_name, item_width, item_value))

        if not item_details:
            return []

        # DP 테이블 초기화
        dp_value = [[float('-inf')] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_parent = [[None] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_value[0][0] = 0.0

        # DP 테이블 채우기
        for pieces in range(piece_limit + 1):
            for width in range(width_limit + 1):
                current_value = dp_value[pieces][width]
                if current_value == float('-inf'):
                    continue
                
                for item_name, item_width, item_value in item_details:
                    next_pieces = pieces + 1
                    next_width = width + item_width
                    
                    if next_pieces > piece_limit or next_width > width_limit:
                        continue
                    
                    new_value = current_value + item_value
                    if new_value > dp_value[next_pieces][next_width] + 1e-9:
                        dp_value[next_pieces][next_width] = new_value
                        dp_parent[next_pieces][next_width] = (pieces, width, item_name)

        # 후보 패턴 추출
        candidate_patterns = []
        seen_patterns = set()

        for pieces in range(self.min_pieces, piece_limit + 1):
            for width in range(self.min_width, width_limit + 1):
                value = dp_value[pieces][width]
                if value <= 1.0 + 1e-6:
                    continue
                
                parent = dp_parent[pieces][width]
                if not parent:
                    continue

                # 패턴 재구성
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

                if not pattern or cur_pieces != 0 or cur_width != 0:
                    continue

                pattern_key = frozenset(pattern.items())
                if pattern_key in seen_patterns:
                    continue

                total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                if total_width < self.min_width or total_width > self.max_width:
                    continue

                # [Constraint] User Request: Check disallowed combinations
                if not self._is_pattern_valid(pattern):
                     continue

                # Reduced Cost 계산
                reduced_cost = value - (1.0 + PIECE_COUNT_PENALTY * (pieces ** 2))

                seen_patterns.add(pattern_key)
                candidate_patterns.append({
                    'pattern': pattern,
                    'value': reduced_cost,
                    'width': total_width,
                    'pieces': pieces
                })

        if not candidate_patterns:
            return []
        
        candidate_patterns.sort(key=lambda x: x['value'], reverse=True)
        return [cand['pattern'] for cand in candidate_patterns[:CG_SUBPROBLEM_TOP_N]]

    def run_optimize(self, start_prod_seq=0):
        """
        최적화를 실행하고 결과를 반환합니다.
        
        자체적으로 패턴을 생성하고 Column Generation을 수행합니다.
        - Small Problem: 전체 패턴 탐색 (_generate_all_patterns)
        - Large Problem: 열 생성 기법 (_generate_initial_patterns + Column Generation)
        
        Args:
            start_prod_seq (int): 생산 시퀀스 시작 번호 (기본값: 0)
        
        Returns:
            dict: 최적화 결과
        """
        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2
        
        # 패턴이 외부에서 주입되지 않은 경우 자체 생성
        if not self.patterns:
            # DB에서 사용자 편집 패턴을 먼저 불러와 추가합니다.
            self._generate_initial_patterns_db()
            
            if len(self.order_widths) <= SMALL_PROBLEM_THRESHOLD:
                logging.info(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 모든 패턴을 탐색합니다 (Small-scale) ---")
                self._generate_all_patterns()
            else:
                logging.info(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 열 생성 기법을 시작합니다 (Large-scale) ---")
                self._generate_initial_patterns()
                
                if not self.patterns:
                    return {"error": "초기 유효 패턴을 생성할 수 없습니다. 제약조건이 너무 엄격할 수 있습니다."}

                # 열 생성 루프
                no_improvement_count = 0
                for iteration in range(CG_MAX_ITERATIONS):
                    master_solution = self._solve_master_problem_ilp()
                    if not master_solution or 'duals' not in master_solution:
                        break

                    new_patterns = self._solve_subproblem_dp(master_solution['duals'])
                    
                    patterns_added = 0
                    if new_patterns:
                        current_pattern_keys = {frozenset(p['composition'].items()) for p in self.patterns}
                        for new_pattern in new_patterns:
                            if frozenset(new_pattern.items()) not in current_pattern_keys:
                                pattern_width = sum(self.item_info[item] * count for item, count in new_pattern.items())
                                if pattern_width >= self.min_width:
                                    loss_per_roll = self.max_width - pattern_width
                                    self.patterns.append({
                                        'composition': new_pattern,
                                        'length': pattern_length,
                                        'loss_per_roll': loss_per_roll
                                    })
                                    patterns_added += 1
                    
                    if patterns_added > 0:
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    if no_improvement_count >= CG_NO_IMPROVEMENT_LIMIT:
                        logging.info(f"--- {CG_NO_IMPROVEMENT_LIMIT}번의 반복 동안 개선이 없어 수렴으로 간주하고 종료합니다 (반복 {iteration}). ---")
                        break

        if not self.patterns:
            return {"error": "유효한 패턴을 생성할 수 없습니다."}

        logging.info(f"--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        logging.info("--- 생성된 패턴 목록 ---")
        for idx, pattern in enumerate(self.patterns):
            logging.info(f"  P{idx}: {pattern['composition']} | length={pattern['length']} | loss={pattern['loss_per_roll']}")
        
        # [Debug] Check Item Generation
        logging.info(f"[DEBUG] Generated Items (Top 20): {list(self.item_info.items())[:20]}")
        logging.info(f"[DEBUG] Constraints: min_sc_width={self.min_sc_width}, max_sc_width={self.max_sc_width}")
        
        # 1. Column Generation (패턴 생성)
        # 초기 패턴만으로도 충분한지 확인 (Small Problem)
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
        # [Constraint] std_roll_cnt (복합폭 개수) 배수 적용
        # 롤 개수가 std_roll_cnt 의 배수가 되도록 길이 조절
        if self.std_roll_cnt and self.std_roll_cnt >= 1:
            # 1. 계산된 솔루션의 실제 생산량과 주문량을 비교하여 과생산 비율(Scale Factor) 계산
            #    (Solver가 고정 길이 제약으로 인해 과생산한 경우 이를 보정하기 위함)
            scale_factor = 1.0
            
            # 패턴별 생산량 집계
            solver_prod_by_width = {} # {width: total_meters}
            for j, count in final_solution['pattern_counts'].items():
                if count < 0.01: continue
                pat_len = self.patterns[j]['length']
                for item_name, item_count in self.patterns[j]['composition'].items():
                    # item_composition: {base_width: num}
                    for base_w, base_num in self.item_composition[item_name].items():
                        solver_prod_by_width[base_w] = solver_prod_by_width.get(base_w, 0) + (count * pat_len * item_count * base_num)

            # 주문량 대비 생산량 비율 확인 (가장 타이트한 비율 찾기)
            # 생산량이 주문량보다 크다면(비율 < 1) 줄여야 함.
            # 모든 주문을 만족해야 하므로 Max(Required / Produced) 를 사용.
            max_ratio = 0.0
            has_check = False
            for w, demand_m in self.demands_in_meters.items():
                prod_m = solver_prod_by_width.get(w, 0)
                if prod_m > 0:
                    ratio = demand_m / prod_m
                    if ratio > max_ratio:
                        max_ratio = ratio
                    has_check = True
            
            if has_check and max_ratio > 0:
                scale_factor = max_ratio
                logging.info(f"[Constraint Adjustment] Solver Over/Under-production Scale Factor: {scale_factor:.4f} (based on demand)")

            # [New] Aggregated Roll Count Calculation
            # "roll_std_cnt를 전체 패턴에서에서 복합폭기준으로 해서 최소로만 적용하면 되도록 해줘"
            item_aggregated_counts = {}
            for j, count in final_solution['pattern_counts'].items():
                if count < 0.99: continue
                count_int = int(round(count))
                for item_name in self.patterns[j]['composition']:
                    item_aggregated_counts[item_name] = item_aggregated_counts.get(item_name, 0) + count_int

            # [New] 지폭별 scale factor 개별 계산
            # 각 지폭의 ratio를 미리 계산해둠
            width_ratios = {}
            for w, demand_m in self.demands_in_meters.items():
                prod_m = solver_prod_by_width.get(w, 0)
                if prod_m > 0:
                    width_ratios[w] = demand_m / prod_m
                else:
                    width_ratios[w] = 1.0  # 생산량 없으면 기본값

            # 2. 패턴별 조정 적용 (패턴별 개별 scale factor 적용)
            for j, count in list(final_solution['pattern_counts'].items()):
                if count < 0.99: 
                    continue
                
                count_int = int(round(count))
                current_length = self.patterns[j]['length']

                # [Fix] 패턴별 개별 scale factor 계산
                # 해당 패턴에 포함된 지폭들의 min ratio 사용 (과생산 최소화)
                pattern_scale_factor = float('inf')
                for item_name in self.patterns[j]['composition']:
                    for base_w in self.item_composition[item_name]:
                        if base_w in width_ratios:
                            if width_ratios[base_w] < pattern_scale_factor:
                                pattern_scale_factor = width_ratios[base_w]
                
                # 생산량 없는 경우 기본값
                if pattern_scale_factor == float('inf'):
                    pattern_scale_factor = 1.0
                
                # [Safety Margin] 2% 여유 적용
                pattern_scale_factor = pattern_scale_factor * 1.02

                # 보정된 필요 총 길이 (과생산 제거)
                optimized_total_len = (count_int * current_length) * pattern_scale_factor
                
                # [Modified] Ensure count is at least std_roll_cnt IF aggregated count is insufficient
                # Only enforce minimum if ANY item in this pattern has a LOW aggregated count.
                # If all items in this pattern are produced in sufficient quantity (>= std_roll_cnt) across all patterns,
                # then we can allow this specific pattern to have a small count (e.g. 3).
                
                needs_increase = False
                for item_name in self.patterns[j]['composition']:
                    if item_aggregated_counts.get(item_name, 0) < self.std_roll_cnt:
                        needs_increase = True
                        break
                
                if count_int < self.std_roll_cnt and needs_increase:
                    new_count = self.std_roll_cnt
                    logging.info(f"[Constraint Adjustment] P{j} Count {count_int} increased to min {self.std_roll_cnt} (Aggregated insufficient)")
                else:
                    new_count = count_int 
                
                # 이미 배수지만 scale_factor로 인해 길이가 줄어들어야 하는 경우도 처리하기 위해
                if new_count == 0: new_count = max(1, self.std_roll_cnt)

                new_length = optimized_total_len / new_count
                
                # [User Request] 롤 길이를 100 단위로 반올림 (초과생산 감수)
                new_length = round(new_length / 100.0) * 100.0
                
                # 제약조건 확인 (최소 롤 길이) - 경고만 남기고 적용
                if new_length < self.min_sheet_roll_length:
                    logging.warning(f"[Constraint Warning] 패턴 {j} 길이 조정: {current_length:.1f} -> {new_length:.1f} (최소 {self.min_sheet_roll_length} 미만). Over-production 보정 및 배수 적용 결과.")

                logging.info(f"[Constraint Adjustment] Pattern {j} Count {count_int} -> {new_count} (Min {self.std_roll_cnt}). Length {current_length:.1f} -> {new_length:.1f} (Scale: {pattern_scale_factor:.4f})")
                
                # 결과 업데이트
                final_solution['pattern_counts'][j] = float(new_count)
                self.patterns[j]['length'] = new_length

        return self._format_results(final_solution, start_prod_seq)

    def _format_results(self, final_solution, start_prod_seq=0):
        """
        최적화 결과를 포맷팅하여 반환합니다.
        
        최적화 솔루션을 받아 DB 저장용 상세 정보를 생성하고,
        주문 이행 요약을 작성합니다.
        
        Args:
            final_solution (dict): 솔버로부터 얻은 최적 해 (패턴별 생산 횟수 포함)
            start_prod_seq (int): 생산 시퀀스 시작 번호
        
        Returns:
            dict: 포맷팅된 결과
                - pattern_result: 패턴 요약 DataFrame
                - pattern_details_for_db: TH_ROLL_SEQUENCE 테이블용 데이터
                - pattern_roll_details_for_db: TH_ROLL_DETAIL 테이블용 데이터
                - pattern_roll_cut_details_for_db: TH_CUT_DETAIL 테이블용 데이터
                - fulfillment_summary: 주문 이행 요약
                - last_prod_seq: 마지막 생산 시퀀스 번호
        """
        # 패턴 상세 정보 생성 (DB 저장용 및 주문 이행 추적)
        result_patterns, pattern_details_for_db, pattern_roll_details_for_db, pattern_roll_cut_details_for_db, demand_tracker, last_prod_seq = self._build_pattern_details(final_solution, start_prod_seq)
        
        # 결과 DataFrame 생성
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['pattern', 'wd_width', 'roll_length', 'count', 'loss_per_roll']]

        # 주문 이행 요약 생성
        fulfillment_summary = self._build_fulfillment_summary(demand_tracker)

        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]") 
        logging.info("[주문 이행 요약 (그룹오더별)]")
        
        return {
            "pattern_result": df_patterns.sort_values('count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "last_prod_seq": last_prod_seq
        }

    def _build_pattern_details(self, final_solution, start_prod_seq=0):
        """
        최적화 결과로부터 DB 저장용 상세 정보를 생성합니다.
        
        이 메서드는 3가지 수준의 상세 데이터를 생성합니다:
        1. pattern_details_for_db: 패턴 수준 (TH_PATTERN_SEQUENCE)
        2. pattern_roll_details_for_db: 복합폭/롤 수준 (TH_ROLL_SEQUENCE)
        3. pattern_roll_cut_details_for_db: 개별 지폭 커팅 수준 (TH_CUT_SEQUENCE)
        
        또한 주문 이행 상황을 추적하여 demand_tracker를 업데이트합니다.
        
        Args:
            final_solution (dict): 최적화 솔루션 (패턴별 생산 횟수)
            start_prod_seq (int): 생산 시퀀스 시작 번호
        
        Returns:
            tuple: (result_patterns, pattern_details_for_db, pattern_roll_details_for_db, 
                   pattern_roll_cut_details_for_db, demand_tracker, last_prod_seq)
        """
        # ========================================================================
        # 주문 이행 추적용 DataFrame 초기화
        # - 각 주문별로 필요 미터 대비 생산된 미터를 추적
        # ========================================================================
        demand_tracker = self.df_orders.copy()
        demand_tracker['original_order_idx'] = demand_tracker.index  # 원본 인덱스 보존
        demand_tracker = demand_tracker[['original_order_idx', 'group_order_no', '지폭', 'meters']].copy()
        demand_tracker['fulfilled_meters'] = 0.0  # 생산된 미터 초기화
        demand_tracker = demand_tracker.sort_values(by=['지폭', 'group_order_no']).reset_index(drop=True)

        # 결과 저장 리스트 초기화
        result_patterns = []               # 요약 결과용
        pattern_details_for_db = []        # TH_PATTERN_SEQUENCE 테이블용
        pattern_roll_details_for_db = []   # TH_ROLL_SEQUENCE 테이블용
        pattern_roll_cut_details_for_db = []  # TH_CUT_SEQUENCE 테이블용
        prod_seq_counter = start_prod_seq  # 생산 시퀀스 카운터
        total_cut_seq_counter = 0          # 전체 커팅 시퀀스 카운터

        # 헬퍼 함수: 안전한 정수 변환
        def safe_int(val):
            """None이나 문자열을 안전하게 정수로 변환합니다."""
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        # ========================================================================
        # 공통 속성 추출 (DB 저장 시 모든 레코드에 동일하게 적용)
        # - 쉬트지는 지름(diameter), 코어(core)를 사용하지 않음
        # ========================================================================
        first_row = self.df_orders.iloc[0]
        common_props = {
            'diameter': 0, # 쉬트지는 지름 없음
            'color': first_row.get('color', ''),
            'luster': safe_int(first_row.get('luster', 0)),
            'p_lot': self.lot_no,
            'core': 0, # 쉬트지는 코어 없음
            'order_pattern': first_row.get('order_pattern', '')
        }

        # ========================================================================
        # 각 패턴별로 상세 정보 생성
        # ========================================================================
        for j, count in final_solution['pattern_counts'].items():
            # 생산 횟수가 1 미만인 패턴은 무시
            if count < 0.99:
                continue
            
            roll_count = int(round(count))  # 롤 생산 횟수 (정수로 반올림)
            pattern = self.patterns[j]
            pattern_comp = pattern['composition']  # 복합 아이템 구성
            pattern_length = pattern['length']     # 패턴 롤 길이

            prod_seq_counter += 1  # 생산 시퀀스 증가

            # 복합 아이템을 폭 내림차순으로 정렬 (큰 폭부터 배치)
            sorted_pattern_items = sorted(pattern_comp.items(), key=lambda item: self.item_info[item[0]], reverse=True)
            pattern_item_strs = []  # 패턴 문자열 표현용
            total_width = 0         # 패턴 총 폭
            all_base_pieces_in_roll = []  # 이 패턴에 포함된 모든 기본 지폭 목록

            # ----------------------------------------------------------------
            # Step 1: 패턴 요약 정보 생성
            # ----------------------------------------------------------------
            for item_name, num in sorted_pattern_items:
                width = self.item_info[item_name]  # 복합 아이템의 총 폭
                total_width += width * num
                
                # 복합 아이템을 기본 지폭으로 분해
                base_width_dict = self.item_composition[item_name]
                for base_key, num_base in base_width_dict.items():
                    # [Mod] 키가 튜플이면 첫 번째 요소(지폭)만 추출
                    if isinstance(base_key, tuple):
                        base_width = base_key[0]
                    else:
                        base_width = base_key
                    # 기본 지폭 목록에 추가 (실제 생산되는 개수만큼)
                    all_base_pieces_in_roll.extend([base_key] * (num_base * num))

                # [Mod] 패턴 문자열 포맷팅 - 세로 포함 형식(788_545x2) 처리
                # 형식: "788_545x2" → "1596(788*2)"
                sub_items = item_name.split('+')
                if len(sub_items) > 1 or 'x' not in item_name:
                     formatted_name = f"{width}({item_name})"  # 혼합 복합폭
                else:
                    try:
                        # 새로운 형식: "788_545x2" 또는 기존 형식: "788x2"
                        parts = item_name.split('x')
                        multiplier = int(parts[1])
                        width_part = parts[0]
                        
                        # "_"가 있으면 세로 포함 형식
                        if '_' in width_part:
                            base_width_val = int(width_part.split('_')[0])
                        else:
                            base_width_val = int(width_part)
                        
                        formatted_name = f"{width}({base_width_val}*{multiplier})"  # 단일 복합폭
                    except ValueError:
                        formatted_name = f"{width}({item_name})"
                pattern_item_strs.extend([formatted_name] * num)
            
            # 요약 결과 추가
            result_patterns.append({
                'pattern': ' + '.join(pattern_item_strs),  # 패턴 문자열 (예: "1420(710*2) + 1560(710x1+850x1)")
                'wd_width': total_width,                   # 총 사용 폭
                'roll_length': round(pattern_length, 2),   # 롤 길이
                'count': roll_count,                       # 생산 횟수
                'loss_per_roll': pattern['loss_per_roll']  # 롤당 손실
            })

            # ----------------------------------------------------------------
            # Step 2: DB 저장용 상세 정보 생성 (TH_ROLL_DETAIL)
            # - 각 복합 아이템별로 상세 레코드 생성
            # - 기본 지폭을 주문에 연결(FIFO 방식으로 주문 소진)
            # ----------------------------------------------------------------
            composite_widths_for_db = []       # 패턴에 포함된 복합폭 목록
            composite_group_nos_for_db = []    # 패턴에 포함된 그룹오더 목록
            
            roll_seq_counter = 0  # 롤 시퀀스 카운터 (패턴 내)
            for item_name, num_of_composite in sorted_pattern_items:
                composite_width = self.item_info[item_name]  # 복합폭
                base_width_dict = self.item_composition[item_name]  # 기본 지폭 구성

                # 각 복합 아이템 개수만큼 반복
                for _ in range(num_of_composite):
                    roll_seq_counter += 1
                    
                    base_widths_for_item = []      # 이 복합 아이템에 포함된 기본 지폭들
                    base_group_nos_for_item = []   # 각 기본 지폭에 매핑된 주문 번호
                    base_rs_gubuns_for_item = []   # 각 기본 지폭에 매핑된 주문 번호
                    assigned_group_no_for_composite = None

                    # [Mod] 기본 키 정렬: 튜플이면 첫 번째 요소(지폭)로 정렬
                    def get_sort_key(item):
                        key = item[0]
                        if isinstance(key, tuple):
                            return key[0]
                        return key
                    
                    # 각 기본 지폭별로 주문 연결 (FIFO 방식) - 지폭 내림차순 정렬
                    for base_key, num_of_base in sorted(base_width_dict.items(), key=get_sort_key, reverse=True):
                        # [Mod] 키가 튜플이면 지폭과 세로 분리
                        if isinstance(base_key, tuple):
                            base_width_val, base_sheet_length = base_key
                        else:
                            base_width_val = base_key
                            base_sheet_length = None
                        
                        for _ in range(num_of_base):
                            # [Mod] double_cutter='N'일 때는 지폭+세로로 매칭
                            if self.double_cutter == 'N' and 'demand_key' in demand_tracker.columns:
                                target_indices = demand_tracker[
                                    (demand_tracker['demand_key'] == base_key) &
                                    (demand_tracker['fulfilled_meters'] < demand_tracker['meters'])
                                ].index
                            else:
                                # 기존 로직: 지폭으로만 매칭
                                target_indices = demand_tracker[
                                    (demand_tracker['지폭'] == base_width_val) &
                                    (demand_tracker['fulfilled_meters'] < demand_tracker['meters'])
                                ].index
                            
                            # 기본값: 과생산 (주문 없음)
                            assigned_group_no = "OVERPROD"
                            if not target_indices.empty:
                                # 미이행 주문이 있으면 첫 번째 주문에 할당
                                target_idx = target_indices.min()
                                assigned_group_no = demand_tracker.loc[target_idx, 'group_order_no']
                            else:
                                # 모든 주문이 이행된 경우, 마지막 주문에 할당 (fallback)
                                if self.double_cutter == 'N' and 'demand_key' in demand_tracker.columns:
                                    fallback_indices = demand_tracker[demand_tracker['demand_key'] == base_key].index
                                else:
                                    fallback_indices = demand_tracker[demand_tracker['지폭'] == base_width_val].index
                                if not fallback_indices.empty:
                                    assigned_group_no = demand_tracker.loc[fallback_indices.min(), 'group_order_no']
                            
                            base_widths_for_item.append(base_width_val)
                            base_group_nos_for_item.append(assigned_group_no)
                            base_rs_gubuns_for_item.append('S')

                            # 복합 아이템의 대표 그룹오더는 첫 번째 기본 지폭의 그룹
                            if assigned_group_no_for_composite is None:
                                assigned_group_no_for_composite = assigned_group_no
                    
                    composite_widths_for_db.append(composite_width)
                    composite_group_nos_for_db.append(assigned_group_no_for_composite if assigned_group_no_for_composite is not None else "")

                    # TH_ROLL_SEQUENCE 레코드 생성
                    pattern_roll_details_for_db.append({
                        'rollwidth': composite_width,
                        'pattern_length': pattern_length,
                        'widths': (base_widths_for_item + [0] * 7)[:7],  # 최대 7개 지폭
                        'roll_widths': ([0] * 7)[:7],  # 최대 7개 지폭
                        'group_nos': (base_group_nos_for_item + [''] * 7)[:7],
                        'rs_gubuns': (base_rs_gubuns_for_item + [''] * 7)[:7],
                        'count': roll_count,
                        'prod_seq': prod_seq_counter,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'T' if self.coating_yn == 'Y' else 'S',  # Sheet 구분
                        'trim_loss': self.ww_trim_size,
                        'sc_trim': self.sheet_trim,
                        'sl_trim': self.ww_trim_size_sheet if self.coating_yn == 'Y' else 0, 
                        **common_props
                    })

                    # ----------------------------------------------------------------
                    # Step 3: 커팅 상세 정보 생성 (TH_CUT_DETAIL)
                    # - 각 기본 지폭별로 커팅 레코드 생성
                    # ----------------------------------------------------------------
                    cut_seq_counter = 0
                    for i in range(len(base_widths_for_item)):
                        width = base_widths_for_item[i]
                        if width > 0:
                            cut_seq_counter += 1
                            total_cut_seq_counter += 1
                            group_no = base_group_nos_for_item[i]
                            
                            # 무게 계산: 평량(g/m²) × 폭(m) × 길이(m)
                            weight = (self.b_wgt * (width / 1000) * pattern_length)

                            # TH_CUT_SEQUENCE 레코드 생성
                            pattern_roll_cut_details_for_db.append({
                                'prod_seq': prod_seq_counter,
                                'unit_no': prod_seq_counter,
                                'seq': total_cut_seq_counter,
                                'roll_seq': roll_seq_counter,
                                'cut_seq': cut_seq_counter,
                                'width': width,
                                'group_no': group_no,
                                'weight': weight,
                                'pattern_length': pattern_length,
                                'count': roll_count,
                                'cut_cnt': roll_count,
                                'rs_gubun': 'T' if self.coating_yn == 'Y' else 'S',  # Sheet 구분
                                **common_props
                            })

            # TH_PATTERN_SEQUENCE 레코드 생성 (패턴 수준)
            pattern_details_for_db.append({
                'pattern_length': pattern_length,
                'count': roll_count,
                'widths': (composite_widths_for_db + [0] * 8)[:8],  # 최대 8개 복합폭
                'group_nos': (composite_group_nos_for_db + [''] * 8)[:8],
                'prod_seq': prod_seq_counter,
                'rs_gubun': 'T' if self.coating_yn == 'Y' else 'S',  # Sheet 구분
                **common_props # [Mod] Add common props (color, p_lot, etc)
            })

            # ----------------------------------------------------------------
            # Step 4: 주문 이행 상황 업데이트
            # - 이 패턴에서 생산된 기본 지폭별로 주문 이행량 카운트
            # - FIFO 방식으로 주문 소진
            # ----------------------------------------------------------------
            base_counts_in_roll = Counter(all_base_pieces_in_roll)
            for base_key, num_in_roll in base_counts_in_roll.items():
                # 이 패턴에서 생산된 해당 지폭의 총 미터
                produced_meters = num_in_roll * pattern_length * roll_count
                
                # [Mod] double_cutter='N'일 때는 demand_key로 매칭
                if self.double_cutter == 'N' and 'demand_key' in demand_tracker.columns:
                    relevant_orders = demand_tracker[demand_tracker['demand_key'] == base_key].index
                else:
                    # 기존 로직: 지폭으로만 매칭
                    if isinstance(base_key, tuple):
                        base_width_val = base_key[0]
                    else:
                        base_width_val = base_key
                    relevant_orders = demand_tracker[demand_tracker['지폭'] == base_width_val].index
                
                for order_idx in relevant_orders:
                    if produced_meters <= 0:
                        break
                    
                    # 이 주문에서 아직 필요한 미터
                    needed = demand_tracker.loc[order_idx, 'meters'] - demand_tracker.loc[order_idx, 'fulfilled_meters']
                    if needed > 0:
                        # 필요량과 생산량 중 작은 값만큼 이행 (일단 채움)
                        fulfill_amount = min(needed, produced_meters)
                        demand_tracker.loc[order_idx, 'fulfilled_meters'] += fulfill_amount
                        produced_meters -= fulfill_amount
                
                # [Fix] 과생산분 반영: 만약 모든 오더를 채우고도 생산량이 남았다면, 해당 지폭의 마지막 오더에 합산하여 표기
                if produced_meters > 0 and not relevant_orders.empty:
                    last_idx = relevant_orders[-1]
                    demand_tracker.loc[last_idx, 'fulfilled_meters'] += produced_meters

        return result_patterns, pattern_details_for_db, pattern_roll_details_for_db, pattern_roll_cut_details_for_db, demand_tracker, prod_seq_counter

    def _build_fulfillment_summary(self, demand_tracker):
        """
        주문 이행 요약 보고서를 생성합니다.
        
        각 주문별로 주문량, 생산량, 과부족을 톤 및 미터 단위로 계산합니다.
        
        Args:
            demand_tracker (pd.DataFrame): 주문 이행 추적 데이터
                - fulfilled_meters: 각 주문별 생산된 미터
        
        Returns:
            pd.DataFrame: 주문 이행 요약
                - group_order_no: 그룹오더 번호
                - 가로, 세로, 수출내수, 등급: 주문 정보
                - 주문량(톤), 생산량(톤), 과부족(톤): 톤 단위 비교
                - 필요길이(m), 생산길이(m), 과부족(m): 미터 단위 비교
        """
        # 기본 주문 정보 복사
        # self.df_orders에는 '가로' 대신 '지폭' 컬럼이 존재함 (_calculate_demand_meters에서 rename 됨)
        summary_df = self.df_orders[['group_order_no', '지폭', '세로', '수출내수', '등급', '주문톤', 'meters']].copy()
        # 리포팅용으로 다시 '가로'로 이름 변경
        summary_df.rename(columns={'지폭': '가로', 'meters': '필요길이(m)', '주문톤': '주문량(톤)'}, inplace=True)
        
        # 이행 정보 병합
        summary_df = pd.merge(summary_df, demand_tracker[['original_order_idx', 'fulfilled_meters']], 
                              left_index=True, right_on='original_order_idx', how='left')
        summary_df.rename(columns={'fulfilled_meters': '생산길이(m)'}, inplace=True)
        summary_df.drop(columns=['original_order_idx'], inplace=True)

        summary_df['생산길이(m)'] = summary_df['생산길이(m)'].fillna(0)
        
        # 과부족 계산 (미터 단위)
        summary_df['과부족(m)'] = summary_df['생산길이(m)'] - summary_df['필요길이(m)']
        
        # 미터당 톤수 계산 (톤 단위 변환용)
        tons_per_meter = (summary_df['주문량(톤)'] / summary_df['필요길이(m)']).replace([float('inf'), -float('inf')], 0).fillna(0)
        summary_df['생산량(톤)'] = (summary_df['생산길이(m)'] * tons_per_meter).round(2)
        summary_df['과부족(톤)'] = (summary_df['생산량(톤)'] - summary_df['주문량(톤)']).round(2)

        # 출력 커럼 순서 정의
        final_cols = [
            'group_order_no', '가로', '세로', '수출내수', '등급', 
            '주문량(톤)', '생산량(톤)', '과부족(톤)',
            '필요길이(m)', '생산길이(m)', '과부족(m)'
        ]
        
        # 소수점 정리
        for col in ['필요길이(m)', '생산길이(m)', '과부족(m)']:
            summary_df[col] = summary_df[col].round(2)

        return summary_df[final_cols]

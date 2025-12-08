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

# --- 최적화 설정 상수 ---
# 비용 상수 (모든 목적 함수 항을 '비용'으로 통일하기 위해 사용)
COST_PER_ROLL = 5000.0          # 롤 1개 교체/사용에 대한 비용 (예시)
COST_PER_METER_MATERIAL = 0.8  # 원자재 1미터당 비용 (예시)

# 페널티 값
OVER_PROD_PENALTY = 200.0    # 과생산에 대한 페널티
UNDER_PROD_PENALTY = 100000.0  # 부족생산에 대한 페널티
PATTERN_COMPLEXITY_PENALTY = 0.01  # 패턴 복잡성에 대한 페널티
PIECE_COUNT_PENALTY = 10           # 패턴 내 롤(piece) 개수에 대한 페널티 (적은 롤 선호)
TRIM_PENALTY = 0          # 트림(loss) 면적(mm^2)당 페널티. 폐기물 비용.
ITEM_SINGLE_STRIP_PENALTIES = {}
DEFAULT_SINGLE_STRIP_PENALTY = 1000  # 지정되지 않은 단일폭은 기본적으로 패널티 없음
DISALLOWED_SINGLE_BASE_WIDTHS = {}  # 단일 사용을 금지할 주문 폭 집합

# 솔버 멀티스레딩
NUM_THREADS = 4

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 1      # 패턴에 포함될 수 있는 최소 폭(piece)의 수
SMALL_PROBLEM_THRESHOLD = 8     # 전체 탐색을 수행할 최대 주문 지폭 종류 수
SOLVER_TIME_LIMIT_MS = 180000    # 최종 MIP 솔버의 최대 실행 시간 (밀리초)
CG_MAX_ITERATIONS = 1000         # 열 생성(Column Generation) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 100    # 개선 없는 경우, 열 생성 조기 종료 조건
CG_SUBPROBLEM_TOP_N = 1         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴
# 나이프 로드 제약: 패턴 생산 횟수는 k1*a + k2*b 형태여야 함 (a,b>=0 정수)
KNIFE_LOAD_K1 = 1
KNIFE_LOAD_K2 = 1

class SheetOptimizeCa:
    """
    쉬트지 복합폭(Composite Width) 최적화 클래스.
    
    이 클래스는 여러 지폭을 조합한 '복합 아이템'을 기반으로 최적 절단 패턴을 찾습니다.
    일반 쉬트 최적화와 달리, 복합폭 개념을 통해 슬리터 칼 배치를 더 유연하게 처리합니다.
    """
    
    def __init__(
            self,
            df_spec_pre,
            max_width,
            min_width,
            max_pieces,
            b_wgt,
            min_sheet_roll_length,
            max_sheet_roll_length,
            sheet_trim,
            min_sc_width,
            max_sc_width,
            db=None,
            lot_no=None,
            version=None
    ):
        """
        SheetOptimizeCa 생성자.
        
        Args:
            df_spec_pre (pd.DataFrame): 주문 데이터. 필수 컬럼: '가로', '세로', '주문톤', 'group_order_no'
            max_width (int): 원지(Jumbo Roll) 최대 폭 (mm)
            min_width (int): 패턴에서 허용하는 최소 총 폭 (mm)
            max_pieces (int): 패턴당 최대 허용 조수 (복합 아이템 개수)
            b_wgt (float): 평량 (g/m²) - 무게 계산에 사용
            min_sheet_roll_length (float): 쉬트 롤 최소 길이 (m)
            max_sheet_roll_length (float): 쉬트 롤 최대 길이 (m)
            sheet_trim (int): 쉬트 트림 사이즈 (mm) - 복합폭 계산 시 추가되는 손실분
            min_sc_width (int): 슬리터 칼(SC) 최소 폭 (mm) - 복합 아이템 폭의 하한
            max_sc_width (int): 슬리터 칼(SC) 최대 폭 (mm) - 복합 아이템 폭의 상한
            db: 데이터베이스 연결 객체 (선택)
            lot_no (str): 롯트 번호 (선택)
            version (str): 버전 (선택)
        """
        # 주문 데이터의 '가로' 컬럼을 '지폭'으로 복사 (내부 처리용)
        df_spec_pre['지폭'] = df_spec_pre['가로']

        # 평량 및 롤 길이 제약조건 저장
        self.b_wgt = b_wgt
        self.min_sheet_roll_length = min_sheet_roll_length
        self.max_sheet_roll_length = max_sheet_roll_length
        self.sheet_trim = sheet_trim  # 복합폭 계산 시 추가되는 트림 손실
        self.original_max_width = max_width  # 원지 최대 폭 저장
        
        # 주문량을 미터 단위로 변환하여 수요 계산
        self.df_orders, self.demands_in_meters, self.order_sheet_lengths = self._calculate_demand_meters(df_spec_pre)
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

        # 제약조건 저장
        self.max_width = max_width
        self.min_width = min_width
        self.min_pieces = MIN_PIECES_PER_PATTERN
        self.max_pieces = int(max_pieces)
        self.min_sc_width = min_sc_width
        self.max_sc_width = max_sc_width
        self.db = db
        self.lot_no = lot_no
        self.version = version
        print(f"--- 패턴 제약조건: 최소 {self.min_pieces}폭, 최대 {self.max_pieces}폭 ---")

        # 패턴 저장소 초기화 (외부에서 패턴을 주입해야 함)
        self.patterns = []

    def _prepare_items(self, min_sc_width, max_sc_width):
        """
        복합 아이템(Composite Items)을 생성합니다.
        
        복합 아이템이란 하나 이상의 기본 지폭을 조합하여 슬리터 칼로
        한 번에 절단할 수 있는 단위를 말합니다.
        
        생성되는 아이템 유형:
        1. 단일 지폭 복합: "710x2" → 710mm 2개 = 1420mm + trim
        2. 혼합 지폭 복합: "710x1+850x1" → 710mm 1개 + 850mm 1개 = 1560mm + trim
        
        Args:
            min_sc_width (int): 슬리터 칼 최소 폭 제약 (이 값 이상이어야 유효)
            max_sc_width (int): 슬리터 칼 최대 폭 제약 (이 값 이하여야 유효)
        
        Returns:
            tuple: (items, item_info, item_composition)
                - items: 복합 아이템 이름 목록 (예: ["710x1", "710x2", "710x1+850x1"])
                - item_info: {아이템명: 총 폭(mm)} 매핑
                - item_composition: {아이템명: {기본지폭: 개수}} 구성 정보
        """
        from itertools import combinations_with_replacement
        
        items = []
        item_info = {}  # item_name -> 총 폭 (mm)
        item_composition = {}  # composite_item_name -> {original_width: count}
        
        # 하나의 복합 아이템에 포함될 수 있는 최대 기본 지폭 개수
        max_pieces_in_composite = 4 

        # ============================================================
        # Step 1: 단일 지폭 복합 아이템 생성 (같은 지폭 N개 조합)
        # 예: 710mm x 1 = 710mm, 710mm x 2 = 1420mm, ...
        # ============================================================
        for width in self.order_widths:
            for i in range(1, max_pieces_in_composite + 1):
                # 복합폭 계산: (기본 지폭 × 개수) + 트림 손실
                base_width = width * i + self.sheet_trim
                
                # 슬리터 칼 제약조건 체크
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                # 아이템 명명: "지폭x개수" 형식 (예: "710x2")
                item_name = f"{width}x{i}"
                if base_width <= self.original_max_width:
                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = base_width
                        item_composition[item_name] = {width: i}

        # ============================================================
        # Step 2: 혼합 지폭 복합 아이템 생성 (다른 지폭 조합)
        # 예: 710mm + 850mm = 1560mm
        # ============================================================
        for i in range(2, max_pieces_in_composite + 1):
            # 중복 조합(combinations_with_replacement) 생성
            for combo in combinations_with_replacement(self.order_widths, i):
                # 단일 지폭만으로 구성된 조합은 Step 1에서 이미 처리됨
                if len(set(combo)) == 1:
                    continue

                # 복합폭 계산: 모든 지폭 합계 + 트림 손실
                base_width = sum(combo) + self.sheet_trim
                
                # 슬리터 칼 제약조건 체크
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                if base_width <= self.original_max_width:
                    # 조합 구성 카운팅 (예: (710, 710, 850) → {710: 2, 850: 1})
                    comp_counts = Counter(combo)
                    # 아이템 명명: 정렬된 "지폭x개수" 조합 (예: "710x2+850x1")
                    item_name = "+".join(sorted([f"{w}x{c}" for w, c in comp_counts.items()]))

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
        
        Args:
            df_orders (pd.DataFrame): 주문 데이터 (가로, 세로, 주문톤 컬럼 필수)
        
        Returns:
            tuple: (df_copy, demand_meters, order_sheet_lengths)
                - df_copy: meters 컬럼이 추가된 주문 DataFrame
                - demand_meters: {지폭: 필요미터} 딕셔너리
                - order_sheet_lengths: {지폭: 세로길이} 딕셔너리 (첫 번째 값)
        """
        df_copy = df_orders.copy()

        def calculate_meters(row):
            """개별 주문 행에 대한 필요 생산 길이(m)를 계산합니다."""
            width_mm = row['가로']    # 지폭 (mm)
            length_mm = row['세로']   # 장당 세로 길이 (mm)
            order_ton = row['주문톤']  # 주문량 (톤)

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

        # 각 주문 행에 대해 필요 미터 계산
        df_copy['meters'] = df_copy.apply(calculate_meters, axis=1)
        # 지폭별로 필요 미터 합계
        demand_meters = df_copy.groupby('지폭')['meters'].sum().to_dict()
        # 지폭별 대표 세로 길이 (첫 번째 값 사용)
        order_sheet_lengths = df_copy.groupby('지폭')['세로'].first().to_dict()

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        print("--- 지폭별 필요 총 길이 ---")
        print("--------------------------")

        return df_copy, demand_meters, order_sheet_lengths

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
        print(f"--- 전체 탐색으로 {len(self.patterns)}개의 패턴 생성됨 ---")

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
                    
                    # 남은 공간에 맞는 가장 큰 아이템을 찾음 (First-Fit)
                    best_fit_item = next((i for i in sorted_items if self.item_info[i] <= remaining_width), None)
                    
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

        print(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")

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
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        
        if hasattr(solver, 'SetNumThreads'):
            solver.SetNumThreads(NUM_THREADS)

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
        total_complexity_penalty = solver.Sum(
            PATTERN_COMPLEXITY_PENALTY * len(self.patterns[j]['composition']) * x[j]
            for j in range(len(self.patterns))
        )
        
        # 패턴 내 롤 개수에 대한 페널티 (Quadratic)
        total_piece_penalty = solver.Sum(
            PIECE_COUNT_PENALTY * (sum(
                count for item, count in self.patterns[j]['composition'].items()
            ) ** 2) * x[j]
            for j in range(len(self.patterns))
        )

        solver.Minimize(
            total_rolls + total_over_prod_penalty + total_under_prod_penalty + 
            total_complexity_penalty + total_piece_penalty
        )
        
        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            solution = {
                'objective': solver.Objective().Value(),
                'pattern_counts': {j: var.solution_value() for j, var in x.items()},
                'over_production': {w: var.solution_value() for w, var in over_prod_vars.items()},
                'under_production': {w: var.solution_value() for w, var in under_prod_vars.items()}
            }
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
            if len(self.order_widths) <= SMALL_PROBLEM_THRESHOLD:
                print(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 모든 패턴을 탐색합니다 (Small-scale) ---")
                self._generate_all_patterns()
            else:
                print(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 열 생성 기법을 시작합니다 (Large-scale) ---")
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
                        print(f"--- {CG_NO_IMPROVEMENT_LIMIT}번의 반복 동안 개선이 없어 수렴으로 간주하고 종료합니다 (반복 {iteration}). ---")
                        break

        if not self.patterns:
            return {"error": "유효한 패턴을 생성할 수 없습니다."}

        print(f"--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
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

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]") 
        print("[주문 이행 요약 (그룹오더별)]")
        
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
        1. pattern_details_for_db: 패턴 수준 (TH_ROLL_SEQUENCE)
        2. pattern_roll_details_for_db: 복합폭/롤 수준 (TH_ROLL_DETAIL)
        3. pattern_roll_cut_details_for_db: 개별 지폭 커팅 수준 (TH_CUT_DETAIL)
        
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
        pattern_details_for_db = []        # TH_ROLL_SEQUENCE 테이블용
        pattern_roll_details_for_db = []   # TH_ROLL_DETAIL 테이블용
        pattern_roll_cut_details_for_db = []  # TH_CUT_DETAIL 테이블용
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
                for base_width, num_base in base_width_dict.items():
                    # 기본 지폭 목록에 추가 (실제 생산되는 개수만큼)
                    all_base_pieces_in_roll.extend([base_width] * (num_base * num))

                # 패턴 문자열 포맷팅 (예: "1420(710*2)" 또는 "1560(710x1+850x1)")
                sub_items = item_name.split('+')
                if len(sub_items) > 1 or 'x' not in item_name:
                     formatted_name = f"{width}({item_name})"  # 혼합 복합폭
                else:
                    try:
                        base_width, multiplier = map(int, item_name.split('x'))
                        formatted_name = f"{width}({base_width}*{multiplier})"  # 단일 복합폭
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
                    assigned_group_no_for_composite = None

                    # 각 기본 지폭별로 주문 연결 (FIFO 방식)
                    for base_width, num_of_base in base_width_dict.items():
                        for _ in range(num_of_base):
                            # 아직 이행되지 않은 주문 중 해당 지폭 찾기
                            target_indices = demand_tracker[
                                (demand_tracker['지폭'] == base_width) &
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
                                fallback_indices = demand_tracker[demand_tracker['지폭'] == base_width].index
                                if not fallback_indices.empty:
                                    assigned_group_no = demand_tracker.loc[fallback_indices.min(), 'group_order_no']
                            
                            base_widths_for_item.append(base_width)
                            base_group_nos_for_item.append(assigned_group_no)

                            # 복합 아이템의 대표 그룹오더는 첫 번째 기본 지폭의 그룹
                            if assigned_group_no_for_composite is None:
                                assigned_group_no_for_composite = assigned_group_no
                    
                    composite_widths_for_db.append(composite_width)
                    composite_group_nos_for_db.append(assigned_group_no_for_composite if assigned_group_no_for_composite is not None else "")

                    # TH_ROLL_DETAIL 레코드 생성
                    pattern_roll_details_for_db.append({
                        'rollwidth': composite_width,
                        'pattern_length': pattern_length,
                        'widths': (base_widths_for_item + [0] * 7)[:7],  # 최대 7개 지폭
                        'group_nos': (base_group_nos_for_item + [''] * 7)[:7],
                        'count': roll_count,
                        'prod_seq': prod_seq_counter,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'S',  # Sheet 구분
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

                            # TH_CUT_DETAIL 레코드 생성
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
                                'rs_gubun': 'S',  # Sheet 구분
                                **common_props
                            })

            # TH_ROLL_SEQUENCE 레코드 생성 (패턴 수준)
            pattern_details_for_db.append({
                'pattern_length': pattern_length,
                'count': roll_count,
                'widths': (composite_widths_for_db + [0] * 8)[:8],  # 최대 8개 복합폭
                'group_nos': (composite_group_nos_for_db + [''] * 8)[:8],
                'prod_seq': prod_seq_counter,
                'rs_gubun': 'S',  # Sheet 구분
            })

            # ----------------------------------------------------------------
            # Step 4: 주문 이행 상황 업데이트
            # - 이 패턴에서 생산된 기본 지폭별로 주문 이행량 카운트
            # - FIFO 방식으로 주문 소진
            # ----------------------------------------------------------------
            base_counts_in_roll = Counter(all_base_pieces_in_roll)
            for base_width, num_in_roll in base_counts_in_roll.items():
                # 이 패턴에서 생산된 해당 지폭의 총 미터
                produced_meters = num_in_roll * pattern_length * roll_count
                
                # 해당 지폭의 주문들에 순차적으로 할당
                relevant_orders = demand_tracker[demand_tracker['지폭'] == base_width].index
                
                for order_idx in relevant_orders:
                    if produced_meters <= 0:
                        break
                    
                    # 이 주문에서 아직 필요한 미터
                    needed = demand_tracker.loc[order_idx, 'meters'] - demand_tracker.loc[order_idx, 'fulfilled_meters']
                    if needed > 0:
                        # 필요량과 생산량 중 작은 값만큼 이행
                        fulfill_amount = min(needed, produced_meters)
                        demand_tracker.loc[order_idx, 'fulfilled_meters'] += fulfill_amount
                        produced_meters -= fulfill_amount

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
        summary_df = self.df_orders[['group_order_no', '가로', '세로', '수출내수', '등급', '주문톤', 'meters']].copy()
        summary_df.rename(columns={'meters': '필요길이(m)', '주문톤': '주문량(톤)'}, inplace=True)
        
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

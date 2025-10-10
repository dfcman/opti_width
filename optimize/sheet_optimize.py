import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import math
import random

# --- 최적화 설정 상수 ---
# 페널티 값
OVER_PROD_PENALTY = 200.0    # 과생산에 대한 페널티
UNDER_PROD_PENALTY = 500.0  # 부족생산에 대한 페널티
PATTERN_COMPLEXITY_PENALTY = 0.01  # 패턴 복잡성에 대한 페널티

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 2      # 패턴에 포함될 수 있는 최소 폭(piece)의 수
SMALL_PROBLEM_THRESHOLD = 8     # 전체 탐색을 수행할 최대 주문 지폭 종류 수
SOLVER_TIME_LIMIT_MS = 60000    # 최종 MIP 솔버의 최대 실행 시간 (밀리초)
CG_MAX_ITERATIONS = 100         # 열 생성(Column Generation) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 100    # 개선 없는 경우, 열 생성 조기 종료 조건
CG_SUBPROBLEM_TOP_N = 3         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴

class SheetOptimize:
    def __init__(
            self,
            df_spec_pre,
            max_width,
            min_width,
            max_pieces,
            b_wgt,
            sheet_roll_length,
            sheet_trim,
            min_sc_width,
            max_sc_width,
            db=None,
            lot_no=None,
            version=None
    ):
        df_spec_pre['지폭'] = df_spec_pre['가로']

        self.b_wgt = b_wgt
        self.sheet_roll_length = sheet_roll_length
        self.sheet_trim = sheet_trim
        self.original_max_width = max_width
        
        # _calculate_demand_rolls에서 'rolls' 열이 추가된 데이터프레임을 받음
        self.df_orders, self.demands_in_rolls = self._calculate_demand_rolls(df_spec_pre)
        self.order_widths = list(self.demands_in_rolls.keys()) 

        width_summary = {}
        # 'rolls' 열이 추가되었으므로 self.df_orders를 사용
        tons_per_width = self.df_orders.groupby('지폭')['주문톤'].sum()
        for width, required_rolls in self.demands_in_rolls.items():
            order_tons = tons_per_width.get(width, 0)
            width_summary[width] = {'order_tons': order_tons}
        self.width_summary = width_summary 

        self.items, self.item_info, self.item_composition = self._prepare_items(min_sc_width, max_sc_width)

        self.max_width = max_width
        self.min_width = min_width
        self.min_pieces = MIN_PIECES_PER_PATTERN
        self.max_pieces = int(max_pieces)
        self.min_sc_width = min_sc_width
        self.max_sc_width = max_sc_width
        self.db = db
        self.lot_no = lot_no
        self.version = version
        print(f"\n--- 패턴 제약조건: 최소 {self.min_pieces}폭, 최대 {self.max_pieces}폭 ---")

        self.patterns = []

    def _prepare_items(self, min_sc_width, max_sc_width):
        """복합폭 아이템(패턴의 구성요소)을 생성합니다."""
        items = []
        item_info = {}  # item_name -> width
        item_composition = {}  # composite_item_name -> {original_width: count}

        for width in self.order_widths:
            for i in range(1, 5): # 1, 2, 3, 4폭까지 고려
                base_width = width * i + self.sheet_trim
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                item_name = f"{width}x{i}"
                composite_width = base_width
                if composite_width <= self.original_max_width:
                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = composite_width
                        item_composition[item_name] = {width: i}
        return items, item_info, item_composition

    def _calculate_demand_rolls(self, df_orders):
        """주문량을 바탕으로 지폭별 필요 롤 수를 계산하고, 원본 데이터프레임에 'rolls' 열을 추가하여 반환합니다."""
        df_copy = df_orders.copy()
        sheet_roll_length_mm = self.sheet_roll_length * 1000

        def calculate_rolls(row):
            width_mm = row['가로']
            length_mm = row['세로']
            order_ton = row['주문톤']

            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0 or order_ton <= 0:
                return 0
            
            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1000000
            if sheet_weight_g <= 0:
                return 0

            total_sheets_needed = (order_ton * 1000000) / sheet_weight_g
            sheets_per_roll_length = math.floor(sheet_roll_length_mm / length_mm)
            if sheets_per_roll_length <= 0:
                return 0

            sheets_per_roll = sheets_per_roll_length * 1 # num_across is always 1
            return round(total_sheets_needed / sheets_per_roll, 0)

        df_copy['rolls'] = df_copy.apply(calculate_rolls, axis=1).astype(int)
        demand_rolls = df_copy.groupby('지폭')['rolls'].sum().to_dict()

        print("\n--- 지폭별 필요 롤 수 ---")
        # for width, rolls in demand_rolls.items():
        #     print(f"  지폭 {width}mm: {rolls} 롤")
        print("--------------------------\n")
        
        return df_copy, demand_rolls

    def _generate_initial_patterns_db(self):
        """th_pattern_sequence 테이블의 기존 패턴 데이터를 활용하여 초기 패턴을 생성합니다."""
        if not self.db or not self.lot_no or not self.version:
            print("--- DB 정보가 없어 기존 패턴을 불러올 수 없습니다. ---")
            return

        print("\n--- DB에서 기존 패턴을 불러와 초기 패턴을 생성합니다. ---")
        db_patterns_list = self.db.get_patterns_from_db(self.lot_no, self.version)

        if not db_patterns_list:
            print("--- DB에 저장된 기존 패턴이 없습니다. ---")
            return

        initial_patterns_from_db = []
        for pattern_item_list in db_patterns_list:
            pattern_dict = dict(Counter(pattern_item_list))
            
            # DB의 패턴에 포함된 모든 아이템이 현재 주문에도 유효한지 확인
            is_valid = all(item_name in self.items for item_name in pattern_dict.keys())
            
            if is_valid:
                initial_patterns_from_db.append(pattern_dict)
            else:
                invalid_items = [name for name in pattern_dict if name not in self.items]
                print(f"    - 경고: DB 패턴 {pattern_dict}의 아이템 {invalid_items}이(가) 현재 주문에 없어 해당 패턴을 무시합니다.")

        if initial_patterns_from_db:
            seen_patterns = {frozenset(p.items()) for p in self.patterns}
            added_count = 0
            for p_dict in initial_patterns_from_db:
                if frozenset(p_dict.items()) not in seen_patterns:
                    self.patterns.append(p_dict)
                    added_count += 1
            print(f"--- DB에서 {added_count}개의 신규 초기 패턴을 추가했습니다. ---")

    def _generate_initial_patterns(self):
        """휴리스틱을 사용하여 초기 패턴을 생성합니다."""
        seen_patterns = {frozenset(p.items()) for p in self.patterns}

        # 1. 다양한 휴리스틱을 위한 정렬된 아이템 리스트 생성
        sorted_by_demand = sorted(
            self.items,
            key=lambda i: self.demands_in_rolls.get(list(self.item_composition[i].keys())[0], 0),
            reverse=True
        )
        sorted_by_width_desc = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda i: self.item_info[i], reverse=False)

        heuristics = [sorted_by_demand, sorted_by_width_desc, sorted_by_width_asc]
        
        # 2. 각 휴리스틱에 대해 First-Fit과 유사한 패턴 생성
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

                # 너비가 min_width보다 작은 경우 보정
                while current_width < self.min_width and current_pieces < self.max_pieces:
                    # 추가해도 max_width를 넘지 않는 가장 적절한 아이템 탐색
                    item_to_add = next((i for i in sorted_by_width_desc if current_width + self.item_info[i] <= self.max_width), None)
                    
                    if item_to_add:
                        current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
                        current_width += self.item_info[item_to_add]
                        current_pieces += 1
                    else:
                        break # 더 이상 추가할 아이템이 없으면 종료

                # 최종 유효성 검사 후 패턴 추가
                if self.min_width <= current_width and self.min_pieces <= current_pieces:
                    pattern_key = frozenset(current_pattern.items())
                    if pattern_key not in seen_patterns:
                        self.patterns.append(current_pattern)
                        seen_patterns.add(pattern_key)
        
        print(f"--- {len(self.patterns)}개의 혼합 패턴 생성됨 ---")

        # --- 3. 모든 복합폭에 대해 '순수 품목 패턴' 생성 ---
        pure_patterns_added = 0
        # 순수 패턴은 어떤 정렬이든 상관 없으므로 마지막 정렬(오름차순) 사용
        for item in sorted_by_width_asc:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0: continue

            # 해당 아이템으로만 구성된 패턴 생성 시도
            num_items = min(int(self.max_width / item_width), self.max_pieces)
            
            # 너비가 큰 조합부터 작은 조합까지 순차적으로 확인
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                
                if self.min_width <= total_width and self.min_pieces <= num_items:
                    pattern_key = frozenset(new_pattern.items())
                    if pattern_key not in seen_patterns:
                        self.patterns.append(new_pattern)
                        seen_patterns.add(pattern_key)
                        pure_patterns_added += 1
                        break # 이 아이템으로 만들 수 있는 가장 좋은 순수패턴을 찾았으므로 종료
                
                num_items -= 1

        if pure_patterns_added > 0:
            print(f"--- {pure_patterns_added}개의 순수 품목 패턴 추가됨 ---")

        # --- 4. 폴백 로직: 초기 패턴으로 커버되지 않는 주문이 있는지 최종 확인 ---
        covered_widths = {w for p in self.patterns for item_name in p for w in self.item_composition.get(item_name, {})}
        uncovered_widths = set(self.order_widths) - covered_widths

        if uncovered_widths:
            print(f"--- 경고: 초기 패턴에 포함되지 않은 주문 발견: {uncovered_widths} ---")
            print("--- 해당 주문에 대한 폴백 패턴을 추가 생성합니다. ---")
            
            for width in uncovered_widths:
                print(f"  - 지폭 {width}mm에 대한 순수 품목 패턴 생성 시도...")

                # 1. 이 지폭으로 만들 수 있는 유효한 복합폭 아이템 목록을 찾습니다.
                valid_components = []
                for i in range(1, 5): # 1~4폭 고려
                    item_name = f"{width}x{i}"
                    # 아이템이 이미 생성되었는지 확인
                    if item_name in self.item_info:
                        valid_components.append(item_name)
                    else:
                        # 동적으로 생성 및 유효성 검사
                        composite_width = width * i + self.sheet_trim
                        if (self.min_sc_width <= composite_width <= self.max_sc_width) and \
                           (composite_width <= self.original_max_width):
                            # 유효하면 아이템 정보에 추가
                            if item_name not in self.items: self.items.append(item_name)
                            self.item_info[item_name] = composite_width
                            self.item_composition[item_name] = {width: i}
                            valid_components.append(item_name)

                if not valid_components:
                    print(f"    - 경고: 지폭 {width}mm로 만들 수 있는 유효한 복합폭 아이템이 없습니다. 폴백 패턴을 생성할 수 없습니다.")
                    continue

                # 2. 너비가 넓은 순으로 정렬하여 Greedy 알고리즘 준비
                sorted_components = sorted(valid_components, key=lambda i: self.item_info[i], reverse=True)
                
                # 3. Greedy 방식으로 최적의 단일 품목 패턴 구성
                new_pattern = {}
                current_width = 0
                current_pieces = 0
                
                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    
                    # 남은 공간에 들어갈 수 있는 가장 큰 구성요소 찾기
                    best_fit = next((item for item in sorted_components if self.item_info[item] <= remaining_width), None)
                    
                    if not best_fit:
                        break # 더 이상 추가할 수 있는 구성요소가 없음
                    
                    new_pattern[best_fit] = new_pattern.get(best_fit, 0) + 1
                    current_width += self.item_info[best_fit]
                    current_pieces += 1

                # 4. 생성된 패턴의 유효성 검사 및 추가
                if new_pattern:
                    total_width = sum(self.item_info[name] * count for name, count in new_pattern.items())
                    total_pieces = sum(new_pattern.values())

                    if self.min_width -200 <= total_width and self.min_pieces <= total_pieces:
                        pattern_key = frozenset(new_pattern.items())
                        if pattern_key not in seen_patterns:
                            self.patterns.append(new_pattern)
                            seen_patterns.add(pattern_key)
                            print(f"    -> 생성된 순수 패턴: {new_pattern} (너비: {total_width}mm, 폭 수: {total_pieces}) -> 폴백 패턴으로 추가됨.")
                        else:
                            print(f"    - 생성된 순수 패턴 {new_pattern}은 이미 존재합니다.")
                    else:
                        print(f"    - 생성된 순수 패턴 {new_pattern}이 최종 제약조건(최소너비/폭수)을 만족하지 못합니다. (너비: {total_width}, 폭 수: {total_pieces})")
                else:
                    print(f"    - 지폭 {width}mm에 대한 순수 패턴을 구성하지 못했습니다.")

        # --- 5. 생성된 패턴들을 후처리하여 작은 복합폭들을 통합 ---
        self._consolidate_patterns()

        print(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")
        print(self.patterns)
        print("--------------------------\n")

    def _consolidate_patterns(self):
        """
        생성된 초기 패턴들을 후처리하여 작은 복합폭 아이템들을 가능한 큰 복합폭 아이템으로 통합합니다.
        예: {'814x1': 2}는 {'814x2': 1}로 변경을 시도합니다.
        """
        print("\n--- 생성된 패턴에 대해 후처리(통합)를 시작합니다. ---")
        
        processed_patterns = []
        seen_patterns = set()

        for pattern in self.patterns:
            # 1. 패턴을 기본 지폭 단위로 모두 분해
            base_width_counts = Counter()
            for item_name, count in pattern.items():
                composition = self.item_composition.get(item_name)
                if composition:
                    for base_width, num_base in composition.items():
                        base_width_counts[base_width] += num_base * count

            # 2. 가장 큰 복합폭부터 사용하여 새로운 패턴 재구성
            new_pattern = {}
            current_total_width = 0
            current_total_pieces = 0
            
            sorted_base_widths = sorted(base_width_counts.keys(), reverse=True)

            for base_width in sorted_base_widths:
                remaining_base_count = base_width_counts[base_width]
                
                for i in range(4, 0, -1):
                    if remaining_base_count < i:
                        continue

                    item_name = f"{base_width}x{i}"
                    if item_name in self.item_info:
                        num_to_use = remaining_base_count // i
                        item_width = self.item_info[item_name]
                        
                        if num_to_use > 0 and \
                           current_total_pieces + num_to_use <= self.max_pieces and \
                           current_total_width + item_width * num_to_use <= self.max_width:
                            
                            new_pattern[item_name] = new_pattern.get(item_name, 0) + num_to_use
                            current_total_width += item_width * num_to_use
                            current_total_pieces += num_to_use
                            remaining_base_count -= num_to_use * i
            
            # 3. 재구성된 패턴을 사용할지 결정
            is_new_pattern_valid = (self.min_width <= current_total_width and self.min_pieces <= current_total_pieces)
            
            chosen_pattern = pattern # 기본적으로 원본 유지
            if is_new_pattern_valid and new_pattern and frozenset(new_pattern.items()) != frozenset(pattern.items()):
                chosen_pattern = new_pattern # 유효하고 변경되었으면 새 패턴 선택

            pattern_key = frozenset(chosen_pattern.items())
            if pattern_key not in seen_patterns:
                processed_patterns.append(chosen_pattern)
                seen_patterns.add(pattern_key)

        original_count = len(self.patterns)
        self.patterns = processed_patterns
        print(f"--- 패턴 통합 완료: {original_count}개 -> {len(self.patterns)}개 패턴으로 정리됨 ---")

    def _solve_master_problem_ilp(self, is_final_mip=False):
        """마스터 문제(Master Problem)를 정수계획법으로 해결합니다."""
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        if is_final_mip:
            solver.SetTimeLimit(SOLVER_TIME_LIMIT_MS)

        # 변수 정의
        x = {j: solver.IntVar(0, solver.infinity(), f'P_{j}') if is_final_mip else solver.NumVar(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}
        over_prod_vars = {w: solver.NumVar(0, solver.infinity(), f'Over_{w}') for w in self.demands_in_rolls}
        
        under_prod_vars = {}
        for width, required_rolls in self.demands_in_rolls.items():
            allowed_under_prod = max(1, math.ceil(required_rolls))
            under_prod_vars[width] = solver.NumVar(0, allowed_under_prod, f'Under_{width}')

        # 제약조건: 생산량 + 부족량 = 수요량 + 과생산량
        constraints = {}
        for width, required_rolls in self.demands_in_rolls.items():
            production_for_width = solver.Sum(
                x[j] * sum(self.item_composition[item_name].get(width, 0) * count for item_name, count in self.patterns[j].items())
                for j in range(len(self.patterns))
            )
            constraints[width] = solver.Add(production_for_width + under_prod_vars[width] == required_rolls + over_prod_vars[width], f'demand_{width}')

        # 목적함수: 총 롤 수 + 페널티 최소화
        total_rolls = solver.Sum(x.values())
        total_over_prod_penalty = solver.Sum(OVER_PROD_PENALTY * var for var in over_prod_vars.values())
        total_under_prod_penalty = solver.Sum(UNDER_PROD_PENALTY * var for var in under_prod_vars.values())
        total_complexity_penalty = solver.Sum(PATTERN_COMPLEXITY_PENALTY * len(self.patterns[j]) * x[j] for j in range(len(self.patterns)))
        
        solver.Minimize(total_rolls + total_over_prod_penalty + total_under_prod_penalty + total_complexity_penalty)
        
        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            solution = {
                'objective': solver.Objective().Value(),
                'pattern_counts': {j: var.solution_value() for j, var in x.items()},
                'over_production': {w: var.solution_value() for w, var in over_prod_vars.items()},
                'under_production': {w: var.solution_value() for w, var in under_prod_vars.items()}
            }
            if not is_final_mip:
                solution['duals'] = {w: constraints[w].dual_value() for w in self.demands_in_rolls}
            return solution
        return None

    def _solve_subproblem_dp(self, duals):
        """서브 문제(Sub-problem)를 무한 배낭 방식으로 풀어 새로운 패턴 후보를 찾는다."""
        width_limit = self.max_width
        piece_limit = self.max_pieces

        # item_details 리스트 초기화: (item_name, item_width, item_value)를 저장
        item_details = []
        for item_name in self.items:
            item_width = self.item_info[item_name]
            # duals 값을 이용하여 item_value 계산
            item_value = sum(count * duals.get(width, 0) for width, count in self.item_composition[item_name].items())
            # item_value가 양수인 경우에만 item_details에 추가
            if item_value <= 0:
                continue
            item_details.append((item_name, item_width, item_value))

        # item_details가 비어있으면 빈 리스트 반환
        # 동적 프로그래밍(DP) 테이블 초기화
        if not item_details:
            return []

        dp_value = [[float('-inf')] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_parent = [[None] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_value[0][0] = 0.0

        # DP 테이블 채우기: pieces와 width에 대한 모든 조합 고려
        for pieces in range(piece_limit + 1):
            for width in range(width_limit + 1):
                # 현재 상태의 값 가져오기
                current_value = dp_value[pieces][width]
                # 현재 상태에 도달할 수 없는 경우, 다음 상태로 건너뛰기
                if current_value == float('-inf'):
                    continue
                # 모든 아이템을 반복하여 추가할지 고려
                for item_name, item_width, item_value in item_details:
                    # 다음 상태 계산
                    next_pieces = pieces + 1
                    next_width = width + item_width
                    # 다음 상태가 범위를 벗어나면 건너뛰기
                    if next_pieces > piece_limit or next_width > width_limit:
                        continue
                    # 새로운 값 계산 및 DP 테이블 업데이트
                    new_value = current_value + item_value
                    if new_value > dp_value[next_pieces][next_width] + 1e-9:
                        dp_value[next_pieces][next_width] = new_value
                        dp_parent[next_pieces][next_width] = (pieces, width, item_name)

        candidate_patterns = []
        seen_patterns = set()

        # 최적의 패턴 후보 추출
        for pieces in range(self.min_pieces, piece_limit + 1):
            for width in range(self.min_width, width_limit + 1):
                # 현재 값과 부모 정보 가져오기
                value = dp_value[pieces][width]
                # 값이 특정 임계값보다 작으면 건너뛰기
                if value <= 1.0 + 1e-6:
                    continue
                parent = dp_parent[pieces][width]
                # 부모가 없으면 건너뛰기
                if not parent:
                    continue

                # 패턴 재구성
                pattern = {}
                cur_pieces, cur_width = pieces, width
                # 부모를 따라 현재 패턴 재구성
                while cur_pieces > 0:
                    parent_info = dp_parent[cur_pieces][cur_width]
                    if not parent_info:
                        pattern = None
                        break
                    prev_pieces, prev_width, item_name = parent_info
                    pattern[item_name] = pattern.get(item_name, 0) + 1
                    cur_pieces, cur_width = prev_pieces, prev_width

                # 패턴이 유효하지 않으면 건너뛰기
                if not pattern or cur_pieces != 0 or cur_width != 0:
                    continue

                # 패턴이 이미 seen_patterns에 있으면 건너뛰기
                pattern_key = frozenset(pattern.items())
                if pattern_key in seen_patterns:
                    continue

                total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                if total_width < self.min_width or total_width > self.max_width:
                    # 패턴이 너비 제약 조건을 충족하지 못하면 건너뛰기
                    continue

                seen_patterns.add(pattern_key)
                candidate_patterns.append({'pattern': pattern, 'value': value, 'width': total_width, 'pieces': pieces})

        if not candidate_patterns:
            return []
        
        # 가치에 따라 후보 정렬
        candidate_patterns.sort(key=lambda x: x['value'], reverse=True)

        # 상위 N개 패턴 반환
        return [cand['pattern'] for cand in candidate_patterns[:CG_SUBPROBLEM_TOP_N]]

    def _generate_all_patterns(self):
        """작은 문제에 대해 모든 가능한 패턴을 생성합니다 (Brute-force)."""
        all_patterns = []
        seen_patterns = set()
        item_list = list(self.items)

        def find_combinations_recursive(start_index, current_pattern, current_width, current_pieces):
            if self.min_width <= current_width <= self.max_width and self.min_pieces <= current_pieces <= self.max_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in seen_patterns:
                    all_patterns.append(current_pattern.copy())
                    seen_patterns.add(pattern_key)

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

    def run_optimize(self):
        """최적화 실행 메인 함수"""
        if len(self.order_widths) <= SMALL_PROBLEM_THRESHOLD:
            print(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 모든 패턴을 탐색합니다 (Small-scale) ---")
            self._generate_all_patterns()
        else:
            print(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 열 생성 기법을 시작합니다 (Large-scale) ---")
            # self._generate_initial_patterns_db() # DB에서 가져온 패턴을 먼저 추가
            self._generate_initial_patterns()
            
            initial_pattern_count = len(self.patterns)
            self.patterns = [p for p in self.patterns if sum(self.item_info[i] * c for i, c in p.items()) >= self.min_width - 200]
            print(f"--- 초기 패턴 필터링: {initial_pattern_count}개 -> {len(self.patterns)}개 (최소 너비 {self.min_width}mm 적용)")

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
                    current_pattern_keys = {frozenset(p.items()) for p in self.patterns}
                    for new_pattern in new_patterns:
                        if frozenset(new_pattern.items()) not in current_pattern_keys:
                            pattern_width = sum(self.item_info[item] * count for item, count in new_pattern.items())
                            if pattern_width >= self.min_width:
                                self.patterns.append(new_pattern)
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

        print(f"\n--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)        
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
        return self._format_results(final_solution)

    def _format_results(self, final_solution):
        """최종 결과를 데이터프레임 형식으로 포매팅합니다."""
        
        # 결과 데이터프레임 생성
        result_patterns, pattern_details_for_db, pattern_roll_details_for_db, demand_tracker = self._build_pattern_details(final_solution)
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['Pattern', 'wd_width', 'Count', 'Loss_per_Roll']]

        # 주문 이행 요약 생성 (수정된 _build_fulfillment_summary 호출)
        fulfillment_summary = self._build_fulfillment_summary(demand_tracker)

        print("\n[주문 이행 요약 (그룹오더별)]")
        # print(fulfillment_summary.to_string())

        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "fulfillment_summary": fulfillment_summary
        }

    def _build_pattern_details(self, final_solution):
        """
        패턴 사용 결과와 DB 저장을 위한 상세 정보를 생성합니다.
        이 메서드는 최적화 결과(패턴별 총 생산량)를 개별 group_order_no의 수요에 맞게 분배(disaggregation)합니다.
        """
        # 1. group_order_no별 수요를 추적하기 위한 데이터프레임 생성
        demand_tracker = self.df_orders.copy()
        demand_tracker['original_order_idx'] = demand_tracker.index # 원본 인덱스 보존
        demand_tracker = demand_tracker[['original_order_idx', 'group_order_no', '지폭', 'rolls']].copy()
        demand_tracker['fulfilled'] = 0
        # 할당 순서를 고정하기 위해 정렬
        demand_tracker = demand_tracker.sort_values(by=['지폭', 'group_order_no']).reset_index(drop=True)

        # 최종 결과를 저장할 리스트
        result_patterns = []
        pattern_details_for_db = []
        pattern_roll_details_for_db = []
        
        prod_seq_counter = 0

        # 패턴별로 한 번만 수행할 정보 미리 계산 (for display summary)
        pattern_summary_map = {}
        for j, pattern_dict in enumerate(self.patterns):
            sorted_pattern_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)
            
            pattern_item_strs = []
            total_width_for_pattern = 0
            
            for item_name, num_of_composite in sorted_pattern_items:
                composite_width = self.item_info[item_name]
                total_width_for_pattern += composite_width * num_of_composite
                
                base_width_dict = self.item_composition[item_name]
                base_width, num_of_base = list(base_width_dict.items())[0]
                
                formatted_name = f"{composite_width}({base_width}*{num_of_base})"
                pattern_item_strs.extend([formatted_name] * num_of_composite)
                
            pattern_summary_map[j] = {
                'Pattern': ' + '.join(pattern_item_strs),
                'wd_width': total_width_for_pattern,
                'Loss_per_Roll': self.original_max_width - total_width_for_pattern
            }

        # 2. 최적해의 패턴별로 루프를 돌며 생산량을 분배
        for j, count in final_solution['pattern_counts'].items():
            if count < 0.99:
                continue
            
            roll_count = int(round(count))
            pattern_dict = self.patterns[j]
            
            # 화면 표시용 요약 정보에 집계된 롤 수 추가
            summary = pattern_summary_map[j].copy() # Use copy to avoid modifying map
            summary['Count'] = roll_count
            result_patterns.append(summary)

            # 3. 패턴이 사용된 횟수(roll_count)만큼 반복하여 각 롤에 대한 DB 레코드 생성
            for _ in range(roll_count):
                prod_seq_counter += 1
                
                composite_widths_for_db = []
                composite_group_nos_for_db = []

                roll_seq_counter = 0
                sorted_pattern_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)
                for item_name, num_of_composite in sorted_pattern_items:
                    composite_width = self.item_info[item_name]
                    base_width_dict = self.item_composition[item_name]

                    for _ in range(num_of_composite):
                        roll_seq_counter += 1
                        
                        base_widths_for_item = []
                        base_group_nos_for_item = []
                        assigned_group_no_for_composite = None

                        for base_width, num_of_base in base_width_dict.items():
                            for _ in range(num_of_base):
                                piece_width = base_width
                                
                                # Demand tracking
                                target_indices = demand_tracker[
                                    (demand_tracker['지폭'] == piece_width) &
                                    (demand_tracker['fulfilled'] < demand_tracker['rolls'])
                                ].index
                                
                                assigned_group_no = None
                                if not target_indices.empty:
                                    target_idx = target_indices.min()
                                    demand_tracker.loc[target_idx, 'fulfilled'] += 1
                                    assigned_group_no = demand_tracker.loc[target_idx, 'group_order_no']
                                else:
                                    fallback_indices = demand_tracker[demand_tracker['지폭'] == piece_width].index
                                    if not fallback_indices.empty:
                                        assigned_group_no = demand_tracker.loc[fallback_indices.min(), 'group_order_no']
                                    else:
                                        assigned_group_no = "ERROR" # 데이터 오류
                                
                                base_widths_for_item.append(base_width)
                                base_group_nos_for_item.append(assigned_group_no)

                                if assigned_group_no_for_composite is None:
                                    assigned_group_no_for_composite = assigned_group_no
                        
                        composite_widths_for_db.append(composite_width)
                        composite_group_nos_for_db.append(assigned_group_no_for_composite if assigned_group_no_for_composite is not None else "")

                        pattern_roll_details_for_db.append({
                            'rollwidth': composite_width,
                            'roll_production_length': self.sheet_roll_length,
                            'widths': (base_widths_for_item + [0] * 7)[:7],
                            'group_nos': (base_group_nos_for_item + [''] * 7)[:7],
                            'Count': 1,
                            'Prod_seq': prod_seq_counter,
                            'Roll_seq': roll_seq_counter
                        })

                pattern_details_for_db.append({
                    'roll_production_length': self.sheet_roll_length,
                    'Count': 1,
                    'widths': (composite_widths_for_db + [0] * 8)[:8],
                    'group_nos': (composite_group_nos_for_db + [''] * 8)[:8],
                    'Prod_seq': prod_seq_counter
                })

        return result_patterns, pattern_details_for_db, pattern_roll_details_for_db, demand_tracker

    def _build_fulfillment_summary(self, demand_tracker):
        """주문 이행 요약 데이터프레임을 생성합니다. (개별 주문별)"""
        
        summary_df = self.df_orders[['group_order_no', '가로', '세로', '수출내수', '등급', '주문톤', 'rolls']].copy()
        summary_df.rename(columns={'rolls': '필요롤수', '주문톤': '주문량(톤)'}, inplace=True)
        
        # Merge fulfilled_rolls directly using original_order_idx
        # demand_tracker has 'original_order_idx' from _build_pattern_details
        summary_df = pd.merge(summary_df, demand_tracker[['original_order_idx', 'fulfilled']], 
                              left_index=True, right_on='original_order_idx', how='left')
        summary_df.rename(columns={'fulfilled': '생산롤수'}, inplace=True)
        summary_df.drop(columns=['original_order_idx'], inplace=True) # Drop the temporary merge key

        summary_df['생산롤수'] = summary_df['생산롤수'].fillna(0).astype(int)
        
        # 4. 과부족 및 생산톤 계산
        summary_df['과부족(롤)'] = summary_df['생산롤수'].astype(int) - summary_df['필요롤수'].astype(int)
        
        # 0으로 나누기 오류를 방지하며 롤당 톤 계산
        tons_per_roll = (summary_df['주문량(톤)'] / summary_df['필요롤수']).replace([float('inf'), -float('inf')], 0).fillna(0)
        
        summary_df['생산량(톤)'] = (summary_df['생산롤수'] * tons_per_roll).round(3)
        summary_df['과부족(톤)'] = (summary_df['생산량(톤)'] - summary_df['주문량(톤)']).round(0)

        # 최종 컬럼 순서 정리
        return summary_df[[
            'group_order_no', '가로', '세로', '수출내수', '등급', '주문량(톤)', '생산량(톤)', '과부족(톤)',
            '필요롤수', '생산롤수', '과부족(롤)'
        ]]
    
    def _generate_initial_patterns_test(self):
        """초기 패턴 생성을 위해 First-Fit-Decreasing 휴리스틱을 사용합니다."""
        print("\n--- 유효한 초기 패턴을 생성합니다 ---")
        
        # frozenset으로 패턴 중복 체크를 효율적으로 관리
        seen_patterns = set()

        # # 기존: 너비가 넓은 아이템부터 순서대로 처리 (First-Fit-Decreasing)
        # # sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)

        # 변경: 아이템 순서를 랜덤으로 섞어 다양한 패턴 생성 시도 (First-Fit)
        # 이렇게 하면 매번 실행할 때마다 다른 순서로 패턴 생성을 시도하여
        # 더 다양한 초기 패턴을 탐색할 수 있습니다.
        # randomized_items = list(self.items)
        # random.shuffle(randomized_items)
        
        

        # for item in randomized_items:
        #     item_width = self.item_info[item]
            
        #     current_pattern = {item: 1}
        #     current_width = item_width
        #     current_pieces = 1

        #     while current_pieces < self.max_pieces:
        #         remaining_width = self.max_width - current_width
                
        #         # 남은 공간에 맞는 '첫 번째' 아이템을 찾음 (First-Fit)
        #         # randomized_items가 정렬되어 있지 않으므로, 가장 큰 아이템이 아니라 랜덤 순서에서 처음으로 발견되는 아이템이 선택됩니다.
        #         best_fit_item = next((i for i in randomized_items if self.item_info[i] <= remaining_width), None)
                
        #         if not best_fit_item:
        #             break 

        #         current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
        #         current_width += self.item_info[best_fit_item]
        #         current_pieces += 1

        #     while current_width < self.min_width and current_pieces < self.max_pieces:
        #         # 너비가 min_width보다 작은 경우, 추가 아이템을 탐색하여 보정합니다.
        #         item_to_add = next((i for i in reversed(randomized_items) if current_width + self.item_info[i] <= self.max_width), None)
                
        #         if item_to_add:
        #             current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
        #             current_width += self.item_info[item_to_add]
        #             current_pieces += 1
        #         else:
        #             break # 더 이상 추가할 아이템이 없으면 종료

        #     if self.min_width <= current_width and self.min_pieces <= current_pieces:
        #         pattern_key = frozenset(current_pattern.items())
        #         if pattern_key not in seen_patterns:
        #             self.patterns.append(current_pattern)
        #             seen_patterns.add(pattern_key)
        # print(f"--- {len(self.patterns)}개의 혼합 패턴 생성됨 ---")

        # 너비가 넓은 아이템부터 순서대로 처리 (First-Fit-Decreasing)
        sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)

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

            # 너비가 min_width보다 작은 경우 보정
            while current_width < self.min_width and current_pieces < self.max_pieces:
                # 추가해도 max_width를 넘지 않는 가장 적절한 아이템 탐색
                item_to_add = next((i for i in reversed(sorted_items) if current_width + self.item_info[i] <= self.max_width), None)
                
                if item_to_add:
                    current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
                    current_width += self.item_info[item_to_add]
                    current_pieces += 1
                else:
                    break # 더 이상 추가할 아이템이 없으면 종료

            # 최종 유효성 검사 후 패턴 추가
            if self.min_width <= current_width and self.min_pieces <= current_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in seen_patterns:
                    self.patterns.append(current_pattern)
                    seen_patterns.add(pattern_key)
        print(f"--- {len(self.patterns)}개의 혼합 패턴 생성됨 ---")


        # 너비가 작은 아이템부터 순서대로 처리 (First-Fit-Decreasing)
        sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=False)

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

            # 너비가 min_width보다 작은 경우 보정
            while current_width < self.min_width and current_pieces < self.max_pieces:
                # 추가해도 max_width를 넘지 않는 가장 적절한 아이템 탐색
                item_to_add = next((i for i in reversed(sorted_items) if current_width + self.item_info[i] <= self.max_width), None)
                
                if item_to_add:
                    current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
                    current_width += self.item_info[item_to_add]
                    current_pieces += 1
                else:
                    break # 더 이상 추가할 아이템이 없으면 종료

            # 최종 유효성 검사 후 패턴 추가
            if self.min_width <= current_width and self.min_pieces <= current_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in seen_patterns:
                    self.patterns.append(current_pattern)
                    seen_patterns.add(pattern_key)
        print(f"--- {len(self.patterns)}개의 혼합 패턴 생성됨 ---")

        # --- 2. 모든 복합폭에 대해 '순수 품목 패턴' 생성 ---
        pure_patterns_added = 0
        for item in sorted_items:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0: continue

            # 해당 아이템으로만 구성된 패턴 생성 시도
            num_items = min(int(self.max_width / item_width), self.max_pieces)
            
            # 너비가 큰 조합부터 작은 조합까지 순차적으로 확인
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                
                if self.min_width <= total_width and self.min_pieces <= num_items:
                    pattern_key = frozenset(new_pattern.items())
                    if pattern_key not in seen_patterns:
                        self.patterns.append(new_pattern)
                        seen_patterns.add(pattern_key)
                        pure_patterns_added += 1
                        break # 이 아이템으로 만들 수 있는 가장 좋은 순수패턴을 찾았으므로 종료
                
                num_items -= 1

        if pure_patterns_added > 0:
            print(f"--- {pure_patterns_added}개의 순수 품목 패턴 추가됨 ---")

        # --- 폴백 로직: 초기 패턴으로 커버되지 않는 주문이 있는지 최종 확인 ---
        covered_widths = {w for p in self.patterns for item_name in p for w in self.item_composition.get(item_name, {})}
        uncovered_widths = set(self.order_widths) - covered_widths

        if uncovered_widths:
            print(f"--- 경고: 초기 패턴에 포함되지 않은 주문 발견: {uncovered_widths} ---")
            print("--- 해당 주문에 대한 폴백 패턴을 추가 생성합니다. ---")
            
            for width in uncovered_widths:
                print(f"  - 지폭 {width}mm에 대한 순수 품목 패턴 생성 시도...")

                # 1. 이 지폭으로 만들 수 있는 유효한 복합폭 아이템 목록을 찾습니다.
                valid_components = []
                for i in range(1, 5): # 1~4폭 고려
                    item_name = f"{width}x{i}"
                    # 아이템이 이미 생성되었는지 확인
                    if item_name in self.item_info:
                        valid_components.append(item_name)
                    else:
                        # 동적으로 생성 및 유효성 검사
                        composite_width = width * i + self.sheet_trim
                        if (self.min_sc_width <= composite_width <= self.max_sc_width) and \
                           (composite_width <= self.original_max_width):
                            # 유효하면 아이템 정보에 추가
                            if item_name not in self.items: self.items.append(item_name)
                            self.item_info[item_name] = composite_width
                            self.item_composition[item_name] = {width: i}
                            valid_components.append(item_name)

                if not valid_components:
                    print(f"    - 경고: 지폭 {width}mm로 만들 수 있는 유효한 복합폭 아이템이 없습니다. 폴백 패턴을 생성할 수 없습니다.")
                    continue

                # 2. 너비가 넓은 순으로 정렬하여 Greedy 알고리즘 준비
                sorted_components = sorted(valid_components, key=lambda i: self.item_info[i], reverse=True)
                
                # 3. Greedy 방식으로 최적의 단일 품목 패턴 구성
                new_pattern = {}
                current_width = 0
                current_pieces = 0
                
                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    
                    # 남은 공간에 들어갈 수 있는 가장 큰 구성요소 찾기
                    best_fit = next((item for item in sorted_components if self.item_info[item] <= remaining_width), None)
                    
                    if not best_fit:
                        break # 더 이상 추가할 수 있는 구성요소가 없음
                    
                    new_pattern[best_fit] = new_pattern.get(best_fit, 0) + 1
                    current_width += self.item_info[best_fit]
                    current_pieces += 1

                # 4. 생성된 패턴의 유효성 검사 및 추가
                if new_pattern:
                    total_width = sum(self.item_info[name] * count for name, count in new_pattern.items())
                    total_pieces = sum(new_pattern.values())

                    if self.min_width -200 <= total_width and self.min_pieces <= total_pieces:
                        pattern_key = frozenset(new_pattern.items())
                        if pattern_key not in seen_patterns:
                            self.patterns.append(new_pattern)
                            seen_patterns.add(pattern_key)
                            print(f"    -> 생성된 순수 패턴: {new_pattern} (너비: {total_width}mm, 폭 수: {total_pieces}) -> 폴백 패턴으로 추가됨.")
                        else:
                            print(f"    - 생성된 순수 패턴 {new_pattern}은 이미 존재합니다.")
                    else:
                        print(f"    - 생성된 순수 패턴 {new_pattern}이 최종 제약조건(최소너비/폭수)을 만족하지 못합니다. (너비: {total_width}, 폭 수: {total_pieces})")
                else:
                    print(f"    - 지폭 {width}mm에 대한 순수 패턴을 구성하지 못했습니다.")

        print(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")
        print(self.patterns)
        print("--------------------------\n")
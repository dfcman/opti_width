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
CG_SUBPROBLEM_TOP_N = 1         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴

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
        self.df_orders = df_spec_pre.copy()
        
        self.demands_in_rolls = self._calculate_demand_rolls(self.df_orders)
        self.order_widths = list(self.demands_in_rolls.keys()) 

        width_summary = {}
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
        """주문량을 바탕으로 지폭별 필요 롤 수를 계산합니다."""
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

        df_copy['rolls'] = df_copy.apply(calculate_rolls, axis=1)
        demand_rolls = df_copy.groupby('지폭')['rolls'].sum().astype(int).to_dict()

        print("\n--- 지폭별 필요 롤 수 ---")
        # for width, rolls in demand_rolls.items():
        #     print(f"  지폭 {width}mm: {rolls} 롤")
        print("--------------------------\n")
        
        return demand_rolls

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

        # frozenset으로 패턴 중복 체크를 효율적으로 관리
        seen_patterns = set()
        # 주문량이 많은 아이템부터 순서대로 처리
        sorted_items = sorted(
            self.items,
            key=lambda i: self.demands_in_rolls.get(list(self.item_composition[i].keys())[0], 0),
            reverse=False
        )

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

        # 너비가 큰 아이템부터 순서대로 처리 (First-Fit-Decreasing)
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

        # 너비가 큰 아이템부터 순서대로 처리 (First-Fit-Decreasing)
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

        item_details = []
        for item_name in self.items:
            item_width = self.item_info[item_name]
            item_value = sum(count * duals.get(width, 0) for width, count in self.item_composition[item_name].items())
            if item_value <= 0:
                continue
            item_details.append((item_name, item_width, item_value))

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
                for item_name, item_width, item_value in item_details:
                    next_pieces = pieces + 1
                    next_width = width + item_width
                    if next_pieces > piece_limit or next_width > width_limit:
                        continue
                    new_value = current_value + item_value
                    if new_value > dp_value[next_pieces][next_width] + 1e-9:
                        dp_value[next_pieces][next_width] = new_value
                        dp_parent[next_pieces][next_width] = (pieces, width, item_name)

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

                seen_patterns.add(pattern_key)
                candidate_patterns.append({'pattern': pattern, 'value': value, 'width': total_width, 'pieces': pieces})

        if not candidate_patterns:
            return []

        candidate_patterns.sort(key=lambda x: x['value'], reverse=True)
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
        
        # 최종 생산량 계산
        final_production_rolls = {width: 0 for width in self.order_widths}
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                roll_count = int(round(count))
                for item_name, num_in_pattern in self.patterns[j].items():
                    for width, num_pieces in self.item_composition[item_name].items():
                        if width in final_production_rolls:
                            final_production_rolls[width] += roll_count * num_in_pattern * num_pieces

        # 결과 데이터프레임 생성
        result_patterns, pattern_details_for_db = self._build_pattern_details(final_solution)
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['Pattern', 'Roll_Production_Length', 'Count', 'Loss_per_Roll']]

        # 주문 이행 요약 생성
        fulfillment_summary = self._build_fulfillment_summary(final_production_rolls)

        print("\n[주문 이행 요약]")
        # print(fulfillment_summary.to_string())

        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "fulfillment_summary": fulfillment_summary
        }

    def _build_pattern_details(self, final_solution):
        """패턴 사용 결과와 DB 저장을 위한 상세 정보를 생성합니다."""
        result_patterns = []
        pattern_details_for_db = []

        for j, count in final_solution['pattern_counts'].items():
            if count < 0.99:
                continue
            
            roll_count = int(round(count))
            pattern_dict = self.patterns[j]
            
            sorted_pattern_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)

            db_widths, db_group_nos, pattern_item_strs = [], [], []
            total_width = 0

            for item_name, num in sorted_pattern_items:
                width = self.item_info[item_name]
                total_width += width * num
                db_widths.extend([width] * num)
                db_group_nos.extend([item_name] * num)
                
                base_width, multiplier = map(int, item_name.split('x'))
                formatted_name = f"{self.item_info[item_name]}({base_width}*{multiplier})"
                pattern_item_strs.extend([formatted_name] * num)

            result_patterns.append({
                'Pattern': ' + '.join(pattern_item_strs),
                'Roll_Production_Length': total_width,
                'Count': roll_count,
                'Loss_per_Roll': self.original_max_width - total_width
            })
            pattern_details_for_db.append({
                'Count': roll_count,
                'widths': (db_widths + [0] * 8)[:8],
                'group_nos': (db_group_nos + [''] * 8)[:8]
            })
        return result_patterns, pattern_details_for_db

    def _build_fulfillment_summary(self, final_production_rolls):
        """주문 이행 요약 데이터프레임을 생성합니다."""
        summary_data = []
        for width, required_rolls in self.demands_in_rolls.items():
            produced_rolls = final_production_rolls.get(width, 0)
            order_tons = self.width_summary[width]['order_tons']
            
            if required_rolls > 0:
                tons_per_roll = order_tons / required_rolls
                produced_tons = produced_rolls * tons_per_roll
            else:
                produced_tons = 0
            
            over_prod_tons = produced_tons - order_tons

            summary_data.append({
                '지폭': width,
                '주문량(톤)': order_tons,
                '생산량(톤)': round(produced_tons, 2),
                '과부족(톤)': round(over_prod_tons, 2),
                '필요롤수': required_rolls,
                '생산롤수': produced_rolls,
                '과부족(롤)': produced_rolls - required_rolls,
            })
        
        return pd.DataFrame(summary_data)[[
            '지폭', '주문량(톤)', '생산량(톤)', '과부족(톤)',
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

import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import math
import random
import time

# --- 최적화 설정 상수 ---
# 페널티 값
OVER_PROD_PENALTY = 200.0    # 과생산에 대한 페널티
UNDER_PROD_PENALTY = 500.0  # 부족생산에 대한 페널티
PATTERN_COMPLEXITY_PENALTY = 0.01  # 패턴 복잡성에 대한 페널티
ITEM_SINGLE_STRIP_PENALTIES = {}
DEFAULT_SINGLE_STRIP_PENALTY = 100.0  # 지정되지 않은 단일폭은 기본적으로 패널티 없음
DISALLOWED_SINGLE_BASE_WIDTHS = {}  # 단일 사용을 금지할 주문 폭 집합

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 2      # 패턴에 포함될 수 있는 최소 폭(piece)의 수
SMALL_PROBLEM_THRESHOLD = 8     # 전체 탐색을 수행할 최대 주문 지폭 종류 수
SOLVER_TIME_LIMIT_MS = 300000    # 최종 MIP 솔버의 최대 실행 시간 (밀리초)
CG_MAX_ITERATIONS = 100000         # 열 생성(Column Generation) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 100    # 개선 없는 경우, 열 생성 조기 종료 조건
CG_SUBPROBLEM_TOP_N = 10         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴
# 나이프 로드 제약: 패턴 생산 횟수는 k1*a + k2*b 형태여야 함 (a,b>=0 정수)
KNIFE_LOAD_K1 = 3
KNIFE_LOAD_K2 = 4

class SheetOptimizeVar:
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
        df_spec_pre['지폭'] = df_spec_pre['가로']

        self.b_wgt = b_wgt
        self.min_sheet_roll_length = min_sheet_roll_length
        self.max_sheet_roll_length = max_sheet_roll_length
        self.sheet_trim = sheet_trim
        self.original_max_width = max_width
        
        # 수요 단위를 '롤'에서 '미터'로 변경하고, 'meters' 열이 추가된 데이터프레임을 받음
        self.df_orders, self.demands_in_meters, self.order_sheet_lengths = self._calculate_demand_meters(df_spec_pre)
        self.order_widths = list(self.demands_in_meters.keys())

        width_summary = {}
        tons_per_width = self.df_orders.groupby('지폭')['주문톤'].sum()
        for width, required_meters in self.demands_in_meters.items():
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
        print(f"--- 패턴 제약조건: 최소 {self.min_pieces}폭, 최대 {self.max_pieces}폭 ---")

        # self.patterns는 이제 {'composition': dict, 'length': float, 'penalty': float} 형태의 딕셔너리 리스트가 됨
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

    def _calculate_demand_meters(self, df_orders):
        """주문량을 바탕으로 지폭별 필요 총 길이(미터)를 계산하고, 'meters' 열이 추가된 데이터프레임을 반환합니다."""
        df_copy = df_orders.copy()

        def calculate_meters(row):
            width_mm = row['가로']
            length_mm = row['세로']
            order_ton = row['주문톤']

            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0 or order_ton <= 0:
                return 0

            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1000000
            if sheet_weight_g <= 0:
                return 0

            total_sheets_needed = (order_ton * 1000000) / sheet_weight_g
            total_meters_needed = total_sheets_needed * (length_mm / 1000)
            return total_meters_needed

        df_copy['meters'] = df_copy.apply(calculate_meters, axis=1)
        demand_meters = df_copy.groupby('지폭')['meters'].sum().to_dict()
        order_sheet_lengths = df_copy.groupby('지폭')['세로'].first().to_dict()

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        print("--- 지폭별 필요 총 길이 ---")
        print("--------------------------")

        return df_copy, demand_meters, order_sheet_lengths

    def _generate_initial_patterns_db(self):
        """DB의 기존 패턴을 불러와 초기 패턴 '조합'을 생성합니다."""
        if not self.db or not self.lot_no or not self.version:
            print("--- DB 정보가 없어 기존 패턴을 불러올 수 없습니다. ---")
            return []

        print("--- DB에서 기존 패턴을 불러와 초기 패턴 조합을 생성합니다. ---")
        db_patterns_list = self.db.get_patterns_from_db(self.lot_no, self.version)

        if not db_patterns_list:
            print("--- DB에 저장된 기존 패턴이 없습니다. ---")
            return []

        initial_compositions = []
        for pattern_item_list in db_patterns_list:
            pattern_dict = dict(Counter(pattern_item_list))
            
            is_valid = all(item_name in self.items for item_name in pattern_dict.keys())
            if is_valid:
                initial_compositions.append(pattern_dict)
            else:
                invalid_items = [name for name in pattern_dict if name not in self.items]
                print(f"- 경고: DB 패턴 {pattern_dict}의 아이템 {invalid_items}이(가) 현재 주문에 없어 해당 패턴을 무시합니다.")
        
        return initial_compositions

    def _generate_initial_patterns(self):
        """휴리스틱을 사용하여 초기 패턴 '조합'을 생성합니다."""
        
        compositions = []
        seen_compositions = set()

        # 다양한 휴리스틱을 적용하여 초기 조합 생성
        # 1. 주문량 기반 FFD (수요가 적은 폭부터 채우기)
        sorted_items_demand = sorted(
            self.items,
            key=lambda i: self.demands_in_meters.get(list(self.item_composition[i].keys())[0], 0),
            reverse=False
        )
        # 2. 너비 기반 FFD (넓은 폭부터 채우기)
        sorted_items_width_desc = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        # 3. 너비 기반 FFD (좁은 폭부터 채우기)
        sorted_items_width_asc = sorted(self.items, key=lambda i: self.item_info[i], reverse=False)

        heuristics = [sorted_items_demand, sorted_items_width_desc, sorted_items_width_asc]

        for sorted_items in heuristics:
            for item in sorted_items:
                item_width = self.item_info[item]
                
                current_pattern = {item: 1}
                current_width = item_width
                current_pieces = 1

                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    best_fit_item = next((i for i in sorted_items if self.item_info[i] <= remaining_width), None)
                    
                    if not best_fit_item:
                        break 

                    current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
                    current_width += self.item_info[best_fit_item]
                    current_pieces += 1

                while current_width < self.min_width and current_pieces < self.max_pieces:
                    item_to_add = next((i for i in reversed(sorted_items) if current_width + self.item_info[i] <= self.max_width), None)
                    if item_to_add:
                        current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
                        current_width += self.item_info[item_to_add]
                        current_pieces += 1
                    else:
                        break

                if self.min_width <= current_width and self.min_pieces <= current_pieces:
                    comp_key = frozenset(current_pattern.items())
                    if comp_key not in seen_compositions:
                        compositions.append(current_pattern)
                        seen_compositions.add(comp_key)

        # 순수 품목 패턴 추가
        for item in self.items:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0: continue
            num_items = min(int(self.max_width / item_width), self.max_pieces)
            
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                if self.min_width <= total_width and self.min_pieces <= num_items:
                    comp_key = frozenset(new_pattern.items())
                    if comp_key not in seen_compositions:
                        compositions.append(new_pattern)
                        seen_compositions.add(comp_key)
                        break
                num_items -= 1
        
        # 폴백 로직: 커버되지 않은 주문에 대한 순수 패턴 생성
        covered_widths = {w for p in compositions for item_name in p for w in self.item_composition.get(item_name, {})}
        uncovered_widths = set(self.order_widths) - covered_widths
        if uncovered_widths:
            for width in uncovered_widths:
                # ... (폴백 로직은 단순화를 위해 생략, 필요시 복원) ...
                pass

        print(f"--- {len(compositions)}개의 초기 패턴 조합 생성됨 ---")

        # --- 생성된 패턴 조합들을 후처리하여 작은 복합폭들을 통합 ---
        consolidated_compositions = self._consolidate_compositions(compositions)
        
        return consolidated_compositions

    def _consolidate_compositions(self, compositions):
        """
        생성된 초기 패턴 조합들을 후처리하여 작은 복합폭 아이템들을 가능한 큰 복합폭 아이템으로 통합합니다.
        """
        print("\n--- 생성된 패턴 조합에 대해 후처리(통합)를 시작합니다. ---")
        
        processed_compositions = []
        # frozenset을 키로 하고 원본 comp를 값으로 하는 딕셔셔너리로 변경하여 중복 방지 및 원본 추적
        seen_compositions_map = {}

        for comp in compositions:
            # 1. 패턴을 기본 지폭 단위로 분해
            base_width_counts = Counter()
            for item_name, count in comp.items():
                composition_details = self.item_composition.get(item_name)
                if composition_details:
                    for base_width, num_base in composition_details.items():
                        base_width_counts[base_width] += num_base * count
            
            # 2. DP를 사용하여 각 기본 지폭별로 최적의 아이템 조합을 찾아 새로운 패턴 재구성
            new_comp = {}
            sorted_base_widths = sorted(base_width_counts.keys(), reverse=True)

            for base_width in sorted_base_widths:
                total_base_count = base_width_counts[base_width]
                
                # DP 테이블: dp[j] = j개의 기본 지폭을 만드는 데 필요한 최소 아이템 수
                dp = [float('inf')] * (total_base_count + 1)
                dp_path = [None] * (total_base_count + 1)
                dp[0] = 0

                # 이 기본 지폭으로 만들 수 있는 아이템 목록 (예: 700x1, 700x2, 700x3, 700x4)
                possible_items = []
                for i in range(1, 5):
                    item_name = f"{base_width}x{i}"
                    if item_name in self.item_info:
                        possible_items.append({'name': item_name, 'pieces': i})

                # DP 계산
                for j in range(1, total_base_count + 1):
                    for item in possible_items:
                        pieces = item['pieces']
                        if j >= pieces and dp[j - pieces] + 1 < dp[j]:
                            dp[j] = dp[j - pieces] + 1
                            dp_path[j] = item['name']
                
                # DP 경로를 역추적하여 최적 조합 구성
                best_combination = Counter()
                current_count = total_base_count
                while current_count > 0 and dp_path[current_count]:
                    item_name_to_add = dp_path[current_count]
                    best_combination[item_name_to_add] += 1
                    
                    # item_name에서 'x' 뒤의 숫자를 파싱하여 pieces를 구함
                    try:
                        pieces_to_subtract = int(item_name_to_add.split('x')[1])
                        current_count -= pieces_to_subtract
                    except (ValueError, IndexError):
                        # 예외 처리: item_name 형식이 예상과 다를 경우
                        break # 루프 중단

                # new_comp에 최적 조합 추가
                for item_name, count in best_combination.items():
                    new_comp[item_name] = new_comp.get(item_name, 0) + count

            # 3. 재구성된 패턴의 유효성 검사 및 선택
            new_total_width = sum(self.item_info[name] * num for name, num in new_comp.items())
            new_total_pieces = sum(new_comp.values())
            is_new_comp_valid = (self.min_width <= new_total_width <= self.max_width and 
                                 self.min_pieces <= new_total_pieces <= self.max_pieces)
            
            chosen_comp = comp # 기본적으로 원본 유지
            # 새 조합이 유효하고, 원본과 다르며, 더 적은 수의 아이템을 사용한다면 선택
            if is_new_comp_valid and new_comp and sum(new_comp.values()) < sum(comp.values()):
                chosen_comp = new_comp # 유효하고 변경되었으면 새 조합 선택

            comp_key = frozenset(chosen_comp.items())
            # 더 적은 아이템을 사용하는 더 나은 조합이 발견되면 기존 것을 교체
            if comp_key not in seen_compositions_map or sum(chosen_comp.values()) < sum(seen_compositions_map[comp_key].values()):
                seen_compositions_map[comp_key] = chosen_comp

        original_count = len(compositions)
        processed_compositions = list(seen_compositions_map.values())
        print(f"--- 패턴 조합 통합 완료: {original_count}개 -> {len(processed_compositions)}개 조합으로 정리됨 ---")
        return processed_compositions

    def _solve_master_problem_ilp(self, is_final_mip=False):
        """마스터 문제(Master Problem)를 정수계획법으로 해결합니다."""
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        if is_final_mip:
            solver.SetTimeLimit(SOLVER_TIME_LIMIT_MS)

        # --- 변수 정의 ---
        if is_final_mip and (KNIFE_LOAD_K1 > 0 or KNIFE_LOAD_K2 > 0):
            print(f"--- 최종 최적화에 나이프 로드 제약(k1={KNIFE_LOAD_K1}, k2={KNIFE_LOAD_K2})을 적용합니다. ---")
            x = {j: solver.IntVar(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}
            y = {j: solver.BoolVar(f'use_P_{j}') for j in range(len(self.patterns))}
            a = {j: solver.IntVar(0, solver.infinity(), f'a_{j}') for j in range(len(self.patterns))}
            b = {j: solver.IntVar(0, solver.infinity(), f'b_{j}') for j in range(len(self.patterns))}
            M = 1_000_000
            for j in range(len(self.patterns)):
                solver.Add(x[j] == KNIFE_LOAD_K1 * a[j] + KNIFE_LOAD_K2 * b[j])
                solver.Add(a[j] <= M * y[j])
                solver.Add(b[j] <= M * y[j])
                solver.Add(a[j] + b[j] >= y[j])
        else:
            x = {j: solver.IntVar(0, solver.infinity(), f'P_{j}') if is_final_mip else solver.NumVar(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}

        over_prod_vars = {w: solver.NumVar(0, solver.infinity(), f'Over_{w}') for w in self.demands_in_meters}
        under_prod_vars = {w: solver.NumVar(0, self.demands_in_meters.get(w, 1000), f'Under_{w}') for w in self.demands_in_meters}

        constraints = {}
        for width, required_meters in self.demands_in_meters.items():
            num_strips_per_pattern = {
                j: sum(self.item_composition[item_name].get(width, 0) * count for item_name, count in self.patterns[j]['composition'].items())
                for j in range(len(self.patterns))
            }
            production_for_width = solver.Sum(
                x[j] * self.patterns[j]['length'] * num_strips_per_pattern[j]
                for j in range(len(self.patterns))
            )
            constraints[width] = solver.Add(production_for_width + under_prod_vars[width] == required_meters + over_prod_vars[width], f'demand_{width}')

        total_roll_length_used = solver.Sum(x[j] * self.patterns[j]['length'] for j in range(len(self.patterns)))
        pattern_usage_penalty = solver.Sum(self.patterns[j]['penalty'] * x[j] for j in range(len(self.patterns)))
        
        METER_BASED_OVER_PROD_PENALTY = OVER_PROD_PENALTY / 1000
        METER_BASED_UNDER_PROD_PENALTY = UNDER_PROD_PENALTY / 1000
        total_over_prod_penalty = solver.Sum(METER_BASED_OVER_PROD_PENALTY * var for var in over_prod_vars.values())
        total_under_prod_penalty = solver.Sum(METER_BASED_UNDER_PROD_PENALTY * var for var in under_prod_vars.values())
        
        solver.Minimize(total_roll_length_used + total_over_prod_penalty + total_under_prod_penalty + pattern_usage_penalty)
        
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
        """서브 문제(Sub-problem)를 무한 배낭 방식으로 풀어 새로운 패턴 '조합'을 찾는다."""
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

        for pieces in range(piece_limit):
            for width in range(width_limit + 1):
                current_value = dp_value[pieces][width]
                if current_value == float('-inf'):
                    continue
                for item_name, item_width, item_value in item_details:
                    next_pieces = pieces + 1
                    next_width = width + item_width
                    if next_width > width_limit:
                        continue
                    new_value = current_value + item_value
                    if new_value > dp_value[next_pieces][next_width] + 1e-9:
                        dp_value[next_pieces][next_width] = new_value
                        dp_parent[next_pieces][next_width] = (pieces, width, item_name)

        candidate_compositions = []
        seen_compositions = set()

        for pieces in range(self.min_pieces, piece_limit + 1):
            for width in range(self.min_width, width_limit + 1):
                value = dp_value[pieces][width]
                if value <= 1.0 + 1e-9:
                    continue

                pattern = {}
                cur_pieces, cur_width = pieces, width
                path_valid = True
                while cur_pieces > 0 and cur_width > 0:
                    parent_info = dp_parent[cur_pieces][cur_width]
                    if not parent_info:
                        path_valid = False
                        break
                    prev_pieces, prev_width, item_name = parent_info
                    pattern[item_name] = pattern.get(item_name, 0) + 1
                    cur_pieces, cur_width = prev_pieces, prev_width
                
                if not path_valid or cur_pieces != 0 or cur_width != 0:
                    continue

                comp_key = frozenset(pattern.items())
                if comp_key in seen_compositions:
                    continue
                
                # reduced cost 계산은 run_optimize에서 수행
                seen_compositions.add(comp_key)
                candidate_compositions.append({'composition': pattern, 'value': value})

        if not candidate_compositions:
            return []

        candidate_compositions.sort(key=lambda x: x['value'], reverse=True)
        return [cand['composition'] for cand in candidate_compositions[:CG_SUBPROBLEM_TOP_N]]

    def _generate_all_patterns(self):
        """작은 문제에 대해 모든 가능한 패턴 '조합'을 생성합니다.
        max_pieces를 패턴에 포함된 '복합폭(item)'의 개수로 계산합니다.
        """
        compositions = []
        seen_compositions = set()
        
        valid_items = []
        for item in self.items:
            item_width = self.item_info.get(item)
            if item_width and self.min_sc_width <= item_width <= self.max_sc_width:
                valid_items.append(item)
            else:
                print(f"--- 경고: _generate_all_patterns에서 유효하지 않은 복합폭 아이템 발견 및 제외: {item} (너비: {item_width}mm) ---")
        
        item_list = list(valid_items)
        n = len(item_list)

        # current_item_count는 패턴에 포함된 복합폭(item)의 총 개수를 추적합니다.
        def find_combinations_recursive(start_index, current_pattern, current_width, current_item_count):
            # 유효한 패턴은 너비와 '복합폭 개수' 제약조건을 만족해야 합니다.
            if self.min_width <= current_width <= self.max_width and self.min_pieces <= current_item_count <= self.max_pieces:
                comp_key = frozenset(current_pattern.items())
                if comp_key not in seen_compositions:
                    compositions.append(current_pattern.copy())
                    seen_compositions.add(comp_key)

            # 복합폭 개수가 max_pieces에 도달하면 더 이상 추가하지 않습니다.
            if current_item_count >= self.max_pieces or start_index >= n:
                return

            for i in range(start_index, n):
                item = item_list[i]
                item_width = self.item_info[item]

                # 너비와 '복합폭 개수'를 기준으로 추가할지 결정합니다.
                if current_width + item_width <= self.max_width and current_item_count + 1 <= self.max_pieces:
                    current_pattern[item] = current_pattern.get(item, 0) + 1
                    # 복합폭 개수(current_item_count)를 1 증가시켜 재귀 호출합니다.
                    find_combinations_recursive(i, current_pattern, current_width + item_width, current_item_count + 1)
                    current_pattern[item] -= 1
                    if current_pattern[item] == 0:
                        del current_pattern[item]

        # 초기 호출 시, 복합폭 개수는 0입니다.
        find_combinations_recursive(0, {}, 0, 0)
        return compositions

    def _calculate_heuristic_pattern_length(self, pattern_comp):
        """패턴 조합에 대한 휴리스틱 기반 롤 길이를 계산합니다."""
        pattern_widths = {w for item_name in pattern_comp for w in self.item_composition.get(item_name, {})}
        if not pattern_widths:
            return self.max_sheet_roll_length

        dominant_width = max(pattern_widths, key=lambda w: self.demands_in_meters.get(w, 0), default=next(iter(pattern_widths)))
        
        order_len_m = self.order_sheet_lengths.get(dominant_width, self.max_sheet_roll_length) / 1000
        if order_len_m <= 0:
            return self.max_sheet_roll_length

        l_candidate = math.floor(self.max_sheet_roll_length / (order_len_m * 1000)) * (order_len_m * 1000)
        
        return max(l_candidate, self.min_sheet_roll_length)
        
    def _pattern_has_forbidden_single(self, pattern_comp):
        """금지 폭이 단일 장으로 포함되어 있는지 확인합니다."""
        for item_name, count in pattern_comp.items():
            try:
                base_width, multiplier = map(int, item_name.split("x"))
            except ValueError:
                continue
            if base_width in DISALLOWED_SINGLE_BASE_WIDTHS and multiplier == 1 and count > 0:
                return True
        return False

    def _calculate_pattern_usage_penalty(self, pattern_comp):
        """각 패턴 조합에 포함된 폭별 페널티를 계산합니다."""
        penalty = 0.0
        for item_name, count in pattern_comp.items():
            try:
                base_width, multiplier = map(int, item_name.split("x"))
            except ValueError:
                continue
            base_penalty = ITEM_SINGLE_STRIP_PENALTIES.get(base_width, DEFAULT_SINGLE_STRIP_PENALTY)
            if base_penalty > 0 and multiplier == 1 and count > 0:
                penalty += base_penalty * count
        return penalty

    def run_optimize(self):
        """최적화 실행 메인 함수"""
        
        # 1. 초기 패턴 '조합' 생성
        if len(self.order_widths) <= SMALL_PROBLEM_THRESHOLD:
            print(f"--- 주문 종류가 {len(self.order_widths)}개 이므로, 모든 패턴 조합을 탐색합니다 (Small-scale) ---")
            compositions = self._generate_all_patterns()
        else:
            print(f"--- 주문 종류가 {len(self.order_widths)}개 이므로, 열 생성 기법을 시작합니다 (Large-scale) ---")
            compositions = self._generate_initial_patterns()
            # compositions.extend(self._generate_initial_patterns_db()) # 필요시 DB 패턴 추가

        # 2. 생성된 조합을 여러 롤 길이와 결합하여 최종 패턴 목록 확장
        print(f"--- {len(compositions)}개의 패턴 조합을 여러 롤 길이와 결합하여 최종 패턴을 생성합니다. ---")
        seen_patterns = set() # (frozenset(comp.items()), length)
        
        for comp in compositions:
            # 금지된 단일폭 포함 시 제외
            if self._pattern_has_forbidden_single(comp):
                continue
            
            # 후보 롤 길이 목록 생성
            length_candidates = {
                self.min_sheet_roll_length,
                self.max_sheet_roll_length,
                int((self.min_sheet_roll_length + self.max_sheet_roll_length) / 2 / 10) * 10
            }
            length_candidates.add(self._calculate_heuristic_pattern_length(comp))

            # 각 후보 길이에 대해 패턴 생성
            for length in sorted(list(length_candidates)):
                if not (self.min_sheet_roll_length <= length <= self.max_sheet_roll_length):
                    continue

                pattern_key = (frozenset(comp.items()), length)
                if pattern_key not in seen_patterns:
                    penalty = self._calculate_pattern_usage_penalty(comp)
                    self.patterns.append({'composition': comp, 'length': length, 'penalty': penalty})
                    seen_patterns.add(pattern_key)

        if not self.patterns:
            return {"error": "초기 유효 패턴을 생성할 수 없습니다. 제약조건이 너무 엄격할 수 있습니다."}
        
        print(f"--- 총 {len(self.patterns)}개의 (조합+길이) 패턴으로 최적화를 시작합니다. ---")

        # 3. 열 생성 루프 (Large-scale 문제의 경우)
        if len(self.order_widths) > SMALL_PROBLEM_THRESHOLD:
            no_improvement_count = 0
            for iteration in range(CG_MAX_ITERATIONS):
                master_solution = self._solve_master_problem_ilp()
                if not master_solution or 'duals' not in master_solution:
                    break

                new_compositions = self._solve_subproblem_dp(master_solution['duals'])
                
                patterns_added = 0
                if new_compositions:
                    length_candidates = {
                        self.min_sheet_roll_length,
                        self.max_sheet_roll_length,
                        (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2
                    }
                    for new_comp in new_compositions:
                        if self._pattern_has_forbidden_single(new_comp):
                            continue
                        
                        # 새 조합에 대해서도 여러 길이 테스트
                        heuristic_l = self._calculate_heuristic_pattern_length(new_comp)
                        current_lengths = length_candidates | {heuristic_l}

                        for length in sorted(list(current_lengths)):
                             if not (self.min_sheet_roll_length <= length <= self.max_sheet_roll_length):
                                continue
                             
                             pattern_key = (frozenset(new_comp.items()), length)
                             if pattern_key not in seen_patterns:
                                penalty = self._calculate_pattern_usage_penalty(new_comp)
                                self.patterns.append({'composition': new_comp, 'length': length, 'penalty': penalty})
                                seen_patterns.add(pattern_key)
                                patterns_added += 1
                
                if patterns_added > 0:
                    no_improvement_count = 0
                    print(f"--- CG 반복 {iteration}: {patterns_added}개의 신규 패턴 추가 (총 {len(self.patterns)}개) ---")
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= CG_NO_IMPROVEMENT_LIMIT:
                    print(f"--- {CG_NO_IMPROVEMENT_LIMIT}번의 반복 동안 개선이 없어 수렴으로 간주하고 종료합니다 (반복 {iteration}). ---")
                    break

        # 4. 최종 최적화
        if not self.patterns:
            return {"error": "유효한 패턴을 생성할 수 없습니다."}

        print(f"--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
        return self._format_results(final_solution)

    def _format_results(self, final_solution):
        """최종 결과를 데이터프레임 형식으로 포매팅합니다."""
        
        # _build_pattern_details에서 모든 것을 처리하고, 분배 결과인 demand_tracker를 반환받음
        result_patterns, pattern_details_for_db, pattern_roll_details_for_db, demand_tracker = self._build_pattern_details(final_solution)
        
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['Pattern', 'wd_width', 'Roll_Length', 'Count', 'Loss_per_Roll']]

        # 주문 이행 요약 생성 (분배된 상세 데이터를 기반으로)
        fulfillment_summary = self._build_fulfillment_summary(demand_tracker)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]") 
        print("[주문 이행 요약 (그룹오더별)]")
        
        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "fulfillment_summary": fulfillment_summary
        }

    def _build_pattern_details(self, final_solution):
        """패턴 사용 결과와 DB 저장을 위한 상세 정보를 생성하고, 생산량을 그룹오더별로 분배합니다."""
        
        # 1. 수요 추적기 생성 (group_order_no별 필요 미터)
        demand_tracker = self.df_orders.copy()
        demand_tracker['original_order_idx'] = demand_tracker.index # 원본 인덱스 보존
        demand_tracker = demand_tracker[['original_order_idx', 'group_order_no', '지폭', 'meters']].copy()
        demand_tracker['fulfilled_meters'] = 0.0
        demand_tracker = demand_tracker.sort_values(by=['지폭', 'group_order_no']).reset_index(drop=True)

        result_patterns = []
        pattern_details_for_db = []
        pattern_roll_details_for_db = []
        prod_seq_counter = 0

        for j, count in final_solution['pattern_counts'].items():
            if count < 0.99:
                continue
            
            roll_count = int(round(count))
            pattern = self.patterns[j]
            pattern_comp = pattern['composition']
            pattern_len = pattern['length']

            # --- 요약 결과 생성 ---
            sorted_pattern_items = sorted(pattern_comp.items(), key=lambda item: self.item_info[item[0]], reverse=True)
            pattern_item_strs = []
            total_width = 0
            for item_name, num in sorted_pattern_items:
                width = self.item_info[item_name]
                total_width += width * num
                base_width, multiplier = map(int, item_name.split('x'))
                formatted_name = f"{self.item_info[item_name]}({base_width}*{multiplier})"
                pattern_item_strs.extend([formatted_name] * num)
            
            result_patterns.append({
                'Pattern': ' + '.join(pattern_item_strs),
                'wd_width': total_width,
                'Roll_Length': round(pattern_len, 2),
                'Count': roll_count,
                'Loss_per_Roll': self.original_max_width - total_width
            })

            # --- DB 저장을 위한 분배 로직 ---
            for _ in range(roll_count):
                prod_seq_counter += 1
                
                composite_widths_for_db = []
                composite_group_nos_for_db = []
                
                roll_seq_counter = 0
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
                                    (demand_tracker['fulfilled_meters'] < demand_tracker['meters'])
                                ].index
                                
                                assigned_group_no = None
                                if not target_indices.empty:
                                    target_idx = target_indices.min()
                                    demand_tracker.loc[target_idx, 'fulfilled_meters'] += pattern_len
                                    assigned_group_no = demand_tracker.loc[target_idx, 'group_order_no']
                                else:
                                    fallback_indices = demand_tracker[demand_tracker['지폭'] == piece_width].index
                                    if not fallback_indices.empty:
                                        assigned_group_no = demand_tracker.loc[fallback_indices.min(), 'group_order_no']
                                    else:
                                        assigned_group_no = "ERROR"
                                
                                base_widths_for_item.append(base_width)
                                base_group_nos_for_item.append(assigned_group_no)

                                if assigned_group_no_for_composite is None:
                                    assigned_group_no_for_composite = assigned_group_no
                        
                        composite_widths_for_db.append(composite_width)
                        composite_group_nos_for_db.append(assigned_group_no_for_composite if assigned_group_no_for_composite is not None else "")

                        pattern_roll_details_for_db.append({
                            'rollwidth': composite_width,
                            'roll_production_length': pattern_len,
                            'widths': (base_widths_for_item + [0] * 7)[:7],
                            'group_nos': (base_group_nos_for_item + [''] * 7)[:7],
                            'Count': 1,
                            'Prod_seq': prod_seq_counter,
                            'Roll_seq': roll_seq_counter
                        })

                pattern_details_for_db.append({
                    'roll_production_length': pattern_len,
                    'Count': 1,
                    'widths': (composite_widths_for_db + [0] * 8)[:8],
                    'group_nos': (composite_group_nos_for_db + [''] * 8)[:8],
                    'Prod_seq': prod_seq_counter
                })
                
        return result_patterns, pattern_details_for_db, pattern_roll_details_for_db, demand_tracker

    def _build_fulfillment_summary(self, demand_tracker):
        """주문 이행 요약 데이터프레임을 생성합니다. (개별 주문별)"""
        
        summary_df = self.df_orders[['group_order_no', '가로', '세로', '수출내수', '등급', '주문톤', 'meters']].copy()
        summary_df.rename(columns={'meters': '필요길이(m)', '주문톤': '주문량(톤)'}, inplace=True)
        
        # Merge fulfilled_meters directly using original_order_idx
        # demand_tracker has 'original_order_idx' from _build_pattern_details
        summary_df = pd.merge(summary_df, demand_tracker[['original_order_idx', 'fulfilled_meters']], 
                              left_index=True, right_on='original_order_idx', how='left')
        summary_df.rename(columns={'fulfilled_meters': '생산길이(m)'}, inplace=True)
        summary_df.drop(columns=['original_order_idx'], inplace=True) # Drop the temporary merge key

        summary_df['생산길이(m)'] = summary_df['생산길이(m)'].fillna(0)
        
        summary_df['과부족(m)'] = summary_df['생산길이(m)'] - summary_df['필요길이(m)']
        tons_per_meter = (summary_df['주문량(톤)'] / summary_df['필요길이(m)']).replace([float('inf'), -float('inf')], 0).fillna(0)
        summary_df['생산량(톤)'] = (summary_df['생산길이(m)'] * tons_per_meter).round(2)
        summary_df['과부족(톤)'] = (summary_df['생산량(톤)'] - summary_df['주문량(톤)']).round(2)

        final_cols = [
            'group_order_no', '가로', '세로', '수출내수', '등급', 
            '주문량(톤)', '생산량(톤)', '과부족(톤)',
            '필요길이(m)', '생산길이(m)', '과부족(m)'
        ]
        
        for col in ['필요길이(m)', '생산길이(m)', '과부족(m)']:
            summary_df[col] = summary_df[col].round(2)

        return summary_df[final_cols]
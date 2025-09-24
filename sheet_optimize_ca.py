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
TRIM_PENALTY = 0          # 트림(loss) 면적(mm^2)당 페널티. 폐기물 비용.
ITEM_SINGLE_STRIP_PENALTIES = {}
DEFAULT_SINGLE_STRIP_PENALTY = 1000  # 지정되지 않은 단일폭은 기본적으로 패널티 없음
DISALLOWED_SINGLE_BASE_WIDTHS = {}  # 단일 사용을 금지할 주문 폭 집합

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
        self.df_orders = df_spec_pre.copy()
        self.width_to_group_order_no = self.df_orders.drop_duplicates('지폭').set_index('지폭')['group_order_no'].to_dict()

        # 수요 단위를 '롤'에서 '미터'로 변경
        self.demands_in_meters, self.order_sheet_lengths = self._calculate_demand_meters(self.df_orders)
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

        self.patterns = []

    def _prepare_items(self, min_sc_width, max_sc_width):
        """복합폭 아이템(패턴의 구성요소)을 생성합니다. 단일 지폭 및 다른 지폭 조합을 모두 포함합니다."""
        from itertools import combinations_with_replacement
        
        items = []
        item_info = {}  # item_name -> width
        item_composition = {}  # composite_item_name -> {original_width: count}
        
        max_pieces_in_composite = 4 

        for width in self.order_widths:
            for i in range(1, max_pieces_in_composite + 1):
                base_width = width * i + self.sheet_trim
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                item_name = f"{width}x{i}"
                if base_width <= self.original_max_width:
                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = base_width
                        item_composition[item_name] = {width: i}

        for i in range(2, max_pieces_in_composite + 1):
            for combo in combinations_with_replacement(self.order_widths, i):
                if len(set(combo)) == 1:
                    continue

                base_width = sum(combo) + self.sheet_trim
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                if base_width <= self.original_max_width:
                    comp_counts = Counter(combo)
                    item_name = "+".join(sorted([f"{w}x{c}" for w, c in comp_counts.items()]))

                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = base_width
                        item_composition[item_name] = dict(comp_counts)

        return items, item_info, item_composition

    def _calculate_demand_meters(self, df_orders):
        """주문량을 바탕으로 지폭별 필요 총 길이(미터)를 계산합니다."""
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

        return demand_meters, order_sheet_lengths

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
        get_pieces = lambda item_name: sum(self.item_composition[item_name].values())

        sorted_items_demand = sorted(
            self.items,
            key=lambda i: sum(self.demands_in_meters.get(w, 0) * c for w, c in self.item_composition[i].items()),
            reverse=False
        )
        sorted_items_width_desc = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        sorted_items_width_asc = sorted(self.items, key=lambda i: self.item_info[i], reverse=False)

        heuristics = [sorted_items_demand, sorted_items_width_desc, sorted_items_width_asc]

        for sorted_items in heuristics:
            for item in sorted_items:
                item_width = self.item_info[item]
                item_pieces = get_pieces(item)
                
                current_pattern = {item: 1}
                current_width = item_width
                current_pieces = item_pieces

                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    
                    best_fit_item = None
                    for i in sorted_items:
                        fit_item_pieces = get_pieces(i)
                        if self.item_info[i] <= remaining_width and current_pieces + fit_item_pieces <= self.max_pieces:
                             best_fit_item = i
                             break
                    
                    if not best_fit_item:
                        break 

                    best_fit_item_pieces = get_pieces(best_fit_item)
                    current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
                    current_width += self.item_info[best_fit_item]
                    current_pieces += best_fit_item_pieces

                while current_width < self.min_width and current_pieces < self.max_pieces:
                    item_to_add = None
                    for i in reversed(sorted_items):
                        add_item_pieces = get_pieces(i)
                        if current_width + self.item_info[i] <= self.max_width and current_pieces + add_item_pieces <= self.max_pieces:
                            item_to_add = i
                            break
                            
                    if item_to_add:
                        item_to_add_pieces = get_pieces(item_to_add)
                        current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
                        current_width += self.item_info[item_to_add]
                        current_pieces += item_to_add_pieces
                    else:
                        break

                if self.min_width <= current_width and self.min_pieces <= current_pieces:
                    comp_key = frozenset(current_pattern.items())
                    if comp_key not in seen_compositions:
                        compositions.append(current_pattern)
                        seen_compositions.add(comp_key)

        for item in self.items:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0: continue
            
            item_pieces = get_pieces(item)
            if item_pieces == 0: continue

            num_items = min(int(self.max_width / item_width), int(self.max_pieces / item_pieces))
            
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                total_pieces = item_pieces * num_items
                if self.min_width <= total_width and self.min_pieces <= total_pieces <= self.max_pieces:
                    comp_key = frozenset(new_pattern.items())
                    if comp_key not in seen_compositions:
                        compositions.append(new_pattern)
                        seen_compositions.add(comp_key)
                        break
                num_items -= 1
        
        covered_widths = {w for p in compositions for item_name in p for w in self.item_composition.get(item_name, {})}
        uncovered_widths = set(self.order_widths) - covered_widths
        if uncovered_widths:
            for width in uncovered_widths:
                pass

        print(f"--- {len(compositions)}개의 초기 패턴 조합 생성됨 ---")
        return compositions

    def _solve_master_problem_ilp(self, is_final_mip=False):
        """마스터 문제(Master Problem)를 정수계획법으로 해결합니다."""
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        if is_final_mip:
            solver.SetTimeLimit(SOLVER_TIME_LIMIT_MS)

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
                # [버그 수정] 패턴 길이는 mm, 수요는 meter 단위이므로 1000으로 나누어 단위를 통일합니다.
                x[j] * (self.patterns[j]['length'] ) * num_strips_per_pattern[j]
                for j in range(len(self.patterns))
            )
            constraints[width] = solver.Add(production_for_width + under_prod_vars[width] == required_meters + over_prod_vars[width], f'demand_{width}')

        # --- 목적 함수 설정 ---
        # 모든 항을 '비용' 단위로 통일하여 목적 함수를 재구성합니다.

        # 1. 총 롤 사용 비용 (롤 교체/준비 비용)
        total_rolls_cost = solver.Sum(COST_PER_ROLL * x[j] for j in range(len(self.patterns)))

        # 2. 총 원자재 사용 비용
        # self.patterns[j]['length']는 mm 단위이므로 미터로 변환 ( )
        total_material_cost = solver.Sum(COST_PER_METER_MATERIAL * (self.patterns[j]['length'] ) * x[j] for j in range(len(self.patterns)))

        # 3. 패턴 사용 페널티 (롤당 부과되는 비용)
        pattern_usage_penalty = solver.Sum(self.patterns[j]['penalty'] * x[j] for j in range(len(self.patterns)))
        
        # 4. 과생산 페널티 (미터당 부과되는 비용)
        total_over_prod_penalty = solver.Sum(OVER_PROD_PENALTY * var for var in over_prod_vars.values())

        # 5. 부족생산 페널티 (미터당 부과되는 비용)
        total_under_prod_penalty = solver.Sum(UNDER_PROD_PENALTY * var for var in under_prod_vars.values())

        print(f"--- 총 롤 사용 비용 {total_rolls_cost} ---")
        print(f"--- 총 원자재 사용 비용 {total_material_cost} ---")
        print(f"--- 총 과생산 페널티 {total_over_prod_penalty} ---")
        print(f"--- 총 부족생산 페널티 {total_under_prod_penalty} ---")
        print(f"--- 총 패턴 사용 페널티 {pattern_usage_penalty} ---")

        # 6. 트림 페널티 (폐기물 비용)
        total_trim_penalty = solver.Sum(TRIM_PENALTY * self.patterns[j]['loss_per_roll'] * self.patterns[j]['length'] * x[j] for j in range(len(self.patterns)))
        print(f"--- 총 트림 페널티 {total_trim_penalty} ---")

        # 최종 목적 함수: 모든 비용의 합계를 최소화
        solver.Minimize(
            total_rolls_cost + total_material_cost + total_over_prod_penalty + 
            total_under_prod_penalty + pattern_usage_penalty + total_trim_penalty
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
            if count <= 0:
                continue
            
            sub_items = item_name.split('+')
            for sub_item in sub_items:
                try:
                    base_width, multiplier = map(int, sub_item.split('x'))
                except ValueError:
                    continue
                
                if base_width in DISALLOWED_SINGLE_BASE_WIDTHS and multiplier == 1:
                    return True
        return False

    def _calculate_pattern_usage_penalty(self, pattern_comp):
        """각 패턴 조합에 포함된 폭별 페널티를 계산합니다."""
        penalty = 0.0
        for item_name, count in pattern_comp.items():
            sub_items = item_name.split('+')
            for sub_item in sub_items:
                try:
                    base_width, multiplier = map(int, sub_item.split('x'))
                except ValueError:
                    continue
                base_penalty = ITEM_SINGLE_STRIP_PENALTIES.get(base_width, DEFAULT_SINGLE_STRIP_PENALTY)
                if base_penalty > 0 and multiplier == 1:
                    penalty += base_penalty * count
        return penalty

    def run_optimize(self):
        """최적화 실행 메인 함수"""
        
        if len(self.order_widths) <= SMALL_PROBLEM_THRESHOLD:
            print(f"--- 주문 종류가 {len(self.order_widths)}개 이므로, 모든 패턴 조합을 탐색합니다 (Small-scale) ---")
            compositions = self._generate_all_patterns()
        else:
            print(f"--- 주문 종류가 {len(self.order_widths)}개 이므로, 열 생성 기법을 시작합니다 (Large-scale) ---")
            compositions = self._generate_initial_patterns()

        print(f"--- {len(compositions)}개의 패턴 조합을 여러 롤 길이와 결합하여 최종 패턴을 생성합니다. ---")
        seen_patterns = set()
        
        for comp in compositions:
            if self._pattern_has_forbidden_single(comp):
                continue
            
            length_candidates = {
                self.min_sheet_roll_length,
                self.max_sheet_roll_length,
                int((self.min_sheet_roll_length + self.max_sheet_roll_length) / 2 / 10) * 10
            }
            length_candidates.add(self._calculate_heuristic_pattern_length(comp))

            for length in sorted(list(length_candidates)):
                if not (self.min_sheet_roll_length <= length <= self.max_sheet_roll_length):
                    continue

                pattern_key = (frozenset(comp.items()), length)
                if pattern_key not in seen_patterns:
                    total_width = sum(self.item_info[item_name] * num for item_name, num in comp.items())
                    loss_per_roll = self.original_max_width - total_width
                    penalty = self._calculate_pattern_usage_penalty(comp)
                    self.patterns.append({
                        'composition': comp, 
                        'length': length, 
                        'penalty': penalty,
                        'loss_per_roll': loss_per_roll
                    })
                    seen_patterns.add(pattern_key)

        if not self.patterns:
            return {"error": "초기 유효 패턴을 생성할 수 없습니다. 제약조건이 너무 엄격할 수 있습니다."}
        
        print(f"--- 총 {len(self.patterns)}개의 (조합+길이) 패턴으로 최적화를 시작합니다. ---")

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
                        
                        heuristic_l = self._calculate_heuristic_pattern_length(new_comp)
                        current_lengths = length_candidates | {heuristic_l}

                        for length in sorted(list(current_lengths)):
                             if not (self.min_sheet_roll_length <= length <= self.max_sheet_roll_length):
                                continue
                             
                             pattern_key = (frozenset(new_comp.items()), length)
                             if pattern_key not in seen_patterns:
                                total_width = sum(self.item_info[item_name] * num for item_name, num in new_comp.items())
                                loss_per_roll = self.original_max_width - total_width
                                penalty = self._calculate_pattern_usage_penalty(new_comp)
                                self.patterns.append({
                                    'composition': new_comp, 
                                    'length': length, 
                                    'penalty': penalty,
                                    'loss_per_roll': loss_per_roll
                                })
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

        if not self.patterns:
            return {"error": "유효한 패턴을 생성할 수 없습니다."}

        print(f"----최종패턴 {self.patterns}")
        print(f"--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
        return self._format_results(final_solution)

    def _format_results(self, final_solution):
        """최종 결과를 데이터프레임 형식으로 포매팅합니다."""
        final_production_meters = {width: 0 for width in self.order_widths}
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                roll_count = int(round(count))
                pattern = self.patterns[j]
                
                for item_name, num_in_pattern in pattern['composition'].items():
                    for width, num_pieces in self.item_composition[item_name].items():
                        if width in final_production_meters:
                            num_strips = num_in_pattern * num_pieces
                            final_production_meters[width] += roll_count * pattern['length'] * num_strips

        result_patterns, pattern_details_for_db = self._build_pattern_details(final_solution)
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['Pattern', 'Roll_Production_Width', 'Roll_Length', 'Count', 'Loss_per_Roll']]

        fulfillment_summary = self._build_fulfillment_summary(final_production_meters)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]") 
        print("[주문 이행 요약]")
        
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
            pattern = self.patterns[j]
            pattern_dict = pattern['composition']
            pattern_len = pattern['length']
            
            sorted_pattern_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)

            db_widths, db_group_nos, pattern_item_strs = [], [], []
            total_width = 0

            for item_name, num in sorted_pattern_items:
                width = self.item_info[item_name]
                total_width += width * num
                db_widths.extend([width] * num)

                # group_order_no를 결정하는 로직 수정
                if '+' in item_name:
                    # 여러 종류의 폭이 조합된 복합폭은 첫 번째 아이템의 group_order_no를 사용
                    try:
                        first_item = item_name.split('+')[0]
                        base_width_str = first_item.split('x')[0]
                        base_width = int(base_width_str)
                        group_no = self.width_to_group_order_no.get(base_width, item_name)
                    except (ValueError, IndexError):
                        group_no = item_name # 예외 발생 시 안전하게 item_name 사용
                else:
                    # 단일 종류의 폭으로 구성된 복합폭 (e.g., '788x3')
                    try:
                        base_width_str = item_name.split('x')[0]
                        base_width = int(base_width_str)
                        # self.width_to_group_order_no 딕셔너리에서 group_order_no를 찾음
                        group_no = self.width_to_group_order_no.get(base_width, item_name)
                    except (ValueError, IndexError):
                        # 예외 발생 시 안전하게 item_name 사용
                        group_no = item_name
                db_group_nos.extend([group_no] * num)
                
                if '+' in item_name:
                    formatted_name = f"{self.item_info[item_name]}({item_name})"
                else:
                    try:
                        base_width, multiplier = map(int, item_name.split('x'))
                        formatted_name = f"{self.item_info[item_name]}({base_width}*{multiplier})"
                    except ValueError:
                        formatted_name = f"{self.item_info[item_name]}({item_name})"
                pattern_item_strs.extend([formatted_name] * num)

            result_patterns.append({
                'Pattern': ' + '.join(pattern_item_strs),
                'Roll_Production_Width': total_width,
                'Roll_Length': round(pattern_len, 2),
                'Count': roll_count,
                'Loss_per_Roll': pattern['loss_per_roll']
            })
            pattern_details_for_db.append({
                'Count': roll_count,
                'widths': (db_widths + [0] * 8)[:8],
                'group_nos': (db_group_nos + [''] * 8)[:8]
            })
        return result_patterns, pattern_details_for_db

    def _build_fulfillment_summary(self, final_production_meters):
        """주문 이행 요약 데이터프레임을 생성합니다. (단위: 톤, 미터)"""
        summary_data = []
        for width, required_meters in self.demands_in_meters.items():
            produced_meters = final_production_meters.get(width, 0)
            order_tons = self.width_summary[width]['order_tons']
            
            if required_meters > 0:
                tons_per_meter = order_tons / required_meters
                produced_tons = produced_meters * tons_per_meter
            else:
                produced_tons = 0
            
            over_prod_tons = produced_tons - order_tons
            over_prod_meters = produced_meters - required_meters

            summary_data.append({
                '지폭': width,
                '주문량(톤)': order_tons,
                '생산량(톤)': round(produced_tons, 2),
                '과부족(톤)': round(over_prod_tons, 2),
                '필요길이(m)': round(required_meters, 2),
                '생산길이(m)': round(produced_meters, 2),
                '과부족(m)': round(over_prod_meters, 2),
            })
        
        return pd.DataFrame(summary_data)[[
            '지폭', '주문량(톤)', '생산량(톤)', '과부족(톤)',
            '필요길이(m)', '생산길이(m)', '과부족(m)'
        ]]
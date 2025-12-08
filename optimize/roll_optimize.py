import pandas as pd
from ortools.linear_solver import pywraplp
import random
import logging
import time

"""
[파일 설명: roll_optimize.py]
롤(Roll) 제품의 생산 최적화를 위한 핵심 모듈입니다.
Column Generation(열 생성) 알고리즘을 기반으로 하여, 주문 요구사항(지폭, 수량)을 만족시키면서
폐기물(Trim Loss)과 패턴 교체 비용을 최소화하는 최적의 절단 패턴을 산출합니다.
"""

OVER_PROD_PENALTY = 100000000.0
UNDER_PROD_PENALTY = 100000.0
PATTERN_VALUE_THRESHOLD = 1.0 + 1e-6
CG_MAX_ITERATIONS = 200
CG_NO_IMPROVEMENT_LIMIT = 100  # Increased from 25
CG_SUBPROBLEM_TOP_N = 20      # Increased from 3
SMALL_PROBLEM_THRESHOLD = 10
FINAL_MIP_TIME_LIMIT_MS = 120000
PATTERN_SETUP_COST = 50000.0 # 새로운 패턴 종류를 1개 사용할 때마다 50000mm의 손실과 동일한 페널티
TRIM_LOSS_PENALTY = 10.0      # 자투리 손실 1mm당 페널티
MIXING_PENALTY = 100000.0       # 공백이 1개 섞인 경우는 페널티 비용 팬턴 생성비용과 비교

NUM_THREADS = 4

class RollOptimize:
    """
    [RollOptimize 클래스 분석 및 기능 요약]

    이 클래스는 롤(Roll) 제품의 생산 최적화를 담당합니다.
    주어진 원지(Jumbo Roll) 폭과 설비 제약 조건을 고려하여, 주문받은 규격(지폭)을 
    가장 효율적으로 생산할 수 있는 절단 패턴(Cutting Pattern)과 생산 수량을 산출합니다.

    핵심 기능 및 알고리즘:
    1.  **초기화 (__init__)**:
        -   주문 데이터(df_spec_pre)를 로드하고, 지폭별 수요량(demands) 및 제약 조건(최대/최소 폭, 최대 조수 등)을 설정합니다.

    2.  **초기 패턴 생성 (_generate_initial_patterns)**:
        -   Column Generation의 시작점이 될 초기 패턴 집합을 생성합니다.
        -   다양한 휴리스틱(수요순, 너비순 정렬 등)과 무작위 섞기를 결합한 First-Fit 알고리즘을 사용합니다.
        -   모든 주문을 커버할 수 있도록 단일 품목 패턴 및 폴백(Fallback) 로직도 포함합니다.

    3.  **열 생성법 (Column Generation) 기반 최적화 (run_optimize)**:
        -   대규모 조합 최적화 문제를 효율적으로 풀기 위해 반복적인 과정을 수행합니다.
        -   **Master Problem (_solve_master_problem)**: 
            현재 확보된 패턴들을 조합하여 비용(과생산, 폐기물, 패턴 교체 비용 등)을 최소화하는 해를 찾습니다. 
            (초기에는 LP로 완화하여 풀고, 마지막에 MIP로 정수해를 구합니다.)
        -   **Sub Problem (_solve_subproblem)**: 
            Master Problem의 Dual Value(잠재 가격)를 활용하여, 현재 해를 개선할 수 있는(Reduced Cost > 0) 
            새로운 유망 패턴(Knapsack Problem 해)을 동적 계획법(DP)으로 찾아냅니다.
    
    4.  **결과 생성 (_format_results)**:
        -   최적화된 패턴별 생산 수량을 바탕으로 구체적인 작업 지시 데이터(생산 순서, 롤별 구성 등)를 생성합니다.
        -   과생산/부족생산 및 로스율 등의 지표를 집계하여 요약 정보를 제공합니다.
    """
    
    def __init__(self, df_spec_pre, max_width=1000, min_width=0, max_pieces=8, lot_no=None):
        self.df_spec_pre = df_spec_pre
        self.max_width = max_width
        self.min_width = min_width
        self.max_pieces = max_pieces
        self.lot_no = lot_no
        self.patterns = []
        self.pattern_keys = set()
        self.demands = df_spec_pre.groupby('group_order_no')['주문수량'].sum().to_dict()
        self.items = list(self.demands.keys())
        self.item_info = df_spec_pre.set_index('group_order_no')['지폭'].to_dict()
        self.length_info = df_spec_pre.set_index('group_order_no')['롤길이'].to_dict()
        
        # sep_qt 정보 저장 (없으면 빈 문자열로 가정)
        if 'sep_qt' in df_spec_pre.columns:
            self.sep_qt_info = df_spec_pre.set_index('group_order_no')['sep_qt'].to_dict()
        else:
            self.sep_qt_info = {item: '' for item in self.items}

    def _clear_patterns(self):
        self.patterns = []
        self.pattern_keys = set()

    def _rebuild_pattern_cache(self):
        self.pattern_keys = {frozenset(p.items()) for p in self.patterns}

    def _add_pattern(self, pattern):
        key = frozenset(pattern.items())
        if key in self.pattern_keys:
            return False
        self.patterns.append(dict(pattern))
        self.pattern_keys.add(key)
        return True

    def _generate_initial_patterns(self):
        self._clear_patterns()
        if not self.items:
            return

        sorted_by_demand = sorted(self.items, key=lambda item: self.demands.get(item, 0), reverse=True)
        sorted_by_demand_asc = sorted(self.items, key=lambda item: self.demands.get(item, 0))
        sorted_by_width_desc = sorted(self.items, key=lambda item: self.item_info.get(item, 0), reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda item: self.item_info.get(item, 0))

        # New Heuristics
        # 1. Width * Demand (Area proxy)
        # sorted_by_area_desc = sorted(self.items, key=lambda item: self.item_info.get(item, 0) * self.demands.get(item, 0), reverse=True)
        
        # 2. Random Shuffles (add multiple to increase diversity)
        random.seed(41) # Ensure determinism
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

        for sorted_items in heuristics:
            for item in sorted_items:
                current_pattern = {item: 1}
                current_width = self.item_info[item]
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
                    item_to_add = next((i for i in sorted_by_width_desc if current_width + self.item_info[i] <= self.max_width), None)
                    
                    if item_to_add:
                        current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
                        current_width += self.item_info[item_to_add]
                        current_pieces += 1
                    else:
                        break

                if current_width >= self.min_width:
                    self._add_pattern(current_pattern)

        for item in self.items:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0: continue

            num_items = min(int(self.max_width / item_width), self.max_pieces)
            
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                
                if total_width >= self.min_width:
                    if self._add_pattern(new_pattern):
                        break
                
                num_items -= 1

        covered_items = {item for pattern in self.patterns for item in pattern}
        uncovered_items = set(self.items) - covered_items

        if uncovered_items:
            for item in uncovered_items:
                pattern = {item: 1}
                width = self.item_info[item]
                pieces = 1
                
                while pieces < self.max_pieces and width < self.min_width:
                    remaining_width = self.max_width - width
                    
                    candidate = next((i for i in sorted_by_width_desc if self.item_info[i] <= remaining_width), None)
                    if not candidate:
                        break
                        
                    pattern[candidate] = pattern.get(candidate, 0) + 1
                    width += self.item_info[candidate]
                    pieces += 1
                
                if width >= self.min_width:
                    self._add_pattern(pattern)
                else:
                    self._add_pattern({item: 1})

    def _solve_master_problem(self, is_final_mip=False, max_patterns=None):
        """
        마스터 문제(Master Problem)를 선형계획법(LP) 또는 정수계획법(MIP)으로 해결합니다.
        
        목적 함수:
        1. 과생산 페널티 최소화 (주문량 준수)
        2. 패턴 교체 비용(Setup Cost) 최소화
        3. 폐기물(Trim Loss) 최소화 (옵션)
        4. 혼합 생산(Mixing) 페널티 최소화 (동일 패턴 내 이질적인 제품 혼합 지양)
        
        Args:
            is_final_mip (bool): True이면 정수해(Integer Solution)를 구하고, 
                               False이면 열 생성을 위한 실수해(Relaxed LP)와 Dual Value를 구합니다.
        """
        solver_name = 'SCIP' if is_final_mip else 'GLOP'
        solver = pywraplp.Solver.CreateSolver(solver_name)
        if not solver:
            return None
        if is_final_mip and hasattr(solver, 'SetTimeLimit'):
            solver.SetTimeLimit(FINAL_MIP_TIME_LIMIT_MS)
        
        # Enable multi-threading
        if hasattr(solver, 'SetNumThreads'):
            solver.SetNumThreads(NUM_THREADS)

        x = {j: (solver.IntVar if is_final_mip else solver.NumVar)(0, solver.infinity(), f'P_{j}')
             for j in range(len(self.patterns))}
        over_prod_vars = {item: solver.NumVar(0, solver.infinity(), f'Over_{item}') for item in self.demands}

        constraints = {}
        total_trim_loss = solver.Sum(
            (self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())) * x[j]
            for j, pattern in enumerate(self.patterns)
        )

        for item, demand in self.demands.items():
            production_expr = solver.Sum(self.patterns[j].get(item, 0) * x[j] for j in range(len(self.patterns)))
            constraints[item] = solver.Add(
                production_expr == demand + over_prod_vars[item],
                f'demand_{item}'
            )

        # Dynamic penalty calculation based on the number of UNIQUE widths
        # This ensures penalty > potential pattern reduction savings
        unique_widths_count = len(set(self.item_info.values()))
        dynamic_over_prod_penalty = max(OVER_PROD_PENALTY, PATTERN_SETUP_COST * unique_widths_count * 20)

        total_over_penalty = solver.Sum(dynamic_over_prod_penalty * over_prod_vars[item]
                                        for item in self.demands)
        
        objective = total_over_penalty

        if is_final_mip:
            # y_j = 1 if pattern j is used, 0 otherwise
            y = {j: solver.BoolVar(f'y_{j}') for j in range(len(self.patterns))}

            # A loose upper bound on how many times a pattern can be used.
            M = sum(self.demands.values()) + 1
            for j in range(len(self.patterns)):
                solver.Add(x[j] <= M * y[j])

            # Add setup cost to objective
            total_setup_cost = solver.Sum(
                y[j] * PATTERN_SETUP_COST for j in range(len(self.patterns))
            )
            objective += total_setup_cost

            # Add trim loss to objective
            # total_trim_loss = solver.Sum(
            #     (self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())) * x[j]
            #     for j, pattern in enumerate(self.patterns)
            # )
            # objective += total_trim_loss * TRIM_LOSS_PENALTY

            # Add mixing penalty (sep_qt)
            total_mixing_penalty = 0
            for j, pattern in enumerate(self.patterns):
                # Get all non-empty sep_qts in this pattern
                sep_qts = [self.sep_qt_info[item] for item in pattern for _ in range(pattern[item]) if self.sep_qt_info[item].strip()]
                unique_sep_qts = set(sep_qts)
                
                # Case 1: Mixing different non-empty sep_qts (e.g. 'A' and 'B') -> High Penalty
                if len(unique_sep_qts) > 1:
                    total_mixing_penalty += x[j] * MIXING_PENALTY * 100 # Strong avoidance
                
                # Case 2: Mixing sep_qt with empty sep_qt
                elif len(unique_sep_qts) == 1:
                    # Count items with empty sep_qt
                    empty_count = sum(count for item, count in pattern.items() if not self.sep_qt_info[item].strip())
                    
                    if empty_count > 0:
                        # Penalty increases with the number of empty items
                        # User said "1 is suboptimal, 2 is worse".
                        total_mixing_penalty += x[j] * MIXING_PENALTY * empty_count
            
            objective += total_mixing_penalty

            # if max_patterns is not None:
            #     solver.Add(solver.Sum(y[j] for j in range(len(self.patterns))) <= max_patterns)

        solver.Minimize(objective)

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None

        solution = {
            'objective': solver.Objective().Value(),
            'pattern_counts': {j: var.solution_value() for j, var in x.items()},
            'over_production': {item: over_prod_vars[item].solution_value() for item in self.demands},
        }
        if not is_final_mip:
            solution['duals'] = {item: constraints[item].dual_value() for item in self.demands}
        return solution

    def _solve_subproblem(self, duals):
        """
        서브 문제(Sub-problem)를 동적 계획법(Dynamic Programming, Knapsack-like)으로 해결합니다.
        
        Master Problem에서 얻은 Dual Value(잠재 가격)를 가치(Value)로 간주하여, 
        제한된 원지 폭(Knapsack Capacity) 내에서 가치 합이 최대가 되는 새로운 패턴을 찾습니다.
        (Reduced Cost가 양수인 패턴을 찾아 Master Problem에 추가)
        """
        def run_dp(candidate_items):
            width_limit = self.max_width
            piece_limit = self.max_pieces
            item_details = []
            for item in candidate_items:
                item_width = self.item_info[item]
                item_value = duals.get(item, 0)
                if item_value <= 0:
                    continue
                item_details.append((item, item_width, item_value))
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

            local_candidates = []
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
                    if not pattern or cur_pieces != 0 or cur_width != 0:
                        continue
                    key = frozenset(pattern.items())
                    if key in seen_patterns:
                        continue
                    total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                    if total_width < self.min_width or total_width > self.max_width:
                        continue
                    seen_patterns.add(key)
                    local_candidates.append({'pattern': pattern, 'value': value, 'width': total_width, 'pieces': pieces})
            return local_candidates

        # 1. Run DP for ALL items (standard)
        all_candidates = run_dp(self.items)
        
        # 2. Run DP for each sep_qt group
        # This encourages generating pure patterns for each sep_qt type
        unique_sep_qts = set(self.sep_qt_info.values())
        for qt in unique_sep_qts:
             qt_items = [item for item in self.items if self.sep_qt_info.get(item) == qt]
             if qt_items:
                 all_candidates.extend(run_dp(qt_items))

        # Sort by value and return top N
        # Note: Mixed patterns from step 1 might have high 'value' (dual sum) but will be penalized in Master.
        # Pure patterns from step 2/3 will have lower 'value' potentially but no penalty in Master.
        # We should return enough candidates to let Master decide.
        
        # Remove duplicates based on pattern content
        unique_candidates = []
        seen_keys = set()
        for cand in all_candidates:
            key = frozenset(cand['pattern'].items())
            if key not in seen_keys:
                seen_keys.add(key)
                unique_candidates.append(cand)

        unique_candidates.sort(key=lambda x: x['value'], reverse=True)
        return unique_candidates[:CG_SUBPROBLEM_TOP_N * 3] # Return more candidates to cover different types    

    def _filter_patterns_by_lp(self, keep_top_n=300):
        # 1. Solve LP relaxation
        lp_solution = self._solve_master_problem(is_final_mip=False)
        if not lp_solution or 'duals' not in lp_solution:
            return

        duals = lp_solution['duals']
        kept_indices = set()

        # 2. Keep Basis patterns (x > 0)
        for j, count in lp_solution['pattern_counts'].items():
            if count > 1e-6:
                kept_indices.add(j)

        # 3. Score patterns by "implied value" (sum of duals)
        # We want patterns that cover high-value demands (high duals).
        patterns_data = []
        for j, pattern in enumerate(self.patterns):
            if j in kept_indices:
                continue
            val = sum(duals.get(item, 0) * count for item, count in pattern.items())
            patterns_data.append({'index': j, 'val': val})

        # 4. Keep Top N by score
        patterns_data.sort(key=lambda x: x['val'], reverse=True)
        for i in range(min(keep_top_n, len(patterns_data))):
            kept_indices.add(patterns_data[i]['index'])

        # 5. Filter patterns
        self.patterns = [self.patterns[j] for j in kept_indices]
        self._rebuild_pattern_cache()

    def _generate_all_patterns(self):
        all_patterns = []
        seen_patterns = set()
        item_list = sorted(list(self.items), key=lambda item: self.item_info[item], reverse=True)

        def add_pattern_from_state(pattern):
            if not pattern:
                return
            key = frozenset(pattern.items())
            if key in seen_patterns:
                return
            total_width = sum(self.item_info[item] * count for item, count in pattern.items())
            if self.min_width <= total_width <= self.max_width:
                all_patterns.append(dict(pattern))
                seen_patterns.add(key)

        def find_combinations_recursive(start_index, current_pattern, current_width, current_pieces):
            add_pattern_from_state(current_pattern)
            if current_pieces >= self.max_pieces or start_index >= len(item_list):
                return

            for i in range(start_index, len(item_list)):
                item = item_list[i]
                item_width = self.item_info[item]
                if current_width + item_width > self.max_width:
                    continue
                current_pattern[item] = current_pattern.get(item, 0) + 1
                find_combinations_recursive(i, current_pattern, current_width + item_width, current_pieces + 1)
                current_pattern[item] -= 1
                if current_pattern[item] == 0:
                    del current_pattern[item]

        find_combinations_recursive(0, {}, 0, 0)
        self.patterns = [dict(p) for p in all_patterns]
        self._rebuild_pattern_cache()
        logging.info(f"Generated {len(self.patterns)} patterns in _generate_all_patterns")
 
    def run_optimize(self, start_prod_seq=0):
        logging.info(f"Starting run_optimize with {len(self.items)} items")
        start_time = time.time()
        
        if len(self.items) <= SMALL_PROBLEM_THRESHOLD:
            logging.info("Using _generate_all_patterns (Small Problem)")
            self._generate_all_patterns()
                       
            # filter_start = time.time()
            # self._filter_patterns_by_lp(keep_top_n=600) # Optimize by filtering patterns
            # logging.info(f"Pattern filtering took {time.time() - filter_start:.4f}s. Remaining patterns: {len(self.patterns)}")
        else:
            logging.info("Using Column Generation (Large Problem)")
            self._generate_initial_patterns()
            if not self.patterns:
                return {"error": "초기 유효 패턴을 생성하지 못했습니다."}

            no_improvement = 0
            for _ in range(CG_MAX_ITERATIONS):
                master_solution = self._solve_master_problem()
                if not master_solution or 'duals' not in master_solution:
                    break

                candidate_patterns = self._solve_subproblem(master_solution['duals'])
                patterns_added = 0
                for candidate in candidate_patterns:
                    pattern = candidate['pattern']
                    pattern_width = candidate['width']
                    if pattern_width < self.min_width:
                        continue
                    if self._add_pattern(pattern):
                        patterns_added += 1

                if patterns_added == 0:
                    no_improvement += 1
                else:
                    no_improvement = 0

                if no_improvement >= CG_NO_IMPROVEMENT_LIMIT:
                    break

        if not self.patterns:
            return {"error": "유효한 패턴을 생성하지 못했습니다."}

        self.patterns = [
            pattern for pattern in self.patterns
            if sum(self.item_info[item] * count for item, count in pattern.items()) >= self.min_width
        ]
        if not self.patterns:
            return {"error": f"{self.min_width}mm 이상으로 조합할 수 있는 패턴이 없습니다."}

        self._rebuild_pattern_cache()

        final_solution = self._solve_master_problem(is_final_mip=True)
        if not final_solution:
            return {"error": f"최종 해를 찾을 수 없습니다. {self.min_width}mm 이상을 충족하는 주문이 부족했을 수 있습니다."}


        return self._format_results(final_solution, start_prod_seq)

    def _format_results(self, final_solution, start_prod_seq=0):
        result_patterns = []
        pattern_details_for_db = []
        pattern_roll_details_for_db = []
        pattern_roll_cut_details_for_db = []
        production_counts = {item: 0 for item in self.demands}
        prod_seq = start_prod_seq
        prod_seq = start_prod_seq
        total_cut_seq_counter = 0

        # Extract common properties from the first row of the dataframe (since they are grouped)
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
            'p_lot': self.lot_no, # Use lot_no passed to init
            'core': safe_int(first_row.get('core', 0)),
            'order_pattern': first_row.get('order_pattern', '')
        }

        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                pattern_dict = self.patterns[j]
                prod_seq += 1
                
                db_widths, db_group_nos, db_lengths = [], [], []
                for group_no, num in pattern_dict.items():
                    width = self.item_info[group_no]
                    length = self.length_info.get(group_no, 0)
                    db_widths.extend([width] * num)
                    db_group_nos.extend([group_no] * num)
                    db_lengths.extend([length] * num)
                    production_counts[group_no] += int(round(count)) * num
                
                pattern_str = ', '.join(map(str, sorted(db_widths, reverse=True)))
                total_width = sum(db_widths)
                loss = self.max_width - total_width
                
                pattern_length = db_lengths[0] if db_lengths and db_lengths[0] is not None else 0

                result_patterns.append({
                    'pattern': pattern_str,
                    'pattern_width': total_width,
                    'loss_per_roll': loss,
                    'count': int(round(count)),
                    'prod_seq': prod_seq,
                    'rs_gubun': 'R',
                    'pattern_length': pattern_length
                })
                pattern_details_for_db.append({
                    'widths': (db_widths + [0] * 8)[:8],
                    'group_nos': (db_group_nos + [''] * 8)[:8],
                    'count': int(round(count)),
                    'prod_seq': prod_seq,
                    'rs_gubun': 'R',
                    'pattern_length': pattern_length,
                    **common_props
                })
                
                roll_seq_counter = 0
                for i in range(len(db_widths)):
                    roll_width = db_widths[i]
                    group_no = db_group_nos[i]
                    roll_length = self.length_info.get(group_no, 0)
                    roll_seq_counter += 1
                    
                    new_widths = [0] * 8
                    new_widths[0] = roll_width
                    
                    new_group_nos = [''] * 8
                    new_group_nos[0] = group_no
                    
                    pattern_roll_details_for_db.append({
                        'rollwidth': roll_width,
                        'pattern_length': roll_length,
                        'widths': new_widths,
                        'group_nos': new_group_nos,
                        'count': int(round(count)),
                        'prod_seq': prod_seq,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'R',
                        **common_props
                    })

                    cut_seq_counter = 0
                    if roll_width > 0:
                        cut_seq_counter += 1
                        total_cut_seq_counter += 1
                        pattern_roll_cut_details_for_db.append({
                            'prod_seq': prod_seq,
                            'unit_no': prod_seq,
                            'seq': total_cut_seq_counter,
                            'roll_seq': roll_seq_counter,
                            'cut_seq': cut_seq_counter,
                            'rs_gubun': 'R',
                            'width': roll_width,
                            'group_no': group_no,
                            'weight': 0,  # Weight calculation might be needed here
                            'pattern_length': roll_length,
                            'count': int(round(count)),
                            **common_props
                        })

        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['pattern', 'pattern_width', 'count', 'loss_per_roll', 'pattern_length']]

        df_demand = pd.DataFrame.from_dict(self.demands, orient='index', columns=['필요롤수'])
        df_demand.index.name = 'group_order_no'
        
        df_prod = pd.DataFrame.from_dict(production_counts, orient='index', columns=['생산롤수'])
        df_prod.index.name = 'group_order_no'
        
        df_summary = df_demand.join(df_prod)
        df_summary['과부족(롤)'] = df_summary['생산롤수'] - df_summary['필요롤수']

        info_cols = ['group_order_no', '지폭', '롤길이', '등급', '수출내수']
        available_info_cols = [c for c in self.df_spec_pre.columns if c in info_cols]
        group_info_df = self.df_spec_pre[available_info_cols].drop_duplicates(subset=['group_order_no'])

        fulfillment_summary = pd.merge(group_info_df, df_summary.reset_index(), on='group_order_no')
        
        fulfillment_summary.rename(columns={'지폭': '가로', '롤길이': '세로'}, inplace=True)
        
        final_cols = ['group_order_no', '가로', '세로', '수출내수', '등급', '필요롤수', '생산롤수', '과부족(롤)']
        available_final_cols = [c for c in final_cols if c in fulfillment_summary.columns]
        fulfillment_summary = fulfillment_summary[available_final_cols]

        return {
            "pattern_result": df_patterns.sort_values('count', ascending=False),
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "last_prod_seq": prod_seq
        }
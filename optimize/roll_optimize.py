import pandas as pd
from ortools.linear_solver import pywraplp

OVER_PROD_PENALTY = 200.0
UNDER_PROD_PENALTY = 10000.0
PATTERN_VALUE_THRESHOLD = 1.0 + 1e-6
CG_MAX_ITERATIONS = 200
CG_NO_IMPROVEMENT_LIMIT = 25
CG_SUBPROBLEM_TOP_N = 3
SMALL_PROBLEM_THRESHOLD = 10
FINAL_MIP_TIME_LIMIT_MS = 60000

class RollOptimize:
    
    def __init__(self, df_spec_pre, max_width=1000, min_width=0, max_pieces=8):
        self.df_spec_pre = df_spec_pre
        self.max_width = max_width
        self.min_width = min_width
        self.max_pieces = max_pieces
        self.patterns = []
        self.pattern_keys = set()
        self.demands = df_spec_pre.groupby('group_order_no')['주문수량'].sum().to_dict()
        self.items = list(self.demands.keys())
        self.item_info = df_spec_pre.set_index('group_order_no')['지폭'].to_dict()

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
        sorted_by_width_desc = sorted(self.items, key=lambda item: self.item_info.get(item, 0), reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda item: self.item_info.get(item, 0))

        heuristics = [sorted_by_demand, sorted_by_width_desc, sorted_by_width_asc]

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

    def _solve_master_problem(self, is_final_mip=False):
        solver_name = 'SCIP' if is_final_mip else 'GLOP'
        solver = pywraplp.Solver.CreateSolver(solver_name)
        if not solver:
            return None
        if is_final_mip and hasattr(solver, 'SetTimeLimit'):
            solver.SetTimeLimit(FINAL_MIP_TIME_LIMIT_MS)

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

        total_over_penalty = solver.Sum(OVER_PROD_PENALTY * self.item_info[item] * over_prod_vars[item]
                                        for item in self.demands)
        
        solver.Minimize(total_trim_loss + total_over_penalty)

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
        width_limit = self.max_width
        piece_limit = self.max_pieces
        item_details = []
        for item in self.items:
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
                if not pattern or cur_pieces != 0 or cur_width != 0:
                    continue
                key = frozenset(pattern.items())
                if key in seen_patterns:
                    continue
                total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                if total_width < self.min_width or total_width > self.max_width:
                    continue
                seen_patterns.add(key)
                candidate_patterns.append({'pattern': pattern, 'value': value, 'width': total_width, 'pieces': pieces})

        candidate_patterns.sort(key=lambda x: x['value'], reverse=True)
        return candidate_patterns[:CG_SUBPROBLEM_TOP_N]

    def _generate_all_patterns(self):
        all_patterns = []
        seen_patterns = set()
        item_list = list(self.items)

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

    def run_optimize(self, start_prod_seq=0):
        if len(self.items) <= SMALL_PROBLEM_THRESHOLD:
            self._generate_all_patterns()
        else:
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
        total_cut_seq_counter = 0

        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                pattern_dict = self.patterns[j]
                prod_seq += 1
                
                db_widths, db_group_nos = [], []
                for group_no, num in pattern_dict.items():
                    width = self.item_info[group_no]
                    db_widths.extend([width] * num)
                    db_group_nos.extend([group_no] * num)
                    production_counts[group_no] += int(round(count)) * num
                
                pattern_str = ', '.join(map(str, sorted(db_widths, reverse=True)))
                total_width = sum(db_widths)
                loss = self.max_width - total_width

                result_patterns.append({
                    'Pattern': pattern_str,
                    'Pattern_Width': total_width,
                    'Loss_per_Roll': loss,
                    'Count': int(round(count)),
                    'Prod_seq': prod_seq
                })
                pattern_details_for_db.append({
                    'widths': (db_widths + [0] * 8)[:8],
                    'group_nos': (db_group_nos + [''] * 8)[:8],
                    'Count': int(round(count)),
                    'Prod_seq': prod_seq
                })
                
                roll_seq_counter = 0
                for i in range(len(db_widths)):
                    roll_width = db_widths[i]
                    group_no = db_group_nos[i]
                    roll_seq_counter += 1
                    
                    new_widths = [0] * 8
                    new_widths[0] = roll_width
                    
                    new_group_nos = [''] * 8
                    new_group_nos[0] = group_no
                    
                    pattern_roll_details_for_db.append({
                        'rollwidth': roll_width,
                        'widths': new_widths,
                        'group_nos': new_group_nos,
                        'Count': int(round(count)),
                        'Prod_seq': prod_seq,
                        'Roll_seq': roll_seq_counter
                    })

                    cut_seq_counter = 0
                    if roll_width > 0:
                        cut_seq_counter += 1
                        total_cut_seq_counter += 1
                        pattern_roll_cut_details_for_db.append({
                            'PROD_SEQ': prod_seq,
                            'UNIT_NO': prod_seq,
                            'SEQ': total_cut_seq_counter,
                            'ROLL_SEQ': roll_seq_counter,
                            'CUT_SEQ': cut_seq_counter,
                            'WIDTH': roll_width,
                            'GROUP_NO': group_no,
                            'WEIGHT': 0,  # Weight calculation might be needed here
                            'TOTAL_LENGTH': 0, # Length calculation might be needed here
                            'CUT_CNT': int(round(count))
                        })

        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['Pattern', 'Pattern_Width', 'Count', 'Loss_per_Roll']]

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
            "pattern_result": df_patterns.sort_values('Count', ascending=False),
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "last_prod_seq": prod_seq
        }
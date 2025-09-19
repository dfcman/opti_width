import pandas as pd
from ortools.linear_solver import pywraplp
import math

# --- ����ȭ ���� ��� ---
# ���Ƽ ��
OVER_PROD_PENALTY = 20.0    # �����꿡 ���� ���Ƽ
UNDER_PROD_PENALTY = 500.0  # �������꿡 ���� ���Ƽ
PATTERN_COMPLEXITY_PENALTY = 0.001  # ���� ���⼺�� ���� ���Ƽ

# �˰��� �Ķ����
MIN_PIECES_PER_PATTERN = 1      # ���Ͽ� ���Ե� �� �ִ� �ּ� ��(piece)�� ��
SMALL_PROBLEM_THRESHOLD = 8     # ��ü Ž���� ������ �ִ� �ֹ� ���� ���� ��
SOLVER_TIME_LIMIT_MS = 60000    # ���� MIP �ֹ��� �ִ� ���� �ð� (�и���)
CG_MAX_ITERATIONS = 1000         # �� ����(Column Generation) �ִ� �ݺ� Ƚ��
CG_NO_IMPROVEMENT_LIMIT = 100    # ���� ���� ���, �� ���� ���� ���� ����
CG_SUBPROBLEM_TOP_N = 5         # �� ���� ��, �� �ݺ����� �߰��� ���� N�� �ű� ����

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
            max_sc_width
    ):
        df_spec_pre['����'] = df_spec_pre['����']

        self.b_wgt = b_wgt
        self.sheet_roll_length = sheet_roll_length
        self.sheet_trim = sheet_trim
        self.original_max_width = max_width
        self.original_df_orders = df_spec_pre.copy() # ���� �ֹ� ����
        self.df_orders = df_spec_pre.copy()
        
        self.demands_in_rolls = self._calculate_demand_rolls(self.df_orders)
        self.order_widths = list(self.demands_in_rolls.keys()) 

        width_summary = {}
        tons_per_width = self.df_orders.groupby('����')['�ֹ���'].sum()
        for width, required_rolls in self.demands_in_rolls.items():
            order_tons = tons_per_width.get(width, 0)
            width_summary[width] = {'order_tons': order_tons}
        self.width_summary = width_summary 

        self.min_sc_width = min_sc_width
        self.max_sc_width = max_sc_width
        self.items, self.item_info, self.item_composition = self._prepare_items(min_sc_width, max_sc_width)

        self.max_width = max_width
        self.min_width = min_width
        self.min_pieces = MIN_PIECES_PER_PATTERN
        self.max_pieces = int(max_pieces)
        print(f"\n--- ���� ��������: �ּ� {self.min_pieces}��, �ִ� {self.max_pieces}�� ---")

        self.patterns = []

    def _prepare_items(self, min_sc_width, max_sc_width):
        """������ ������(������ �������)�� �����մϴ�."""
        items = []
        item_info = {}  # item_name -> width
        item_composition = {}  # composite_item_name -> {original_width: count}

        for width in self.order_widths:
            for i in range(1, 5): # 1, 2, 3, 4������ ���
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
        """�ֹ����� �������� ������ �ʿ� �� ���� ����մϴ�."""
        df_copy = df_orders.copy()
        sheet_roll_length_mm = self.sheet_roll_length * 1000

        def calculate_rolls(row):
            width_mm = row['����']
            length_mm = row['����']
            order_ton = row['�ֹ���']

            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0 or order_ton <= 0:
                return 0
            
            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1_000_000
            if sheet_weight_g <= 0:
                return 0

            total_sheets_needed = (order_ton * 1_000_000) / sheet_weight_g
            sheets_per_roll_length = math.floor(sheet_roll_length_mm / length_mm)
            if sheets_per_roll_length <= 0:
                return 0

            sheets_per_roll = sheets_per_roll_length * 1 # num_across is always 1
            return round(total_sheets_needed / sheets_per_roll, 0)

        df_copy['rolls'] = df_copy.apply(calculate_rolls, axis=1)
        demand_rolls = df_copy.groupby('����')['rolls'].sum().astype(int).to_dict()

        print("\n--- ������ �ʿ� �� �� ---")
        for width, rolls in demand_rolls.items():
            print(f"  ���� {width}mm: {rolls} ��")
        print("--------------------------\n")
        
        return demand_rolls

    def _generate_initial_patterns(self):
        """
        �ʱ� ���� ������ ���� �� ���� ������ ����մϴ�.
        1. First-Fit-Decreasing �޸���ƽ���� �پ��� ǰ���� ���� ������ �����մϴ�.
        2. �� ������ �������� �ִ��� ä��� '���� ǰ�� ����'�� �����Ͽ� �پ缺�� Ȯ���մϴ�.
        """
        print("\n--- ��ȿ�� �ʱ� ������ �����մϴ� ---")
        
        sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        seen_patterns = set()

        # --- 1. FFD (First-Fit-Decreasing) �޸���ƽ���� ȥ�� ���� ���� ---
        for item in sorted_items:
            current_pattern = {item: 1}
            current_width = self.item_info[item]
            current_pieces = 1

            # ���� ä��� (Greedy)
            while current_pieces < self.max_pieces:
                remaining_width = self.max_width - current_width
                best_fit_item = next((i for i in sorted_items if self.item_info[i] <= remaining_width), None)
                if not best_fit_item:
                    break 
                current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
                current_width += self.item_info[best_fit_item]
                current_pieces += 1

            # �ּ� �ʺ� ������Ű��
            while current_width < self.min_width and current_pieces < self.max_pieces:
                item_to_add = next((i for i in reversed(sorted_items) if current_width + self.item_info[i] <= self.max_width), None)
                if item_to_add:
                    current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
                    current_width += self.item_info[item_to_add]
                    current_pieces += 1
                else:
                    break

            # ���� ��ȿ�� �˻� �� �߰�
            if self.min_width <= current_width and self.min_pieces <= current_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in seen_patterns:
                    self.patterns.append(current_pattern)
                    seen_patterns.add(pattern_key)

        print(f"--- {len(self.patterns)}���� ȥ�� ���� ������ ---")

        # --- 2. ��� �������� ���� '���� ǰ�� ����' ���� ---
        pure_patterns_added = 0
        for item in sorted_items:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0: continue

            # �ش� ���������θ� ������ ���� ���� �õ�
            num_items = min(int(self.max_width / item_width), self.max_pieces)
            
            # �ʺ� ū ���պ��� ���� ���ձ��� ���������� Ȯ��
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                
                if self.min_width - 100 <= total_width and self.min_pieces <= num_items:
                    pattern_key = frozenset(new_pattern.items())
                    if pattern_key not in seen_patterns:
                        self.patterns.append(new_pattern)
                        seen_patterns.add(pattern_key)
                        pure_patterns_added += 1
                        break # �� ���������� ���� �� �ִ� ���� ���� ���������� ã�����Ƿ� ����
                
                num_items -= 1

        if pure_patterns_added > 0:
            print(f"--- {pure_patterns_added}���� ���� ǰ�� ���� �߰��� ---")

        # --- ���� ����: �ʱ� �������� Ŀ������ �ʴ� �ֹ��� �ִ��� ���� Ȯ�� ---
        covered_widths = {w for p in self.patterns for item_name in p for w in self.item_composition.get(item_name, {})}
        uncovered_widths = set(self.order_widths) - covered_widths

        if uncovered_widths:
            print(f"--- ���: �ʱ� ���Ͽ� ���Ե��� ���� �ֹ� �߰�: {uncovered_widths} ---")
            print("--- �ش� �ֹ��� ���� ���� ������ �߰� �����մϴ�. ---")
            
            for width in uncovered_widths:
                print(f"  - ���� {width}mm�� ���� ���� ǰ�� ���� ���� �õ�...")

                # 1. �� �������� ���� �� �ִ� ��ȿ�� ������ ������ ����� ã���ϴ�.
                valid_components = []
                for i in range(1, 5): # 1~4�� ���
                    item_name = f"{width}x{i}"
                    # �������� �̹� �����Ǿ����� Ȯ��
                    if item_name in self.item_info:
                        valid_components.append(item_name)
                    else:
                        # �������� ���� �� ��ȿ�� �˻�
                        composite_width = width * i + self.sheet_trim
                        if (self.min_sc_width <= composite_width <= self.max_sc_width) and \
                           (composite_width <= self.original_max_width):
                            # ��ȿ�ϸ� ������ ������ �߰�
                            if item_name not in self.items: self.items.append(item_name)
                            self.item_info[item_name] = composite_width
                            self.item_composition[item_name] = {width: i}
                            valid_components.append(item_name)

                if not valid_components:
                    print(f"    - ���: ���� {width}mm�� ���� �� �ִ� ��ȿ�� ������ �������� �����ϴ�. ���� ������ ������ �� �����ϴ�.")
                    continue

                # 2. �ʺ� ���� ������ �����Ͽ� Greedy �˰��� �غ�
                sorted_components = sorted(valid_components, key=lambda i: self.item_info[i], reverse=True)
                
                # 3. Greedy ������� ������ ���� ǰ�� ���� ����
                new_pattern = {}
                current_width = 0
                current_pieces = 0
                
                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    
                    # ���� ������ �� �� �ִ� ���� ū ������� ã��
                    best_fit = next((item for item in sorted_components if self.item_info[item] <= remaining_width), None)
                    
                    if not best_fit:
                        break # �� �̻� �߰��� �� �ִ� ������Ұ� ����
                    
                    new_pattern[best_fit] = new_pattern.get(best_fit, 0) + 1
                    current_width += self.item_info[best_fit]
                    current_pieces += 1

                # 4. ������ ������ ��ȿ�� �˻� �� �߰�
                if new_pattern:
                    total_width = sum(self.item_info[name] * count for name, count in new_pattern.items())
                    total_pieces = sum(new_pattern.values())

                    if self.min_width - 1000 <= total_width and self.min_pieces <= total_pieces:
                        pattern_key = frozenset(new_pattern.items())
                        if pattern_key not in seen_patterns:
                            self.fallback_patterns.append(new_pattern)
                            seen_patterns.add(pattern_key)
                            print(f"    -> ������ ���� ����: {new_pattern} (�ʺ�: {total_width}mm, �� ��: {total_pieces}) -> ���� �������� �߰���.")
                        else:
                            print(f"    - ������ ���� ���� {new_pattern}�� �̹� �����մϴ�.")
                    else:
                        print(f"    - ������ ���� ���� {new_pattern}�� ���� ��������(�ּҳʺ�/����)�� �������� ���մϴ�. (�ʺ�: {total_width}, �� ��: {total_pieces})")
                else:
                    print(f"    - ���� {width}mm�� ���� ���� ������ �������� ���߽��ϴ�.")

        print(f"--- �� {len(self.patterns)}���� �ʱ� ���� ������ ---")
        print(self.patterns)
        print

    def _solve_master_problem_ilp(self, is_final_mip=False):
        """������ ����(Master Problem)�� ������ȹ������ �ذ��մϴ�."""
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        if is_final_mip:
            solver.SetTimeLimit(SOLVER_TIME_LIMIT_MS)

        # ���� ����
        x = {j: solver.IntVar(0, solver.infinity(), f'P_{j}') if is_final_mip else solver.NumVar(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}
        over_prod_vars = {w: solver.NumVar(0, solver.infinity(), f'Over_{w}') for w in self.demands_in_rolls}
        
        under_prod_vars = {}
        for width, required_rolls in self.demands_in_rolls.items():
            allowed_under_prod = max(1, math.ceil(required_rolls * 0.05))
            under_prod_vars[width] = solver.NumVar(0, allowed_under_prod, f'Under_{width}')

        # ��������: ���귮 + ������ = ���䷮ + �����귮
        constraints = {}
        for width, required_rolls in self.demands_in_rolls.items():
            production_for_width = solver.Sum(
                x[j] * sum(self.item_composition[item_name].get(width, 0) * count for item_name, count in self.patterns[j].items())
                for j in range(len(self.patterns))
            )
            constraints[width] = solver.Add(production_for_width + under_prod_vars[width] == required_rolls + over_prod_vars[width], f'demand_{width}')

        # �����Լ�: �� �� �� + ���Ƽ �ּ�ȭ
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
        """���깮��(Sub-problem)�� ���� ���α׷������� �ذ��Ͽ� ���ο� ������ ã���ϴ�."""
        dp = [[(0, []) for _ in range(self.max_width + 1)] for _ in range(self.max_pieces + 1)]

        item_details = []
        for item_name in self.items:
            item_width = self.item_info[item_name]
            item_value = sum(count * duals.get(width, 0) for width, count in self.item_composition[item_name].items())
            if item_value > 0:
                item_details.append((item_name, item_width, item_value))

        for k in range(1, self.max_pieces + 1):
            for w in range(1, self.max_width + 1):
                dp[k][w] = dp[k-1][w]
                for name, i_width, i_value in item_details:
                    if w >= i_width:
                        prev_value, prev_items = dp[k-1][w - i_width]
                        new_value = prev_value + i_value
                        if new_value > dp[k][w][0]:
                            dp[k][w] = (new_value, prev_items + [name])

        candidate_patterns = []
        for k in range(self.min_pieces, self.max_pieces + 1):
            for w in range(self.min_width, self.max_width + 1):
                reduced_cost = dp[k][w][0]
                if reduced_cost > 1.0:
                    pattern_dict = {}
                    for item in dp[k][w][1]:
                        pattern_dict[item] = pattern_dict.get(item, 0) + 1
                    candidate_patterns.append({'pattern': pattern_dict, 'cost': reduced_cost})

        if not candidate_patterns:
            return []

        # �ߺ� ���� �� ���� N�� ����
        seen_patterns = set()
        unique_candidates = []
        for cand in sorted(candidate_patterns, key=lambda x: x['cost'], reverse=True):
            pattern_key = frozenset(cand['pattern'].items())
            if pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                unique_candidates.append(cand['pattern'])
        
        return unique_candidates[:CG_SUBPROBLEM_TOP_N]

    def _generate_all_patterns(self):
        """���� ������ ���� ��� ������ ������ �����մϴ� (Brute-force)."""
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

            # ���� �������� �������� �ʰ� �������� �Ѿ
            find_combinations_recursive(start_index + 1, current_pattern, current_width, current_pieces)

            # ���� �������� �����Ͽ� ��� ȣ��
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

    def _find_unproducible_width(self):
        """���� ���� ������� ������ �� ���� ���� ������ ã���ϴ�."""
        for width, demand in self.demands_in_rolls.items():
            can_produce = False
            for p in self.patterns:
                for item_name, count in p.items():
                    if self.item_composition.get(item_name, {}).get(width, 0) > 0:
                        can_produce = True
                        break
                if can_produce:
                    break
            if not can_produce:
                print(f"[���� ����] ���� {width} (����: {demand}��)�� ������ �� �ִ� ������ ���� ��Ͽ� �����ϴ�.")
                return width
        
        print("[���� �Ҹ�] ��� ���� ������ ���� ���� ������ ����������, �ٸ� �������� ���� �ظ� ã�� ���߽��ϴ�.")
        return None

    def run_optimize(self):
        """����ȭ ���� ���� �Լ�"""
        excluded_widths = []
        final_results = {}
        
        while True:
            # 1. ���� ���� ����� ������� �ֹ� �� ���� �Ӽ� �缳��
            self.df_orders = self.original_df_orders[~self.original_df_orders['����'].isin(excluded_widths)]
            if self.df_orders.empty:
                final_results = {"error": "��� �ֹ��� ���ܵǾ� ����ȭ�� ������ �� �����ϴ�."}
                break

            self.demands_in_rolls = self._calculate_demand_rolls(self.df_orders)
            self.order_widths = list(self.demands_in_rolls.keys()) 
            
            width_summary = {}
            tons_per_width = self.df_orders.groupby('����')['�ֹ���'].sum()
            for width, required_rolls in self.demands_in_rolls.items():
                order_tons = tons_per_width.get(width, 0)
                width_summary[width] = {'order_tons': order_tons}
            self.width_summary = width_summary
            
            self.items, self.item_info, self.item_composition = self._prepare_items(self.min_sc_width, self.max_sc_width)
            self.patterns = []

            # 2. ���� ����
            if len(self.order_widths) <= SMALL_PROBLEM_THRESHOLD:
                print(f"\n--- �ֹ� ������ {len(self.order_widths)}�� �̹Ƿ�, ��� ������ Ž���մϴ� (Small-scale) ---")
                self._generate_all_patterns()
            else:
                print(f"\n--- �ֹ� ������ {len(self.order_widths)}�� �̹Ƿ�, �� ���� ����� �����մϴ� (Large-scale) ---")
                self._generate_initial_patterns()
                
                initial_pattern_count = len(self.patterns)
                self.patterns = [p for p in self.patterns if sum(self.item_info.get(i, 0) * c for i, c in p.items()) >= self.min_width - 100]
                if len(self.patterns) < initial_pattern_count:
                    print(f"--- �ʱ� ���� ���͸�: {initial_pattern_count}�� -> {len(self.patterns)}�� (�ּ� �ʺ� {self.min_width}mm ����)")

                if not self.patterns:
                    final_results = {"error": "�ʱ� ��ȿ ������ ������ �� �����ϴ�. ���������� �ʹ� ������ �� �ֽ��ϴ�."}
                    break

                # �� ���� ����
                no_improvement_count = 0
                should_restart = False
                for iteration in range(CG_MAX_ITERATIONS):
                    master_solution = self._solve_master_problem_ilp()
                    if not master_solution or 'duals' not in master_solution:
                        print("\n--- ������ ������ �ذ��� �� �����ϴ�. ���� �м� ���� ---")
                        unproducible_width = self._find_unproducible_width()
                        if unproducible_width:
                            print(f"--- �ش� ����({unproducible_width})�� �����ϰ� ����ȭ�� �ٽ� �õ��մϴ�. ---")
                            excluded_widths.append(unproducible_width)
                            should_restart = True
                        else:
                            final_results = {"error": "������ ������ �ذ��� �� ������, Ư�� ���� ������ ã�� ���߽��ϴ�."}
                        break # Exit iteration loop

                    if should_restart: break

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
                    
                    if patterns_added > 0: no_improvement_count = 0
                    else: no_improvement_count += 1
                    
                    if no_improvement_count >= CG_NO_IMPROVEMENT_LIMIT:
                        print(f"--- {CG_NO_IMPROVEMENT_LIMIT}���� �ݺ� ���� ������ ���� �������� �����ϰ� �����մϴ� (�ݺ� {iteration}). ---")
                        break
                
                if should_restart:
                    continue # Restart while loop

            # 3. ���� ����ȭ
            if not self.patterns:
                final_results = {"error": "��ȿ�� ������ ������ �� �����ϴ�."}
                break

            print(f"\n--- �� {len(self.patterns)}���� �������� ���� ����ȭ�� �����մϴ�. ---")
            final_solution = self._solve_master_problem_ilp(is_final_mip=True)
            
            if not final_solution:
                print("\n--- ���� ����ȭ �ظ� ã�� �� �����ϴ�. ���� �м� ���� ---")
                unproducible_width = self._find_unproducible_width()
                if unproducible_width:
                    print(f"--- �ش� ����({unproducible_width})�� �����ϰ� ����ȭ�� �ٽ� �õ��մϴ�. ---")
                    excluded_widths.append(unproducible_width)
                    continue # Restart while loop
                else:
                    final_results = {"error": "���� �ظ� ã�� �� �����ϴ�."}
                    break
            
            # 4. ���� �� ��� ������
            final_results = self._format_results(final_solution)
            if excluded_widths:
                print("\n[����ȭ ���� ���]")
                excluded_summary = []
                for w in excluded_widths:
                    print(f"  - ����: {w} mm (���� ������ ������ ã�� ����)")
                    excluded_summary.append({"����": w, "����": "���� ���� ���� ����"})
                final_results['excluded_summary'] = excluded_summary
            
            break # while ���� ����

        return final_results

    def _format_results(self, final_solution):
        """���� ����� ������������ �������� �������մϴ�."""
        
        # ���� ���귮 ���
        final_production_rolls = {width: 0 for width in self.order_widths}
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                roll_count = int(round(count))
                for item_name, num_in_pattern in self.patterns[j].items():
                    for width, num_pieces in self.item_composition[item_name].items():
                        if width in final_production_rolls:
                            final_production_rolls[width] += roll_count * num_in_pattern * num_pieces

        # ��� ������������ ����
        result_patterns, pattern_details_for_db = self._build_pattern_details(final_solution)
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['Pattern', 'Roll_Production_Length', 'Count', 'Loss_per_Roll']]

        # �ֹ� ���� ��� ����
        fulfillment_summary = self._build_fulfillment_summary(final_production_rolls)

        print("\n[�ֹ� ���� ���]")
        print(fulfillment_summary.to_string())

        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "fulfillment_summary": fulfillment_summary
        }

    def _build_pattern_details(self, final_solution):
        """���� ��� ����� DB ������ ���� �� ������ �����մϴ�."""
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
        """�ֹ� ���� ��� �������������� �����մϴ�."""
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
                '����': width,
                '�ֹ���(��)': order_tons,
                '���귮(��)': round(produced_tons, 2),
                '������(��)': round(over_prod_tons, 2),
                '�ʿ�Ѽ�': required_rolls,
                '����Ѽ�': produced_rolls,
                '������(��)': produced_rolls - required_rolls,
            })
        
        return pd.DataFrame(summary_data)[[
            '����', '�ֹ���(��)', '���귮(��)', '������(��)',
            '�ʿ�Ѽ�', '����Ѽ�', '������(��)'
        ]]
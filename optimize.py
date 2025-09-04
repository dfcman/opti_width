import pandas as pd
from ortools.linear_solver import pywraplp
import time

class Optimize:
    
    def __init__(self, df_spec_pre, max_width=1000, min_width=0, max_pieces=8):
        self.df_spec_pre = df_spec_pre
        self.max_width = max_width
        self.min_width = min_width
        self.max_pieces = max_pieces
        self.patterns = []
        self.demands = df_spec_pre.groupby('group_order_no')['주문수량'].sum().to_dict()
        self.items = list(self.demands.keys())
        self.item_info = df_spec_pre.set_index('group_order_no')['지폭'].to_dict()

    def _generate_initial_patterns(self):
        for item in self.items:
            self.patterns.append({item: 1})

    def _solve_master_problem(self, is_final_mip=False):
        solver = pywraplp.Solver.CreateSolver('GLOP' if not is_final_mip else 'SCIP')
        x = {j: (solver.IntVar if is_final_mip else solver.NumVar)(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}
        
        production_exprs = {}
        constraints = {}
        for item, demand in self.demands.items():
            production_expr = solver.Sum(p.get(item, 0) * x[j] for j, p in enumerate(self.patterns))
            production_exprs[item] = production_expr
            constraints[item] = solver.Add(production_expr >= demand, f'demand_{item}')

        total_rolls = solver.Sum(x[j] for j in range(len(self.patterns)))
        total_over_production = solver.Sum(production_exprs[item] - demand for item, demand in self.demands.items())
        penalty_weight = 0.001
        solver.Minimize(total_rolls + penalty_weight * total_over_production)

        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            solution = {
                'objective': solver.Objective().Value(),
                'pattern_counts': {j: var.solution_value() for j, var in x.items()}
            }
            if not is_final_mip:
                solution['duals'] = {item: constraints[item].dual_value() for item in self.demands}
            return solution
        return None

    def _solve_subproblem(self, duals):
        dp = [[(0, []) for _ in range(self.max_pieces + 1)] for _ in range(self.max_width + 1)]
        for item in self.items:
            value = duals.get(item, 0)
            weight = self.item_info[item]
            for w in range(self.max_width, weight - 1, -1):
                for k in range(self.max_pieces, 0, -1):
                    if dp[w - weight][k - 1][0] + value > dp[w][k][0]:
                        dp[w][k] = (dp[w - weight][k - 1][0] + value, dp[w - weight][k - 1][1] + [item])
        best_pattern_items = []
        max_value = 1.0
        for w in range(self.min_width, self.max_width + 1):
            for k in range(1, self.max_pieces + 1):
                if dp[w][k][0] > max_value:
                    max_value = dp[w][k][0]
                    best_pattern_items = dp[w][k][1]
        if not best_pattern_items:
            return None
        new_pattern = {}
        for item in best_pattern_items:
            new_pattern[item] = new_pattern.get(item, 0) + 1
        return new_pattern

    def run_optimize(self):
        self._generate_initial_patterns()
        if not self.patterns:
            return {"error": "초기 패턴을 생성할 수 없습니다."}
        for iteration in range(50):
            master_solution = self._solve_master_problem()
            if not master_solution or 'duals' not in master_solution:
                break
            new_pattern = self._solve_subproblem(master_solution['duals'])
            if new_pattern is None:
                break
            if new_pattern not in self.patterns:
                self.patterns.append(new_pattern)

        original_pattern_count = len(self.patterns)
        
        good_patterns = [
            p for p in self.patterns
            if sum(self.item_info[item] * count for item, count in p.items()) >= self.min_width
        ]
        
        covered_items = set()
        for p in good_patterns:
            covered_items.update(p.keys())
            
        uncovered_items = set(self.items) - covered_items
        
        if uncovered_items:
            for item_to_cover in uncovered_items:
                candidate_patterns = [p for p in self.patterns if item_to_cover in p]
                
                if not candidate_patterns:
                    continue

                best_candidate = max(
                    candidate_patterns,
                    key=lambda p: sum(self.item_info[item] * count for item, count in p.items())
                )
                
                if best_candidate not in good_patterns:
                    good_patterns.append(best_candidate)

        self.patterns = good_patterns

        final_solution = self._solve_master_problem(is_final_mip=True)
        if not final_solution:
            return {"error": f"최종 해를 찾을 수 없습니다. {self.min_width}mm 이상의 폭으로 조합할 수 없는 주문이 포함되었을 수 있습니다."}
        return self._format_results(final_solution)

    def _format_results(self, final_solution):
        result_patterns = []
        pattern_details_for_db = []
        production_counts = {item: 0 for item in self.demands}

        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                pattern_dict = self.patterns[j]
                db_widths, db_group_nos = [], []
                for group_no, num in pattern_dict.items():
                    width = self.item_info[group_no]
                    db_widths.extend([width] * num)
                    db_group_nos.extend([group_no] * num)
                    production_counts[group_no] += int(round(count)) * num
                
                pattern_str = ', '.join(map(str, sorted(db_widths, reverse=True)))
                loss = self.max_width - sum(db_widths)

                result_patterns.append({
                    'Pattern': pattern_str,
                    'Count': int(round(count)),
                    'Loss_per_Roll': loss
                })
                pattern_details_for_db.append({
                    'Count': int(round(count)),
                    'widths': (db_widths + [0] * 8)[:8],
                    'group_nos': (db_group_nos + [''] * 8)[:8]
                })

        df_patterns = pd.DataFrame(result_patterns)

        df_demand = pd.DataFrame.from_dict(self.demands, orient='index', columns=['Total_Ordered_per_Group'])
        df_demand.index.name = 'group_order_no'
        df_prod = pd.DataFrame.from_dict(production_counts, orient='index', columns=['Total_Produced_per_Group'])
        df_prod.index.name = 'group_order_no'
        
        df_summary = df_demand.join(df_prod)
        df_summary['Over_Production'] = df_summary['Total_Produced_per_Group'] - df_summary['Total_Ordered_per_Group']

        group_info_cols = self.df_spec_pre[['group_order_no', '지폭', '롤길이', '등급']].drop_duplicates()
        fulfillment_summary = pd.merge(group_info_cols, df_summary.reset_index(), on='group_order_no')

        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False),
            "pattern_details_for_db": pattern_details_for_db,
            "fulfillment_summary": fulfillment_summary
        }
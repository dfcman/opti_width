import pandas as pd
from ortools.linear_solver import pywraplp
import time
import math

class SheetOptimize:
    def __init__(self, df_spec_pre, max_width, min_width, max_pieces, b_wgt, sheet_roll_length, sheet_trim):
        df_spec_pre['지폭'] = df_spec_pre['가로']

        MIN_ITEM_WIDTH = 850
        MAX_ITEM_WIDTH = 2600

        self.b_wgt = b_wgt
        self.sheet_roll_length = sheet_roll_length
        self.sheet_trim = sheet_trim
        self.original_max_width = max_width
        
        df_calculated = self._calculate_demand_length(df_spec_pre)
        self.original_orders = df_calculated.copy()

        # 각 group_order_no에 대한 sheets_per_length를 계산하여 맵으로 저장
        self.sheets_per_length_map = {}
        sheet_roll_length_mm = self.sheet_roll_length * 1000 # 미터를 밀리미터로 변환  
        for _, row in self.original_orders.iterrows():
            seiro = row['세로']
            if seiro > 0:
                self.sheets_per_length_map[row['group_order_no']] = math.floor(sheet_roll_length_mm / seiro)    
            else:
                self.sheets_per_length_map[row['group_order_no']] = 0

        self.composite_item_map = {}
        producible_item_rows = []

        for _, row in df_calculated.iterrows():
            original_group_no = row['group_order_no']
            dim = row['지폭']

            for i in range(1, 5):
                base_width = dim * i

                if not (MIN_ITEM_WIDTH <= base_width <= MAX_ITEM_WIDTH):
                    continue
                
                producible_width = base_width + self.sheet_trim

                if producible_width > self.original_max_width:
                    continue

                producible_group_no = f"{original_group_no}{i}"
                
                if i == 1:
                    producible_group_no = original_group_no
                else:
                    self.composite_item_map[producible_group_no] = {
                        'original_group': original_group_no,
                        'num_pieces': i,
                        'original_dim': dim
                    }

                new_row = row.copy()
                new_row['group_order_no'] = producible_group_no
                new_row['지폭'] = producible_width
                new_row['주문수량'] = 0
                new_row['주문톤'] = 0
                new_row['등급'] = 'producible'
                producible_item_rows.append(new_row)

        if not producible_item_rows:
            raise ValueError("No producible items could be generated from the orders.")

        df_producible_items = pd.DataFrame(producible_item_rows).drop_duplicates(subset=['group_order_no'])

        self.df_spec_pre = df_producible_items
        self.items = list(self.df_spec_pre['group_order_no'])
        self.item_info = self.df_spec_pre.set_index('group_order_no')['지폭'].to_dict()
        self.demands = df_calculated.set_index('group_order_no')['주문수량'].to_dict()
        
        self.max_width = max_width
        self.min_width = min_width
        self.max_pieces = max_pieces
        self.patterns = []

    def _calculate_demand_length(self, df_orders):
        """
        주문톤(order_ton_cnt)을 총 필요 장수로 변환하여 '주문수량' 컬럼에 설정합니다.
        """
        df_copy = df_orders.copy() 
        
        group_cols = ['group_order_no', '지폭', '가로', '세로', '등급']
        df_grouped = df_copy.groupby(group_cols, as_index=False)['주문톤'].sum()

        required_sheets_list = []
        for _, row in df_grouped.iterrows():
            width_mm = row['지폭']
            length_mm = row['세로']
            order_ton = row['주문톤']

            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0:
                required_sheets_list.append(0)
                continue

            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1_000_000
            if sheet_weight_g <= 0:
                required_sheets_list.append(0)
                continue
            
            total_sheets_needed = (order_ton * 1_000_000) / sheet_weight_g
            required_sheets_list.append(total_sheets_needed)

        df_grouped['주문수량'] = required_sheets_list
        
        return df_grouped

    def _generate_initial_patterns(self):
        sorted_items = sorted(self.items, key=lambda item: self.item_info[item], reverse=True)
        
        for item in self.items:
            self.patterns.append({item: 1})

        for _ in range(30):
            new_pattern = {}
            current_width = 0
            current_pieces = 0
            for item in sorted_items:
                item_width = self.item_info[item]
                if current_width + item_width <= self.max_width and current_pieces < self.max_pieces:
                    new_pattern[item] = new_pattern.get(item, 0) + 1
                    current_width += item_width
                    current_pieces += 1
            
            if new_pattern and new_pattern not in self.patterns:
                self.patterns.append(new_pattern)
                used_item = list(new_pattern.keys())[0]
                sorted_items.remove(used_item)
                sorted_items.append(used_item)

    def _solve_master_problem(self, is_final_mip=False):
        # is_final_mip: 최종 정수해를 구하는 단계인지(True), 아니면 열 생성 과정 중인지(False)를 나타내는 플래그입니다.

        solver = pywraplp.Solver.CreateSolver('GLOP' if not is_final_mip else 'SCIP')

        if is_final_mip:
            solver.SetTimeLimit(60000)
        x = {j: (solver.IntVar if is_final_mip else solver.NumVar)(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}
        
        production_exprs = {}
        constraints = {}

        for original_item_group, demand in self.demands.items():
            sheets_per_len = self.sheets_per_length_map.get(original_item_group, 0)
            if sheets_per_len == 0:
                # 이 품목은 생산이 불가능하므로, 제약조건에서 제외하거나 수요를 0으로 처리
                continue

            # 1롤 당 생산되는 총 장수 (가로폭에 들어가는 품목 수의 합)
            total_pieces_produced = solver.Sum(
                x[j] * (
                    self.patterns[j].get(original_item_group, 0) +
                    sum(
                        self.patterns[j].get(composite_group, 0) * composite_info['num_pieces']
                        for composite_group, composite_info in self.composite_item_map.items()
                        if composite_info['original_group'] == original_item_group
                    )
                )
                for j in range(len(self.patterns))
            )
            
            # 총 생산량 = 롤 수 * 롤 당 생산 장수
            total_production_for_item = total_pieces_produced * sheets_per_len

            production_exprs[original_item_group] = total_production_for_item
            constraints[original_item_group] = solver.Add(total_production_for_item >= demand * 0.95, f'demand_{original_item_group}')

        pattern_penalties = []
        for p in self.patterns:
            penalty = 0
            for item, count_in_pattern in p.items():
                num_pieces = self.composite_item_map.get(item, {'num_pieces': 1})['num_pieces']
                if num_pieces == 3:
                    penalty += 0.5 * count_in_pattern
                elif num_pieces == 4:
                    penalty += 1.0 * count_in_pattern
            pattern_penalties.append(penalty)

        total_rolls = solver.Sum(x[j] for j in range(len(self.patterns)))
        
        # 과생산 페널티 계산 시, 수요 단위를 장수에서 롤수로 변환하여 비교해야 함
        total_over_production_rolls = solver.Sum(
            (production_exprs[item] / self.sheets_per_length_map.get(item, 1)) - (demand / self.sheets_per_length_map.get(item, 1))
            for item, demand in self.demands.items() if self.sheets_per_length_map.get(item, 0) > 0
        )

        total_piece_penalty = solver.Sum(x[j] * pattern_penalties[j] for j in range(len(self.patterns)))

        over_production_penalty_weight = 1.0
        piece_penalty_weight = 0.2

        solver.Minimize(total_rolls +
                        piece_penalty_weight * total_piece_penalty +
                        over_production_penalty_weight * total_over_production_rolls)
        
        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            solution = {
                'objective': solver.Objective().Value(),
                'pattern_counts': {j: var.solution_value() for j, var in x.items()}
            }
            if not is_final_mip:
                solution['duals'] = {item: constraints[item].dual_value() for item in self.demands if item in constraints}
            return solution
        return None

    def _solve_subproblem_old(self, duals):
        dp = [[(0, []) for _ in range(self.max_pieces + 1)] for _ in range(self.max_width + 1)]
        
        for item_group_no in self.items:
            original_group = item_group_no
            num_pieces = 1
            if item_group_no in self.composite_item_map:
                info = self.composite_item_map[item_group_no]
                original_group = info['original_group']
                num_pieces = info['num_pieces']
            
            sheets_per_len = self.sheets_per_length_map.get(original_group, 0)
            dual_value = duals.get(original_group, 0)
            
            value = num_pieces * dual_value * sheets_per_len
            weight = self.item_info[item_group_no]

            for w in range(self.max_width, weight - 1, -1):
                for k in range(self.max_pieces, 0, -1):
                    if dp[w - weight][k - 1][0] + value > dp[w][k][0]:
                        dp[w][k] = (dp[w - weight][k - 1][0] + value, dp[w - weight][k - 1][1] + [item_group_no])
                        
        best_pattern_items = []
        max_value = 1.0
        for w in range(self.min_width, self.max_width + 1):
            for k in range(1, self.max_pieces + 1):
                if dp[w][k][0] > max_value:
                    max_value = dp[w][k][0]
                    best_pattern_items = dp[w][k][1]

        if not best_pattern_items:
            return None, 0

        new_pattern = {}
        for item in best_pattern_items:
            new_pattern[item] = new_pattern.get(item, 0) + 1
        return new_pattern, max_value

    def _solve_subproblem_new(self, duals):
        dp = [[(0, []) for _ in range(self.max_width + 1)] for _ in range(self.max_pieces + 1)]

        for k in range(1, self.max_pieces + 1):
            for w in range(1, self.max_width + 1):
                for item_group_no in self.items:
                    weight = self.item_info[item_group_no]
                    
                    original_group = item_group_no
                    num_pieces = 1
                    if item_group_no in self.composite_item_map:
                        info = self.composite_item_map[item_group_no]
                        original_group = info['original_group']
                        num_pieces = info['num_pieces']
                    
                    sheets_per_len = self.sheets_per_length_map.get(original_group, 0)
                    dual_value = duals.get(original_group, 0)
                    value = num_pieces * dual_value * sheets_per_len
                    
                    if weight <= w:
                        prev_value, prev_items = dp[k-1][w-weight]
                        
                        if prev_value > 0 or k == 1:
                            if prev_value + value > dp[k][w][0]:
                                dp[k][w] = (prev_value + value, prev_items + [item_group_no])

        best_pattern_items = []
        max_value = 1.0
        for k in range(1, self.max_pieces + 1):
            for w in range(self.min_width, self.max_width + 1):
                if dp[k][w][0] > max_value:
                    max_value = dp[k][w][0]
                    best_pattern_items = dp[k][w][1]
        
        if not best_pattern_items:
            return None, 0
        
        new_pattern = {}
        for item in best_pattern_items:
            new_pattern[item] = new_pattern.get(item, 0) + 1
        return new_pattern, max_value

    def run_optimize(self):
        self._generate_initial_patterns()
        if not self.patterns:
            return {"error": "초기 패턴을 생성할 수 없습니다."}
        
        for iteration in range(200):
            master_solution = self._solve_master_problem()
            if not master_solution or 'duals' not in master_solution:
                break

            new_pattern_1, val1 = self._solve_subproblem_old(master_solution['duals'])
            new_pattern_2, val2 = self._solve_subproblem_new(master_solution['duals'])
            
            added_pattern = False
            if new_pattern_1 and val1 > 1.0 and new_pattern_1 not in self.patterns:
                self.patterns.append(new_pattern_1)
                added_pattern = True

            if new_pattern_2 and val2 > 1.0 and new_pattern_2 not in self.patterns:
                self.patterns.append(new_pattern_2)
                added_pattern = True

            if not added_pattern:
                break

        self.patterns = [p for p in self.patterns if sum(self.item_info[item] * count for item, count in p.items()) >= self.min_width]

        final_solution = self._solve_master_problem(is_final_mip=True)
        
        if not final_solution:
            return {"error": f"최종 해를 찾을 수 없습니다. {self.min_width}mm 이상의 폭으로 조합할 수 없는 주문이 포함되었을 수 있습니다."}
        
        return self._format_results(final_solution)

    def _format_results(self, final_solution):
        result_patterns = []
        pattern_details_for_db = []
        
        # 최종 생산량 계산 (롤 수 기준)
        final_production_sheets = {item: 0 for item in self.demands}
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                roll_count = int(round(count))
                pattern_dict = self.patterns[j]
                for producible_group, num_in_pattern in pattern_dict.items():
                    original_group = producible_group
                    num_pieces = 1
                    if producible_group in self.composite_item_map:
                        info = self.composite_item_map[producible_group]
                        original_group = info['original_group']
                        num_pieces = info['num_pieces']
                    
                    if original_group in final_production_sheets:
                        sheets_per_len = self.sheets_per_length_map.get(original_group, 0)
                        final_production_sheets[original_group] += roll_count * num_in_pattern * num_pieces * sheets_per_len

        # 패턴 결과 생성
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                pattern_dict = self.patterns[j]
                db_widths, db_group_nos = [], []
                pattern_item_strs = []
                
                sorted_pattern_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)

                for group_no, num in sorted_pattern_items:
                    width = self.item_info[group_no]
                    db_widths.extend([width] * num)
                    db_group_nos.extend([group_no] * num)

                    if group_no in self.composite_item_map:
                        info = self.composite_item_map[group_no]
                        item_str = f"{width}({info['original_dim']}*{info['num_pieces']})"
                    else:
                        original_dim = width - self.sheet_trim
                        item_str = f"{width}({int(original_dim)}*1)"
                    pattern_item_strs.extend([item_str] * num)

                pattern_str = '*'.join(pattern_item_strs)
                total_width = sum(db_widths)
                loss = self.original_max_width - total_width

                result_patterns.append({
                    'Pattern': pattern_str,
                    'Roll_Production_Length': total_width,
                    'Count': int(round(count)),
                    'Loss_per_Roll': loss
                })
                pattern_details_for_db.append({
                    'Count': int(round(count)),
                    'widths': (db_widths + [0] * 8)[:8],
                    'group_nos': (db_group_nos + [''] * 8)[:8]
                })
        
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['Pattern', 'Roll_Production_Length', 'Count', 'Loss_per_Roll']]


        # --- 주문 이행 요약 로직 수정 ---
        group_info = self.original_orders.groupby('group_order_no').first()
        df_prod_sheets = pd.DataFrame.from_dict(final_production_sheets, orient='index', columns=['Total_Produced_Sheets'])
        df_summary = group_info.join(df_prod_sheets, how='left').fillna(0)

        df_summary['Total_Produced_Area_m2'] = df_summary['Total_Produced_Sheets'] * (df_summary['가로'] / 1000) * (df_summary['세로'] / 1000)
        df_summary['Total_Produced_Tons'] = (df_summary['Total_Produced_Area_m2'] * self.b_wgt) / 1_000_000
        df_summary['Over_Production_Tons'] = df_summary['Total_Produced_Tons'] - df_summary['주문톤']
        
        fulfillment_summary = df_summary.reset_index()[
            ['group_order_no', '가로', '세로', '등급', '주문톤', 'Total_Produced_Tons', 'Over_Production_Tons']
        ].rename(columns={
            '주문톤': 'Total_Ordered_per_Group',
            'Total_Produced_Tons': 'Total_Produced_per_Group',
            'Over_Production_Tons': 'Over_Production'
        })
        fulfillment_summary['Total_Produced_per_Group'] = fulfillment_summary['Total_Produced_per_Group'].round(3)
        fulfillment_summary['Over_Production'] = fulfillment_summary['Over_Production'].round(3)

        # --- 단독생산시 롤수 및 최적해 사용롤수 계산 로직 수정 ---
        rolls_in_solution_dict = {group_no: 0 for group_no in self.demands}
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                roll_count = int(round(count))
                pattern_dict = self.patterns[j]
                total_pattern_width = sum(self.item_info[item] * num for item, num in pattern_dict.items())
                if total_pattern_width == 0: continue

                for producible_group, num_in_pattern in pattern_dict.items():
                    original_group = producible_group
                    if producible_group in self.composite_item_map:
                        original_group = self.composite_item_map[producible_group]['original_group']
                    if original_group in self.demands:
                        item_width = self.item_info[producible_group]
                        width_fraction = (item_width * num_in_pattern) / total_pattern_width
                        rolls_in_solution_dict[original_group] += roll_count * width_fraction

        rolls_if_single_dict = {}
        for _, row in self.original_orders.iterrows():
            group_no = row['group_order_no']
            demand_in_sheets = row['주문수량']
            if demand_in_sheets == 0: continue

            # 단일 품목(1-piece) 기준
            producible_width = self.item_info.get(group_no)
            sheets_per_len = self.sheets_per_length_map.get(group_no, 0)
            if not producible_width or not sheets_per_len: continue

            num_across = self.original_max_width // producible_width
            if num_across == 0: continue

            sheets_per_roll = num_across * sheets_per_len
            if sheets_per_roll > 0:
                rolls_needed = math.ceil(demand_in_sheets / sheets_per_roll)
                rolls_if_single_dict[group_no] = rolls_needed

        fulfillment_summary['단독생산시_롤수'] = fulfillment_summary['group_order_no'].map(rolls_if_single_dict).fillna(0).astype(int)
        fulfillment_summary['최적해_사용롤수'] = fulfillment_summary['group_order_no'].map(rolls_in_solution_dict).fillna(0).round(2)

        cols = fulfillment_summary.columns.tolist()
        cols.insert(cols.index('Over_Production'), cols.pop(cols.index('단독생산시_롤수')))
        cols.insert(cols.index('Over_Production'), cols.pop(cols.index('최적해_사용롤수')))
        fulfillment_summary = fulfillment_summary[cols]

        print("\n[주문 이행 요약]")
        print(fulfillment_summary.to_string())

        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "fulfillment_summary": fulfillment_summary
        }

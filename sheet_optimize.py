import pandas as pd
from ortools.linear_solver import pywraplp
import time

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
        
        # '가로' 컬럼을 그룹핑에 포함시켜 데이터 유실을 방지합니다.
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
        # First Fit Decreasing 휴리스틱을 사용한 초기 패턴 생성
        # 1. 품목을 폭 기준으로 내림차순 정렬
        sorted_items = sorted(self.items, key=lambda item: self.item_info[item], reverse=True)
        
        # 2. 단일 품목으로만 구성된 기본 패턴 추가 (기존 방식)
        for item in self.items:
            self.patterns.append({item: 1})

        # 3. 휴리스틱을 사용하여 폭이 넓은 조합 패턴 몇 개를 추가
        # 최대 10개의 추가 패턴 생성 시도
        for _ in range(30):
            new_pattern = {}
            current_width = 0
            current_pieces = 0
            # 정렬된 품목 리스트를 순회하며 패턴에 추가
            for item in sorted_items:
                item_width = self.item_info[item]
                # 현재 패턴에 품목을 추가할 수 있는지 확인
                if current_width + item_width <= self.max_width and current_pieces < self.max_pieces:
                    new_pattern[item] = new_pattern.get(item, 0) + 1
                    current_width += item_width
                    current_pieces += 1
            
            # 유효하고 새로운 패턴인 경우에만 추가
            if new_pattern and new_pattern not in self.patterns:
                self.patterns.append(new_pattern)
                # 다음 휴리스틱 실행을 위해 사용된 품목은 리스트에서 뒤로 보냄 (다양성 확보)
                used_item = list(new_pattern.keys())[0]
                sorted_items.remove(used_item)
                sorted_items.append(used_item)

    def _solve_master_problem(self, is_final_mip=False):
        solver = pywraplp.Solver.CreateSolver('GLOP' if not is_final_mip else 'SCIP')
        if is_final_mip:
            solver.SetTimeLimit(60000)
        x = {j: (solver.IntVar if is_final_mip else solver.NumVar)(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}
        
        production_exprs = {}
        constraints = {}

        for original_item_group, demand in self.demands.items():
            # 주문단위가 '장수'이므로, 생산량도 '장수'로 계산
            total_production_for_item = solver.Sum(
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
            
            production_exprs[original_item_group] = total_production_for_item
            constraints[original_item_group] = solver.Add(total_production_for_item >= demand, f'demand_{original_item_group}')

        # 3-piece, 4-piece 아이템에 대한 페널티 계산
        pattern_penalties = []
        for p in self.patterns:
            penalty = 0
            for item, count_in_pattern in p.items():
                num_pieces = self.composite_item_map.get(item, {'num_pieces': 1})['num_pieces']
                if num_pieces == 3:
                    penalty += 0.5 * count_in_pattern  # 3-piece 페널티
                elif num_pieces == 4:
                    penalty += 1.0 * count_in_pattern  # 4-piece 페널티 (더 높게)
            pattern_penalties.append(penalty)

        total_rolls = solver.Sum(x[j] for j in range(len(self.patterns)))
        total_over_production = solver.Sum(production_exprs[item] - demand for item, demand in self.demands.items())
        total_piece_penalty = solver.Sum(x[j] * pattern_penalties[j] for j in range(len(self.patterns)))

        over_production_penalty_weight = 0.001
        piece_penalty_weight = 0.2  # 페널티 가중치 (조정 가능)

        solver.Minimize(total_rolls +
                        piece_penalty_weight * total_piece_penalty +
                        over_production_penalty_weight * total_over_production)
        
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
        
        for item_group_no in self.items: # item_group_no is for a producible item
            
            value = 0
            # duals의 단위가 '롤/장' 이므로, sheet_roll_length를 곱하지 않음
            # dual 값의 의미: 특정 품목의 수요(장)를 1장 줄였을 때, 총 롤 개수가 얼마나 줄어드는가?
            # value의 의미: 이 품목을 새 패턴에 추가했을 때, 총 롤 개수를 얼마나 절약할 수 있는가?
            
            if item_group_no in self.composite_item_map:
                info = self.composite_item_map[item_group_no]
                original_group = info['original_group']
                num_pieces = info['num_pieces']
                value = num_pieces * duals.get(original_group, 0)

            elif item_group_no in duals:
                value = duals.get(item_group_no, 0)
            
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
            return None
        new_pattern = {}
        for item in best_pattern_items:
            new_pattern[item] = new_pattern.get(item, 0) + 1
        return new_pattern

    def run_optimize(self):
        self._generate_initial_patterns()
        if not self.patterns:
            return {"error": "초기 패턴을 생성할 수 없습니다."}
        
        for iteration in range(100):
            master_solution = self._solve_master_problem()
            if not master_solution or 'duals' not in master_solution:
                break
            new_pattern = self._solve_subproblem(master_solution['duals'])
            if new_pattern is None:
                break
            if new_pattern not in self.patterns:
                self.patterns.append(new_pattern)

        # min_width 제약조건을 강제로 필터링
        self.patterns = [p for p in self.patterns if sum(self.item_info[item] * count for item, count in p.items()) >= self.min_width]

        final_solution = self._solve_master_problem(is_final_mip=True)
        
        if not final_solution:
            print("\n[ANALYSIS] 최종 해를 찾지 못했습니다. 아래는 최종적으로 사용하려던 패턴의 목록입니다.")
            for i, p in enumerate(self.patterns):
                p_str_parts = []
                for group_no, count_in_pattern in p.items():
                    producible_width = self.item_info[group_no]
                    
                    repr_str = str(producible_width)
                    if group_no in self.composite_item_map:
                        info = self.composite_item_map[group_no]
                        repr_str = f"{info['original_dim']}*{info['num_pieces']}"
                    
                    final_item_str = f"{repr_str}({producible_width})"
                    if count_in_pattern > 1:
                        final_item_str += f" (x{count_in_pattern}개)"
                    p_str_parts.append(final_item_str)
                
                p_str = ' + '.join(p_str_parts)
                p_width = sum(self.item_info[k] * v for k, v in p.items())
                print(f"  - 패턴 {i+1}: {p_str} (총폭: {p_width})")
            
            print("\n[주문 이행 요약 (실패)]")
            df_demand = pd.DataFrame.from_dict(self.demands, orient='index', columns=['Total_Ordered_per_Group'])
            df_demand.index.name = 'group_order_no'
            df_demand['Total_Produced_per_Group'] = 0
            df_demand['Over_Production'] = df_demand['Total_Produced_per_Group'] - df_demand['Total_Ordered_per_Group']
            
            group_info_cols = self.original_orders[['group_order_no', '가로', '세로', '등급']].drop_duplicates()
            fulfillment_summary = pd.merge(group_info_cols, df_demand.reset_index(), on='group_order_no')
            print(fulfillment_summary.to_string())

            return {"error": f"최종 해를 찾을 수 없습니다. {self.min_width}mm 이상의 폭으로 조합할 수 없는 주문이 포함되었을 수 있습니다."}
        
        return self._format_results(final_solution)

    def _format_results(self, final_solution):
        """
        결과 포맷팅을 쉬트지에 맞게 재정의합니다.
        """
        result_patterns = []
        pattern_details_for_db = []
        
        raw_production_counts = {}
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                pattern_dict = self.patterns[j]
                for group_no, num in pattern_dict.items():
                    raw_production_counts[group_no] = raw_production_counts.get(group_no, 0) + int(round(count)) * num

        final_production_counts = {item: 0 for item in self.demands}
        for producible_group, prod_count in raw_production_counts.items():
            if producible_group in self.composite_item_map:
                info = self.composite_item_map[producible_group]
                original_group = info['original_group']
                num_pieces = info['num_pieces']
                if original_group in final_production_counts:
                    final_production_counts[original_group] += prod_count * num_pieces
            elif producible_group in final_production_counts:
                final_production_counts[producible_group] += prod_count

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
                loss = self.original_max_width - sum(db_widths)

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

        # --- Start of new summary logic ---

        # Get original order info (tons, dimensions)
        group_info = self.original_orders.groupby('group_order_no').first()

        # final_production_counts는 각 품목별로 생산된 총 '장 수'를 담고 있음
        df_prod_sheets = pd.DataFrame.from_dict(final_production_counts, orient='index', columns=['Total_Produced_Sheets'])
        df_prod_sheets.index.name = 'group_order_no'

        # Join with group_info to get dimensions and original tonnage
        df_summary = group_info.join(df_prod_sheets, how='left').fillna(0)

        # 생산된 총 톤(ton)을 계산
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

        # --- Start of new roll calculation logic ---
        rolls_in_solution_dict = {group_no: 0 for group_no in self.demands}
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                roll_count = int(round(count))
                pattern_dict = self.patterns[j]
                
                total_pattern_width = sum(self.item_info[item] * num for item, num in pattern_dict.items())
                if total_pattern_width == 0:
                    continue

                for producible_group, num_in_pattern in pattern_dict.items():
                    original_group = producible_group
                    if producible_group in self.composite_item_map:
                        original_group = self.composite_item_map[producible_group]['original_group']

                    if original_group in self.demands:
                        item_width = self.item_info[producible_group]
                        width_fraction = (item_width * num_in_pattern) / total_pattern_width
                        rolls_in_solution_dict[original_group] += roll_count * width_fraction

        rolls_if_single_dict = {}
        for group_no in fulfillment_summary['group_order_no']:
            order_info = self.original_orders[self.original_orders['group_order_no'] == group_no]
            if order_info.empty:
                rolls_if_single_dict[group_no] = 0
                continue
            
            demand_in_sheets = order_info['주문수량'].iloc[0]
            if demand_in_sheets == 0:
                rolls_if_single_dict[group_no] = 0
                continue

            if group_no in self.item_info:
                producible_width = self.item_info[group_no]
                is_single_item = group_no not in self.composite_item_map
                
                if is_single_item and producible_width > 0:
                    num_per_roll = self.original_max_width // producible_width
                    if num_per_roll > 0:
                        rolls_needed = -(-demand_in_sheets // num_per_roll)
                        rolls_if_single_dict[group_no] = rolls_needed
                    else:
                        rolls_if_single_dict[group_no] = float('inf')
                else:
                    rolls_if_single_dict[group_no] = float('inf')
            else:
                rolls_if_single_dict[group_no] = float('inf')

        rolls_if_single_series = fulfillment_summary['group_order_no'].map(rolls_if_single_dict)
        rolls_if_single_series.replace(float('inf'), -1, inplace=True)
        fulfillment_summary['단독생산시_롤수'] = rolls_if_single_series.fillna(0).astype(int)

        fulfillment_summary['최적해_사용롤수'] = fulfillment_summary['group_order_no'].map(rolls_in_solution_dict).fillna(0).round(2)

        cols = fulfillment_summary.columns.tolist()
        cols.insert(cols.index('Over_Production'), cols.pop(cols.index('단독생산시_롤수')))
        cols.insert(cols.index('Over_Production'), cols.pop(cols.index('최적해_사용롤수')))
        fulfillment_summary = fulfillment_summary[cols]

        print("\n[주문 이행 요약]")
        print(fulfillment_summary.to_string())

        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False),
            "pattern_details_for_db": pattern_details_for_db,
            "fulfillment_summary": fulfillment_summary
        }

import pandas as pd
from ortools.linear_solver import pywraplp
import time
import math
"""
쉬트지 최적화를 롤수를 기준으로 해서 최적해를 구하는 코드입니다.
오더량을 평량*가로*세로로 구한 1장당 무게로 나누어서 쉬트장수를 구하고
표준길이로 생산할수 있는 쉬트장수를 계산하여 1폭기준으로 몇롤이 필요한지를 계산함.
"""
class SheetOptimize:
    def __init__(
            self, df_spec_pre, max_width, min_width, max_pieces, b_wgt, sheet_roll_length, sheet_trim, min_sc_width, max_sc_width
    ):
        df_spec_pre['지폭'] = df_spec_pre['가로']

        self.b_wgt = b_wgt
        self.sheet_roll_length = sheet_roll_length
        self.sheet_trim = sheet_trim
        self.original_max_width = max_width
        self.df_orders = df_spec_pre.copy()
        
        # --- 1. 수요 계산 (롤 단위로 통일) ---
        self.demands_in_rolls = self._calculate_demand_rolls(self.df_orders)
        self.order_widths = list(self.demands_in_rolls.keys()) 

        # 주문 정보 요약
        width_summary = {}
        tons_per_width = self.df_orders.groupby('지폭')['주문톤'].sum()
        for width, required_rolls in self.demands_in_rolls.items():
            order_tons = tons_per_width.get(width, 0)
            width_summary[width] = {
                'order_tons': order_tons,
            }
        self.width_summary = width_summary 

        # --- 2. 아이템(패턴 구성요소) 정의 ---
        self.items = []
        self.item_info = {} # item_name -> width
        self.item_composition = {} # composite_item_name -> {original_width: count}

        for width in self.order_widths:
            # 단일폭 아이템: 1폭도 너비 제약조건(850~2600)을 만족해야 함
            base_width = width + self.sheet_trim
            composite_width = base_width
            if min_sc_width <= base_width <= max_sc_width:
                item_name = str(base_width)
                if item_name not in self.items:
                    self.items.append(item_name)
                    self.item_info[item_name] = composite_width
                    self.item_composition[item_name] = {width: 1}

            # 복합폭 아이템 (2, 3, 4폭)
            for i in range(2, 5):
                base_width = width * i + self.sheet_trim
                # 복합지폭의 너비 제약조건(850~2600) 추가
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                item_name = f"{width}x{i}"
                composite_width = base_width
                if composite_width <= self.original_max_width:
                    if item_name not in self.items:
                        self.items.append(item_name)
                        self.item_info[item_name] = composite_width
                        self.item_composition[item_name] = {width: i}

        self.max_width = max_width
        self.min_width = min_width
        self.min_pieces = 2
        self.max_pieces = int(max_pieces)
        print(f"\n--- 패턴 제약조건: 최소 {self.min_pieces}폭, 최대 {self.max_pieces}폭 ---")

        self.patterns = []

    def _calculate_demand_rolls(self, df_orders):
        """
        주문톤(order_ton_cnt)을 지폭별 필요 롤 수로 변환합니다.
        """
        df_copy = df_orders.copy()
        
        required_rolls_list = []
        sheet_roll_length_mm = self.sheet_roll_length * 1000

        for _, row in df_copy.iterrows():
            width_mm = row['가로']
            length_mm = row['세로']
            order_ton = row['주문톤']

            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0 or order_ton <= 0:
                required_rolls_list.append({'지폭': width_mm, 'rolls': 0})
                continue

            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1_000_000
            if sheet_weight_g <= 0:
                required_rolls_list.append({'지폭': width_mm, 'rolls': 0})
                continue
            
            total_sheets_needed = (order_ton * 1_000_000) / sheet_weight_g

            sheets_per_roll_length = math.floor(sheet_roll_length_mm / length_mm)
            if sheets_per_roll_length <= 0:
                required_rolls_list.append({'지폭': width_mm, 'rolls': 0})
                continue

            num_across = 1
            sheets_per_roll = sheets_per_roll_length * num_across
            # rolls_needed = math.ceil(total_sheets_needed / sheets_per_roll)
            rolls_needed = round(total_sheets_needed / sheets_per_roll, 0)
            required_rolls_list.append({'지폭': width_mm, 'rolls': rolls_needed})

        df_rolls = pd.DataFrame(required_rolls_list)
        demand_rolls = df_rolls.groupby('지폭')['rolls'].sum().astype(int).to_dict()

        print("\n--- 지폭별 필요 롤 수 ---")
        for width, rolls in demand_rolls.items():
            print(f"  지폭 {width}mm: {rolls} 롤")
        print("--------------------------\n")        

        return demand_rolls

    def _generate_initial_patterns(self):
        """
        초기 패턴 생성 - First-Fit-Decreasing 휴리스틱 사용
        """
        print("\n--- 유효한 초기 패턴을 생성합니다 ---")
        
        sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)

        for item in sorted_items:
            item_width = self.item_info[item]
            
            current_pattern = {item: 1}
            current_width = item_width
            current_pieces = 1

            while current_pieces < self.max_pieces:
                remaining_width = self.max_width - current_width
                
                candidates = [i for i in sorted_items if self.item_info[i] <= remaining_width]
                if not candidates:
                    break 

                best_fit_item_name = candidates[0]
                
                current_pattern[best_fit_item_name] = current_pattern.get(best_fit_item_name, 0) + 1
                current_width += self.item_info[best_fit_item_name]
                current_pieces += 1

            if self.min_width <= current_width and self.min_pieces <= current_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in [frozenset(p.items()) for p in self.patterns]:
                    self.patterns.append(current_pattern)

        print(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")

    def _solve_master_problem_ilp(self, is_final_mip=False):
        """
        개선된 마스터 문제 해결 - 정수계획법 사용
        수요를 정확히 충족하면서 과생산을 최소화
        """
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        if is_final_mip:
            solver.SetTimeLimit(60000)

        # 패턴 사용 횟수 변수
        x = {}
        for j in range(len(self.patterns)):
            if is_final_mip:
                x[j] = solver.IntVar(0, solver.infinity(), f'P_{j}')
            else:
                x[j] = solver.NumVar(0, solver.infinity(), f'P_{j}')
        
        # 과생산 변수 (수요 초과분)
        over_prod_vars = {}
        for width in self.demands_in_rolls:
            over_prod_vars[width] = solver.NumVar(0, solver.infinity(), f'Over_{width}')

        # 제약조건: 생산량 >= 수요량
        constraints = {}
        for width, required_rolls in self.demands_in_rolls.items():
            production_for_width = solver.Sum(
                x[j] * sum(self.item_composition[item_name].get(width, 0) * count
                           for item_name, count in self.patterns[j].items())
                for j in range(len(self.patterns))
            )
            
            # 생산량 = 수요량 + 과생산량
            constraints[width] = solver.Add(
                production_for_width == required_rolls + over_prod_vars[width], 
                f'demand_width_{width}'
            )

        # 목적함수: 총 롤 수 + 과생산 페널티 최소화
        total_rolls = solver.Sum(x[j] for j in range(len(self.patterns)))
        
        # 과생산에 대한 강한 페널티
        OVER_PROD_PENALTY = 10.0
        total_over_prod_penalty = solver.Sum(OVER_PROD_PENALTY * over_prod_vars[width] for width in self.demands_in_rolls)
        
        # 패턴 복잡도 페널티 (많은 아이템을 포함한 패턴 회피)
        PATTERN_COMPLEXITY_PENALTY = 0.01
        total_complexity_penalty = solver.Sum(
            PATTERN_COMPLEXITY_PENALTY * len(self.patterns[j]) * x[j] 
            for j in range(len(self.patterns))
        )

        solver.Minimize(total_rolls + total_over_prod_penalty + total_complexity_penalty)
        
        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            solution = {
                'objective': solver.Objective().Value(),
                'pattern_counts': {j: var.solution_value() for j, var in x.items()},
                'over_production': {width: var.solution_value() for width, var in over_prod_vars.items()}
            }
            
            if not is_final_mip:
                solution['duals'] = {width: constraints[width].dual_value() for width in self.demands_in_rolls}
            
            return solution
        return None

    def _solve_subproblem_dp(self, duals):
        """
        동적 프로그래밍을 사용한 서브문제 해결
        더 효율적인 패턴 생성
        """
        # dp[k][w] = (최대 가치, 아이템 리스트)
        dp = [[(0, []) for _ in range(self.max_width + 1)] for _ in range(self.max_pieces + 1)]

        # 아이템별 가치 계산
        item_details = []
        for item_name in self.items:
            item_width = self.item_info[item_name]
            item_value = sum(count * duals.get(width, 0) 
                             for width, count in self.item_composition[item_name].items())
            if item_value > 0:  # 양의 가치를 가진 아이템만 고려
                item_details.append((item_name, item_width, item_value))

        # DP 계산
        for k in range(1, self.max_pieces + 1):
            for w in range(1, self.max_width + 1):
                # 이전 상태 복사
                dp[k][w] = dp[k-1][w]
                
                # 각 아이템 추가 시도
                for name, i_width, i_value in item_details:
                    if w >= i_width and k >= 1:
                        prev_value, prev_items = dp[k-1][w - i_width]
                        new_value = prev_value + i_value
                        if new_value > dp[k][w][0]:
                            dp[k][w] = (new_value, prev_items + [name])

        # 최적 패턴 찾기 (reduced cost > 1인 패턴)
        best_pattern_items = []
        max_reduced_cost = 1.0  # 임계값
        
        for k in range(self.min_pieces, self.max_pieces + 1):
            for w in range(self.min_width, self.max_width + 1):
                if dp[k][w][0] > max_reduced_cost:
                    max_reduced_cost = dp[k][w][0]
                    best_pattern_items = dp[k][w][1]

        if not best_pattern_items:
            return None, 0
        
        # 패턴 딕셔너리로 변환
        new_pattern = {}
        for item in best_pattern_items:
            new_pattern[item] = new_pattern.get(item, 0) + 1
        
        return new_pattern, max_reduced_cost

    def _generate_all_patterns(self):
        """
        작은 문제를 위한 모든 가능한 패턴 생성
        """
        all_patterns = []
        seen_patterns = set()
        item_list = list(self.items)

        def find_combinations_recursive(start_index, current_pattern, current_width, current_pieces):
            # 유효한 패턴 확인
            if self.min_width <= current_width <= self.max_width and self.min_pieces <= current_pieces <= self.max_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in seen_patterns:
                    all_patterns.append(current_pattern.copy())
                    seen_patterns.add(pattern_key)

            # 종료 조건
            if current_pieces >= self.max_pieces or start_index >= len(item_list):
                return

            # 현재 아이템 스킵
            find_combinations_recursive(start_index + 1, current_pattern, current_width, current_pieces)

            # 현재 아이템 추가
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
        """
        최적화 실행 메인 함수
        """
        # 작은 문제는 완전 탐색, 큰 문제는 열 생성
        if len(self.order_widths) <= 8:
            print("\n--- 모든 가능한 패턴을 생성합니다 (Small-scale) ---")
            self._generate_all_patterns()
        else:
            print("\n--- 열 생성 기법을 시작합니다 (Large-scale) ---")
            self._generate_initial_patterns()
            
            if not self.patterns:
                return {"error": "초기 유효 패턴을 생성할 수 없습니다. 제약조건이 너무 엄격할 수 있습니다."}

            # 열 생성 반복
            max_iterations = 400
            no_improvement_count = 0
            
            for iteration in range(max_iterations):
                # 마스터 문제 해결
                master_solution = self._solve_master_problem_ilp()
                if not master_solution or 'duals' not in master_solution:
                    break

                # 서브문제 해결 (새 패턴 생성)
                new_pattern, reduced_cost = self._solve_subproblem_dp(master_solution['duals'])
                
                if new_pattern and reduced_cost > 1.0:
                    # 중복 체크
                    pattern_key = frozenset(new_pattern.items())
                    if pattern_key not in [frozenset(p.items()) for p in self.patterns]:
                        self.patterns.append(new_pattern)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
                
                # 수렴 확인
                if no_improvement_count >= 10:
                    print(f"--- 반복 {iteration}에서 수렴 ---")
                    break

        if not self.patterns:
            return {"error": "유효한 패턴을 생성할 수 없습니다."}

        print(f"\n--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        
        # 최종 정수 해 구하기
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
        return self._format_results(final_solution)

    def _format_results(self, final_solution):
        """
        결과 포매팅
        """
        result_patterns = []
        pattern_details_for_db = []
        
        # 최종 생산량 계산
        final_production_rolls = {width: 0 for width in self.order_widths}

        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                roll_count = int(round(count))
                pattern_dict = self.patterns[j]
                
                # 패턴의 실제 생산량 계산
                for item_name, num_in_pattern in pattern_dict.items():
                    composition = self.item_composition[item_name]
                    for width, num_pieces in composition.items():
                        if width in final_production_rolls:
                            final_production_rolls[width] += roll_count * num_in_pattern * num_pieces

        # 패턴 결과 생성
        for j, count in final_solution['pattern_counts'].items():
            if count > 0.99:
                pattern_dict = self.patterns[j]
                db_widths, db_group_nos = [], []
                pattern_item_strs = []
                
                sorted_pattern_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)

                total_width = 0
                for item_name, num in sorted_pattern_items:
                    width = self.item_info[item_name]
                    total_width += width * num
                    db_widths.extend([width] * num)
                    db_group_nos.extend([item_name] * num)
                    
                    # 패턴 문자열 생성
                    if 'x' in item_name:
                        parts = item_name.split('x')
                        base_width = int(parts[0])
                        multiplier = int(parts[1])
                        total_with_trim = self.item_info[item_name]
                        formatted_name = f"{total_with_trim}({base_width}*{multiplier})"
                    else:
                        base_width = list(self.item_composition[item_name].keys())[0]
                        total_with_trim = self.item_info[item_name]
                        formatted_name = f"{total_with_trim}({base_width}*1)"
                    
                    pattern_item_strs.extend([formatted_name] * num)

                pattern_str = ' + '.join(pattern_item_strs)
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

        # 주문 이행 요약
        summary_data = []
        for width, required_rolls in self.demands_in_rolls.items():
            produced_rolls = final_production_rolls.get(width, 0)
            over_prod_rolls = produced_rolls - required_rolls
            
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
                '생산량(톤)': produced_tons,
                '과부족(톤)': over_prod_tons,
                '필요롤수': required_rolls,
                '생산롤수': produced_rolls,
                '과부족(롤)': over_prod_rolls,
            })
        
        fulfillment_summary = pd.DataFrame(summary_data)
        if not fulfillment_summary.empty:
            fulfillment_summary['생산량(톤)'] = fulfillment_summary['생산량(톤)'].round(2)
            fulfillment_summary['과부족(톤)'] = fulfillment_summary['과부족(톤)'].round(2)
            fulfillment_summary = fulfillment_summary[[
                '지폭', 
                '주문량(톤)', '생산량(톤)', '과부족(톤)',
                '필요롤수', '생산롤수', '과부족(롤)'
            ]]

        print("\n[주문 이행 요약]")
        print(fulfillment_summary.to_string())

        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "fulfillment_summary": fulfillment_summary
        }

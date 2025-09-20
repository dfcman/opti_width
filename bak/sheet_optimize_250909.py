import pandas as pd
from ortools.linear_solver import pywraplp
import math

# --- 최적화 설정 상수 ---
# 페널티 값
OVER_PROD_PENALTY = 20.0    # 과생산에 대한 페널티
UNDER_PROD_PENALTY = 500.0  # 부족생산에 대한 페널티
PATTERN_COMPLEXITY_PENALTY = 0.01  # 패턴 복잡성에 대한 페널티

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 2      # 패턴에 포함될 수 있는 최소 폭(piece)의 수
SMALL_PROBLEM_THRESHOLD = 8     # 전체 탐색을 수행할 최대 주문 지폭 종류 수
SOLVER_TIME_LIMIT_MS = 60000    # 최종 MIP 솔버의 최대 실행 시간 (밀리초)
CG_MAX_ITERATIONS = 400         # 열 생성(Column Generation) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 10    # 개선 없는 경우, 열 생성 조기 종료 조건
CG_SUBPROBLEM_TOP_N = 5         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴

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
        demand_rolls = df_copy.groupby('지폭')['rolls'].sum().astype(int).to_dict()

        print("\n--- 지폭별 필요 롤 수 ---")
        for width, rolls in demand_rolls.items():
            print(f"  지폭 {width}mm: {rolls} 롤")
        print("--------------------------\n")
        
        return demand_rolls

    def _generate_initial_patterns(self):
        """초기 패턴 생성을 위해 First-Fit-Decreasing 휴리스틱을 사용합니다."""
        print("\n--- 유효한 초기 패턴을 생성합니다 ---")
        
        sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        
        # frozenset으로 패턴 중복 체크를 효율적으로 관리
        seen_patterns = set()

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

        print(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")

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
            allowed_under_prod = max(1, math.ceil(required_rolls * 0.05))
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
        """서브문제(Sub-problem)를 동적 프로그래밍으로 해결하여 새로운 패턴을 찾습니다."""
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

        # 중복 제거 및 상위 N개 선택
        seen_patterns = set()
        unique_candidates = []
        for cand in sorted(candidate_patterns, key=lambda x: x['cost'], reverse=True):
            pattern_key = frozenset(cand['pattern'].items())
            if pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                unique_candidates.append(cand['pattern'])
        
        return unique_candidates[:CG_SUBPROBLEM_TOP_N]

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
            self._generate_initial_patterns()
            
            initial_pattern_count = len(self.patterns)
            self.patterns = [p for p in self.patterns if sum(self.item_info[i] * c for i, c in p.items()) >= self.min_width]
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
        print(fulfillment_summary.to_string())

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

import pandas as pd
from ortools.linear_solver import pywraplp
import time

class Optimize:
    
    def __init__(self, df_spec_pre, max_width=1000, min_width=0, max_pieces=8):
        """
        최적화 클래스 초기화 (칼럼 생성법 적용)
        """
        self.df_spec_pre = df_spec_pre
        self.demands = df_spec_pre.groupby('지폭')['주문수량'].sum().to_dict()
        self.widths = sorted(list(self.demands.keys()))
        self.max_width = max_width
        self.min_width = min_width
        self.max_pieces = max_pieces
        self.patterns = []

    def _generate_initial_patterns(self):
        """초기 패턴 생성: 모든 수요를 충족하는 유효한 초기 패턴 조합을 보장 (Greedy FFD 변형)"""
        patterns = []
        demands_copy = self.demands.copy()
        
        # 수요가 모두 0이 될 때까지 반복
        while any(v > 0 for v in demands_copy.values()):
            new_pattern_dict = {}
            current_width = 0
            current_pieces = 0

            # 큰 지폭부터 우선적으로 채워넣기
            for width in sorted(demands_copy.keys(), reverse=True):
                while demands_copy[width] > 0 and current_width + width <= self.max_width and current_pieces < self.max_pieces:
                    new_pattern_dict[width] = new_pattern_dict.get(width, 0) + 1
                    demands_copy[width] -= 1
                    current_width += width
                    current_pieces += 1
            
            # 유효한 패턴이 만들어졌는지 확인
            if not new_pattern_dict:
                # 남은 롤 중 가장 큰 롤 하나로 강제 생성 (매우 큰 롤 처리)
                for width in sorted(demands_copy.keys(), reverse=True):
                    if demands_copy[width] > 0:
                        new_pattern_dict = {width: 1}
                        demands_copy[width] -= 1
                        break

            if new_pattern_dict and new_pattern_dict not in patterns:
                patterns.append(new_pattern_dict)
        
        return patterns if patterns else [{self.widths[0]: 1}] # 비어있을 경우에 대한 최종 방어

    def _solve_master_problem(self, is_final_mip=False):
        """메인 문제(Master Problem)를 LP 또는 MIP로 해결 (표준 절단 최적화 공식)"""
        solver = pywraplp.Solver.CreateSolver('GLOP' if not is_final_mip else 'SCIP')

        # 변수: 각 패턴의 사용 횟수
        if is_final_mip:
            x = {j: solver.IntVar(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}
        else:
            x = {j: solver.NumVar(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}

            """ 이전 제약조건 총 롤 사용량 조건.
            # 제약조건: 생산량 >= 주문량
            constraints = {}
            for width, demand in self.demands.items():
                production_expr = solver.Sum(p.get(width, 0) * x[j] for j, p in enumerate(self.patterns))
                constraints[width] = solver.Add(production_expr >= demand, f'demand_{width}')
   
            # 목표: 총 사용 롤 개수 최소화
            objective_expr = solver.Sum(x[j] for j in range(len(self.patterns)))
            solver.Minimize(objective_expr)
            """





        # 제약조건: 생산량 >= 주문량 (유지)
        # 그리고 초과생산량을 계산하기 위해 생산량 표현식을 별도로 저장합니다.
        production_exprs = {}
        constraints = {}
        for width, demand in self.demands.items():
            # 각 지폭별 총 생산량 표현식
            production_expr = solver.Sum(p.get(width, 0) * x[j] for j, p in enumerate(self.patterns))
            production_exprs[width] = production_expr
            constraints[width] = solver.Add(production_expr >= demand, f'demand_{width}')

        # --- 목표 함수 수정 ---
        # 1. 기본 목표: 총 사용 롤 개수
        total_rolls = solver.Sum(x[j] for j in range(len(self.patterns)))
        
        # 2. 패널티 대상: 총 초과생산량 계산
        # (생산량 - 주문량)의 총합
        total_over_production = solver.Sum(production_exprs[width] - demand for width, demand in self.demands.items())

        # 3. 초과생산에 대한 패널티 가중치 (이 값을 조절하여 우선순위를 정합니다)
        #    - 1.0 보다 크면: 롤 1개를 더 쓰는 것보다 초과생산 1개를 줄이는 것을 더 중요하게 생각
        #    - 1.0 보다 작으면: 초과생산을 감수하더라도 롤 사용량을 줄이는 것을 더 중요하게 생각
        penalty_weight = 1.5 

        # 4. 새로운 목표 설정
        solver.Minimize(total_rolls + penalty_weight * total_over_production)

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            solution = {
                'objective': solver.Objective().Value(),
                'pattern_counts': {j: var.solution_value() for j, var in x.items()}
            }
            if not is_final_mip:
                solution['duals'] = {width: constraints[width].dual_value() for width in self.demands}
            return solution
        return None

    def _solve_subproblem(self, duals):
        """하위 문제(Subproblem) 해결: 가장 수익성 있는 새 패턴 찾기 (Knapsack)"""
        dp = [[(0, []) for _ in range(self.max_pieces + 1)] for _ in range(self.max_width + 1)]

        # 표준 절단 최적화 문제에서, 아이템의 가치는 dual 값입니다.
        for width_item in self.widths:
            value = duals.get(width_item, 0)
            # Unbounded Knapsack에서는 너비(w) 루프가 증가해야 합니다.
            for w in range(width_item, self.max_width + 1):
                for k in range(1, self.max_pieces + 1):
                    # 현재 아이템(width_item)을 추가하여 더 나은 해를 만들 수 있는지 확인
                    if dp[w - width_item][k - 1][0] + value > dp[w][k][0]:
                        dp[w][k] = (dp[w - width_item][k - 1][0] + value, dp[w - width_item][k - 1][1] + [width_item])

        best_pattern = None
        # 새 패턴의 reduced cost가 음수(가치의 합 > 1)인지 확인
        max_value = 1.0

        for w in range(self.min_width, self.max_width + 1):
            for k in range(1, self.max_pieces + 1):
                if dp[w][k][0] > max_value:
                    max_value = dp[w][k][0]
                    new_p = {}
                    for item in dp[w][k][1]:
                        new_p[item] = new_p.get(item, 0) + 1
                    best_pattern = new_p
        
        return best_pattern

    def run_optimize(self):
        """칼럼 생성법을 사용하여 최적화 실행"""
        start_time = time.time()
        self.patterns = self._generate_initial_patterns()

        if not self.patterns:
            return {"error": "초기 패턴을 생성할 수 없습니다."}

        iteration = 0
        # 최대 반복 횟수 제한 대신 시간제한(300초)을 메인으로 사용
        while time.time() - start_time < 300:
            iteration += 1
            master_solution = self._solve_master_problem()

            if not master_solution or 'duals' not in master_solution:
                # LP에서 해를 못찾으면, 현재까지의 패턴으로 MIP를 시도
                print("LP 해를 찾지 못했거나 dual 값을 얻을 수 없습니다. 최종 최적화를 시도합니다.")
                break

            new_pattern = self._solve_subproblem(master_solution['duals'])

            if new_pattern is None:
                print(f"\nIteration {iteration}: 개선 가능한 새 패턴 없음. 최종 최적화 시작.")
                break
            
            if new_pattern in self.patterns:
                print(f"\nIteration {iteration}: 이미 존재하는 패턴 발견. 루프 종료.")
                break

            # print(f"Iteration {iteration}: 새로운 패턴 발견: {new_pattern}")
            self.patterns.append(new_pattern)
        else:
            print("\n최대 반복 횟수에 도달했습니다. 현재까지의 패턴으로 최종 최적화를 시작합니다.")

        print("\n최종 MIP 문제 해결 중...")
        final_solution = self._solve_master_problem(is_final_mip=True)

        if not final_solution:
            return {"error": "최종 MIP 문제 해결 실패."}

        return self._format_results(final_solution)

    def _format_results(self, final_solution):
        """최적화 결과를 데이터프레임 형식으로 정리"""
        # 1. 결과 저장을 위한 변수 초기화
        result_patterns = []    # 패턴별 결과를 저장할 리스트
        production_counts = {w: 0 for w in self.demands}    # 지폭별 총 생산량을 계산할 딕셔너리

        # 2. 솔버가 찾아낸 모든 패턴을 하나씩 확인
        for j, count in final_solution['pattern_counts'].items():
            # 3. 위에서 설명한 '사용하기로 결정된' 패턴만 필터링
            if count > 0.99:
                # 4. 패턴 정보 가공
                pattern_dict = self.patterns[j]
                # 사람이 읽기 좋은 문자열로 변환 (예: '1200mm x 2, 1500mm x 1')
                pattern_str = ', '.join([f'{w}mm x {c}' for w, c in sorted(pattern_dict.items())])
                # 이 패턴을 한 번 사용할 때 발생하는 손실(loss) 계산
                loss = self.max_width - sum(w * c for w, c in pattern_dict.items())
                
                # Loss가 음수인 경우(max_width 초과)는 버그이므로 확인
                if loss < 0:
                    print(f"[경고] 너비 초과 패턴 발견: {pattern_dict}, Loss: {loss}")

                # 5. 최종 'pattern_result' DataFrame에 들어갈 내용 추가
                result_patterns.append({
                    'Pattern': pattern_str,
                    'Count': int(round(count)),
                    'Loss_per_Roll': loss
                })

                # 6. 지폭별 총 생산량 업데이트
                # 이 패턴이 5번 사용되고, 패턴 안에 1200mm 지폭이 2개 있다면,
                # 1200mm 지폭의 총 생산량은 5 * 2 = 10 만큼 늘어납니다
                for width, num in pattern_dict.items():
                    production_counts[width] += int(round(count)) * num

        # 7. 가공된 패턴 목록으로 DataFrame 생성
        df_patterns = pd.DataFrame(result_patterns)

        # 8. 지폭별 주문량 vs 생산량 비교 데이터 생성 (df_stats)
        production_stats = {}
        for width, demand in self.demands.items():
            produced = production_counts.get(width, 0)
            production_stats[width] = {
                'Total_Ordered_per_Width': demand,          # 이 지폭의 총 주문량
                'Total_Produced_per_Width': produced,       # 이 지폭의 총 생산량
                'Over_production': produced - demand        # 초과 생산량
            }
        
        df_stats = pd.DataFrame.from_dict(production_stats, orient='index')
        df_stats.index.name = '지폭'

        # 9. 최종 'fulfillment_summary' DataFrame 생성
        # 기존 주문 정보(오더번호, 지폭, 주문수량)와
        # 위에서 계산한 지폭별 생산량 통계(df_stats)를 '지폭' 기준으로 합칩니다
        fulfillment_summary = pd.merge(
            self.df_spec_pre[['오더번호', '지폭', '롤길이', '주문수량']],
            df_stats.reset_index(),
            on='지폭'
        )
        fulfillment_summary.rename(columns={'주문수량': 'Ordered_per_Order'}, inplace=True)
        
        # 10. 최종 결과 반환
        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False),
            "fulfillment_summary": fulfillment_summary
        }
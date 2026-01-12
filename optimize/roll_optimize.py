import pandas as pd
from ortools.linear_solver import pywraplp
import random
import logging
import time
import gurobipy as gp
from gurobipy import GRB

"""
[파일 설명: roll_optimize.py]
롤(Roll) 제품의 생산 최적화를 위한 핵심 모듈입니다.
Column Generation(열 생성) 알고리즘을 기반으로 하여, 주문 요구사항(지폭, 수량)을 만족시키면서
폐기물(Trim Loss)과 패턴 교체 비용을 최소화하는 최적의 절단 패턴을 산출합니다.
"""

OVER_PROD_PENALTY = 100000000.0
UNDER_PROD_PENALTY = 10000.0
PATTERN_VALUE_THRESHOLD = 1.0 + 1e-6
CG_MAX_ITERATIONS = 1000 # 안전장치 (최대 반복 횟수)
CG_NO_IMPROVEMENT_LIMIT = 50  # 200 -> 50: 초기 패턴이 우수하면 빨리 넘어가도록 단축
CG_SUBPROBLEM_TOP_N = 10      # Increased from 3
SMALL_PROBLEM_THRESHOLD = 6  # Increased to force exhaustive search for narrow width ranges
PATTERN_SETUP_COST = 1000000.0 # 새로운 패턴 종류를 1개 사용할 때마다 1000mm의 손실과 동일한 페널티
TRIM_LOSS_PENALTY = 5.0      # 자투리 손실 1mm당 페널티
MIXING_PENALTY = 100.0       # 공백이 1개 섞인 경우는 페널티 비용 팬텀 생성비용과 비교



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
    
    def __init__(
        self, 
        db=None, 
        plant=None, 
        pm_no=None, 
        schedule_unit=None, 
        lot_no=None, 
        version=None, 
        paper_type=None, 
        b_wgt=None, 
        color=None, 
        p_type=None, 
        p_wgt=None, 
        p_color=None, 
        p_machine=None, 
        df_spec_pre=None, 
        min_width=0, 
        max_width=1000, 
        max_pieces=8, 
        time_limit=300000, 
        num_threads=4
    ):
        self.db = db
        self.plant = plant
        self.pm_no = pm_no
        self.schedule_unit = schedule_unit
        self.lot_no = lot_no
        self.version = version
        self.paper_type = paper_type
        self.b_wgt = b_wgt
        self.color = color
        self.p_type = paper_type
        self.p_wgt = b_wgt
        self.p_color = color
        self.p_machine = pm_no
        self.df_spec_pre = df_spec_pre
        self.min_width = min_width
        self.max_width = max_width
        self.max_pieces = max_pieces
        self.solver_time_limit_ms = time_limit  # 밀리초 단위 시간 제한
        self.num_threads = num_threads
        self.patterns = []
        self.pattern_keys = set()
        self.demands = df_spec_pre.groupby('group_order_no')['주문수량'].sum().to_dict()
        self.items = list(self.demands.keys())
        self.item_info = df_spec_pre.set_index('group_order_no')['지폭'].to_dict()
        self.length_info = df_spec_pre.set_index('group_order_no')['롤길이'].to_dict()
        
        # 지폭 -> group_order_no 역매핑 (DB 패턴 변환용)
        self.width_to_items = {}
        for item, width in self.item_info.items():
            if width not in self.width_to_items:
                self.width_to_items[width] = []
            self.width_to_items[width].append(item)
        
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

    def _generate_initial_patterns_db(self):
        """DB에서 기존 롤 패턴(rollwidth)을 가져와 초기 패턴으로 추가합니다."""
        if not self.db or not self.lot_no or not self.version:
            logging.info("--- DB 정보가 없어 기존 롤 패턴을 불러올 수 없습니다. ---")
            return

        logging.info("\n--- DB에서 기존 롤 패턴을 불러와 초기 패턴을 생성합니다. ---")
        db_patterns = self.db.get_roll_patterns_from_db(self.lot_no, self.version)

        if not db_patterns:
            logging.info("--- DB에 저장된 기존 롤 패턴이 없습니다. ---")
            return

        added_count = 0
        invalid_width_count = 0
        out_of_range_count = 0
        duplicate_count = 0
        
        invalid_width_details = []
        out_of_range_details = []
        
        for pattern_widths in db_patterns:
            # 지폭 값들을 group_order_no로 변환
            pattern_dict = {}
            is_valid = True
            
            for width in pattern_widths:
                # 해당 지폭에 매핑되는 아이템 찾기
                matching_items = self.width_to_items.get(width, [])
                if not matching_items:
                    is_valid = False
                    invalid_width_count += 1
                    invalid_width_details.append(pattern_widths)
                    break
                
                # 첫 번째 매칭되는 아이템 사용 (동일 지폭이면 어떤 것이든 상관없음)
                item = matching_items[0]
                pattern_dict[item] = pattern_dict.get(item, 0) + 1
            
            if is_valid and pattern_dict:
                # 패턴 너비 검증
                total_width = sum(self.item_info[item] * count for item, count in pattern_dict.items())
                if self.min_width <= total_width <= self.max_width:
                    if self._add_pattern(pattern_dict):
                        added_count += 1
                    else:
                        duplicate_count += 1
                else:
                    out_of_range_count += 1
                    out_of_range_details.append((pattern_widths, total_width))
        
        logging.info(f"--- DB 패턴 처리 결과: 추가={added_count}, 중복={duplicate_count}, 지폭없음={invalid_width_count}, 범위초과={out_of_range_count} ---")
        
        if invalid_width_details:
             logging.info(f"--- 지폭없음(매핑실패) 상세 ({len(invalid_width_details)}건) ---")
             for p in invalid_width_details:
                 logging.info(f"  Widths: {p} -> 매핑 가능한 지폭 없음")

        if out_of_range_details:
             logging.info(f"--- 범위초과 상세 ({len(out_of_range_details)}건) [Min:{self.min_width}, Max:{self.max_width}] ---")
            #  for p, t_width in out_of_range_details:
            #      logging.info(f"  Widths: {p}, Total: {t_width}")

    def _generate_initial_patterns(self):
        self._clear_patterns()
        if not self.items:
            return
        
        # DB에서 기존 패턴 먼저 로드 (있는 경우)
        self._generate_initial_patterns_db()

        sorted_by_demand = sorted(self.items, key=lambda item: self.demands.get(item, 0), reverse=True)
        sorted_by_demand_asc = sorted(self.items, key=lambda item: self.demands.get(item, 0))
        sorted_by_width_desc = sorted(self.items, key=lambda item: self.item_info.get(item, 0), reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda item: self.item_info.get(item, 0))

        # New Heuristics
        # 1. Width * Demand (Area proxy)
        # sorted_by_area_desc = sorted(self.items, key=lambda item: self.item_info.get(item, 0) * self.demands.get(item, 0), reverse=True)
        
        # 2. Random Shuffles (add multiple to increase diversity)
        # random.seed(41) # Ensure determinism
        random_shuffles = []
        for _ in range(5):  # Increased from 8 to 10
            items_copy = list(self.items)
            random.shuffle(items_copy)
            random_shuffles.append(items_copy)

        heuristics = [
            sorted_by_demand,  
            sorted_by_width_asc, 
            sorted_by_width_desc, 
            sorted_by_demand_asc,
            # sorted_by_area_desc,
        ] + random_shuffles

        # === 기존 First-Fit 휴리스틱 ===
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

                # min_width 충족을 위한 채우기
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

        # === 새로운 휴리스틱 1: Best-Fit (남은 공간에 가장 잘 맞는 아이템 선택) ===
        for item in self.items:
            current_pattern = {item: 1}
            current_width = self.item_info[item]
            current_pieces = 1

            while current_pieces < self.max_pieces and current_width < self.max_width:
                remaining_width = self.max_width - current_width
                
                # 남은 공간에 가장 잘 맞는 (가장 큰) 아이템 선택
                candidates = [(i, self.item_info[i]) for i in self.items if self.item_info[i] <= remaining_width]
                if not candidates:
                    break
                
                # 남은 공간을 가장 많이 채우는 아이템 선택
                best_item = max(candidates, key=lambda x: x[1])[0]
                current_pattern[best_item] = current_pattern.get(best_item, 0) + 1
                current_width += self.item_info[best_item]
                current_pieces += 1

            if current_width >= self.min_width:
                self._add_pattern(current_pattern)

        # === 새로운 휴리스틱 2: min_width 타겟팅 (min_width에 가장 가까운 조합 찾기) ===
        target_width = (self.min_width + self.max_width) // 2  # 중간값 타겟
        
        for item in self.items:
            current_pattern = {item: 1}
            current_width = self.item_info[item]
            current_pieces = 1

            while current_pieces < self.max_pieces and current_width < target_width:
                # 목표까지 남은 너비
                remaining_to_target = target_width - current_width
                remaining_to_max = self.max_width - current_width
                
                # 목표에 가장 가깝게 채울 수 있는 아이템 선택
                candidates = [(i, self.item_info[i]) for i in self.items 
                              if self.item_info[i] <= remaining_to_max]
                if not candidates:
                    break
                
                # 목표와의 차이가 가장 작은 아이템 선택
                best_item = min(candidates, key=lambda x: abs(remaining_to_target - x[1]))[0]
                current_pattern[best_item] = current_pattern.get(best_item, 0) + 1
                current_width += self.item_info[best_item]
                current_pieces += 1

            if current_width >= self.min_width:
                self._add_pattern(current_pattern)

        # === 새로운 휴리스틱 3: 역순 채우기 (큰 아이템부터 시작하여 작은 것으로 채우기) ===
        for item in sorted_by_width_desc:
            current_pattern = {item: 1}
            current_width = self.item_info[item]
            current_pieces = 1

            # 같은 아이템으로 최대한 채우기
            item_width = self.item_info[item]
            while current_pieces < self.max_pieces and current_width + item_width <= self.max_width:
                current_pattern[item] = current_pattern.get(item, 0) + 1
                current_width += item_width
                current_pieces += 1

            # 작은 아이템으로 min_width까지 채우기
            for small_item in sorted_by_width_asc:
                if current_width >= self.min_width:
                    break
                if current_pieces >= self.max_pieces:
                    break
                    
                small_width = self.item_info[small_item]
                while current_pieces < self.max_pieces and current_width + small_width <= self.max_width:
                    current_pattern[small_item] = current_pattern.get(small_item, 0) + 1
                    current_width += small_width
                    current_pieces += 1
                    
                    if current_width >= self.min_width:
                        break

            if current_width >= self.min_width:
                self._add_pattern(current_pattern)

        # === 순수 아이템 패턴 (동일 아이템만으로 구성) ===
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

        # === 강화된 폴백: 커버되지 않은 아이템 처리 ===
        covered_items = {item for pattern in self.patterns for item in pattern}
        uncovered_items = set(self.items) - covered_items

        if uncovered_items:
            logging.info(f"[폴백] 커버되지 않은 지폭: {[self.item_info[i] for i in uncovered_items]}")
            for item in uncovered_items:
                item_width = self.item_info[item]
                
                # 시도 1: 같은 아이템 반복 + 다른 큰 아이템으로 채우기
                pattern = {item: 1}
                width = item_width
                pieces = 1
                
                while pieces < self.max_pieces and width + item_width <= self.max_width:
                    pattern[item] = pattern.get(item, 0) + 1
                    width += item_width
                    pieces += 1
                
                if width < self.min_width:
                    for other_item in sorted_by_width_desc:
                        if pieces >= self.max_pieces:
                            break
                        other_width = self.item_info[other_item]
                        
                        while pieces < self.max_pieces and width + other_width <= self.max_width:
                            pattern[other_item] = pattern.get(other_item, 0) + 1
                            width += other_width
                            pieces += 1
                            
                            if width >= self.min_width:
                                break
                        
                        if width >= self.min_width:
                            break
                
                if width >= self.min_width:
                    self._add_pattern(pattern)
                    logging.info(f"  -> 폴백 패턴 생성: {pattern} (너비: {width}mm)")
                else:
                    # 시도 2: 작은 아이템부터 채워서 min_width 충족 시도
                    pattern2 = {item: 1}
                    width2 = item_width
                    pieces2 = 1
                    
                    for other_item in sorted_by_width_asc:
                        if pieces2 >= self.max_pieces:
                            break
                        other_width = self.item_info[other_item]
                        
                        while pieces2 < self.max_pieces and width2 + other_width <= self.max_width:
                            pattern2[other_item] = pattern2.get(other_item, 0) + 1
                            width2 += other_width
                            pieces2 += 1
                    
                    if width2 >= self.min_width:
                        self._add_pattern(pattern2)
                        logging.info(f"  -> 폴백 패턴 생성 (시도2): {pattern2} (너비: {width2}mm)")
                    else:
                        logging.warning(f"  -> [경고] 지폭 {item_width}mm에 대해 min_width({self.min_width}mm)를 충족하는 패턴을 생성하지 못함.")
                        # 최소한 패턴은 추가하여 MIP에서 검토
                        self._add_pattern(pattern)

        logging.info(f"[초기 패턴] 총 {len(self.patterns)}개의 초기 패턴 생성됨")

    def _solve_master_problem(self, is_final_mip=False, max_patterns=None):
        """
        마스터 문제(Master Problem)를 선형계획법(LP) 또는 정수계획법(MIP)으로 해결합니다.
        
        목적 함수:
        1. 과생산 페널티 최소화 (주문량 준수)
        2. 패턴 교체 비용(Setup Cost) 최소화
        3. 폐기물(Trim Loss) 최소화 (옵션)
        4. 혼합 생산(Mixing) 페널티 최소화 (동일 패턴 내 이질적인 제품 혼합 지양)
        """
        
        # 1. [Final MIP 단계] Gurobi 직접 호출 시도 (Size-Limited License 활용)
        if is_final_mip:
            try:
                
                
                logging.info(f"[Final MIP] 총 {len(self.patterns)}개의 패턴 생성됨")
                logging.info("Trying Gurobi Direct Solver roll_optimization(gurobipy)...")
                
                # Suppress Gurobi output
                model = gp.Model("RollOptimization")
                model.setParam("OutputFlag", 0)
                model.setParam("LogToConsole", 0)
                if hasattr(self, 'num_threads'):
                    model.setParam("Threads", self.num_threads)
                
                # Variables
                x = {} # pattern count (integer)
                y = {} # pattern used? (binary)
                for j in range(len(self.patterns)):
                    x[j] = model.addVar(vtype=GRB.INTEGER, name=f"P_{j}")
                    y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
                
                over_prod_vars = {}
                for item in self.demands:
                    over_prod_vars[item] = model.addVar(vtype=GRB.CONTINUOUS, name=f"Over_{item}")
                    
                model.update()
                
                # Constraints
                # 1. Demand Satisfaction
                for item, demand in self.demands.items():
                    production_expr = gp.quicksum(self.patterns[j].get(item, 0) * x[j] for j in range(len(self.patterns)))
                    model.addConstr(production_expr == int(demand) + over_prod_vars[item], name=f"demand_{item}")
                
                # 2. Link x and y (Big-M)
                M = sum(self.demands.values()) + 10
                for j in range(len(self.patterns)):
                    model.addConstr(x[j] <= M * y[j], name=f"link_{j}")

                # Objective Function
                obj_terms = []
                
                # (1) Over-production Penalty
                unique_widths_count = len(set(self.item_info.values()))
                dynamic_over_prod_penalty = max(OVER_PROD_PENALTY, PATTERN_SETUP_COST * unique_widths_count * 20)
                for item in self.demands:
                    obj_terms.append(over_prod_vars[item] * dynamic_over_prod_penalty)
                    
                # (2) Setup Cost
                for j in range(len(self.patterns)):
                    obj_terms.append(y[j] * PATTERN_SETUP_COST)
                    
                # (3) Trim Loss
                for j, pattern in enumerate(self.patterns):
                    loss = self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())
                    if loss > 0:
                        obj_terms.append(x[j] * loss * TRIM_LOSS_PENALTY)
                        
                # (4) Mixing Penalty
                for j, pattern in enumerate(self.patterns):
                    sep_qts = [self.sep_qt_info[item] for item in pattern for _ in range(pattern[item]) if self.sep_qt_info[item].strip()]
                    unique_sep_qts = set(sep_qts)
                    if len(unique_sep_qts) > 1:
                        obj_terms.append(x[j] * MIXING_PENALTY * 50)
                    elif len(unique_sep_qts) == 1:
                        empty_count = sum(count for item, count in pattern.items() if not self.sep_qt_info[item].strip())
                        if empty_count > 0:
                            obj_terms.append(x[j] * MIXING_PENALTY * empty_count)

                model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
                
                # Optimize
                model.setParam('TimeLimit', self.solver_time_limit_ms / 1000.0)
                model.optimize()
                
                # Check for optimal or feasible solution (even if time limit reached)
                if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                    status_msg = "Optimal" if model.Status == GRB.OPTIMAL else "Feasible (TimeLimit)"
                    logging.info(f"Using solver: GUROBI for Final MIP (Success: {status_msg}, Obj={model.ObjVal})")
                    solution = {
                        'objective': model.ObjVal,
                        'pattern_counts': {j: x[j].X for j in range(len(self.patterns))},
                        'over_production': {item: over_prod_vars[item].X for item in self.demands}
                    }
                    return solution
                else:
                    logging.warning(f"Gurobi failed to find optimal solution (Status={model.Status}). Fallback to SCIP.")

            except Exception as e:
                logging.warning(f"Gurobi direct execution failed: {e}. Fallback to SCIP.")

        # 2. [Fallback/LP 단계] OR-Tools Solver (SCIP or GLOP)
        # Gurobi 실패 시 또는 LP 단계일 경우 실행
        
        solver_name = 'SCIP' if is_final_mip else 'GLOP'
        solver = pywraplp.Solver.CreateSolver(solver_name)
        solver.SetNumThreads(self.num_threads)
        
        if not solver:
            return None
            
        if is_final_mip:
             logging.info(f"Using solver: {solver_name} (Fallback/Default)")

        if is_final_mip and hasattr(solver, 'SetTimeLimit'):
            solver.SetTimeLimit(self.solver_time_limit_ms)
        
        if hasattr(solver, 'SetNumThreads'):
            solver.SetNumThreads(self.num_threads)

        x = {j: (solver.IntVar if is_final_mip else solver.NumVar)(0, solver.infinity(), f'P_{j}')
             for j in range(len(self.patterns))}
        over_prod_vars = {item: solver.NumVar(0, solver.infinity(), f'Over_{item}') for item in self.demands}

        constraints = {}
        objective = 0
        
        # Constraints: Demand
        for item, demand in self.demands.items():
            production_expr = solver.Sum(self.patterns[j].get(item, 0) * x[j] for j in range(len(self.patterns)))
            constraints[item] = solver.Add(production_expr == demand + over_prod_vars[item], f'demand_{item}')

        # Objective: Over-production Penalty (LP & MIP common)
        unique_widths_count = len(set(self.item_info.values()))
        dynamic_over_prod_penalty = max(OVER_PROD_PENALTY, PATTERN_SETUP_COST * unique_widths_count * 20)
        objective += solver.Sum(dynamic_over_prod_penalty * over_prod_vars[item] for item in self.demands)

        # MIP Specific Objective Terms
        if is_final_mip:
            y = {j: solver.BoolVar(f'y_{j}') for j in range(len(self.patterns))}
            
            # Big-M Constraint
            M = sum(self.demands.values()) + 1
            for j in range(len(self.patterns)):
                solver.Add(x[j] <= M * y[j])

            # Setup Cost (MIP only, uses binary variable y)
            objective += solver.Sum(y[j] * PATTERN_SETUP_COST for j in range(len(self.patterns)))

        # [Common Objective] Trim Loss
        # LP 단계에서도 트림 손실 비용을 반영해야, Dual 값이 "트림을 줄이는 패턴"에 대한 가치를 반영하게 됨.
        objective += solver.Sum(
            (self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())) * x[j] * TRIM_LOSS_PENALTY
            for j, pattern in enumerate(self.patterns)
        )

        # [Common Objective] Mixing Penalty
        total_mixing_penalty = 0
        for j, pattern in enumerate(self.patterns):
            sep_qts = [self.sep_qt_info[item] for item in pattern for _ in range(pattern[item]) if self.sep_qt_info[item].strip()]
            unique_sep_qts = set(sep_qts)
            
            if len(unique_sep_qts) > 1:
                total_mixing_penalty += x[j] * MIXING_PENALTY * 50
            elif len(unique_sep_qts) == 1:
                empty_count = sum(count for item, count in pattern.items() if not self.sep_qt_info[item].strip())
                if empty_count > 0:
                    total_mixing_penalty += x[j] * MIXING_PENALTY * empty_count
        
        objective += total_mixing_penalty

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
            # Cost-Adjusted DP Logic:
            # Objective: Maximize (Sum(Dual * Count) - PatternCost)
            # PatternCost = SetupCost + TrimPenalty + MixingPenalty
            # Since TrimPenalty = (MaxW - Sum(Width * Count)) * TRIM_LOSS_PENALTY
            #                   = MaxW * TRIM_LOSS_PENALTY - Sum(Width * Count * TRIM_LOSS_PENALTY)
            #
            # Reduced Cost = Sum(Dual * Count) - [SetupCost + (MaxW - Sum(Width * Count)) * TRIM_LOSS_PENALTY]
            #              = Sum( (Dual + Width * TRIM_LOSS_PENALTY) * Count ) - (SetupCost + MaxW * TRIM_LOSS_PENALTY)
            #
            # So, Effective Item Value in DP = Dual + Width * TRIM_LOSS_PENALTY
            # Threshold to beat = SetupCost + MaxW * TRIM_LOSS_PENALTY
            
            target_threshold = PATTERN_SETUP_COST + self.max_width * TRIM_LOSS_PENALTY
            
            # [DEBUG] 로그: Dual 값 통계 및 Threshold 확인
            max_dual = max(duals.values()) if duals else 0
            avg_dual = sum(duals.values()) / len(duals) if duals else 0
            # logging.info(f"[CG Debug] Max Dual: {max_dual:.2f}, Avg Dual: {avg_dual:.2f}, Threshold: {target_threshold:.2f}")

            item_details = []
            for item in candidate_items:
                item_width = self.item_info[item]
                item_dual = duals.get(item, 0)
                
                # Add implicit value from saving trim loss
                item_effective_value = item_dual + item_width * TRIM_LOSS_PENALTY
                
                if item_effective_value <= 0:
                    continue
                item_details.append((item, item_width, item_effective_value))
            
            # logging.info(f"[CG Debug] Candidate Items with +Value: {len(item_details)}/{len(candidate_items)}")
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
                    
                    # Reduced Cost Check
                    # RC = Value_Computed - Target_Threshold
                    # We want RC > 0 (or > small epsilon)
                    reduced_cost = value - target_threshold
                    
                    if reduced_cost <= 1.0: # Minimum improvement margin
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
                    
                    # Adjust for Mixing Penalty (Simplified)
                    # If mixed, we subtract mixing penalty from reduced cost to see if it's still viable
                    sep_qts = [self.sep_qt_info[item] for item in pattern for _ in range(pattern[item]) if self.sep_qt_info[item].strip()]
                    unique_sep_qts = set(sep_qts)
                    mixing_cost = 0
                    if len(unique_sep_qts) > 1:
                        mixing_cost = MIXING_PENALTY * 50
                    elif len(unique_sep_qts) == 1:
                        empty_count = sum(count for item, count in pattern.items() if not self.sep_qt_info[item].strip())
                        if empty_count > 0:
                            mixing_cost = MIXING_PENALTY * empty_count
                            
                    real_reduced_cost = reduced_cost - mixing_cost
                    if real_reduced_cost <= 1.0:
                        continue

                    key = frozenset(pattern.items())
                    if key in seen_patterns:
                        continue
                    total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                    if total_width < self.min_width or total_width > self.max_width:
                        continue
                    seen_patterns.add(key)
                    local_candidates.append({'pattern': pattern, 'value': real_reduced_cost, 'width': total_width, 'pieces': pieces})
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
        
        # Gurobi Size-Limited 라이선스는 변수 2000개까지 허용하므로, 
        # 패턴 수가 2000개 이하이면 굳이 필터링할 필요가 없음. (오히려 정수해 탐색에 방해될 수 있음)
        SAFE_PATTERN_LIMIT = 2000
        if len(self.patterns) > SAFE_PATTERN_LIMIT:
            logging.info(f"패턴 수가 많아 LP 필터링 수행 중... (현재: {len(self.patterns)}개)")
            self._filter_patterns_by_lp(keep_top_n=SAFE_PATTERN_LIMIT)
            logging.info(f"LP 필터링 완료. 남은 패턴: {len(self.patterns)}개")

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
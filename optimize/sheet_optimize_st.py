"""
===============================================================================
SheetOptimizeSt Module - 쉬트지 최적화 모듈 (ST / Plant 8000)
===============================================================================

[모듈 개요]
sheet_optimize_ca.py의 가변 생산길이 로직과 sheet_optimize.py의 패턴 생성 로직을
결합한 쉬트지 최적화 모듈입니다.

- 미터(m) 기반 수요 계산
- (지폭, 세로) 튜플 키로 세로가 다른 주문 분리
- sheet_trim만 적용 (coating, double_cutter 없음)
- 가변 생산길이 (min/max_sheet_roll_length) + std_roll_cnt 배수 적용
===============================================================================
"""

import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import math
import random
import time
import logging
import gurobipy as gp
from gurobipy import GRB
import itertools

# 페널티 값
OVER_PROD_PENALTY = 100000.0
UNDER_PROD_PENALTY = 50000.0
PATTERN_COUNT_PENALTY = 100000.0
SINGLE_STRIP_PENALTY = 50000.0
PATTERN_COMPLEXITY_PENALTY = 1.0

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 1
SMALL_PROBLEM_THRESHOLD = 8
SOLVER_TIME_LIMIT_MS = 180000
CG_MAX_ITERATIONS = 1000
CG_NO_IMPROVEMENT_LIMIT = 50
CG_SUBPROBLEM_TOP_N = 10
PIECE_COUNT_PENALTY = 100


class SheetOptimizeSt:
    """쉬트지 최적화 클래스 (ST / Plant 8000)."""

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
            df_spec_pre=None,
            min_width=None,
            max_width=None,
            max_pieces=None,
            time_limit=300000,
            min_sheet_roll_length=None,
            max_sheet_roll_length=None,
            std_roll_cnt=None,
            sheet_trim=None,
            min_sc_width=None,
            max_sc_width=None,
            num_threads=4
    ):
        self.df_orders = df_spec_pre.copy()
        self.max_width = max_width
        self.min_width = min_width
        self.max_pieces = int(max_pieces) if max_pieces else 8
        self.b_wgt = b_wgt
        self.min_sheet_roll_length = min_sheet_roll_length
        self.max_sheet_roll_length = max_sheet_roll_length
        self.sheet_trim = sheet_trim or 0
        self.min_sc_width = min_sc_width
        self.max_sc_width = max_sc_width
        self.color = color
        self.paper_type = paper_type
        self.std_roll_cnt = std_roll_cnt
        self.db = db
        self.lot_no = lot_no
        self.version = version
        self.num_threads = num_threads
        self.solver_time_limit_ms = time_limit
        self.original_max_width = max_width
        self.min_pieces = MIN_PIECES_PER_PATTERN

        # 주문 데이터의 '가로' 컬럼을 '지폭'으로 복사
        self.df_orders['지폭'] = self.df_orders['가로']

        # 주문량을 미터 단위로 변환하여 수요 계산
        self.df_orders, self.demands_in_meters, self.order_sheet_lengths, self.demands_in_rolls = self._calculate_demand_meters(df_spec_pre)
        self.order_widths = list(self.demands_in_meters.keys())

        # 지폭별 주문 톤수 요약
        width_summary = {}
        tons_per_width = self.df_orders.groupby('지폭')['주문톤'].sum()
        for width, required_meters in self.demands_in_meters.items():
            base_w = width[0] if isinstance(width, tuple) else width
            order_tons = tons_per_width.get(base_w, 0)
            width_summary[width] = {'order_tons': order_tons}
        self.width_summary = width_summary

        # 복합 아이템 생성
        self.items, self.item_info, self.item_composition = self._prepare_items(min_sc_width, max_sc_width)
        logging.info(f"--- 패턴 제약조건: 최소 {self.min_pieces}폭, 최대 {self.max_pieces}폭 ---")

        self.patterns = []

    def _prepare_items(self, min_sc_width, max_sc_width):
        """복합 아이템을 생성합니다. (지폭, 세로) 튜플 키 기반."""
        items = []
        item_info = {}
        item_composition = {}

        for key in self.order_widths:
            if isinstance(key, tuple):
                width, sheet_length = key
            else:
                width = key
                sheet_length = self.order_sheet_lengths.get(key, 0)

            for i in range(1, 5):  # 1~4폭
                base_width = width * i + self.sheet_trim

                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                if isinstance(key, tuple):
                    item_name = f"{width}_{sheet_length}x{i}"
                else:
                    item_name = f"{width}x{i}"

                if base_width <= self.original_max_width:
                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = base_width
                        item_composition[item_name] = {key: i}

        logging.info(f"\n--- 생성된 복합폭 아이템 ({len(items)}개) ---")
        for item in items:
            logging.info(f"  {item}: {item_info[item]}mm")

        return items, item_info, item_composition

    def _calculate_demand_meters(self, df_orders):
        """쉬트지 주문량(톤)을 필요 생산 길이(미터)로 변환합니다."""
        df_copy = df_orders.copy()

        def calculate_meters(row):
            width_mm = row.get('지폭', row.get('width', 0))
            length_mm = row.get('세로', row.get('length', 0))
            order_ton = row.get('주문톤', row.get('order_ton_cnt', 0))
            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0 or order_ton <= 0:
                return 0
            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1000000
            if sheet_weight_g <= 0:
                return 0
            total_sheets_needed = (order_ton * 1000000) / sheet_weight_g
            total_meters_needed = total_sheets_needed * (length_mm / 1000)
            return total_meters_needed

        df_copy.columns = [c.lower() for c in df_copy.columns]
        rename_map = {
            'width': '지폭', 'length': '세로',
            'order_ton_cnt': '주문톤', '가로': '지폭'
        }
        df_copy = df_copy.rename(columns=rename_map)
        df_copy['meters'] = df_copy.apply(calculate_meters, axis=1)

        # (지폭, 세로) 튜플을 키로 사용
        df_copy['demand_key'] = list(zip(df_copy['지폭'].astype(int), df_copy['세로'].astype(int)))
        demand_meters = df_copy.groupby('demand_key')['meters'].sum().to_dict()
        order_sheet_lengths = df_copy.groupby('demand_key')['세로'].first().to_dict()

        std_roll_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2
        demand_rolls = {key: meters / std_roll_length for key, meters in demand_meters.items()}

        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        logging.info(f"--- (지폭, 세로)별 필요 총 길이 (표준롤길이: {std_roll_length}m) ---")
        logging.info("-" * 50)
        for key, meters in demand_meters.items():
            rolls = demand_rolls[key]
            logging.info(f"  {key}: {meters:.2f}m ({rolls:.2f}롤)")

        return df_copy, demand_meters, order_sheet_lengths, demand_rolls

    # === 패턴 생성 ===

    def _generate_all_patterns(self):
        """작은 문제에 대해 모든 가능한 패턴을 생성합니다 (Brute-force)."""
        all_patterns = []
        seen_patterns = set()
        item_list = list(self.items)
        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2

        def find_combinations_recursive(start_index, current_pattern, current_width, current_pieces):
            if self.min_width <= current_width <= self.max_width and self.min_pieces <= current_pieces <= self.max_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in seen_patterns:
                    loss_per_roll = self.max_width - current_width
                    all_patterns.append({
                        'composition': current_pattern.copy(),
                        'length': pattern_length,
                        'loss_per_roll': loss_per_roll
                    })
                    seen_patterns.add(pattern_key)

            if current_pieces >= self.max_pieces or start_index >= len(item_list):
                return

            find_combinations_recursive(start_index + 1, current_pattern, current_width, current_pieces)

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
        logging.info(f"--- 전체 탐색으로 {len(self.patterns)}개의 패턴 생성됨 ---")

    def _generate_initial_patterns_db(self):
        """DB에서 사용자 편집 패턴을 불러와 초기 패턴을 생성합니다."""
        if not self.db or not self.lot_no:
            logging.info("--- DB 정보가 없어 기존 패턴을 불러올 수 없습니다. ---")
            return
        logging.info("\n--- DB(th_pattern_tot_sheet)에서 사용자 편집 패턴을 불러옵니다. ---")
        db_patterns_list = self.db.get_sheet_ca_patterns_from_db(self.lot_no)
        if not db_patterns_list:
            logging.info("--- DB에 저장된 사용자 편집 패턴이 없습니다. ---")
            return

        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2
        initial_patterns_from_db = []

        for pattern_item_list in db_patterns_list:
            pattern_dict = dict(Counter(pattern_item_list))
            all_items_valid = all(item_name in self.items for item_name in pattern_dict.keys())
            if all_items_valid:
                current_total_width = sum(self.item_info[name] * count for name, count in pattern_dict.items())
                current_total_pieces = sum(pattern_dict.values())
                if self.min_width <= current_total_width <= self.max_width and self.min_pieces <= current_total_pieces <= self.max_pieces:
                    loss_per_roll = self.max_width - current_total_width
                    initial_patterns_from_db.append({
                        'composition': pattern_dict,
                        'length': pattern_length,
                        'loss_per_roll': loss_per_roll
                    })

        if initial_patterns_from_db:
            seen_patterns = {frozenset(p['composition'].items()) for p in self.patterns}
            added_count = 0
            for pat in initial_patterns_from_db:
                key = frozenset(pat['composition'].items())
                if key not in seen_patterns:
                    self.patterns.append(pat)
                    seen_patterns.add(key)
                    added_count += 1
            logging.info(f"--- DB에서 {added_count}개의 사용자 편집 패턴을 추가했습니다. ---")

    def _generate_initial_patterns(self):
        """휴리스틱을 사용하여 초기 패턴을 생성합니다."""
        seen_patterns = {frozenset(p['composition'].items()) for p in self.patterns}
        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2

        # 정렬 전략
        sorted_by_demand = sorted(self.items, key=lambda i: self.demands_in_meters.get(list(self.item_composition[i].keys())[0], 0), reverse=True)
        sorted_by_demand_asc = sorted(self.items, key=lambda i: self.demands_in_meters.get(list(self.item_composition[i].keys())[0], 0))
        sorted_by_width_desc = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda i: self.item_info[i])

        random.seed(42)
        random_shuffles = []
        for _ in range(100):
            items_copy = list(self.items)
            random.shuffle(items_copy)
            random_shuffles.append(items_copy)

        heuristics = [sorted_by_demand, sorted_by_width_desc, sorted_by_width_asc, sorted_by_demand_asc] + random_shuffles

        def add_pattern(pat):
            nonlocal seen_patterns
            key = frozenset(pat.items())
            if key not in seen_patterns:
                total_w = sum(self.item_info[i] * c for i, c in pat.items())
                if self.min_width <= total_w <= self.max_width and self.min_pieces <= sum(pat.values()):
                    loss = self.max_width - total_w
                    self.patterns.append({'composition': pat.copy(), 'length': pattern_length, 'loss_per_roll': loss})
                    seen_patterns.add(key)

        # 소량 주문 우선 패턴
        small_demand_items = sorted(self.items, key=lambda i: self.demands_in_meters.get(list(self.item_composition[i].keys())[0], float('inf')))
        for primary_item in small_demand_items[:20]:
            primary_width = self.item_info[primary_item]
            for secondary_item in sorted_by_width_desc:
                if secondary_item == primary_item:
                    continue
                secondary_width = self.item_info[secondary_item]
                for primary_count in range(1, self.max_pieces + 1):
                    remaining_width = self.max_width - (primary_width * primary_count)
                    remaining_pieces = self.max_pieces - primary_count
                    if remaining_width <= 0 or remaining_pieces <= 0:
                        continue
                    secondary_count = min(int(remaining_width / secondary_width), remaining_pieces)
                    if secondary_count <= 0:
                        continue
                    add_pattern({primary_item: primary_count, secondary_item: secondary_count})

        logging.info(f"--- {len(self.patterns)}개의 소량주문 우선 패턴 생성됨 ---")

        # First-Fit 휴리스틱
        for sorted_items in heuristics:
            for item in sorted_items:
                item_width = self.item_info[item]
                current_pattern = {item: 1}
                current_width = item_width
                current_pieces = 1
                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    best_fit_item = next((i for i in sorted_items if self.item_info[i] <= remaining_width), None)
                    if not best_fit_item:
                        break
                    current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
                    current_width += self.item_info[best_fit_item]
                    current_pieces += 1
                add_pattern(current_pattern)

        logging.info(f"--- {len(self.patterns)}개 First-Fit 패턴 생성됨 ---")

        # Best-Fit 휴리스틱
        for item in sorted_by_demand_asc:
            item_width = self.item_info[item]
            current_pattern = {item: 1}
            current_width = item_width
            current_pieces = 1
            while current_pieces < self.max_pieces:
                remaining_width = self.max_width - current_width
                best_fit_item = None
                min_waste = float('inf')
                for candidate in self.items:
                    cw = self.item_info[candidate]
                    if cw <= remaining_width and remaining_width - cw < min_waste:
                        min_waste = remaining_width - cw
                        best_fit_item = candidate
                if not best_fit_item:
                    break
                current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
                current_width += self.item_info[best_fit_item]
                current_pieces += 1
            add_pattern(current_pattern)

        logging.info(f"--- {len(self.patterns)}개 Best-Fit 패턴 생성됨 ---")

        # 2폭 조합 체계적 생성
        items_list = list(self.items)
        for i, item1 in enumerate(items_list):
            w1 = self.item_info[item1]
            for item2 in items_list[i:]:
                w2 = self.item_info[item2]
                for c1 in range(1, self.max_pieces):
                    for c2 in range(1, self.max_pieces - c1 + 1):
                        tw = w1 * c1 + w2 * c2
                        tp = c1 + c2
                        if self.min_width <= tw <= self.max_width and self.min_pieces <= tp <= self.max_pieces:
                            new_pat = {item1: c1}
                            if item1 != item2:
                                new_pat[item2] = c2
                            else:
                                new_pat[item1] += c2
                            add_pattern(new_pat)

        logging.info(f"--- {len(self.patterns)}개 2폭 조합 패턴 생성됨 ---")

        # 순수 품목 패턴
        for item in sorted_by_width_asc:
            iw = self.item_info.get(item, 0)
            if iw <= 0:
                continue
            num_items = min(int(self.max_width / iw), self.max_pieces)
            while num_items > 0:
                new_pat = {item: num_items}
                tw = iw * num_items
                if self.min_width <= tw and self.min_pieces <= num_items:
                    add_pattern(new_pat)
                    break
                num_items -= 1

        # 폴백: 커버되지 않는 지폭 확인
        covered = set()
        for p in self.patterns:
            for item_name in p['composition']:
                covered.update(self.item_composition.get(item_name, {}).keys())
        uncovered = set(self.order_widths) - covered
        if uncovered:
            logging.info(f"--- 경고: 초기 패턴에 포함되지 않은 지폭: {uncovered} ---")
            for key in uncovered:
                w = key[0] if isinstance(key, tuple) else key
                for i in range(4, 0, -1):
                    if isinstance(key, tuple):
                        iname = f"{w}_{key[1]}x{i}"
                    else:
                        iname = f"{w}x{i}"
                    if iname in self.item_info:
                        num = min(int(self.max_width / self.item_info[iname]), self.max_pieces)
                        if num > 0:
                            add_pattern({iname: num})
                            break

        logging.info(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")

    # === 솔버 ===

    def _solve_master_problem_ilp(self, is_final_mip=False):
        """마스터 문제를 ILP/LP로 풀어 최적 패턴 조합을 찾습니다."""
        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2
        num_patterns = len(self.patterns)
        demands = self.demands_in_meters
        order_width_list = list(demands.keys())
        num_widths = len(order_width_list)

        try:
            return self._solve_master_gurobi(is_final_mip, pattern_length, num_patterns, demands, order_width_list, num_widths)
        except Exception as e:
            logging.warning(f"Gurobi 실패 ({e}), OR-Tools로 전환합니다.")
            return self._solve_master_ortools(is_final_mip, pattern_length, num_patterns, demands, order_width_list, num_widths)

    def _solve_master_gurobi(self, is_final_mip, pattern_length, num_patterns, demands, order_width_list, num_widths):
        """Gurobi로 마스터 문제를 풀어 최적 솔루션을 찾습니다."""
        model = gp.Model("SheetOptimizeST_Master")
        model.Params.OutputFlag = 0
        model.Params.Threads = self.num_threads
        model.Params.TimeLimit = self.solver_time_limit_ms / 1000

        # 변수 생성
        if is_final_mip:
            x = model.addVars(num_patterns, vtype=GRB.INTEGER, name="x", lb=0)
        else:
            x = model.addVars(num_patterns, vtype=GRB.CONTINUOUS, name="x", lb=0)
        over_prod = model.addVars(num_widths, vtype=GRB.CONTINUOUS, name="over", lb=0)
        under_prod = model.addVars(num_widths, vtype=GRB.CONTINUOUS, name="under", lb=0)

        # 수요 제약: 각 (지폭,세로)별 미터 기반
        for w_idx, width_key in enumerate(order_width_list):
            demand_m = demands[width_key]
            supply = gp.LinExpr()
            for j in range(num_patterns):
                pattern = self.patterns[j]
                pat_len = pattern['length']
                contribution = 0
                for item_name, item_count in pattern['composition'].items():
                    for base_key, base_num in self.item_composition.get(item_name, {}).items():
                        if base_key == width_key:
                            contribution += item_count * base_num * pat_len
                if contribution > 0:
                    supply.add(x[j], contribution)
            model.addConstr(supply + under_prod[w_idx] - over_prod[w_idx] == demand_m, name=f"demand_{w_idx}")

        # 목적함수
        obj = gp.LinExpr()
        # 총 롤 수 최소화
        for j in range(num_patterns):
            obj.add(x[j], 1.0)
        # 과/부족 페널티
        for w_idx in range(num_widths):
            obj.add(over_prod[w_idx], OVER_PROD_PENALTY)
            obj.add(under_prod[w_idx], UNDER_PROD_PENALTY)
        # 패턴 수 페널티
        if is_final_mip:
            y = model.addVars(num_patterns, vtype=GRB.BINARY, name="y")
            for j in range(num_patterns):
                max_rolls = max(1, int(sum(demands.values()) / pattern_length) + 10)
                model.addConstr(x[j] <= max_rolls * y[j])
            obj.add(gp.quicksum(y[j] for j in range(num_patterns)), PATTERN_COUNT_PENALTY)
        # 패턴 복잡도 페널티
        for j in range(num_patterns):
            num_distinct = len(self.patterns[j]['composition'])
            obj.add(x[j], PATTERN_COMPLEXITY_PENALTY * num_distinct)
        # 단일 스트립 페널티
        for j in range(num_patterns):
            total_pieces = sum(self.patterns[j]['composition'].values())
            if total_pieces == 1:
                obj.add(x[j], SINGLE_STRIP_PENALTY)
        # 지폭 수 페널티
        for j in range(num_patterns):
            total_pieces = sum(self.patterns[j]['composition'].values())
            if total_pieces > 2:
                obj.add(x[j], PIECE_COUNT_PENALTY * (total_pieces - 2))

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            return None

        # 결과 추출
        solution = {'pattern_counts': {}, 'duals': {}}
        for j in range(num_patterns):
            val = x[j].X
            if val > 0.001:
                solution['pattern_counts'][j] = val

        if not is_final_mip:
            for w_idx in range(num_widths):
                try:
                    constr = model.getConstrByName(f"demand_{w_idx}")
                    solution['duals'][order_width_list[w_idx]] = constr.Pi
                except:
                    solution['duals'][order_width_list[w_idx]] = 0

        return solution

    def _solve_master_ortools(self, is_final_mip, pattern_length, num_patterns, demands, order_width_list, num_widths):
        """OR-Tools로 마스터 문제를 풀어 최적 솔루션을 찾습니다."""
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        if not solver:
            return None
        solver.SetTimeLimit(self.solver_time_limit_ms)

        # 변수 생성
        if is_final_mip:
            x = [solver.IntVar(0, solver.infinity(), f"x_{j}") for j in range(num_patterns)]
        else:
            x = [solver.NumVar(0, solver.infinity(), f"x_{j}") for j in range(num_patterns)]
        over_prod = [solver.NumVar(0, solver.infinity(), f"over_{i}") for i in range(num_widths)]
        under_prod = [solver.NumVar(0, solver.infinity(), f"under_{i}") for i in range(num_widths)]

        # 수요 제약
        demand_constraints = []
        for w_idx, width_key in enumerate(order_width_list):
            demand_m = demands[width_key]
            constraint = solver.Constraint(demand_m, demand_m)
            for j in range(num_patterns):
                pat_len = self.patterns[j]['length']
                contribution = 0
                for item_name, item_count in self.patterns[j]['composition'].items():
                    for base_key, base_num in self.item_composition.get(item_name, {}).items():
                        if base_key == width_key:
                            contribution += item_count * base_num * pat_len
                if contribution > 0:
                    constraint.SetCoefficient(x[j], contribution)
            constraint.SetCoefficient(over_prod[w_idx], -1)
            constraint.SetCoefficient(under_prod[w_idx], 1)
            demand_constraints.append(constraint)

        # 목적함수
        objective = solver.Objective()
        for j in range(num_patterns):
            objective.SetCoefficient(x[j], 1.0)
            num_distinct = len(self.patterns[j]['composition'])
            objective.SetCoefficient(x[j], 1.0 + PATTERN_COMPLEXITY_PENALTY * num_distinct)
            total_pieces = sum(self.patterns[j]['composition'].values())
            if total_pieces == 1:
                objective.SetCoefficient(x[j], objective.GetCoefficient(x[j]) + SINGLE_STRIP_PENALTY)
            if total_pieces > 2:
                objective.SetCoefficient(x[j], objective.GetCoefficient(x[j]) + PIECE_COUNT_PENALTY * (total_pieces - 2))
        for w_idx in range(num_widths):
            objective.SetCoefficient(over_prod[w_idx], OVER_PROD_PENALTY)
            objective.SetCoefficient(under_prod[w_idx], UNDER_PROD_PENALTY)
        objective.SetMinimization()

        status = solver.Solve()
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None

        solution = {'pattern_counts': {}, 'duals': {}}
        for j in range(num_patterns):
            val = x[j].solution_value()
            if val > 0.001:
                solution['pattern_counts'][j] = val

        if not is_final_mip:
            for w_idx in range(num_widths):
                try:
                    solution['duals'][order_width_list[w_idx]] = demand_constraints[w_idx].dual_value()
                except:
                    solution['duals'][order_width_list[w_idx]] = 0

        return solution

    def _solve_subproblem_dp(self, duals):
        """DP를 사용하여 새로운 후보 패턴을 찾습니다."""
        new_patterns = []
        item_list = list(self.items)
        num_items = len(item_list)

        for _ in range(CG_SUBPROBLEM_TOP_N):
            best_pattern = None
            best_value = 0

            for start_item_idx in range(num_items):
                current_pattern = {}
                current_width = 0
                current_pieces = 0
                current_value = 0.0

                indices = list(range(start_item_idx, num_items)) + list(range(0, start_item_idx))
                for idx in indices:
                    item = item_list[idx]
                    item_width = self.item_info[item]
                    while current_width + item_width <= self.max_width and current_pieces + 1 <= self.max_pieces:
                        # 이 아이템이 기여하는 dual value 계산
                        item_dual = 0
                        for base_key, base_num in self.item_composition.get(item, {}).items():
                            item_dual += duals.get(base_key, 0) * base_num
                        if item_dual <= 0:
                            break
                        current_pattern[item] = current_pattern.get(item, 0) + 1
                        current_width += item_width
                        current_pieces += 1
                        current_value += item_dual

                if current_value > best_value and self.min_width <= current_width:
                    reduced_cost = current_value - 1.0
                    if reduced_cost > 0.001:
                        best_value = current_value
                        best_pattern = current_pattern.copy()

            if best_pattern:
                pattern_key = frozenset(best_pattern.items())
                existing_keys = {frozenset(p.items()) for p in new_patterns}
                if pattern_key not in existing_keys:
                    new_patterns.append(best_pattern)
            else:
                break

        return new_patterns if new_patterns else None

    # === 메인 최적화 실행 ===

    def run_optimize(self, start_prod_seq=0):
        """최적화를 실행합니다."""
        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2

        # 패턴 생성
        if not self.patterns:
            self._generate_initial_patterns_db()

            if len(self.order_widths) <= SMALL_PROBLEM_THRESHOLD:
                logging.info(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 모든 패턴을 탐색합니다 (Small-scale) ---")
                self._generate_all_patterns()
            else:
                logging.info(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 열 생성 기법을 시작합니다 (Large-scale) ---")
                self._generate_initial_patterns()

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
                        current_pattern_keys = {frozenset(p['composition'].items()) for p in self.patterns}
                        for new_pattern in new_patterns:
                            if frozenset(new_pattern.items()) not in current_pattern_keys:
                                pattern_width = sum(self.item_info[item] * count for item, count in new_pattern.items())
                                if pattern_width >= self.min_width:
                                    loss = self.max_width - pattern_width
                                    self.patterns.append({
                                        'composition': new_pattern,
                                        'length': pattern_length,
                                        'loss_per_roll': loss
                                    })
                                    patterns_added += 1

                    if patterns_added > 0:
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= CG_NO_IMPROVEMENT_LIMIT:
                        logging.info(f"--- {CG_NO_IMPROVEMENT_LIMIT}번의 반복 동안 개선이 없어 수렴으로 간주하고 종료합니다 (반복 {iteration}). ---")
                        break

        if not self.patterns:
            return {"error": "유효한 패턴을 생성할 수 없습니다."}

        logging.info(f"--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        for idx, pattern in enumerate(self.patterns):
            logging.info(f"  P{idx}: {pattern['composition']} | length={pattern['length']} | loss={pattern['loss_per_roll']}")

        # 최종 MIP 솔루션
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}

        # std_roll_cnt 배수 적용 및 길이 조절
        if self.std_roll_cnt and self.std_roll_cnt >= 1:
            # 패턴별 생산량 집계
            solver_prod_by_width = {}
            for j, count in final_solution['pattern_counts'].items():
                if count < 0.01:
                    continue
                pat_len = self.patterns[j]['length']
                for item_name, item_count in self.patterns[j]['composition'].items():
                    for base_w, base_num in self.item_composition[item_name].items():
                        solver_prod_by_width[base_w] = solver_prod_by_width.get(base_w, 0) + (count * pat_len * item_count * base_num)

            # 지폭별 scale factor 계산
            width_ratios = {}
            for w, demand_m in self.demands_in_meters.items():
                prod_m = solver_prod_by_width.get(w, 0)
                if prod_m > 0:
                    width_ratios[w] = demand_m / prod_m
                else:
                    width_ratios[w] = 1.0

            # Aggregated Roll Count
            item_aggregated_counts = {}
            for j, count in final_solution['pattern_counts'].items():
                if count < 0.99:
                    continue
                count_int = int(round(count))
                for item_name in self.patterns[j]['composition']:
                    item_aggregated_counts[item_name] = item_aggregated_counts.get(item_name, 0) + count_int

            # 패턴별 조정
            for j, count in list(final_solution['pattern_counts'].items()):
                if count < 0.99:
                    continue

                count_int = int(round(count))
                current_length = self.patterns[j]['length']

                # 패턴별 개별 scale factor
                pattern_scale_factor = float('inf')
                for item_name in self.patterns[j]['composition']:
                    for base_w in self.item_composition[item_name]:
                        if base_w in width_ratios:
                            if width_ratios[base_w] < pattern_scale_factor:
                                pattern_scale_factor = width_ratios[base_w]

                if pattern_scale_factor == float('inf'):
                    pattern_scale_factor = 1.0

                # 2% 여유 적용
                pattern_scale_factor = pattern_scale_factor * 1.02

                optimized_total_len = (count_int * current_length) * pattern_scale_factor

                # std_roll_cnt 배수 확인
                needs_increase = False
                for item_name in self.patterns[j]['composition']:
                    if item_aggregated_counts.get(item_name, 0) < self.std_roll_cnt:
                        needs_increase = True
                        break

                if count_int < self.std_roll_cnt and needs_increase:
                    new_count = self.std_roll_cnt
                    logging.info(f"[Constraint] P{j} Count {count_int} -> {self.std_roll_cnt} (Aggregated insufficient)")
                else:
                    new_count = count_int

                if new_count == 0:
                    new_count = max(1, self.std_roll_cnt)

                new_length = optimized_total_len / new_count

                # 롤 길이를 100 단위로 반올림
                new_length = round(new_length / 100.0) * 100.0

                if new_length < self.min_sheet_roll_length:
                    logging.warning(f"[Warning] 패턴 {j} 길이 {new_length:.1f} < 최소 {self.min_sheet_roll_length}")

                logging.info(f"[Adjust] P{j} Count {count_int}->{new_count}, Length {current_length:.1f}->{new_length:.1f} (Scale:{pattern_scale_factor:.4f})")

                final_solution['pattern_counts'][j] = float(new_count)
                self.patterns[j]['length'] = new_length

        return self._format_results(final_solution, start_prod_seq)

    # === 결과 포맷팅 ===

    def _format_results(self, final_solution, start_prod_seq=0):
        """최적화 결과를 포맷팅하여 반환합니다."""
        result_patterns, pattern_details_for_db, pattern_roll_details_for_db, pattern_roll_cut_details_for_db, demand_tracker, last_prod_seq = self._build_pattern_details(final_solution, start_prod_seq)

        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['pattern', 'wd_width', 'roll_length', 'count', 'loss_per_roll']]

        fulfillment_summary = self._build_fulfillment_summary(demand_tracker)

        return {
            "pattern_result": df_patterns.sort_values('count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "last_prod_seq": last_prod_seq
        }

    def _build_pattern_details(self, final_solution, start_prod_seq=0):
        """최적화 결과로부터 DB 저장용 상세 정보를 생성합니다."""
        # 주문 이행 추적용 초기화
        demand_tracker = self.df_orders.copy()
        demand_tracker['original_order_idx'] = demand_tracker.index
        demand_tracker = demand_tracker[['original_order_idx', 'group_order_no', '지폭', 'meters', 'demand_key']].copy()
        demand_tracker['fulfilled_meters'] = 0.0
        demand_tracker = demand_tracker.sort_values(by=['지폭', 'group_order_no']).reset_index(drop=True)

        result_patterns = []
        pattern_details_for_db = []
        pattern_roll_details_for_db = []
        pattern_roll_cut_details_for_db = []
        prod_seq_counter = start_prod_seq
        total_cut_seq_counter = 0

        def safe_int(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        first_row = self.df_orders.iloc[0]
        common_props = {
            'diameter': 0,
            'color': first_row.get('color', ''),
            'luster': safe_int(first_row.get('luster', 0)),
            'p_lot': self.lot_no,
            'core': 0,
            'order_pattern': first_row.get('order_pattern', '')
        }

        for j, count in final_solution['pattern_counts'].items():
            if count < 0.99:
                continue

            roll_count = int(round(count))
            pattern = self.patterns[j]
            pattern_comp = pattern['composition']
            pattern_length = pattern['length']

            prod_seq_counter += 1

            sorted_pattern_items = sorted(pattern_comp.items(), key=lambda item: self.item_info[item[0]], reverse=True)
            pattern_item_strs = []
            total_width = 0
            all_base_pieces_in_roll = []

            # Step 1: 패턴 요약
            for item_name, num in sorted_pattern_items:
                width = self.item_info[item_name]
                total_width += width * num

                base_width_dict = self.item_composition[item_name]
                for base_key, num_base in base_width_dict.items():
                    all_base_pieces_in_roll.extend([base_key] * (num_base * num))

                sub_items = item_name.split('+')
                if len(sub_items) > 1 or 'x' not in item_name:
                    formatted_name = f"{width}({item_name})"
                else:
                    try:
                        parts = item_name.split('x')
                        multiplier = int(parts[1])
                        width_part = parts[0]
                        if '_' in width_part:
                            base_width_val = int(width_part.split('_')[0])
                        else:
                            base_width_val = int(width_part)
                        formatted_name = f"{width}({base_width_val}*{multiplier})"
                    except ValueError:
                        formatted_name = f"{width}({item_name})"
                pattern_item_strs.extend([formatted_name] * num)

            result_patterns.append({
                'pattern': ' + '.join(pattern_item_strs),
                'wd_width': total_width,
                'roll_length': round(pattern_length, 2),
                'count': roll_count,
                'loss_per_roll': pattern['loss_per_roll']
            })

            # Step 2: DB 저장용 상세 정보
            composite_widths_for_db = []
            composite_group_nos_for_db = []
            roll_seq_counter = 0

            for item_name, num_of_composite in sorted_pattern_items:
                composite_width = self.item_info[item_name]
                base_width_dict = self.item_composition[item_name]

                for _ in range(num_of_composite):
                    roll_seq_counter += 1
                    base_widths_for_item = []
                    base_group_nos_for_item = []
                    base_rs_gubuns_for_item = []
                    assigned_group_no_for_composite = None

                    for base_key, num_of_base in sorted(base_width_dict.items(), key=lambda item: item[0][0] if isinstance(item[0], tuple) else item[0], reverse=True):
                        if isinstance(base_key, tuple):
                            base_width_val = base_key[0]
                        else:
                            base_width_val = base_key

                        for _ in range(num_of_base):
                            # demand_key로 매칭
                            target_indices = demand_tracker[
                                (demand_tracker['demand_key'] == base_key) &
                                (demand_tracker['fulfilled_meters'] < demand_tracker['meters'])
                            ].index

                            assigned_group_no = "OVERPROD"
                            if not target_indices.empty:
                                target_idx = target_indices.min()
                                assigned_group_no = demand_tracker.loc[target_idx, 'group_order_no']
                            else:
                                fallback_indices = demand_tracker[demand_tracker['demand_key'] == base_key].index
                                if not fallback_indices.empty:
                                    assigned_group_no = demand_tracker.loc[fallback_indices.min(), 'group_order_no']

                            base_widths_for_item.append(base_width_val)
                            base_group_nos_for_item.append(assigned_group_no)
                            base_rs_gubuns_for_item.append('S')

                            if assigned_group_no_for_composite is None:
                                assigned_group_no_for_composite = assigned_group_no

                    composite_widths_for_db.append(composite_width)
                    composite_group_nos_for_db.append(assigned_group_no_for_composite if assigned_group_no_for_composite else "")

                    pattern_roll_details_for_db.append({
                        'rollwidth': composite_width,
                        'pattern_length': pattern_length,
                        'widths': (base_widths_for_item + [0] * 7)[:7],
                        'roll_widths': ([0] * 7)[:7],
                        'group_nos': (base_group_nos_for_item + [''] * 7)[:7],
                        'rs_gubuns': (base_rs_gubuns_for_item + [''] * 7)[:7],
                        'count': roll_count,
                        'prod_seq': prod_seq_counter,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'S',
                        'trim_loss': 0,
                        'sc_trim': self.sheet_trim,
                        'sl_trim': 0,
                        **common_props
                    })

                    # Step 3: 커팅 상세
                    cut_seq_counter = 0
                    for i in range(len(base_widths_for_item)):
                        width = base_widths_for_item[i]
                        if width > 0:
                            cut_seq_counter += 1
                            total_cut_seq_counter += 1
                            group_no = base_group_nos_for_item[i]
                            weight = (self.b_wgt * (width / 1000) * pattern_length)

                            pattern_roll_cut_details_for_db.append({
                                'prod_seq': prod_seq_counter,
                                'unit_no': prod_seq_counter,
                                'seq': total_cut_seq_counter,
                                'roll_seq': roll_seq_counter,
                                'cut_seq': cut_seq_counter,
                                'width': width,
                                'group_no': group_no,
                                'weight': weight,
                                'pattern_length': pattern_length,
                                'count': roll_count,
                                'cut_cnt': roll_count,
                                'rs_gubun': 'S',
                                **common_props
                            })

            # TH_PATTERN_SEQUENCE
            pattern_details_for_db.append({
                'pattern_length': pattern_length,
                'count': roll_count,
                'widths': (composite_widths_for_db + [0] * 8)[:8],
                'group_nos': (composite_group_nos_for_db + [''] * 8)[:8],
                'prod_seq': prod_seq_counter,
                'rs_gubun': 'S',
                **common_props
            })

            # Step 4: 주문 이행 업데이트
            base_counts_in_roll = Counter(all_base_pieces_in_roll)
            for base_key, num_in_roll in base_counts_in_roll.items():
                produced_meters = num_in_roll * pattern_length * roll_count
                relevant_orders = demand_tracker[demand_tracker['demand_key'] == base_key].index

                for order_idx in relevant_orders:
                    if produced_meters <= 0:
                        break
                    needed = demand_tracker.loc[order_idx, 'meters'] - demand_tracker.loc[order_idx, 'fulfilled_meters']
                    if needed > 0:
                        fulfill_amount = min(needed, produced_meters)
                        demand_tracker.loc[order_idx, 'fulfilled_meters'] += fulfill_amount
                        produced_meters -= fulfill_amount

                if produced_meters > 0 and not relevant_orders.empty:
                    last_idx = relevant_orders[-1]
                    demand_tracker.loc[last_idx, 'fulfilled_meters'] += produced_meters

        return result_patterns, pattern_details_for_db, pattern_roll_details_for_db, pattern_roll_cut_details_for_db, demand_tracker, prod_seq_counter

    def _build_fulfillment_summary(self, demand_tracker):
        """주문 이행 요약 보고서를 생성합니다."""
        summary_df = self.df_orders[['group_order_no', '지폭', '세로', '수출내수', '등급', '주문톤', 'meters']].copy()
        summary_df.rename(columns={'지폭': '가로', 'meters': '필요길이(m)', '주문톤': '주문량(톤)'}, inplace=True)

        summary_df = pd.merge(summary_df, demand_tracker[['original_order_idx', 'fulfilled_meters']],
                              left_index=True, right_on='original_order_idx', how='left')
        summary_df.rename(columns={'fulfilled_meters': '생산길이(m)'}, inplace=True)
        summary_df.drop(columns=['original_order_idx'], inplace=True)

        summary_df['생산길이(m)'] = summary_df['생산길이(m)'].fillna(0)
        summary_df['과부족(m)'] = summary_df['생산길이(m)'] - summary_df['필요길이(m)']

        tons_per_meter = (summary_df['주문량(톤)'] / summary_df['필요길이(m)']).replace([float('inf'), -float('inf')], 0).fillna(0)
        summary_df['생산량(톤)'] = (summary_df['생산길이(m)'] * tons_per_meter).round(2)
        summary_df['과부족(톤)'] = (summary_df['생산량(톤)'] - summary_df['주문량(톤)']).round(2)

        final_cols = [
            'group_order_no', '가로', '세로', '수출내수', '등급',
            '주문량(톤)', '생산량(톤)', '과부족(톤)',
            '필요길이(m)', '생산길이(m)', '과부족(m)'
        ]

        for col in ['필요길이(m)', '생산길이(m)', '과부족(m)']:
            summary_df[col] = summary_df[col].round(2)

        return summary_df[final_cols]

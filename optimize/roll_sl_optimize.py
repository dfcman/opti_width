import pandas as pd
from ortools.linear_solver import pywraplp

OVER_PROD_PENALTY = 2000.0  # 주문량 초과 생산에 대한 페널티
UNDER_PROD_PENALTY = 10000.0  # 주문량 미달 생산에 대한 페널티
PATTERN_VALUE_THRESHOLD = 1.0 + 1e-6  # (열 생성) 새로운 패턴이 유의미하다고 판단하는 기준값
CG_MAX_ITERATIONS = 200  # (열 생성) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 25  # (열 생성) 목적 함수 값 개선이 없을 때 조기 중단을 위한 반복 횟수
CG_SUBPROBLEM_TOP_N = 1  # (열 생성) 각 반복에서 서브문제로부터 가져올 상위 N개 유의미한 패턴
SMALL_PROBLEM_THRESHOLD = 10  # 문제 크기가 이 값보다 작거나 같으면, 모든 가능한 패턴을 생성하여 최적화 시도
FINAL_MIP_TIME_LIMIT_MS = 60000  # 최종 정수 계획법(MIP) 문제 풀이에 대한 시간 제한 (밀리초)

COMPOSITE_MIN_MULTIPLIER = 2  # 복합폭을 구성할 수 있는 최소 롤 개수
COMPOSITE_MAX_MULTIPLIER = 3  # 복합폭을 구성할 수 있는 최대 롤 개수
COMPOSITE_USAGE_PENALTY = 3000.0  # 복합폭 사용 시 추가 부담 (1폭 대비 생산성 저하 반영)
PATTERN_COMPLEXITY_PENALTY = 2000.0  # 패턴의 복잡도(사용된 아이템 개수)에 대한 페널티
COMPOSITE_BASE_CANDIDATES = 20  # 복합폭 생성 시 고려할 기본 롤(가장 수요가 많은)의 후보 개수
COMPOSITE_GENERATION_LIMIT = 2000  # 생성할 수 있는 복합폭 종류의 최대 개수 (성능 제한용)
SMALL_WIDTH_LIMIT = 480  # 소폭 판정 기준(mm)
MAX_SMALL_WIDTH_PER_PATTERN = 2  # 한 패턴에서 허용되는 소폭 롤 수
OVER_PROD_WEIGHT_CAP = 6.0  # 소량 주문에 대한 초과 페널티 가중치 상한
MIXED_COMPOSITE_PENALTY = 500.0  # 서로 다른 규격 조합 복합롤에 대한 추가 패널티


class RollSLOptimize:
    def __init__(
        self,
        df_spec_pre,
        max_width=1000,
        min_width=0,
        max_pieces=8,
        b_wgt=0,
        sl_trim=0,
        min_sl_width=0,
        max_sl_width=0,
        composite_min=COMPOSITE_MIN_MULTIPLIER,
        composite_max=COMPOSITE_MAX_MULTIPLIER,
        composite_penalty=COMPOSITE_USAGE_PENALTY,
        pattern_complexity_penalty=PATTERN_COMPLEXITY_PENALTY,
        small_width_limit=SMALL_WIDTH_LIMIT,
        max_small_width_per_pattern=MAX_SMALL_WIDTH_PER_PATTERN,
    ):
        self.df_spec_pre = df_spec_pre.copy()
        self.max_width = int(max_width)
        self.min_width = int(min_width)
        self.max_pieces = int(max_pieces)
        self.b_wgt = float(b_wgt)
        self.sl_trim = sl_trim
        self.min_sl_width = min_sl_width
        self.max_sl_width = max_sl_width
        self.composite_min = max(2, int(composite_min))
        self.composite_max = max(self.composite_min, int(composite_max))
        self.composite_penalty = float(composite_penalty)
        self.pattern_complexity_penalty = float(pattern_complexity_penalty)
        self.small_width_limit = int(small_width_limit)
        self.max_small_width_per_pattern = int(max_small_width_per_pattern)

        demand_col_candidates = ['주문수량', '주문롤수', 'order_roll_cnt']
        self.demand_column = next((c for c in demand_col_candidates if c in self.df_spec_pre.columns), None)
        if not self.demand_column:
            raise KeyError("df_spec_pre에 주문 수량 컬럼이 필요합니다. (주문수량 / 주문롤수 / order_roll_cnt 중 하나)")

        width_col_candidates = ['지폭', 'width']
        self.width_column = next((c for c in width_col_candidates if c in self.df_spec_pre.columns), None)
        if not self.width_column:
            raise KeyError("df_spec_pre에 지폭(width) 컬럼이 필요합니다. (지폭 / width 중 하나)")

        self.df_spec_pre[self.demand_column] = pd.to_numeric(
            self.df_spec_pre[self.demand_column],
            errors='coerce'
        ).fillna(0)
        self.df_spec_pre[self.width_column] = pd.to_numeric(
            self.df_spec_pre[self.width_column],
            errors='coerce'
        ).fillna(0)

        self.demands = self.df_spec_pre.groupby('group_order_no')[self.demand_column].sum().to_dict()
        self.base_item_widths = self.df_spec_pre.set_index('group_order_no')[self.width_column].to_dict()

        self.patterns = []
        self.pattern_keys = set()

        self.item_info = {}
        self.item_composition = {}
        self.item_piece_count = {}
        self.base_items = []
        self.composite_items = []

        self._prepare_items()

        self.items = list(self.item_info.keys())
        self.max_demand = max(self.demands.values()) if self.demands else 1

    def _prepare_items(self):
        for item, width in self.base_item_widths.items():
            if width <= 0:
                continue
            self.item_info[item] = width
            self.item_composition[item] = {item: 1}
            self.item_piece_count[item] = 1
            self.base_items.append(item)

        # Add pure composite items first
        for base_item, base_width in self.base_item_widths.items():
            if base_width <= 0:
                continue
            for num_repeats in range(self.composite_min, self.composite_max + 1):
                composite_width = base_width * num_repeats
                composite_w_with_trim = composite_width + self.sl_trim
                
                if self.min_sl_width <= composite_w_with_trim <= self.max_sl_width:
                    composition = {base_item: num_repeats}
                    name = self._make_composite_name(composition)
                    if name not in self.item_info:
                        self._register_composite_item(name, composite_w_with_trim, composition, num_repeats)

        max_combo_pieces = min(self.max_pieces, self.composite_max)
        min_combo_pieces = min(max_combo_pieces, self.composite_min)
        if min_combo_pieces > max_combo_pieces:
            return

        base_candidates = sorted(
            self.base_items,
            key=lambda key: (-self.demands.get(key, 0), -self.item_info[key])
        )[:COMPOSITE_BASE_CANDIDATES]
        base_candidates = sorted(base_candidates, key=lambda key: self.item_info[key])

        seen_compositions = set()
        composite_cap = COMPOSITE_GENERATION_LIMIT

        def backtrack(start_idx, composition, total_width, total_pieces):
            nonlocal composite_cap
            
            composite_w_with_trim = total_width + self.sl_trim

            if total_pieces >= min_combo_pieces and self.min_sl_width <= composite_w_with_trim <= self.max_sl_width:
                key = tuple(sorted(composition.items()))
                if key not in seen_compositions:
                    seen_compositions.add(key)
                    composition_snapshot = dict(composition)
                    name = self._make_composite_name(composition_snapshot)
                    if name not in self.item_info:
                        self._register_composite_item(name, composite_w_with_trim, composition_snapshot, total_pieces)
                        composite_cap -= 1
                        if composite_cap <= 0:
                            return True
            
            if total_pieces >= max_combo_pieces:
                return False

            for idx in range(start_idx, len(base_candidates)):
                base_item = base_candidates[idx]
                width = self.item_info[base_item]
                if width <= 0:
                    continue
                
                if (total_width + width + self.sl_trim) > self.max_sl_width:
                    continue
                
                composition[base_item] = composition.get(base_item, 0) + 1
                should_stop = backtrack(idx, composition, total_width + width, total_pieces + 1)
                composition[base_item] -= 1
                if composition[base_item] == 0:
                    del composition[base_item]
                if should_stop:
                    return True
            return False

        backtrack(0, {}, 0, 0)

    def _clear_patterns(self):
        self.patterns = []
        self.pattern_keys = set()

    def _rebuild_pattern_cache(self):
        self.pattern_keys = {frozenset(p.items()) for p in self.patterns}

    def _small_units_for_item(self, item_name):
        """
        Returns the count of small-width (<= limit) rolls contributed by an item.
        Only pure single-width items (단폭) are subject to the cap; composite items are ignored.
        """
        piece_count = self.item_piece_count.get(item_name, 0)
        if piece_count != 1:
            return 0
        composition = self.item_composition.get(item_name, {})
        if len(composition) != 1:
            return 0
        base_item, qty = next(iter(composition.items()))
        base_width = self.base_item_widths.get(base_item, 0)
        if 0 < base_width <= self.small_width_limit:
            return qty
        return 0

    def _count_small_width_units(self, pattern):
        """패턴 내부의 소폭(기준 이하) 롤 수를 계산합니다."""
        return sum(self._small_units_for_item(item_name) * count for item_name, count in pattern.items())

    def _is_mixed_composite(self, item_name):
        composition = self.item_composition.get(item_name, {})
        if not composition:
            return False
        if len(composition) == 1:
            return False
        return True

    def _count_mixed_composites(self, pattern):
        return sum(
            count for item_name, count in pattern.items()
            if self._is_mixed_composite(item_name)
        )

    def _add_pattern(self, pattern):
        key = frozenset(pattern.items())
        if key in self.pattern_keys:
            return False
        if self._count_small_width_units(pattern) > self.max_small_width_per_pattern:
            return False
        self.patterns.append(dict(pattern))
        self.pattern_keys.add(key)
        return True

    def _count_pattern_pieces(self, pattern):
        return sum(self.item_piece_count[item] * count for item, count in pattern.items())

    def _count_pattern_composite_units(self, pattern):
        return sum(max(0, self.item_piece_count[item] - 1) * count for item, count in pattern.items())

    def _effective_demand(self, item):
        composition = self.item_composition[item]
        return sum(self.demands.get(base_item, 0) * qty for base_item, qty in composition.items())

    def _format_width(self, value):
        return int(value) if abs(value - int(value)) < 1e-6 else round(value, 2)

    def _format_item_label(self, item_name):
        composition = self.item_composition[item_name]
        item_width = self.item_info[item_name]
        if len(composition) == 1:
            base_item, qty = next(iter(composition.items()))
            base_width = self.base_item_widths.get(base_item, 0)
            if qty <= 1:
                return str(self._format_width(item_width))
            return f"{self._format_width(item_width)}({self._format_width(base_width)}*{qty})"
        parts = []
        for base_item, qty in sorted(composition.items()):
            base_width = self.base_item_widths.get(base_item, 0)
            parts.append(f"{self._format_width(base_width)}*{qty}")
        return f"{self._format_width(item_width)}(" + '+'.join(parts) + ")"

    def _register_composite_item(self, name, width, composition, piece_count):
        self.item_info[name] = width
        self.item_composition[name] = dict(composition)
        self.item_piece_count[name] = piece_count
        self.composite_items.append(name)

    def _make_composite_name(self, composition):
        items = sorted(composition.items())
        if len(items) == 1:
            item, qty = items[0]
            return f"{item}__x{qty}"
        parts = [f"{item}x{qty}" for item, qty in items]
        return f"mix__{'__'.join(parts)}"

    def _generate_initial_patterns(self):
        self._clear_patterns()
        if not self.items:
            return

        sorted_by_demand = sorted(
            self.items,
            key=lambda i: (self._effective_demand(i), self.item_info[i]),
            reverse=True
        )
        sorted_by_width_desc = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda i: self.item_info[i])

        heuristics = [sorted_by_demand, sorted_by_width_desc, sorted_by_width_asc]

        for sorted_items in heuristics:
            for item in sorted_items:
                item_width = self.item_info[item]
                if item_width <= 0 or item_width > self.max_width:
                    continue
                item_small_units = self._small_units_for_item(item)
                if item_small_units > self.max_small_width_per_pattern:
                    continue

                pattern = {item: 1}
                current_width = item_width
                current_pieces = 1
                current_small_units = item_small_units

                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    best_fit_item = None
                    for candidate in sorted_items:
                        candidate_width = self.item_info[candidate]
                        if candidate_width <= 0:
                            continue
                        if candidate_width > remaining_width:
                            continue
                        candidate_small = self._small_units_for_item(candidate)
                        if current_small_units + candidate_small > self.max_small_width_per_pattern:
                            continue
                        best_fit_item = candidate
                        break
                    if not best_fit_item:
                        break
                    pattern[best_fit_item] = pattern.get(best_fit_item, 0) + 1
                    current_width += self.item_info[best_fit_item]
                    current_pieces += 1
                    current_small_units += self._small_units_for_item(best_fit_item)

                while current_width < self.min_width and current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    add_item = None
                    for candidate in sorted_by_width_desc:
                        candidate_width = self.item_info[candidate]
                        if candidate_width <= 0:
                            continue
                        if candidate_width > remaining_width:
                            continue
                        candidate_small = self._small_units_for_item(candidate)
                        if current_small_units + candidate_small > self.max_small_width_per_pattern:
                            continue
                        add_item = candidate
                        break
                    if not add_item:
                        break
                    pattern[add_item] = pattern.get(add_item, 0) + 1
                    current_width += self.item_info[add_item]
                    current_pieces += 1
                    current_small_units += self._small_units_for_item(add_item)

                if current_width >= self.min_width and current_pieces <= self.max_pieces:
                    self._add_pattern(pattern)

        for item in self.items:
            item_width = self.item_info[item]
            if item_width <= 0:
                continue

            max_repeat_width = int(self.max_width // item_width) if item_width > 0 else 0
            num_items = min(max_repeat_width, self.max_pieces)
            while num_items > 0:
                total_width = item_width * num_items
                small_units = self._small_units_for_item(item) * num_items
                if small_units > self.max_small_width_per_pattern:
                    num_items -= 1
                    continue
                if total_width >= self.min_width and num_items <= self.max_pieces:
                    if self._add_pattern({item: num_items}):
                        break
                num_items -= 1

        covered_items = {name for pattern in self.patterns for name in pattern}
        uncovered_items = set(self.items) - covered_items

        for item in uncovered_items:
            item_width = self.item_info[item]
            if item_width <= 0 or item_width > self.max_width:
                    continue
            item_small_units = self._small_units_for_item(item)
            if item_small_units > self.max_small_width_per_pattern:
                continue

            pattern = {item: 1}
            current_width = item_width
            current_pieces = 1
            current_small_units = item_small_units

            while current_width < self.min_width and current_pieces < self.max_pieces:
                remaining_width = self.max_width - current_width
                addition = None
                for candidate in sorted_by_width_desc:
                    candidate_width = self.item_info[candidate]
                    if candidate_width <= 0:
                        continue
                    if candidate_width > remaining_width:
                        continue
                    candidate_small = self._small_units_for_item(candidate)
                    if current_small_units + candidate_small > self.max_small_width_per_pattern:
                        continue
                    addition = candidate
                    break
                if not addition:
                    break
                pattern[addition] = pattern.get(addition, 0) + 1
                current_width += self.item_info[addition]
                current_pieces += 1
                current_small_units += self._small_units_for_item(addition)

            if current_width >= self.min_width and current_pieces <= self.max_pieces:
                self._add_pattern(pattern)

    def _generate_all_patterns(self):
        all_patterns = []
        seen_patterns = set()
        item_list = list(self.items)

        def add_pattern(pattern):
            if not pattern:
                return
            key = frozenset(pattern.items())
            if key in seen_patterns:
                return
            total_width = sum(self.item_info[item] * count for item, count in pattern.items())
            total_pieces = sum(pattern.values())
            if (
                self.min_width <= total_width <= self.max_width
                and total_pieces <= self.max_pieces
                and self._count_small_width_units(pattern) <= self.max_small_width_per_pattern
            ):
                all_patterns.append(dict(pattern))
                seen_patterns.add(key)

        def backtrack(start_index, current_pattern, current_width, current_pieces, current_small_units):
            add_pattern(current_pattern)
            if current_pieces >= self.max_pieces or start_index >= len(item_list):
                return

            for i in range(start_index, len(item_list)):
                item = item_list[i]
                item_width = self.item_info[item]
                if item_width <= 0:
                    continue
                if current_width + item_width > self.max_width:
                    continue
                if current_pieces + 1 > self.max_pieces:
                    continue
                item_small = self._small_units_for_item(item)
                if current_small_units + item_small > self.max_small_width_per_pattern:
                    continue
                current_pattern[item] = current_pattern.get(item, 0) + 1
                backtrack(i, current_pattern, current_width + item_width, current_pieces + 1, current_small_units + item_small)
                current_pattern[item] -= 1
                if current_pattern[item] == 0:
                    del current_pattern[item]

        backtrack(0, {}, 0, 0, 0)
        self.patterns = all_patterns
        self._rebuild_pattern_cache()

    def _solve_master_problem(self, is_final_mip=False):
        solver_name = 'SCIP' if is_final_mip else 'GLOP'
        solver = pywraplp.Solver.CreateSolver(solver_name)
        if not solver:
            return None
        if is_final_mip and hasattr(solver, 'SetTimeLimit'):
            solver.SetTimeLimit(FINAL_MIP_TIME_LIMIT_MS)

        x = {
            j: (solver.IntVar if is_final_mip else solver.NumVar)(0, solver.infinity(), f'P_{j}')
            for j in range(len(self.patterns))
        }
        over_prod_vars = {item: solver.NumVar(0, solver.infinity(), f'Over_{item}') for item in self.demands}
        under_prod_vars = {
            item: solver.NumVar(0, max(0, self.demands[item]), f'Under_{item}') for item in self.demands
        }

        constraints = {}
        for item, demand in self.demands.items():
            production_expr = solver.Sum(
                sum(self.item_composition[item_name].get(item, 0) * count for item_name, count in self.patterns[j].items()) * x[j]
                for j in range(len(self.patterns))
            )
            constraints[item] = solver.Add(
                production_expr + under_prod_vars[item] == demand + over_prod_vars[item],
                f'demand_{item}'
            )

        pattern_trim = {
            j: self.max_width - sum(self.item_info[item] * count for item, count in pattern.items())
            for j, pattern in enumerate(self.patterns)
        }
        pattern_pieces = {
            j: sum(pattern.values()) # Use chunk count
            for j, pattern in enumerate(self.patterns)
        }
        pattern_composite_units = {
            j: self._count_pattern_composite_units(pattern)
            for j, pattern in enumerate(self.patterns)
        }
        pattern_mixed_counts = {
            j: self._count_mixed_composites(pattern)
            for j, pattern in enumerate(self.patterns)
        }

        total_trim_loss = solver.Sum(pattern_trim[j] * x[j] for j in range(len(self.patterns)))
        over_prod_terms = []
        base_demand = max(self.max_demand, 1)
        for item in self.demands:
            demand = max(self.demands[item], 1)
            weight = min(OVER_PROD_WEIGHT_CAP, base_demand / demand)
            over_prod_terms.append(OVER_PROD_PENALTY * weight * over_prod_vars[item])
        total_over_penalty = solver.Sum(over_prod_terms) if over_prod_terms else solver.Sum([])
        total_under_penalty = solver.Sum(UNDER_PROD_PENALTY * under_prod_vars[item] for item in self.demands)
        total_complexity_penalty = solver.Sum(
            self.pattern_complexity_penalty * pattern_pieces[j] * x[j] for j in range(len(self.patterns))
        )
        total_composite_penalty = solver.Sum(
            self.composite_penalty * pattern_composite_units[j] * x[j] for j in range(len(self.patterns))
        )
        total_mixed_penalty = solver.Sum(
            MIXED_COMPOSITE_PENALTY * pattern_mixed_counts[j] * x[j] for j in range(len(self.patterns))
        )

        solver.Minimize(total_trim_loss + total_over_penalty + total_under_penalty +
                        total_complexity_penalty + total_composite_penalty + total_mixed_penalty)

        # solver.Minimize(total_over_penalty + total_under_penalty +total_composite_penalty)

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None

        solution = {
            'objective': solver.Objective().Value(),
            'pattern_counts': {j: x[j].solution_value() for j in range(len(self.patterns))},
            'over_production': {item: over_prod_vars[item].solution_value() for item in self.demands},
            'under_production': {item: under_prod_vars[item].solution_value() for item in self.demands},
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
            item_pieces = 1 
            if item_width <= 0 or item_pieces > piece_limit:
                continue
            composition = self.item_composition[item]
            item_value = sum(duals.get(base, 0) * qty for base, qty in composition.items())
            if self._is_mixed_composite(item):
                item_value -= MIXED_COMPOSITE_PENALTY * 0.05
            if item_value <= 0:
                continue
            item_details.append((item, item_width, item_pieces, item_value))
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
                for item_name, item_width, item_pieces, item_value in item_details:
                    next_pieces = pieces + item_pieces
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
                if not pattern:
                    continue
                
                total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                if total_width < self.min_width or total_width > self.max_width:
                    continue
                if self._count_small_width_units(pattern) > self.max_small_width_per_pattern:
                    continue
                key = frozenset(pattern.items())
                if key in seen_patterns:
                    continue
                total_pieces = sum(pattern.values())
                composite_units = self._count_pattern_composite_units(pattern)
                adjusted_value = value - (self.pattern_complexity_penalty * total_pieces) - (self.composite_penalty * composite_units)
                if adjusted_value <= PATTERN_VALUE_THRESHOLD:
                    continue
                seen_patterns.add(key)
                candidate_patterns.append({
                    'pattern': pattern,
                    'value': adjusted_value,
                    'width': total_width,
                    'pieces': total_pieces
                })

        candidate_patterns.sort(key=lambda x: x['value'], reverse=True)
        return candidate_patterns[:CG_SUBPROBLEM_TOP_N]

    def run_optimize(self, start_prod_seq=0):
        if len(self.base_items) <= SMALL_PROBLEM_THRESHOLD:
            self._generate_all_patterns()
        else:
            self._generate_initial_patterns()

        if not self.patterns:
             return {"error": "초기 유효 패턴을 생성하지 못했습니다."}

        if not self.patterns:
            return {"error": "유효한 패턴을 생성하지 못했습니다."}

        self.patterns = [
            pattern for pattern in self.patterns
            if self.min_width <= sum(self.item_info[item] * count for item, count in pattern.items()) <= self.max_width
            and sum(pattern.values()) <= self.max_pieces
            and self._count_small_width_units(pattern) <= self.max_small_width_per_pattern
        ]
        if not self.patterns:
            return {"error": f"{self.min_width}mm 이상을 충족하는 패턴이 없습니다."}

        self._rebuild_pattern_cache()

        if len(self.base_items) > SMALL_PROBLEM_THRESHOLD:
            best_objective = None
            stagnation_count = 0
            for iteration in range(CG_MAX_ITERATIONS):
                master_solution = self._solve_master_problem(is_final_mip=False)
                if not master_solution:
                    break
                current_objective = master_solution['objective']
                if best_objective is None or current_objective < best_objective - 1e-6:
                    best_objective = current_objective
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                candidate_patterns = self._solve_subproblem(master_solution.get('duals', {}))
                added_pattern = False
                for candidate in candidate_patterns:
                    if self._add_pattern(candidate['pattern']):
                        added_pattern = True
                if not added_pattern:
                    break
                if stagnation_count >= CG_NO_IMPROVEMENT_LIMIT:
                    break
            self._rebuild_pattern_cache()

        final_solution = self._solve_master_problem(is_final_mip=True)
        if not final_solution:
            return {"error": f"최종 해를 찾지 못했습니다. {self.min_width}mm 이상을 충족하는 주문이 부족합니다."}

        return self._format_results(final_solution, start_prod_seq)

    def _format_results(self, final_solution, start_prod_seq=0):
        result_patterns = []
        pattern_details_for_db = []
        pattern_roll_details_for_db = []
        composite_usage = []

        production_counts = {item: 0 for item in self.demands}
        prod_seq = start_prod_seq

        for j, count in final_solution['pattern_counts'].items():
            if count < 0.99:
                continue
            pattern_dict = self.patterns[j]
            roll_count = int(round(count))
            prod_seq += 1

            sorted_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)

            pattern_labels = []
            total_width = 0
            widths_for_db = []
            group_nos_for_db = []
            composite_meta_for_db = []
            roll_seq_counter = 0

            for item_name, num in sorted_items:
                item_width = self.item_info[item_name]
                composition = self.item_composition[item_name]
                total_width += item_width * num

                pattern_labels.extend([self._format_item_label(item_name)] * num)
                widths_for_db.extend([item_width] * num)
                primary_group = next(iter(composition.keys()))
                group_nos_for_db.extend([primary_group] * num)
                composite_meta_for_db.extend([dict(composition)] * num)

                composite_units = max(0, self.item_piece_count[item_name] - 1) * num
                if composite_units > 0:
                    composite_usage.append({
                        'prod_seq': prod_seq,
                        'item': item_name,
                        'composite_width': item_width,
                        'components': dict(composition),
                        'count': roll_count * num
                    })

                for base_item, qty in composition.items():
                    production_counts[base_item] += roll_count * num * qty

                for _ in range(num):
                    roll_seq_counter += 1
                    expanded_widths = []
                    expanded_groups = []
                    for base_item, qty in composition.items():
                        base_width = self.base_item_widths.get(base_item, 0)
                        if base_width <= 0:
                            base_width = item_width / max(1, self.item_piece_count[item_name])
                        expanded_widths.extend([base_width] * qty)
                        expanded_groups.extend([base_item] * qty)

                    pattern_roll_details_for_db.append({
                        'rollwidth': item_width,
                        'widths': (expanded_widths + [0] * 7)[:7],
                        'group_nos': (expanded_groups + [''] * 7)[:7],
                        'count': roll_count,
                        'prod_seq': prod_seq,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'R',
                    })

            loss = self.max_width - total_width

            result_patterns.append({
                'pattern': ', '.join(pattern_labels),
                'pattern_width': total_width,
                'loss_per_roll': loss,
                'count': roll_count,
                'prod_seq': prod_seq,
                'rs_gubun': 'R',
            })

            pattern_details_for_db.append({
                'widths': (widths_for_db + [0] * 8)[:8],
                'group_nos': (group_nos_for_db + [''] * 8)[:8],
                'count': roll_count,
                'prod_seq': prod_seq,
                'composite_map': composite_meta_for_db,
                'rs_gubun': 'R',
            })
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['pattern', 'pattern_width', 'count', 'loss_per_roll']]

        df_demand = pd.DataFrame.from_dict(self.demands, orient='index', columns=['요구롤수'])
        df_demand.index.name = 'group_order_no'

        df_production = pd.DataFrame.from_dict(production_counts, orient='index', columns=['생산롤수'])
        df_production.index.name = 'group_order_no'

        df_summary = df_demand.join(df_production, how='outer').fillna(0)
        df_summary['과부족량'] = df_summary['생산롤수'] - df_summary['요구롤수']

        info_cols = ['group_order_no', self.width_column, '롤길이', '품질', '출고예정수', 'quality_grade', 'export_yn']
        available_info_cols = [c for c in info_cols if c in self.df_spec_pre.columns]
        group_info_df = self.df_spec_pre[available_info_cols].drop_duplicates(subset=['group_order_no'])

        fulfillment_summary = pd.merge(group_info_df, df_summary.reset_index(), on='group_order_no', how='right')
        rename_map = {
            self.width_column: '가로',
            '롤길이': '세로'
        }
        fulfillment_summary = fulfillment_summary.rename(
            columns={k: v for k, v in rename_map.items() if k in fulfillment_summary.columns}
        )

        final_cols = ['group_order_no', '가로', '세로', '출고예정수', '품질', '요구롤수', '생산롤수', '과부족량']
        final_cols = [c for c in final_cols if c in fulfillment_summary.columns]
        if final_cols:
            fulfillment_summary = fulfillment_summary[final_cols]

        return {
            "pattern_result": df_patterns.sort_values('count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "composite_usage": composite_usage,
            "last_prod_seq": prod_seq
        }

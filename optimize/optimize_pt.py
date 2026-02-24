"""
===============================================================================
OptimizePt Module - 롤지+쉬트지 통합 최적화 모듈
===============================================================================

[모듈 개요]
roll_optimize_ca.py의 롤지 복합폭 로직과 sheet_optimize_st.py의 쉬트지 로직을
통합하여 롤지와 쉬트지 오더를 한꺼번에 지폭조합 최적화하는 모듈입니다.

[핵심 규칙]
- 기본적으로 롤지와 쉬트지는 별도의 패턴으로 구성
- 롤지오더만으로 패턴을 구성할 수 없거나, 쉬트지 복합롤 1개 추가로 
  롤지 전체를 최적화할 수 있는 경우에만 혼합 허용
- 쉬트지 복합롤이 여러 개인 패턴에는 롤지 추가 금지
- 롤지 1폭 복합롤은 ww_trim_size 미적용, 2폭 이상일 때만 트림 적용
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

# === 페널티 값 ===
OVER_PROD_PENALTY = 500000.0
UNDER_PROD_PENALTY = 1000000.0
PATTERN_COUNT_PENALTY = 5000.0
SINGLE_STRIP_PENALTY = 50000.0
PATTERN_COMPLEXITY_PENALTY = 1.0
PIECE_COUNT_PENALTY = 100
COMPOSITE_USAGE_PENALTY = 0
MIXED_COMPOSITE_PENALTY = 50.0
OVER_PROD_WEIGHT_CAP = 6.0

# === 알고리즘 파라미터 ===
MIN_PIECES_PER_PATTERN = 1
SMALL_PROBLEM_THRESHOLD = 10
SOLVER_TIME_LIMIT_MS = 180000
CG_MAX_ITERATIONS = 200
CG_NO_IMPROVEMENT_LIMIT = 25
CG_SUBPROBLEM_TOP_N = 1
PATTERN_VALUE_THRESHOLD = 1.0 + 1e-6

# === 복합폭 파라미터 ===
COMPOSITE_BASE_CANDIDATES = 20
COMPOSITE_GENERATION_LIMIT = 2000
SMALL_WIDTH_LIMIT = 480
MAX_SMALL_WIDTH_PER_PATTERN = 2
ALLOW_DIFF_LENGTH_COMPOSITE = 'N'
TWO_STAGE_TOLERANCE = 0.05


class OptimizePt:
    """롤지+쉬트지 통합 최적화 클래스."""

    def __init__(
            self,
            db=None, plant=None, pm_no=None, schedule_unit=None,
            lot_no=None, version=None,
            paper_type=None, b_wgt=0, color=None,
            p_type=None, p_wgt=0, p_color=None, p_machine=None,
            # 롤지 전용
            df_roll_orders=None,
            coating_yn='N',
            # 쉬트지 전용
            df_sheet_orders=None,
            min_sheet_roll_length=None, max_sheet_roll_length=None,
            std_roll_cnt=None, sheet_trim=None,
            min_sc_width=None, max_sc_width=None,
            # 공통 제약
            min_width=0, max_width=1000, max_pieces=8,
            min_cm_width=0, max_cm_width=0, max_sl_count=5,
            ww_trim_size_sheet=0, ww_trim_size=0,
            num_threads=4, time_limit=180000
    ):
        self.db = db
        self.plant = plant
        self.pm_no = pm_no
        self.schedule_unit = schedule_unit
        self.lot_no = lot_no
        self.version = version
        self.paper_type = paper_type
        self.b_wgt = float(b_wgt) if b_wgt else 0
        self.color = color
        self.p_type = p_type
        self.p_wgt = float(p_wgt) if p_wgt else 0
        self.p_color = p_color
        self.p_machine = p_machine
        self.coating_yn = coating_yn
        self.num_threads = num_threads
        self.solver_time_limit_ms = time_limit

        self.max_width = int(max_width) if max_width else 0
        self.min_width = int(min_width) if min_width else 0
        self.max_pieces = int(max_pieces) if max_pieces else 8
        self.min_pieces = MIN_PIECES_PER_PATTERN

        # 복합폭(CM) 제약
        self.min_cm_width = int(min_cm_width) if min_cm_width else 0
        self.max_cm_width = int(max_cm_width) if max_cm_width else 0
        self.max_sl_count = int(max_sl_count) if max_sl_count else 5
        self.ww_trim_size_sheet = int(ww_trim_size_sheet) if ww_trim_size_sheet else 0
        self.sl_trim = int(ww_trim_size) if ww_trim_size else 0
        self.min_sl_width = self.min_cm_width
        self.max_sl_width = self.max_cm_width

        # 쉬트지 전용
        self.min_sheet_roll_length = min_sheet_roll_length
        self.max_sheet_roll_length = max_sheet_roll_length
        self.std_roll_cnt = std_roll_cnt
        self.sheet_trim = sheet_trim or 0
        self.min_sc_width = min_sc_width
        self.max_sc_width = max_sc_width

        # 복합폭 설정
        self.composite_min = 1
        self.composite_max = self.max_sl_count
        self.composite_penalty = COMPOSITE_USAGE_PENALTY
        self.pattern_count_penalty = PATTERN_COUNT_PENALTY
        self.small_width_limit = SMALL_WIDTH_LIMIT
        self.max_small_width_per_pattern = MAX_SMALL_WIDTH_PER_PATTERN

        # 데이터 저장소 초기화
        self.item_info = {}
        self.item_composition = {}
        self.item_piece_count = {}
        self.item_rs_gubun = {}  # R/S 구분

        self.roll_items = []
        self.sheet_items = []
        self.base_items = []
        self.composite_items = []

        # 롤지 수요
        self.roll_demands = {}
        self.roll_base_item_widths = {}
        self.roll_item_rolls_per_pattern = {}
        self.roll_item_roll_lengths = {}
        self.roll_std_length = 0
        self.roll_pattern_length = 0
        self.has_roll_orders = False

        # 쉬트지 수요
        self.sheet_demands_in_meters = {}
        self.sheet_order_sheet_lengths = {}
        self.sheet_demands_in_rolls = {}
        self.sheet_order_widths = []
        self.has_sheet_orders = False

        self.df_roll_orders = None
        self.df_sheet_orders = None

        # === 롤지 데이터 준비 ===
        if df_roll_orders is not None and not df_roll_orders.empty:
            self.has_roll_orders = True
            self.df_roll_orders = df_roll_orders.copy()
            self._prepare_roll_data()

        # === 쉬트지 데이터 준비 ===
        if df_sheet_orders is not None and not df_sheet_orders.empty:
            self.has_sheet_orders = True
            self.df_sheet_orders = df_sheet_orders.copy()
            self._prepare_sheet_data()

        # === 아이템 준비 ===
        self._prepare_all_items()

        self.items = list(self.item_info.keys())
        self.max_demand = max(self.roll_demands.values()) if self.roll_demands else 1

        self.patterns = []
        self.pattern_keys = set()

        logging.info(f"--- 통합 최적화 초기화 완료: 롤지={self.has_roll_orders}, 쉬트지={self.has_sheet_orders} ---")
        logging.info(f"--- 롤지 아이템: {len(self.roll_items)}개, 쉬트지 아이템: {len(self.sheet_items)}개 ---")
        logging.info(f"--- 전체 아이템: {len(self.items)}개 ---")

    # ================================================================
    # 데이터 준비
    # ================================================================

    def _prepare_roll_data(self):
        """롤지 주문 데이터를 준비합니다."""
        df = self.df_roll_orders

        demand_col = next((c for c in ['주문수량', '주문롤수', 'order_roll_cnt'] if c in df.columns), None)
        width_col = next((c for c in ['지폭', 'width'] if c in df.columns), None)
        if not demand_col or not width_col:
            raise KeyError("df_roll_orders에 주문수량/지폭 컬럼이 필요합니다.")

        df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce').fillna(0)
        df[width_col] = pd.to_numeric(df[width_col], errors='coerce').fillna(0)

        self.roll_demands = df.groupby('group_order_no')[demand_col].sum().to_dict()
        self.roll_base_item_widths = df.set_index('group_order_no')[width_col].to_dict()

        if 'std_length' in df.columns:
            self.roll_std_length = int(pd.to_numeric(df['std_length'].iloc[0], errors='coerce') or 0)
        self.roll_pattern_length = self.roll_std_length if self.roll_std_length > 0 else 0

        for _, row in df.drop_duplicates(subset=['group_order_no']).iterrows():
            gno = row['group_order_no']
            std_len = int(pd.to_numeric(row.get('std_length', 0), errors='coerce') or 0)
            roll_len = int(pd.to_numeric(row.get('롤길이', 0), errors='coerce') or 0)
            self.roll_item_roll_lengths[gno] = roll_len
            self.roll_item_rolls_per_pattern[gno] = std_len // roll_len if std_len > 0 and roll_len > 0 else 1

        self.roll_width_col = width_col
        self.roll_demand_col = demand_col

    def _prepare_sheet_data(self):
        """쉬트지 주문 데이터를 준비합니다."""
        df = self.df_sheet_orders.copy()
        df['지폭'] = df['가로'] if '가로' in df.columns else df.get('지폭', df.get('width', 0))

        df.columns = [c.lower() for c in df.columns]
        rename_map = {'width': '지폭', 'length': '세로', 'order_ton_cnt': '주문톤', '가로': '지폭'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        def calc_meters(row):
            w = row.get('지폭', 0)
            l = row.get('세로', 0)
            t = row.get('주문톤', 0)
            if self.b_wgt <= 0 or w <= 0 or l <= 0 or t <= 0:
                return 0
            sw = (self.b_wgt * w * l) / 1000000
            if sw <= 0:
                return 0
            return (t * 1000000 / sw) * (l / 1000)

        df['meters'] = df.apply(calc_meters, axis=1)
        df['demand_key'] = list(zip(df['지폭'].astype(int), df['세로'].astype(int)))

        self.sheet_demands_in_meters = df.groupby('demand_key')['meters'].sum().to_dict()
        self.sheet_order_sheet_lengths = df.groupby('demand_key')['세로'].first().to_dict()
        self.sheet_order_widths = list(self.sheet_demands_in_meters.keys())

        std_len = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2
        self.sheet_demands_in_rolls = {k: m / std_len for k, m in self.sheet_demands_in_meters.items()}

        self.df_sheet_orders = df

        logging.info(f"--- 쉬트지 (지폭,세로)별 수요 ---")
        for key, meters in self.sheet_demands_in_meters.items():
            logging.info(f"  {key}: {meters:.2f}m ({self.sheet_demands_in_rolls[key]:.2f}롤)")

    # ================================================================
    # 아이템 준비
    # ================================================================

    def _prepare_all_items(self):
        """롤지 + 쉬트지 아이템을 모두 준비합니다."""
        if self.has_roll_orders:
            self._prepare_roll_items()
        if self.has_sheet_orders:
            self._prepare_sheet_items()

        logging.info(f"\n{'='*60}")
        logging.info(f"[아이템 생성 결과] 롤지: {len(self.roll_items)}개, 쉬트지: {len(self.sheet_items)}개")
        logging.info(f"{'='*60}")

    def _prepare_roll_items(self):
        """롤지 아이템을 준비합니다. 1폭은 트림 미적용, 2폭+ 트림 적용."""
        all_base = {}
        for item, width in self.roll_base_item_widths.items():
            if width <= 0:
                continue
            all_base[item] = width

        # 1폭(단폭) - ww_trim_size 미적용
        for item, width in all_base.items():
            if self.min_sl_width <= width <= self.max_sl_width:
                self.item_info[item] = width  # 트림 없음
                self.item_composition[item] = {item: 1}
                self.item_piece_count[item] = 1
                self.item_rs_gubun[item] = 'R'
                self.base_items.append(item)
                self.roll_items.append(item)

        # 순수 복합폭 (동일 규격 반복) - 2폭+ 트림 적용
        for base_item, base_width in all_base.items():
            for n in range(2, self.composite_max + 1):
                cw = base_width * n + self.sl_trim  # 2폭 이상 트림 적용
                if self.min_sl_width <= cw <= self.max_sl_width:
                    comp = {base_item: n}
                    name = self._make_composite_name(comp)
                    if name not in self.item_info:
                        self.item_info[name] = cw
                        self.item_composition[name] = dict(comp)
                        self.item_piece_count[name] = n
                        self.item_rs_gubun[name] = 'R'
                        self.composite_items.append(name)
                        self.roll_items.append(name)

        # 혼합 복합폭 (서로 다른 규격 조합) - 트림 적용
        base_candidates = sorted(
            all_base.keys(),
            key=lambda k: (-self.roll_demands.get(k, 0), -all_base[k])
        )[:COMPOSITE_BASE_CANDIDATES]
        base_candidates = sorted(base_candidates, key=lambda k: all_base[k])

        seen = set()
        cap = COMPOSITE_GENERATION_LIMIT

        def bt_roll(start_idx, comp, tw, tp, cur_rl):
            nonlocal cap
            cw_trim = tw + self.sl_trim  # 혼합은 항상 2폭+ 이므로 트림 적용
            if tp >= 2 and self.min_sl_width <= cw_trim <= self.max_sl_width:
                key = tuple(sorted(comp.items()))
                if key not in seen:
                    seen.add(key)
                    snap = dict(comp)
                    name = self._make_composite_name(snap)
                    if name not in self.item_info:
                        self.item_info[name] = cw_trim
                        self.item_composition[name] = snap
                        self.item_piece_count[name] = tp
                        self.item_rs_gubun[name] = 'R'
                        self.composite_items.append(name)
                        self.roll_items.append(name)
                        cap -= 1
                        if cap <= 0:
                            return True
            if tp >= self.composite_max:
                return False
            for idx in range(start_idx, len(base_candidates)):
                bi = base_candidates[idx]
                w = all_base.get(bi, 0)
                if w <= 0 or (tw + w + self.sl_trim) > self.max_sl_width:
                    continue
                crl = self.roll_item_roll_lengths.get(bi, 0)
                if ALLOW_DIFF_LENGTH_COMPOSITE == 'N' and cur_rl is not None and crl != cur_rl:
                    continue
                nrl = cur_rl if cur_rl is not None else crl
                comp[bi] = comp.get(bi, 0) + 1
                if bt_roll(idx, comp, tw + w, tp + 1, nrl):
                    comp[bi] -= 1
                    if comp[bi] == 0: del comp[bi]
                    return True
                comp[bi] -= 1
                if comp[bi] == 0: del comp[bi]
            return False

        bt_roll(0, {}, 0, 0, None)

        logging.info(f"[롤지 아이템] 단폭: {len(self.base_items)}개, 복합폭: {len([i for i in self.roll_items if i not in self.base_items])}개")

    def _prepare_sheet_items(self):
        """쉬트지 아이템을 준비합니다."""
        for key in self.sheet_order_widths:
            width, sheet_length = key if isinstance(key, tuple) else (key, 0)
            for i in range(1, 5):
                base_width = width * i + self.sheet_trim
                if not (self.min_sc_width <= base_width <= self.max_sc_width):
                    continue
                if base_width > self.max_width:
                    continue
                item_name = f"S_{width}_{sheet_length}x{i}"
                if item_name not in self.item_info:
                    self.item_info[item_name] = base_width
                    self.item_composition[item_name] = {key: i}
                    self.item_piece_count[item_name] = i
                    self.item_rs_gubun[item_name] = 'S'
                    self.sheet_items.append(item_name)

        logging.info(f"[쉬트지 아이템] {len(self.sheet_items)}개 생성")
        for item in self.sheet_items:
            logging.info(f"  {item}: {self.item_info[item]}mm")

    # ================================================================
    # 유틸리티
    # ================================================================

    def _make_composite_name(self, composition):
        items = sorted(composition.items())
        if len(items) == 1:
            item, qty = items[0]
            return f"{item}__x{qty}"
        return f"mix__{'__'.join(f'{i}x{q}' for i, q in items)}"

    def _effective_demand(self, item):
        comp = self.item_composition.get(item, {})
        total = 0
        for base, qty in comp.items():
            if self.item_rs_gubun.get(item, 'R') == 'R':
                total += self.roll_demands.get(base, 0) * qty
            else:
                total += self.sheet_demands_in_meters.get(base, 0) * qty
        return total

    def _small_units_for_item(self, item_name):
        if self.item_piece_count.get(item_name, 0) != 1:
            return 0
        comp = self.item_composition.get(item_name, {})
        if len(comp) != 1:
            return 0
        base_item, qty = next(iter(comp.items()))
        bw = self.roll_base_item_widths.get(base_item, 0)
        return qty if 0 < bw <= self.small_width_limit else 0

    def _count_small_width_units(self, pattern):
        return sum(self._small_units_for_item(i) * c for i, c in pattern.items())

    def _is_mixed_composite(self, item_name):
        return len(self.item_composition.get(item_name, {})) > 1

    def _count_mixed_composites(self, pattern):
        return sum(c for i, c in pattern.items() if self._is_mixed_composite(i))

    def _count_pattern_pieces(self, pat):
        return sum(self.item_piece_count.get(i, 1) * c for i, c in pat.items())

    def _count_pattern_composite_units(self, pat):
        return sum(max(0, self.item_piece_count.get(i, 1) - 1) * c for i, c in pat.items())

    def _format_width(self, v):
        return int(v) if abs(v - int(v)) < 1e-6 else round(v, 2)

    def _format_item_label(self, item_name):
        comp = self.item_composition[item_name]
        iw = self.item_info[item_name]
        if len(comp) == 1:
            base, qty = next(iter(comp.items()))
            if isinstance(base, tuple):
                bw = base[0]
            else:
                bw = self.roll_base_item_widths.get(base, iw)
            if qty <= 1:
                return str(self._format_width(iw))
            return f"{self._format_width(iw)}({self._format_width(bw)}*{qty})"
        parts = []
        for base, qty in sorted(comp.items()):
            if isinstance(base, tuple):
                bw = base[0]
            else:
                bw = self.roll_base_item_widths.get(base, 0)
            parts.append(f"{self._format_width(bw)}*{qty}")
        return f"{self._format_width(iw)}({'+'.join(parts)})"

    def _add_pattern(self, pattern):
        key = frozenset(pattern.items())
        if key in self.pattern_keys:
            return False
        if self._count_small_width_units(pattern) > self.max_small_width_per_pattern:
            return False
        self.patterns.append(dict(pattern))
        self.pattern_keys.add(key)
        return True

    def _clear_patterns(self):
        self.patterns = []
        self.pattern_keys = set()

    def _rebuild_pattern_cache(self):
        self.pattern_keys = {frozenset(p.items()) for p in self.patterns}

    def _get_pattern_rs_type(self, pattern):
        """패턴의 롤/쉬트 타입을 판별합니다."""
        has_roll = any(self.item_rs_gubun.get(i, 'R') == 'R' for i in pattern)
        has_sheet = any(self.item_rs_gubun.get(i, 'S') == 'S' for i in pattern)
        if has_roll and has_sheet:
            return 'M'  # Mixed
        elif has_sheet:
            return 'S'
        else:
            return 'R'

    def _count_sheet_items_in_pattern(self, pattern):
        """패턴 내 쉬트지 복합롤 개수."""
        return sum(c for i, c in pattern.items() if self.item_rs_gubun.get(i) == 'S')

    def _is_valid_mixed_pattern(self, pattern):
        """혼합 패턴의 유효성: 쉬트지 복합롤은 최대 1개만 허용."""
        return self._count_sheet_items_in_pattern(pattern) <= 1

    # ================================================================
    # 패턴 생성
    # ================================================================

    def _generate_all_patterns(self):
        """롤지/쉬트지/혼합 패턴을 생성합니다."""
        self._clear_patterns()

        # 1) 롤지 전용 패턴
        if self.has_roll_orders:
            self._generate_roll_patterns()
            roll_only_count = len(self.patterns)
            logging.info(f"[패턴 생성] 롤지 전용: {roll_only_count}개")

        # 2) 쉬트지 전용 패턴
        if self.has_sheet_orders:
            self._generate_sheet_patterns()
            sheet_count = len(self.patterns) - (roll_only_count if self.has_roll_orders else 0)
            logging.info(f"[패턴 생성] 쉬트지 전용: {sheet_count}개")

        # 3) 혼합 패턴 (조건부)
        if self.has_roll_orders and self.has_sheet_orders:
            self._generate_mixed_patterns()
            mixed_count = len(self.patterns) - roll_only_count - sheet_count
            logging.info(f"[패턴 생성] 혼합: {mixed_count}개")

        logging.info(f"[패턴 생성 완료] 총 {len(self.patterns)}개 패턴")

    def _generate_roll_patterns(self):
        """롤지 아이템만으로 패턴을 생성합니다 (BF + 휴리스틱)."""
        items = self.roll_items
        if not items:
            return

        if len(items) <= SMALL_PROBLEM_THRESHOLD:
            self._generate_patterns_bf(items)
        else:
            self._generate_patterns_heuristic(items)

    def _generate_sheet_patterns(self):
        """쉬트지 아이템만으로 패턴을 생성합니다. 동일 세로끼리만 조합."""
        by_length = {}
        for item in self.sheet_items:
            comp = self.item_composition[item]
            key = next(iter(comp.keys()))
            if isinstance(key, tuple):
                sheet_len = key[1]
            else:
                sheet_len = 0
            by_length.setdefault(sheet_len, []).append(item)

        for sheet_len, items in by_length.items():
            if len(items) <= SMALL_PROBLEM_THRESHOLD:
                self._generate_patterns_bf(items)
            else:
                self._generate_patterns_heuristic(items)

    def _generate_mixed_patterns(self):
        """혼합 패턴을 생성합니다. 쉬트지 복합롤 1개 + 롤지 아이템."""
        for sheet_item in self.sheet_items:
            sw = self.item_info[sheet_item]
            remaining = self.max_width - sw
            if remaining < 0:
                continue
            # 남은 공간에 롤지 아이템 채우기 (greedy)
            roll_items_sorted = sorted(
                self.roll_items,
                key=lambda i: self.item_info[i], reverse=True
            )
            for ri in roll_items_sorted:
                rw = self.item_info[ri]
                if rw <= remaining:
                    pat = {sheet_item: 1, ri: 1}
                    tw = sw + rw
                    if tw >= self.min_width:
                        self._add_pattern(pat)
                    # 같은 롤지 아이템 더 넣기
                    cnt = 1
                    while tw + rw <= self.max_width and cnt < self.max_pieces - 1:
                        cnt += 1
                        tw += rw
                        pat2 = {sheet_item: 1, ri: cnt}
                        if tw >= self.min_width:
                            self._add_pattern(pat2)

            # 2종류 롤지 아이템 조합
            for i, ri1 in enumerate(roll_items_sorted):
                rw1 = self.item_info[ri1]
                if sw + rw1 > self.max_width:
                    continue
                for ri2 in roll_items_sorted[i:]:
                    rw2 = self.item_info[ri2]
                    tw = sw + rw1 + rw2
                    if tw > self.max_width:
                        continue
                    if tw >= self.min_width:
                        if ri1 == ri2:
                            pat = {sheet_item: 1, ri1: 2}
                        else:
                            pat = {sheet_item: 1, ri1: 1, ri2: 1}
                        self._add_pattern(pat)

    def _generate_patterns_bf(self, item_list):
        """브루트포스 패턴 생성."""
        max_counts = {}
        for item in item_list:
            w = self.item_info[item]
            if w > 0:
                max_counts[item] = min(self.max_pieces, int(self.max_width / w))
            else:
                max_counts[item] = 0

        def bf(idx, pat, tw, tp):
            if tw >= self.min_width and pat and tp <= self.max_pieces:
                self._add_pattern(dict(pat))
            if idx >= len(item_list) or tp >= self.max_pieces:
                return
            item = item_list[idx]
            w = self.item_info[item]
            mc = max_counts[item]
            for c in range(0, mc + 1):
                nw = tw + w * c
                if nw > self.max_width:
                    break
                if c > 0:
                    pat[item] = c
                bf(idx + 1, pat, nw, tp + c)
                if c > 0:
                    del pat[item]

        bf(0, {}, 0, 0)

    def _generate_patterns_heuristic(self, item_list):
        """휴리스틱 패턴 생성 (First-Fit, Best-Fit)."""
        sorted_by_width = sorted(item_list, key=lambda i: self.item_info[i], reverse=True)
        sorted_by_demand = sorted(item_list, key=lambda i: (self._effective_demand(i), self.item_info[i]), reverse=True)

        for sorted_items in [sorted_by_width, sorted_by_demand]:
            for start_item in sorted_items:
                pat = {}
                tw = 0
                tp = 0
                for item in sorted_items:
                    w = self.item_info[item]
                    while tw + w <= self.max_width and tp < self.max_pieces:
                        pat[item] = pat.get(item, 0) + 1
                        tw += w
                        tp += 1
                if pat and tw >= self.min_width:
                    self._add_pattern(pat)

        # 단일 아이템 패턴
        for item in item_list:
            w = self.item_info[item]
            mc = min(self.max_pieces, int(self.max_width / w)) if w > 0 else 0
            for c in range(1, mc + 1):
                tw = w * c
                if self.min_width <= tw <= self.max_width:
                    self._add_pattern({item: c})

    # ================================================================
    # 통합 수요 생성
    # ================================================================

    def _build_unified_demands(self):
        """롤지/쉬트지 수요를 통합 인덱스로 만듭니다."""
        order_items = []  # 수요 키 목록
        demands = {}

        if self.has_roll_orders:
            for item in sorted(self.roll_demands.keys()):
                order_items.append(item)
                demands[item] = self.roll_demands[item]

        if self.has_sheet_orders:
            for key in sorted(self.sheet_demands_in_meters.keys()):
                order_items.append(key)
                demands[key] = self.sheet_demands_in_rolls.get(key, 0)

        return order_items, demands

    def _get_pattern_contribution(self, pattern, demand_key):
        """패턴이 특정 수요에 기여하는 양을 계산합니다."""
        total = 0
        for item, count in pattern.items():
            comp = self.item_composition.get(item, {})
            for base_item, qty in comp.items():
                if base_item == demand_key:
                    if self.item_rs_gubun.get(item) == 'R':
                        rpp = self.roll_item_rolls_per_pattern.get(base_item, 1)
                        total += count * qty * rpp
                    else:
                        total += count * qty
        return total

    # ================================================================
    # 솔버
    # ================================================================

    def _solve_master_problem(self, is_final_mip=True):
        """마스터 문제를 풀어 최적해를 구합니다."""
        try:
            return self._solve_gurobi(is_final_mip)
        except Exception as e:
            logging.warning(f"Gurobi 실패 ({e}), OR-Tools로 전환합니다.")
            return self._solve_ortools(is_final_mip)

    def _solve_gurobi(self, is_final_mip=True):
        """Gurobi로 마스터 문제를 풉니다."""
        order_items, demands = self._build_unified_demands()
        num_patterns = len(self.patterns)
        num_demands = len(order_items)

        if num_patterns == 0:
            return {'error': '패턴이 없습니다.'}

        model = gp.Model("OptimizePt")
        model.Params.OutputFlag = 0
        model.Params.Threads = self.num_threads
        model.Params.TimeLimit = self.solver_time_limit_ms / 1000

        vtype = GRB.INTEGER if is_final_mip else GRB.CONTINUOUS
        x = model.addVars(num_patterns, vtype=vtype, lb=0, name="x")

        over_prod = model.addVars(num_demands, vtype=GRB.CONTINUOUS, lb=0, name="over")
        under_prod = model.addVars(num_demands, vtype=GRB.CONTINUOUS, lb=0, name="under")

        # 수요 제약
        for d_idx, demand_key in enumerate(order_items):
            supply = gp.LinExpr()
            for j in range(num_patterns):
                contrib = self._get_pattern_contribution(self.patterns[j], demand_key)
                if contrib > 0:
                    supply.add(x[j], contrib)
            model.addConstr(
                supply + under_prod[d_idx] - over_prod[d_idx] == demands[demand_key],
                name=f"demand_{d_idx}"
            )

        # 패턴별 트림 로스
        pattern_trim = []
        for j, pat in enumerate(self.patterns):
            tw = sum(self.item_info[i] * c for i, c in pat.items())
            pattern_trim.append(self.max_width - tw)

        # 목적함수
        total_trim = gp.quicksum(pattern_trim[j] * x[j] for j in range(num_patterns))
        total_over = gp.quicksum(
            OVER_PROD_PENALTY * gp.min_(over_prod[d], OVER_PROD_WEIGHT_CAP * demands[order_items[d]])
            for d in range(num_demands)
        ) if False else gp.quicksum(OVER_PROD_PENALTY * over_prod[d] for d in range(num_demands))
        total_under = gp.quicksum(UNDER_PROD_PENALTY * under_prod[d] for d in range(num_demands))

        # 복합폭 페널티
        composite_units = [self._count_pattern_composite_units(self.patterns[j]) for j in range(num_patterns)]
        total_comp = gp.quicksum(self.composite_penalty * composite_units[j] * x[j] for j in range(num_patterns))

        # 혼합 복합폭 페널티  
        mixed_counts = [self._count_mixed_composites(self.patterns[j]) for j in range(num_patterns)]
        total_mixed = gp.quicksum(MIXED_COMPOSITE_PENALTY * mixed_counts[j] * x[j] for j in range(num_patterns))

        if is_final_mip:
            y = model.addVars(num_patterns, vtype=GRB.BINARY, name="y")
            M = max(demands.values(), default=1000) * 3
            for j in range(num_patterns):
                model.addConstr(x[j] <= M * y[j])
            total_pattern = gp.quicksum(self.pattern_count_penalty * y[j] for j in range(num_patterns))
        else:
            total_pattern = gp.LinExpr(0)

        model.setObjective(
            total_trim + total_over + total_under + total_comp + total_mixed + total_pattern,
            GRB.MINIMIZE
        )
        model.optimize()

        if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            return {'error': f'Gurobi 해 없음 (status={model.Status})'}

        solution = {j: x[j].X for j in range(num_patterns) if x[j].X > 0.01}
        total_trim_val = sum(pattern_trim[j] * solution.get(j, 0) for j in solution)
        total_under_val = sum(under_prod[d].X for d in range(num_demands))
        total_over_val = sum(over_prod[d].X for d in range(num_demands))

        duals = {}
        if not is_final_mip:
            for d_idx in range(num_demands):
                try:
                    duals[order_items[d_idx]] = model.getConstrByName(f"demand_{d_idx}").Pi
                except:
                    duals[order_items[d_idx]] = 0

        return {
            'pattern_counts': solution,
            'trim_loss': total_trim_val,
            'under_prod': total_under_val,
            'over_prod': total_over_val,
            'pattern_count': len(solution),
            'duals': duals
        }

    def _solve_ortools(self, is_final_mip=True):
        """OR-Tools fallback 솔버."""
        order_items, demands = self._build_unified_demands()
        num_patterns = len(self.patterns)
        num_demands = len(order_items)

        if num_patterns == 0:
            return {'error': '패턴이 없습니다.'}

        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        if not solver:
            return {'error': 'OR-Tools 솔버를 생성할 수 없습니다.'}

        solver.SetTimeLimit(self.solver_time_limit_ms)
        inf = solver.infinity()

        if is_final_mip:
            x = [solver.IntVar(0, inf, f'x_{j}') for j in range(num_patterns)]
        else:
            x = [solver.NumVar(0, inf, f'x_{j}') for j in range(num_patterns)]

        over_prod = [solver.NumVar(0, inf, f'over_{d}') for d in range(num_demands)]
        under_prod = [solver.NumVar(0, inf, f'under_{d}') for d in range(num_demands)]

        for d_idx, demand_key in enumerate(order_items):
            supply = solver.Sum([
                x[j] * self._get_pattern_contribution(self.patterns[j], demand_key)
                for j in range(num_patterns)
            ])
            solver.Add(supply + under_prod[d_idx] - over_prod[d_idx] == demands[demand_key])

        pattern_trim = []
        for j, pat in enumerate(self.patterns):
            tw = sum(self.item_info[i] * c for i, c in pat.items())
            pattern_trim.append(self.max_width - tw)

        obj = solver.Sum([pattern_trim[j] * x[j] for j in range(num_patterns)])
        obj += solver.Sum([OVER_PROD_PENALTY * over_prod[d] for d in range(num_demands)])
        obj += solver.Sum([UNDER_PROD_PENALTY * under_prod[d] for d in range(num_demands)])
        solver.Minimize(obj)
        status = solver.Solve()

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return {'error': 'OR-Tools 해 없음'}

        solution = {j: x[j].solution_value() for j in range(num_patterns) if x[j].solution_value() > 0.01}
        return {
            'pattern_counts': solution,
            'trim_loss': sum(pattern_trim[j] * solution.get(j, 0) for j in solution),
            'under_prod': sum(under_prod[d].solution_value() for d in range(num_demands)),
            'over_prod': sum(over_prod[d].solution_value() for d in range(num_demands)),
            'pattern_count': len(solution),
            'duals': {}
        }

    def solve_two_stage(self):
        """2단계 최적화: 1단계=비용, 2단계=패턴수 최소화."""
        order_items, demands = self._build_unified_demands()
        num_patterns = len(self.patterns)
        num_demands = len(order_items)

        if num_patterns == 0:
            return {'error': '패턴이 없습니다.'}

        model = gp.Model("OptimizePt_2Stage")
        model.Params.OutputFlag = 0
        model.Params.Threads = self.num_threads
        model.Params.TimeLimit = self.solver_time_limit_ms / 1000

        x = model.addVars(num_patterns, vtype=GRB.INTEGER, lb=0, name="x")
        y = model.addVars(num_patterns, vtype=GRB.BINARY, name="y")
        over_prod = model.addVars(num_demands, vtype=GRB.CONTINUOUS, lb=0, name="over")
        under_prod = model.addVars(num_demands, vtype=GRB.CONTINUOUS, lb=0, name="under")

        M = max(demands.values(), default=1000) * 3

        for d_idx, demand_key in enumerate(order_items):
            supply = gp.LinExpr()
            for j in range(num_patterns):
                contrib = self._get_pattern_contribution(self.patterns[j], demand_key)
                if contrib > 0:
                    supply.add(x[j], contrib)
            model.addConstr(supply + under_prod[d_idx] - over_prod[d_idx] == demands[demand_key])

        for j in range(num_patterns):
            model.addConstr(x[j] <= M * y[j])

        pattern_trim = [self.max_width - sum(self.item_info[i] * c for i, c in self.patterns[j].items()) for j in range(num_patterns)]
        composite_units = [self._count_pattern_composite_units(self.patterns[j]) for j in range(num_patterns)]
        mixed_counts = [self._count_mixed_composites(self.patterns[j]) for j in range(num_patterns)]

        # Stage 1: 비용 최소화
        cost_expr = (
            gp.quicksum(pattern_trim[j] * x[j] for j in range(num_patterns)) +
            gp.quicksum(OVER_PROD_PENALTY * over_prod[d] for d in range(num_demands)) +
            gp.quicksum(UNDER_PROD_PENALTY * under_prod[d] for d in range(num_demands)) +
            gp.quicksum(self.composite_penalty * composite_units[j] * x[j] for j in range(num_patterns)) +
            gp.quicksum(MIXED_COMPOSITE_PENALTY * mixed_counts[j] * x[j] for j in range(num_patterns))
        )
        model.setObjective(cost_expr, GRB.MINIMIZE)
        model.optimize()

        if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            return {'error': f'Stage1 실패 (status={model.Status})'}

        stage1_cost = model.ObjVal

        # Stage 2: 패턴수 최소화
        cutoff = stage1_cost * (1.0 + TWO_STAGE_TOLERANCE)
        model.addConstr(cost_expr <= cutoff, "Efficiency")
        model.setObjective(gp.quicksum(y[j] for j in range(num_patterns)), GRB.MINIMIZE)
        model.optimize()

        if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            return {'error': f'Stage2 실패 (status={model.Status})'}

        solution = {j: x[j].X for j in range(num_patterns) if x[j].X > 0.01}
        return {
            'pattern_counts': solution,
            'trim_loss': sum(pattern_trim[j] * solution.get(j, 0) for j in solution),
            'under_prod': sum(under_prod[d].X for d in range(num_demands)),
            'over_prod': sum(over_prod[d].X for d in range(num_demands)),
            'pattern_count': len(solution),
            'duals': {}
        }

    def _select_best_solution(self, sol1, sol2):
        """두 솔루션 중 더 나은 것을 선택합니다."""
        if 'error' in sol1:
            return sol2
        if 'error' in sol2:
            return sol1
        # 우선순위: under_prod → pattern_count → trim_loss
        if abs(sol1['under_prod'] - sol2['under_prod']) > 0.01:
            return sol1 if sol1['under_prod'] < sol2['under_prod'] else sol2
        if sol1['pattern_count'] != sol2['pattern_count']:
            return sol1 if sol1['pattern_count'] < sol2['pattern_count'] else sol2
        return sol1 if sol1['trim_loss'] <= sol2['trim_loss'] else sol2

    # ================================================================
    # 메인 실행
    # ================================================================

    def run_optimize(self, start_prod_seq=0):
        """최적화를 실행하고 결과를 반환합니다."""
        logging.info(f"\n{'='*60}")
        logging.info(f"[통합 최적화 시작] 롤지={self.has_roll_orders}, 쉬트지={self.has_sheet_orders}")
        logging.info(f"{'='*60}")

        if not self.has_roll_orders and not self.has_sheet_orders:
            return {'error': '롤지/쉬트지 오더가 모두 없습니다.'}

        # 1. 패턴 생성
        self._generate_all_patterns()
        if not self.patterns:
            return {'error': '유효한 패턴을 생성할 수 없습니다.'}

        # 2. 솔버 실행 (단일 MIP + 2단계)
        sol_mip = self._solve_master_problem(is_final_mip=True)

        try:
            sol_2stage = self.solve_two_stage()
        except Exception as e:
            logging.warning(f"2단계 최적화 실패: {e}")
            sol_2stage = {'error': str(e)}

        final_solution = self._select_best_solution(sol_mip, sol_2stage)

        if 'error' in final_solution:
            return final_solution

        logging.info(f"\n--- 최적해 선택 완료 ---")
        logging.info(f"패턴 수: {final_solution['pattern_count']}")
        logging.info(f"트림 로스: {final_solution['trim_loss']:.2f}")
        logging.info(f"과소 생산: {final_solution['under_prod']:.2f}")
        logging.info(f"과잉 생산: {final_solution['over_prod']:.2f}")

        # 3. 결과 포맷팅
        return self._format_results(final_solution, start_prod_seq)

    # ================================================================
    # 결과 포맷팅
    # ================================================================

    def _format_results(self, final_solution, start_prod_seq=0):
        """최적화 결과를 DB/출력용 형태로 변환합니다."""
        result_patterns = []
        pattern_details_for_db = []
        pattern_roll_details_for_db = []
        pattern_roll_cut_details_for_db = []
        composite_usage = []

        # 수요 충족 추적 (롤지)
        roll_production = {item: 0 for item in self.roll_demands} if self.has_roll_orders else {}
        # 수요 충족 추적 (쉬트지)
        sheet_production = {key: 0 for key in self.sheet_demands_in_meters} if self.has_sheet_orders else {}

        prod_seq = start_prod_seq

        # common_props
        common_props = self._get_common_props()

        for j, count in final_solution['pattern_counts'].items():
            if count < 0.99:
                continue
            pattern_dict = self.patterns[j]
            roll_count = int(round(count))
            prod_seq += 1
            pat_type = self._get_pattern_rs_type(pattern_dict)

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
                rs = self.item_rs_gubun.get(item_name, 'R')
                total_width += item_width * num

                pattern_labels.extend([self._format_item_label(item_name)] * num)
                widths_for_db.extend([item_width] * num)
                primary_group = next(iter(composition.keys()))
                group_nos_for_db.extend([primary_group] * num)
                composite_meta_for_db.extend([dict(composition)] * num)

                # 복합폭 사용 추적
                cu = max(0, self.item_piece_count[item_name] - 1) * num
                if cu > 0:
                    composite_usage.append({
                        'prod_seq': prod_seq, 'item': item_name,
                        'composite_width': item_width, 'components': dict(composition),
                        'count': roll_count * num
                    })

                # 생산량 추적
                for base_item, qty in composition.items():
                    if rs == 'R':
                        rpp = self.roll_item_rolls_per_pattern.get(base_item, 1)
                        roll_production[base_item] = roll_production.get(base_item, 0) + roll_count * num * qty * rpp
                    else:
                        sheet_production[base_item] = sheet_production.get(base_item, 0) + roll_count * num * qty

                # Roll Details
                for _ in range(num):
                    roll_seq_counter += 1
                    expanded_widths = []
                    expanded_groups = []
                    expanded_rs_gubuns = []

                    for base_item, qty in composition.items():
                        if isinstance(base_item, tuple):
                            bw = base_item[0]
                        else:
                            bw = self.roll_base_item_widths.get(base_item, 0)
                        if bw <= 0:
                            bw = item_width / max(1, self.item_piece_count[item_name])
                        expanded_widths.extend([bw] * qty)
                        expanded_groups.extend([base_item] * qty)
                        expanded_rs_gubuns.extend([rs] * qty)

                    roll_widths_list = (expanded_widths + [0] * 7)[:7]
                    roll_trim = item_width - sum(roll_widths_list)

                    pattern_length = self.roll_pattern_length
                    if rs == 'S' and self.min_sheet_roll_length:
                        pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2

                    pattern_roll_details_for_db.append({
                        'rollwidth': item_width,
                        'roll_widths': roll_widths_list,
                        'widths': roll_widths_list,
                        'group_nos': (expanded_groups + [''] * 7)[:7],
                        'rs_gubuns': (expanded_rs_gubuns + [''] * 7)[:7],
                        'count': roll_count,
                        'prod_seq': prod_seq,
                        'roll_seq': roll_seq_counter,
                        'pattern_length': pattern_length,
                        'loss_per_roll': roll_trim,
                        'rs_gubun': 'W' if rs == 'R' else 'S',
                        'sc_trim': self.ww_trim_size_sheet,
                        'sl_trim': self.sl_trim if self.coating_yn == 'Y' else 0,
                        **common_props
                    })

            loss = self.max_width - total_width

            result_patterns.append({
                'pattern': ', '.join(pattern_labels),
                'pattern_width': total_width,
                'loss_per_roll': loss,
                'count': roll_count,
                'prod_seq': prod_seq,
                'rs_gubun': 'W' if pat_type == 'R' else ('S' if pat_type == 'S' else 'M'),
                **common_props
            })

            pattern_length = self.roll_pattern_length
            if pat_type == 'S' and self.min_sheet_roll_length:
                pattern_length = (self.min_sheet_roll_length + self.max_sheet_roll_length) / 2

            pattern_details_for_db.append({
                'widths': (widths_for_db + [0] * 8)[:8],
                'group_nos': (group_nos_for_db + [''] * 8)[:8],
                'count': roll_count,
                'prod_seq': prod_seq,
                'composite_map': composite_meta_for_db,
                'pattern_length': pattern_length,
                'rs_gubun': 'W' if pat_type == 'R' else ('S' if pat_type == 'S' else 'M'),
                **common_props
            })

        # pattern_roll_cut_details_for_db 생성
        global_cut_seq = 0
        for roll_detail in pattern_roll_details_for_db:
            widths = roll_detail.get('widths', [])
            group_nos = roll_detail.get('group_nos', [])
            rs_gubuns = roll_detail.get('rs_gubuns', [])
            cut_seq_in_roll = 0
            for i, w in enumerate(widths):
                if w > 0:
                    global_cut_seq += 1
                    cut_seq_in_roll += 1
                    g_no = group_nos[i] if i < len(group_nos) else ''
                    rs_g = rs_gubuns[i] if i < len(rs_gubuns) else 'R'
                    pattern_roll_cut_details_for_db.append({
                        'prod_seq': roll_detail['prod_seq'],
                        'unit_no': roll_detail['prod_seq'],
                        'seq': global_cut_seq,
                        'roll_seq': roll_detail['roll_seq'],
                        'cut_seq': cut_seq_in_roll,
                        'rs_gubun': rs_g,
                        'width': w,
                        'group_no': g_no,
                        'weight': 0,
                        'pattern_length': roll_detail.get('pattern_length', 0),
                        'count': roll_detail['count'],
                        'p_lot': roll_detail.get('p_lot'),
                        'diameter': roll_detail.get('diameter'),
                        'core': roll_detail.get('core'),
                        'color': roll_detail.get('color'),
                        'luster': roll_detail.get('luster')
                    })

        # 패턴 결과 DataFrame
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['pattern', 'pattern_width', 'count', 'loss_per_roll']]

        # Fulfillment Summary 생성
        fulfillment_summary = self._build_fulfillment_summary(roll_production, sheet_production)

        return {
            "pattern_result": df_patterns.sort_values('count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "composite_usage": composite_usage,
            "last_prod_seq": prod_seq
        }

    def _get_common_props(self):
        """DB 공통 속성을 추출합니다."""
        props = {
            'p_lot': self.lot_no,
            'color': self.color or '',
            'order_pattern': ''
        }
        df = self.df_roll_orders if self.df_roll_orders is not None else self.df_sheet_orders
        if df is not None and not df.empty:
            first_row = df.iloc[0]
            def si(val):
                try: return int(val)
                except: return 0
            props['diameter'] = si(first_row.get('dia', 0))
            props['luster'] = si(first_row.get('luster', 0))
            props['core'] = si(first_row.get('core', 0))
            props['order_pattern'] = first_row.get('order_pattern', '')
        else:
            props['diameter'] = 0
            props['luster'] = 0
            props['core'] = 0
        return props

    def _build_fulfillment_summary(self, roll_production, sheet_production):
        """주문 충족 현황을 생성합니다."""
        rows = []

        # 롤지
        if self.has_roll_orders and self.df_roll_orders is not None:
            for gno, demand in self.roll_demands.items():
                produced = roll_production.get(gno, 0)
                width = self.roll_base_item_widths.get(gno, 0)
                roll_len = self.roll_item_roll_lengths.get(gno, 0)
                rows.append({
                    'group_order_no': gno, '지폭': width, '롤길이': roll_len,
                    '주문수량': demand, '생산롤수': produced,
                    '과부족(롤)': produced - demand, 'rs_gubun': 'R'
                })

        # 쉬트지
        if self.has_sheet_orders:
            for key, demand_m in self.sheet_demands_in_meters.items():
                produced = sheet_production.get(key, 0)
                demand_rolls = self.sheet_demands_in_rolls.get(key, 0)
                width = key[0] if isinstance(key, tuple) else key
                length = key[1] if isinstance(key, tuple) else 0
                rows.append({
                    'group_order_no': str(key), '지폭': width, '롤길이': length,
                    '주문수량': demand_rolls, '생산롤수': produced,
                    '과부족(롤)': produced - demand_rolls, 'rs_gubun': 'S'
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

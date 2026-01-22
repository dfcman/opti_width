import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import math
import random
import logging
import gurobipy as gp
from gurobipy import GRB

"""
[파일 설명: sheet_optimize.py]
쉬트(Sheet) 제품의 생산 최적화를 위한 핵심 모듈입니다.
Column Generation(열 생성) 알고리즘을 기반으로 하여, 주문 요구사항(지폭, 길이, 수량)을 만족시키면서
폐기물(Trim Loss)과 생산 비용을 최소화하는 최적의 절단 패턴을 산출합니다.
"""

# --- 최적화 설정 상수 ---
# 페널티 값
OVER_PROD_PENALTY  = 1000000.0    # 과생산에 대한 페널티   500000.0
UNDER_PROD_PENALTY = 500000.0     # 부족생산에 대한 페널티 10000000.0
PATTERN_COMPLEXITY_PENALTY = 10   # 패턴 복잡성에 대한 페널티
PIECE_COUNT_PENALTY = 100        # 패턴 내 복합롤 개수에 대한 페널티 

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 1      # 패턴에 포함될 수 있는 최소 폭(piece)의 수
SMALL_PROBLEM_THRESHOLD = 2     # 전체 탐색을 수행할 최대 주문 지폭 종류 수
CG_MAX_ITERATIONS = 1000         # 열 생성(Column Generation) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 50    # 개선 없는 경우, 열 생성 조기 종료 조건
CG_SUBPROBLEM_TOP_N = 10         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴
SC_MIN_WIDTH_FOR_MACHINE = 800



class SheetOptimize:
    def __init__(
            self,
            db=None,
            plant=None,
            pm_no=None,
            schedule_unit=None,
            lot_no=None,
            version=None,
            paper_type=None,
            b_wgt=0,
            color=None,
            df_spec_pre=None,
            min_width=0,
            max_width=1000,
            max_pieces=8,
            time_limit=300000,
            sheet_roll_length=0,
            sheet_trim=0,
            min_sc_width=0,
            max_sc_width=0,
            num_threads=4
    ):
        """
        [SheetOptimize 클래스 분석 및 기능 요약]

        이 클래스는 쉬트(Sheet) 제품의 생산 최적화를 담당합니다.
        주어진 원지(Roll) 폭과 설비 제약 조건을 고려하여, 주문받은 규격(지폭, 길이)을 
        가장 효율적으로 생산할 수 있는 절단 패턴(Cutting Pattern)과 생산 수량을 산출합니다.

        핵심 기능 및 알고리즘:
        1.  **수요 계산 (_calculate_demand_rolls)**:
            -   입력된 주문의 중량(톤)을 바탕으로 생산에 필요한 원지 롤(Roll) 수를 계산합니다.
        
        2.  **복합폭 아이템 생성 (_prepare_items)**:
            -   쉬트 생산 특성상 여러 장을 겹쳐서(예: 2매, 4매) 생산하므로, 
                단일 지폭이 아닌 '지폭 x 배수' 형태의 복합 아이템을 정의합니다.
            -   설비의 최소/최대 칼폭(Scissor Width) 제약을 고려합니다.

        3.  **초기 패턴 생성 (_generate_initial_patterns)**:
            -   Column Generation의 시작점이 될 초기 패턴 집합을 생성합니다.
            -   다양한 휴리스틱(수요순, 너비순 정렬 등)과 무작위 섞기를 결합한 First-Fit 알고리즘을 사용합니다.
            -   모든 주문을 커버할 수 있도록 폴백(Fallback) 로직도 포함합니다.

        4.  **열 생성법 (Column Generation) 기반 최적화 (run_optimize)**:
            -   대규모 조합 최적화 문제를 효율적으로 풀기 위해 반복적인 과정을 수행합니다.
            -   **Master Problem (_solve_master_problem_ilp)**: 
                현재 확보된 패턴들을 조합하여 비용(롤 수 + 페널티)을 최소화하는 해를 찾습니다. 
                (초기에는 LP로 완화하여 풀고, 마지막에 MIP로 정수해를 구합니다.)
            -   **Sub Problem (_solve_subproblem_dp)**: 
                Master Problem의 Dual Value(잠재 가격)를 활용하여, 현재 해를 개선할 수 있는(Reduced Cost < 0) 
                새로운 유망 패턴을 동적 계획법(DP)으로 찾아냅니다.
        
        5.  **패턴 통합 (_consolidate_patterns)**:
            -   생산 효율성을 위해, 작은 배수의 아이템(예: 1매) 여러 개를 큰 배수(예: 2매)로 묶을 수 있는지 확인하고 병합합니다.

        6.  **결과 생성 (_build_pattern_details, _format_results)**:
            -   최적화된 패턴별 생산 수량을 바탕으로 구체적인 작업 지시 데이터(생산 순서, 롤별 구성 등)를 생성합니다.
            -   과생산/부족생산 및 로스율 등의 지표를 집계합니다.
        """
        df_spec_pre['지폭'] = df_spec_pre['가로']

        self.num_threads = num_threads
        self.b_wgt = b_wgt
        self.sheet_roll_length = sheet_roll_length
        self.sheet_trim = sheet_trim
        self.original_max_width = max_width
        self.solver_time_limit_ms = time_limit  # 밀리초 단위 시간 제한
        
        # _calculate_demand_rolls에서 'rolls' 열이 추가된 데이터프레임을 받음
        self.df_orders, self.demands_in_rolls = self._calculate_demand_rolls(df_spec_pre)
        self.order_widths = list(self.demands_in_rolls.keys())  # group_order_no 기준

        # group_order_no별 정보 저장 (가로, max_allowed_width, SKID 정보 등)
        self.group_order_info = {}
        for g_no in self.order_widths:
            row = self.df_orders[self.df_orders['group_order_no'] == g_no].iloc[0]
            self.group_order_info[g_no] = {
                'width': int(row['가로']),  # Python int로 변환 (DB 호환)
                'length': int(row['세로']),  # Python int로 변환 (DB 호환)
                'max_allowed_width': int(row['max_allowed_width']),  # Python int로 변환
                'order_tons': float(self.df_orders[self.df_orders['group_order_no'] == g_no]['주문톤'].sum()),
                # 3폭 제약을 위한 SKID 관련 정보
                'skid_yn': str(row.get('skid_yn', 'N')),
                'export_yn': 'N' if row.get('수출내수', '내수') == '내수' else 'Y',
                'pte_gubun': str(row.get('pte_gubun', '1')),
                'dir_gubun': str(row.get('dir_gubun', ''))
            }

        # group_order_no별 최대 롤지폭 제약조건 계산
        self.width_max_constraints = {}
        for g_no in self.order_widths:
            self.width_max_constraints[g_no] = self.group_order_info[g_no]['max_allowed_width']
        
        logging.info("\n--- 그룹오더별 최대 롤지폭 제약조건 ---")
        for g_no, max_w in sorted(self.width_max_constraints.items(), key=lambda x: str(x[0])):
            info = self.group_order_info[g_no]
            logging.info(f"  {g_no} ({info['width']}x{info['length']}mm): max_allowed = {max_w}mm")

        # 정확한 생산량 준수(Exact Match)가 필요한 지폭 식별
        # 현재 비활성화: 패턴 다양성 부족으로 인해 Exact Match 적용 시 해가 없어지는 문제 발생
        # 과생산 페널티(OVER_PROD_PENALTY)로 초과 생산을 최소화함
        self.exact_match_widths = set()

        self.items, self.item_info, self.item_composition = self._prepare_items(min_sc_width, max_sc_width)

        self.max_width = max_width
        self.min_width = min_width
        self.min_pieces = MIN_PIECES_PER_PATTERN
        self.max_pieces = int(max_pieces)
        self.min_sc_width = min_sc_width
        self.max_sc_width = max_sc_width
        self.db = db
        self.lot_no = lot_no
        self.version = version
        logging.info(f"\n--- 패턴 제약조건: 최소 {self.min_pieces}폭, 최대 {self.max_pieces}폭 ---")

        self.patterns = []

    def _get_max_allowed_width(self, length, dir_gubun):
        """세로 길이와 dir_gubun에 따라 최대 허용 롤지폭을 반환합니다."""
        if dir_gubun == 'ZARL':
            if length < 500:
                return 1680
            elif 500 <= length < 600:
                return 2186
            else:
                return 2450
        
        if length < 500:
            return 1680
        elif 500 <= length < 600:
            return 2186
        else:  # length >= 600
            return 2874
    

    def _prepare_items(self, min_sc_width, max_sc_width):
        """복합폭 아이템(패턴의 구성요소)을 생성합니다.
        group_order_no 기준으로 아이템을 생성하며, 각 그룹오더별로 허용된 max_allowed_width 내에서만 생성합니다.
        item_composition에는 group_order_no를 저장하여 해당 그룹오더 수요에만 기여하도록 합니다.
        """
        items = []
        item_info = {}  # item_name -> width
        item_composition = {}  # composite_item_name -> {원본 지폭: count}

        for g_no in self.order_widths:
            info = self.group_order_info[g_no]
            orig_width = info['width']
            length = info['length']
            max_allowed_width = self.width_max_constraints.get(g_no, self.original_max_width)
            
            # 3폭 제약을 위한 SKID 정보
            skid_yn = info.get('skid_yn', 'N')
            export_yn = info.get('export_yn', 'N')
            pte_gubun = info.get('pte_gubun', '1')

            for i in range(1, 5):  # 1, 2, 3, 4폭까지 고려
                base_width = orig_width * i + self.sheet_trim
                
                # 1폭: min_sc_width 이상이어야 함
                if i == 1:
                    if base_width < min_sc_width:
                        continue
                
                # 2, 4폭: max_allowed_width 이하이어야 함
                if i in [2, 4]:
                    if base_width > max_allowed_width:
                        continue
                
                # 3폭: SKID 관련 추가 제약
                if i == 3:
                    if base_width > max_sc_width:
                        continue
                    if base_width > max_allowed_width:
                        continue
                    # 내수 스키드이고 지폭 600 이하이면 3폭 불가
                    if skid_yn == 'Y' and export_yn == 'N' and orig_width <= 600:
                        continue
                    # 수출 스키드이고 지폭 600 이하이고 pte_gubun='2'이면 3폭 불가
                    if skid_yn == 'Y' and export_yn == 'Y' and orig_width <= 600 and pte_gubun == '2':
                        continue

                if base_width < SC_MIN_WIDTH_FOR_MACHINE:
                    continue

                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                # group_order_no 기반 아이템명 생성 (G 접두어 추가)
                item_name = f"G{g_no}x{i}"
                composite_width = base_width
                if composite_width <= self.original_max_width:
                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = composite_width
                        # group_order_no를 저장하여 해당 그룹오더 수요에만 기여하도록 함
                        item_composition[item_name] = {g_no: i}
        
        logging.info(f"\n--- 생성된 복합폭 아이템 ({len(items)}개) ---")
        for item in items[:60]:  # 처음 60개만 출력
            g_no = list(item_composition[item].keys())[0]
            info = self.group_order_info.get(g_no, {})
            logging.info(f"  {item}: {item_info[item]}mm (group: {g_no}, dims: {info.get('width', '?')}x{info.get('length', '?')}mm)")
        if len(items) > 60:
            logging.info(f"  ... 외 {len(items) - 60}개")
        
        return items, item_info, item_composition

    def _calculate_demand_rolls(self, df_orders):
        """주문량을 바탕으로 그룹오더별 필요 롤 수를 계산합니다.
        개별 오더 단위가 아닌 'group_order_no' 단위로 톤수를 합산하여 롤 수를 계산합니다.
        """
        df_copy = df_orders.copy()
        sheet_roll_length_mm = self.sheet_roll_length * 1000

        # max_allowed_width 계산 (세로 길이에 따라 다름)
        df_copy['max_allowed_width'] = df_copy.apply(
            lambda row: self._get_max_allowed_width(
                row['세로'], 
                str(row.get('dir_gubun', ''))
            ), 
            axis=1
        )

        # 1. Group Order별 총 주문톤 합산
        if 'group_order_no' not in df_copy.columns:
            df_copy['group_order_no'] = df_copy.index

        group_sums = df_copy.groupby('group_order_no')['주문톤'].sum().to_dict()
        group_specs = df_copy.groupby('group_order_no')[['가로', '세로']].first().to_dict('index')

        # 2. Group 단위 롤 수 계산
        group_rolls_map = {}
        for g_no, total_ton in group_sums.items():
            specs = group_specs[g_no]
            width_mm = specs['가로']
            length_mm = specs['세로']
            
            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0 or total_ton <= 0:
                group_rolls_map[g_no] = 0
                continue

            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1000000
            if sheet_weight_g <= 0:
                group_rolls_map[g_no] = 0
                continue
            
            total_sheets_needed = (total_ton * 1000000) / sheet_weight_g
            sheets_per_roll_length = math.floor(sheet_roll_length_mm / length_mm)
            if sheets_per_roll_length <= 0:
                group_rolls_map[g_no] = 0
                continue

            sheets_per_roll = sheets_per_roll_length
            raw_rolls = total_sheets_needed / sheets_per_roll
            decimal_part = raw_rolls - int(raw_rolls)
            group_rolls_map[g_no] = int(raw_rolls) + 1 if decimal_part >= 0.1 else int(raw_rolls)

        # 3. 계산된 그룹 롤 수를 개별 오더(row)에 배분
        def distribute_rolls(group_df):
            g_no = group_df['group_order_no'].iloc[0]
            total_r = group_rolls_map.get(g_no, 0)
            total_t = group_sums.get(g_no, 0)
            
            if total_t == 0 or total_r == 0:
                group_df['rolls'] = 0
                return group_df
            
            group_df['rolls'] = (group_df['주문톤'] / total_t * total_r).round().astype(int)
            
            current_sum = group_df['rolls'].sum()
            diff = total_r - current_sum
            if diff != 0:
                idx = group_df['주문톤'].idxmax()
                group_df.loc[idx, 'rolls'] += diff
            
            return group_df

        df_copy = df_copy.groupby('group_order_no', group_keys=False).apply(distribute_rolls)

        # 그룹오더(group_order_no)별 최종 요구 롤 수 집계
        # 각 그룹오더가 독립적인 수요로 처리됨
        demand_rolls = df_copy.groupby('group_order_no')['rolls'].sum().to_dict()

        logging.info("\n--- 그룹오더별 필요 롤 수 ---")
        for g_no, rolls in sorted(demand_rolls.items(), key=lambda x: str(x[0])):
            # 해당 그룹오더의 가로/세로 정보 가져오기
            row = df_copy[df_copy['group_order_no'] == g_no].iloc[0]
            logging.info(f"  {g_no} ({row['가로']}x{row['세로']}mm): {rolls} 롤")
        logging.info("----------------------------------------------\n")
        
        return df_copy, demand_rolls

    def _generate_initial_patterns_db(self):
        """th_pattern_tot_sheet 테이블의 사용자 편집 패턴 데이터를 활용하여 초기 패턴을 생성합니다."""
        if not self.db or not self.lot_no:
            logging.info("--- DB 정보가 없어 기존 패턴을 불러올 수 없습니다. ---")
            return

        logging.info("\n--- DB(th_pattern_tot_sheet)에서 사용자 편집 패턴을 불러와 초기 패턴을 생성합니다. ---")
        db_patterns_list = self.db.get_sheet_patterns_from_db(self.lot_no)

        if not db_patterns_list:
            logging.info("--- DB에 저장된 사용자 편집 패턴이 없거나, 현재 오더와 일치하는 패턴이 없습니다. ---")
            return

        logging.info(f"--- 현재 생성된 유효 아이템 목록 (총 {len(self.items)}개): {self.items[:20]} ... ---")
        
        # 가로(width) -> group_order_no 매핑 생성 (DB 아이템 변환용)
        width_to_g_no = {}
        for g_no in self.order_widths:
            info = self.group_order_info.get(g_no, {})
            width = info.get('width', 0)
            if width not in width_to_g_no:
                width_to_g_no[width] = []
            width_to_g_no[width].append(g_no)
        
        initial_patterns_from_db = []
        for pattern_item_list in db_patterns_list:
            # DB 아이템 형식 변환: "420x3" -> "G{group_order_no}x3"
            converted_pattern_items = []
            conversion_success = True
            
            for item_name in pattern_item_list:
                # 새 형식인지 확인 (G로 시작)
                if item_name.startswith('G'):
                    converted_pattern_items.append(item_name)
                else:
                    # 기존 형식 파싱: "420x3"
                    try:
                        w_str, c_str = item_name.split('x')
                        w = int(w_str)
                        c = int(c_str)
                        
                        # 해당 width를 가진 group_order_no 찾기
                        if w in width_to_g_no:
                            # 첫 번째 매칭 사용 (동일 가로의 경우)
                            g_no = width_to_g_no[w][0]
                            new_item_name = f"G{g_no}x{c}"
                            converted_pattern_items.append(new_item_name)
                        else:
                            # 매칭되는 가로가 없으면 변환 실패
                            conversion_success = False
                            break
                    except:
                        conversion_success = False
                        break
            
            if not conversion_success:
                continue
            
            pattern_dict = dict(Counter(converted_pattern_items))

            # DB의 패턴에 포함된 모든 아이템이 현재 주문에도 유효한지 확인
            is_valid = all(item_name in self.items for item_name in pattern_dict.keys())
            
            if is_valid:
                initial_patterns_from_db.append(pattern_dict)
            else:
                invalid_items = [name for name in pattern_dict if name not in self.items]
                #logging.info(f"    - 경고: DB 패턴 {pattern_dict}의 아이템 {invalid_items}이(가) 현재 오더에 없어 무시합니다.")

        if initial_patterns_from_db:
            seen_patterns = {frozenset(p.items()) for p in self.patterns}
            added_count = 0
            skipped_count = 0
            
            # 패턴을 원본 가로값으로 변환하는 헬퍼 함수
            def pattern_to_readable(p_dict):
                readable = {}
                for item, count in p_dict.items():
                    # "G{g_no}x{c}" 또는 "{g_no}x{c}" -> "{width}x{c}"
                    if 'x' in item:
                        # G 접두사 제거
                        item_no_g = item[1:] if item.startswith('G') else item
                        g_no_part = item_no_g[:item_no_g.rfind('x')]
                        c_part = item_no_g[item_no_g.rfind('x')+1:]
                        try:
                            # 정수와 문자열 모두 시도
                            g_no_int = int(g_no_part)
                            info = self.group_order_info.get(g_no_int) or self.group_order_info.get(g_no_part) or self.group_order_info.get(str(g_no_int))
                            if info:
                                width = info.get('width', g_no_part)
                                readable[f"{width}x{c_part}"] = count
                            else:
                                readable[item] = count
                        except:
                            readable[item] = count
                    else:
                        readable[item] = count
                return readable
            
            for p_dict in initial_patterns_from_db:
                readable_pattern = pattern_to_readable(p_dict)
                
                # 1. 복합폭 개수(pieces) 제약조건 검증
                pattern_pieces = sum(p_dict.values())
                if pattern_pieces > self.max_pieces:
                    logging.info(f"    - 경고: DB 패턴 {readable_pattern}의 복합폭 개수({pattern_pieces})가 max_pieces({self.max_pieces})를 초과하여 무시합니다.")
                    skipped_count += 1
                    continue
                
                # 2. min_width/max_width 제약조건 검증
                pattern_width = sum(self.item_info.get(item, 0) * count for item, count in p_dict.items())
                if pattern_width > self.max_width:
                    logging.info(f"    - 경고: DB 패턴 {readable_pattern}의 총 폭({pattern_width}mm)이 max_width({self.max_width}mm)를 초과하여 무시합니다.")
                    skipped_count += 1
                    continue
                if pattern_width < self.min_width:
                    logging.info(f"    - 경고: DB 패턴 {readable_pattern}의 총 폭({pattern_width}mm)이 min_width({self.min_width}mm) 미만이어서 무시합니다.")
                    skipped_count += 1
                    continue
                if frozenset(p_dict.items()) not in seen_patterns:
                    self.patterns.append(p_dict)
                    added_count += 1
            if skipped_count > 0:
                logging.info(f"--- DB에서 {skipped_count}개의 패턴이 제약조건 위반(pieces/width)으로 제외되었습니다. ---")
            if added_count > 0:
                logging.info(f"--- DB에서 {added_count}개의 사용자 편집 패턴을 추가했습니다. ---")
            else:
                logging.info(f"--- DB에서 불러온 패턴 중 유효한 것이 없습니다. (총 {len(db_patterns_list)}개 검토, 모두 무효 또는 중복) ---")

    def _generate_initial_patterns(self):
        """휴리스틱을 사용하여 초기 패턴을 생성합니다."""
        seen_patterns = {frozenset(p.items()) for p in self.patterns}

        # demands_in_rolls가 이제 group_order_no 기준이므로 직접 사용
        # item_composition의 키도 group_order_no

        # 1. 다양한 휴리스틱을 위한 정렬된 아이템 리스트 생성
        sorted_by_demand = sorted(
            self.items,
            key=lambda i: self.demands_in_rolls.get(list(self.item_composition[i].keys())[0], 0),
            reverse=True
        )
        sorted_by_demand_asc = sorted(
            self.items,
            key=lambda i: self.demands_in_rolls.get(list(self.item_composition[i].keys())[0], 0),
            reverse=False
        )
        sorted_by_width_desc = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)
        sorted_by_width_asc = sorted(self.items, key=lambda i: self.item_info[i], reverse=False)

        # 2. Random Shuffles (add multiple to increase diversity)
        random.seed(28) # Ensure determinism
        random_shuffles = []
        for _ in range(1000):
            items_copy = list(self.items)
            random.shuffle(items_copy)
            random_shuffles.append(items_copy)

        heuristics = [
            sorted_by_demand, 
            sorted_by_width_desc, 
            sorted_by_width_asc, 
            sorted_by_demand_asc
        ] + random_shuffles
        
        # ====== 개선 1: 소량 주문 그룹오더 우선 패턴 생성 ======
        # 필요 롤수가 적은 그룹오더를 주 아이템으로 하는 패턴을 우선 생성
        small_demand_items = sorted(
            self.items,
            key=lambda i: self.demands_in_rolls.get(list(self.item_composition[i].keys())[0], float('inf'))
        )
        
        for primary_item in small_demand_items[:20]:  # 상위 20개 소량 주문 그룹오더
            primary_width = self.item_info[primary_item]
            primary_g_no = list(self.item_composition[primary_item].keys())[0]
            primary_rolls = self.demands_in_rolls.get(primary_g_no, 0)
            
            # 소량 주문(10롤 이하)에 대해서만 특별 패턴 생성
            if primary_rolls > 10:
                continue
                
            # 이 지폭을 주 아이템으로 하고, 다른 지폭으로 min_width를 맞추는 패턴 탐색
            for secondary_item in sorted_by_width_desc:
                if secondary_item == primary_item:
                    continue
                    
                secondary_width = self.item_info[secondary_item]
                
                # 다양한 조합 시도
                for primary_count in range(1, self.max_pieces + 1):
                    remaining_width = self.max_width - (primary_width * primary_count)
                    remaining_pieces = self.max_pieces - primary_count
                    
                    if remaining_width <= 0 or remaining_pieces <= 0:
                        continue
                    
                    secondary_count = min(int(remaining_width / secondary_width), remaining_pieces)
                    if secondary_count <= 0:
                        continue
                        
                    total_width = primary_width * primary_count + secondary_width * secondary_count
                    total_pieces = primary_count + secondary_count
                    
                    if self.min_width <= total_width <= self.max_width and self.min_pieces <= total_pieces:
                        new_pattern = {primary_item: primary_count, secondary_item: secondary_count}
                        pattern_key = frozenset(new_pattern.items())
                        if pattern_key not in seen_patterns:
                            self.patterns.append(new_pattern)
                            seen_patterns.add(pattern_key)

        logging.info(f"--- {len(self.patterns)}개의 소량주문 우선 패턴 생성됨 ---")




        # ====== 개선 1: 가로 규격이 큰 그룹오더 우선 패턴 생성 ======
        # 가로 규격이 큰 그룹오더를 주 아이템으로 하는 패턴을 우선 생성
        large_width_items = sorted(
            self.items,
            key=lambda i: self.group_order_info.get(list(self.item_composition[i].keys())[0], {}).get('width', 0),
            reverse=True  # 가로가 큰 순서 (내림차순)
        )
        
        for primary_item in large_width_items[:20]:  # 상위 20개 가로 규격이 큰 그룹오더
            primary_width = self.item_info[primary_item]
            primary_g_no = list(self.item_composition[primary_item].keys())[0]
            primary_rolls = self.demands_in_rolls.get(primary_g_no, 0)
                           
            # 이 지폭을 주 아이템으로 하고, 다른 지폭으로 min_width를 맞추는 패턴 탐색
            for secondary_item in sorted_by_width_asc:
                if secondary_item == primary_item:
                    continue
                    
                secondary_width = self.item_info[secondary_item]
                
                # 다양한 조합 시도
                for primary_count in range(1, self.max_pieces + 1):
                    remaining_width = self.max_width - (primary_width * primary_count)
                    remaining_pieces = self.max_pieces - primary_count
                    
                    if remaining_width <= 0 or remaining_pieces <= 0:
                        continue
                    
                    secondary_count = min(int(remaining_width / secondary_width), remaining_pieces)
                    if secondary_count <= 0:
                        continue
                        
                    total_width = primary_width * primary_count + secondary_width * secondary_count
                    total_pieces = primary_count + secondary_count
                    
                    if self.min_width <= total_width <= self.max_width and self.min_pieces <= total_pieces:
                        new_pattern = {primary_item: primary_count, secondary_item: secondary_count}
                        pattern_key = frozenset(new_pattern.items())
                        if pattern_key not in seen_patterns:
                            self.patterns.append(new_pattern)
                            seen_patterns.add(pattern_key)

        logging.info(f"--- {len(self.patterns)}개의 가로가 큰 주문 우선 패턴 생성됨 ---")




        # ====== 기존 로직: First-Fit 휴리스틱 ======
        first_fit_count = len(self.patterns)
        for sorted_items in heuristics:
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
                    item_to_add = next((i for i in sorted_by_width_desc if current_width + self.item_info[i] <= self.max_width), None)
                    
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
        logging.info(f"--- {len(self.patterns)} 생성됨. 기존 로직: First-Fit 휴리스틱 ---")

        # ====== 개선 2: Best-Fit 휴리스틱 추가 (남은 공간을 최소화하는 아이템 선택) ======
        for item in sorted_by_demand_asc:  # 소량 주문 먼저
            item_width = self.item_info[item]
            
            current_pattern = {item: 1}
            current_width = item_width
            current_pieces = 1

            while current_pieces < self.max_pieces:
                remaining_width = self.max_width - current_width
                
                # Best-Fit: 남은 공간에 가장 잘 맞는(공간을 가장 적게 남기는) 아이템 선택
                best_fit_item = None
                min_waste = float('inf')
                for candidate in self.items:
                    candidate_width = self.item_info[candidate]
                    if candidate_width <= remaining_width:
                        waste = remaining_width - candidate_width
                        if waste < min_waste:
                            min_waste = waste
                            best_fit_item = candidate
                
                if not best_fit_item:
                    break 

                current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
                current_width += self.item_info[best_fit_item]
                current_pieces += 1

            if self.min_width <= current_width <= self.max_width and self.min_pieces <= current_pieces:
                pattern_key = frozenset(current_pattern.items())
                if pattern_key not in seen_patterns:
                    self.patterns.append(current_pattern)
                    seen_patterns.add(pattern_key)
        
        logging.info(f"--- {len(self.patterns)} 생성됨. 개선 2: Best-Fit 휴리스틱 추가 ---")


        # ====== 개선 3: 2폭 조합 패턴 체계적 생성 ======
        # 두 가지 아이템만 사용하는 패턴을 체계적으로 생성
        items_list = list(self.items)
        for i, item1 in enumerate(items_list):
            width1 = self.item_info[item1]
            for item2 in items_list[i:]:  # 중복 방지
                width2 = self.item_info[item2]
                
                # 다양한 개수 조합 시도
                for count1 in range(1, self.max_pieces):
                    for count2 in range(1, self.max_pieces - count1 + 1):
                        total_width = width1 * count1 + width2 * count2
                        total_pieces = count1 + count2
                        
                        if self.min_width <= total_width <= self.max_width and self.min_pieces <= total_pieces <= self.max_pieces:
                            new_pattern = {item1: count1}
                            if item1 != item2:
                                new_pattern[item2] = count2
                            else:
                                new_pattern[item1] += count2
                            
                            pattern_key = frozenset(new_pattern.items())
                            if pattern_key not in seen_patterns:
                                self.patterns.append(new_pattern)
                                seen_patterns.add(pattern_key)
        
        logging.info(f"--- {len(self.patterns)} 생성됨.개선 3: 2폭 조합 패턴 체계적 생성 ---")
        logging.info(f"--- {len(self.patterns)}개의 혼합 패턴 생성됨 ---")

        # --- 3. 모든 복합폭에 대해 '순수 품목 패턴' 생성 ---
        pure_patterns_added = 0
        # 순수 패턴은 어떤 정렬이든 상관 없으므로 마지막 정렬(오름차순) 사용
        for item in sorted_by_width_asc:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0: continue

            # 해당 아이템으로만 구성된 패턴 생성 시도
            num_items = min(int(self.max_width / item_width), self.max_pieces)
            
            # 너비가 큰 조합부터 작은 조합까지 순차적으로 확인
            found_valid_pure = False
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                
                if self.min_width <= total_width and self.min_pieces <= num_items:
                    pattern_key = frozenset(new_pattern.items())
                    if pattern_key not in seen_patterns:
                        self.patterns.append(new_pattern)
                        seen_patterns.add(pattern_key)
                        pure_patterns_added += 1
                        found_valid_pure = True
                        break # 이 아이템으로 만들 수 있는 가장 좋은 순수패턴을 찾았으므로 종료
                
                num_items -= 1

            # [Patch] 항상 최적의 순수 패턴(꽉 채운 것)을 후보군에 추가합니다.
            # min_width를 만족하는 패턴을 찾았더라도, 때로는 Trim Loss를 감수하더라도 더 단순한(또는 다른 조합의) 패턴이 필요할 수 있습니다.
            # 특히 Over-Production Penalty가 매우 크므로, 독립 생산 가능한 패턴은 필수입니다.
            fill_num = min(int(self.max_width / item_width), self.max_pieces)
            if fill_num > 0:
                fallback_pattern = {item: fill_num}
                fallback_width = item_width * fill_num
                # min_width 조건 체크 추가 - min_width 미만이면 추가하지 않음
                if fallback_width >= self.min_width:
                    self.patterns.append(fallback_pattern)
                    logging.debug(f"--- [DEBUG] Forced Pure Pattern for {item}: {fallback_pattern} (width: {fallback_width}mm)")
                else:
                    logging.debug(f"--- [DEBUG] Skipped Pure Pattern for {item}: {fallback_pattern} (width: {fallback_width}mm < min_width: {self.min_width}mm)")



        if pure_patterns_added > 0:
            logging.info(f"--- {pure_patterns_added}개의 순수 품목 패턴 추가됨 ---")

        """
        # --- 4. 1폭만 가능한 큰 오더(가로 1200mm+)에 대한 전용 패턴 강제 추가 ---
        # 이 오더들은 2폭이 max_allowed를 초과하므로 1폭만 가능하며, 다른 아이템과 조합해야 함
        large_order_patterns_added = 0
        large_order_candidates = []  # 디버그용
        
        for g_no in self.order_widths:
            info = self.group_order_info.get(g_no, {})
            orig_width = info.get('width', 0)
            max_allowed = self.width_max_constraints.get(g_no, self.original_max_width)
            
            # 2폭이 max_allowed를 초과하는 오더만 처리 (1폭만 가능한 오더)
            two_strip_width = orig_width * 2 + self.sheet_trim
            if two_strip_width <= max_allowed:
                continue  # 2폭 이상 가능한 오더는 스킵
            
            large_order_candidates.append(f"{g_no}({orig_width}mm)")
            
            # 1폭 아이템 확인
            one_strip_item = f"G{g_no}x1"
            if one_strip_item not in self.item_info:
                # 1폭 아이템이 없으면 생성
                one_strip_width = orig_width + self.sheet_trim
                if one_strip_width >= self.min_sc_width and one_strip_width <= self.max_sc_width:
                    if one_strip_item not in self.items:
                        self.items.append(one_strip_item)
                    self.item_info[one_strip_item] = one_strip_width
                    self.item_composition[one_strip_item] = {g_no: 1}
                    logging.info(f"  - 1폭 아이템 생성: {one_strip_item} ({one_strip_width}mm)")
                else:
                    logging.info(f"  - 1폭 아이템 생성 불가 (범위 초과): {one_strip_item} ({one_strip_width}mm)")
                    continue  # 유효하지 않은 1폭은 스킵
            
            one_strip_width = self.item_info[one_strip_item]
            remaining_width = self.max_width - one_strip_width
            
            # 다른 아이템과 조합하여 min_width 충족하는 패턴 생성
            patterns_for_this_order = 0
            for other_item in sorted_by_width_desc:
                if other_item == one_strip_item:
                    continue
                    
                other_width = self.item_info[other_item]
                
                # 남은 공간에 들어갈 수 있는 최대 개수 계산
                max_count = min(int(remaining_width / other_width), self.max_pieces - 1)
                if max_count <= 0:
                    continue
                
                for count in range(max_count, 0, -1):
                    total_width = one_strip_width + other_width * count
                    total_pieces = 1 + count
                    
                    if self.min_width <= total_width <= self.max_width and total_pieces <= self.max_pieces:
                        new_pattern = {one_strip_item: 1, other_item: count}
                        pattern_key = frozenset(new_pattern.items())
                        if pattern_key not in seen_patterns:
                            self.patterns.append(new_pattern)
                            seen_patterns.add(pattern_key)
                            large_order_patterns_added += 1
                            patterns_for_this_order += 1
                        break  # 이 other_item에 대해 가장 좋은 조합 하나만
            
            if patterns_for_this_order > 0:
                logging.info(f"  - {g_no}({orig_width}mm): {patterns_for_this_order}개 패턴 추가됨")

        if large_order_candidates:
            logging.info(f"--- 1폭만 가능한 오더 {len(large_order_candidates)}개 발견: {large_order_candidates[:10]}{'...' if len(large_order_candidates) > 10 else ''} ---")
        
        if large_order_patterns_added > 0:
            logging.info(f"--- {large_order_patterns_added}개의 1폭 전용 오더 조합 패턴 추가됨 ---")

        """ 

        # --- 4. 폴백 로직: 초기 패턴으로 커버되지 않는 그룹오더가 있는지 최종 확인 ---
        # item_composition의 키가 group_order_no이므로 직접 비교 가능
        covered_g_nos = {w for p in self.patterns for item_name in p for w in self.item_composition.get(item_name, {})}
        uncovered_g_nos = set(self.order_widths) - covered_g_nos

        if uncovered_g_nos:
            logging.info(f"--- 경고: 초기 패턴에 포함되지 않은 그룹오더 발견: {uncovered_g_nos} ---")
            logging.info("--- 해당 주문에 대한 폴백 패턴을 추가 생성합니다. ---")
            
            for g_no in uncovered_g_nos:
                info = self.group_order_info.get(g_no, {})
                orig_width = info.get('width', 0)
                
                logging.info(f"  - 그룹오더 {g_no} ({orig_width}x{info.get('length', '?')}mm)에 대한 순수 품목 패턴 생성 시도...")

                # 1. 이 그룹오더로 만들 수 있는 유효한 복합폭 아이템 목록을 찾습니다.
                valid_components = []
                max_allowed_width = self.width_max_constraints.get(g_no, self.original_max_width)

                for i in range(1, 5): # 1~4폭 고려
                    item_name = f"G{g_no}x{i}"
                    # 아이템이 이미 생성되었는지 확인
                    if item_name in self.item_info:
                        valid_components.append(item_name)
                    else:
                        # 동적으로 생성 및 유효성 검사
                        composite_width = orig_width * i + self.sheet_trim
                        
                        # 최대 롤지폭 제약조건 체크
                        if composite_width > max_allowed_width:
                            continue

                        if (self.min_sc_width <= composite_width <= self.max_sc_width) and \
                           (composite_width <= self.original_max_width):
                            # 유효하면 아이템 정보에 추가
                            if item_name not in self.items: self.items.append(item_name)
                            self.item_info[item_name] = composite_width
                            self.item_composition[item_name] = {g_no: i}  # group_order_no 저장
                            valid_components.append(item_name)

                if not valid_components:
                    logging.info(f"    - 경고: 그룹오더 {g_no} ({orig_width}mm)로 만들 수 있는 유효한 복합폭 아이템이 없습니다. 폴백 패턴을 생성할 수 없습니다.")
                    continue

                # 2. 너비가 넓은 순으로 정렬하여 Greedy 알고리즘 준비
                sorted_components = sorted(valid_components, key=lambda i: self.item_info[i], reverse=True)
                
                # 3. Greedy 방식으로 최적의 단일 품목 패턴 구성
                new_pattern = {}
                current_width = 0
                current_pieces = 0
                
                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    
                    # 남은 공간에 들어갈 수 있는 가장 큰 구성요소 찾기
                    best_fit = next((item for item in sorted_components if self.item_info[item] <= remaining_width), None)
                    
                    if not best_fit:
                        break # 더 이상 추가할 수 있는 구성요소가 없음
                    
                    new_pattern[best_fit] = new_pattern.get(best_fit, 0) + 1
                    current_width += self.item_info[best_fit]
                    current_pieces += 1

                # 4. 생성된 패턴의 유효성 검사 및 추가
                if new_pattern:
                    total_width = sum(self.item_info[name] * count for name, count in new_pattern.items())
                    total_pieces = sum(new_pattern.values())

                    if self.min_width <= total_width and self.min_pieces <= total_pieces:
                        pattern_key = frozenset(new_pattern.items())
                        if pattern_key not in seen_patterns:
                            self.patterns.append(new_pattern)
                            seen_patterns.add(pattern_key)
                            logging.info(f"    -> 생성된 순수 패턴: {new_pattern} (너비: {total_width}mm, 폭 수: {total_pieces}) -> 폴백 패턴으로 추가됨.")
                        else:
                            logging.info(f"    - 생성된 순수 패턴 {new_pattern}은 이미 존재합니다.")
                    else:
                        logging.info(f"    - 생성된 순수 패턴 {new_pattern}이 최종 제약조건(최소너비/폭수)을 만족하지 못합니다. (너비: {total_width}, 폭 수: {total_pieces})")
                else:
                    logging.info(f"    - 그룹오더 {g_no} ({orig_width}mm)에 대한 순수 패턴을 구성하지 못했습니다.")


        logging.info(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")
        # logging.info(self.patterns)
        logging.info("--------------------------")

    def _consolidate_patterns(self):
        """
        생성된 초기 패턴들을 후처리하여 작은 복합폭 아이템들을 가능한 큰 복합폭 아이템으로 통합합니다.
        예: {'814x1': 2}는 {'814x2': 1}로 변경을 시도합니다.
        """
        logging.info("\n--- 생성된 패턴에 대해 후처리(통합)을 시작합니다. ---")
        
        processed_patterns = []
        seen_patterns = set()

        for pattern in self.patterns:
            # 1. 패턴을 기본 지폭 단위로 모두 분해
            base_width_counts = Counter()
            for item_name, count in pattern.items():
                composition = self.item_composition.get(item_name)
                if composition:
                    for base_width, num_base in composition.items():
                        base_width_counts[base_width] += num_base * count

            # 2. 가장 큰 복합폭부터 사용하여 새로운 패턴 재구성
            new_pattern = {}
            current_total_width = 0
            current_total_pieces = 0
            
            sorted_base_widths = sorted(base_width_counts.keys(), reverse=True)

            for base_width in sorted_base_widths:
                remaining_base_count = base_width_counts[base_width]
                
                # Special Case: If we have exactly 4 sheets, prefer 2x 2-up over 1x 3-up + 1x 1-up
                # This reduces setup time in the cutter process.
                if remaining_base_count == 4:
                    item_name_2up = f"{base_width}x2"
                    if item_name_2up in self.item_info:
                        item_width_2up = self.item_info[item_name_2up]
                        max_allowed_width = self.width_max_constraints.get(base_width, self.original_max_width)
                        
                        # Check constraints for 2-up
                        if item_width_2up <= max_allowed_width and \
                           current_total_pieces + 2 <= self.max_pieces and \
                           current_total_width + item_width_2up * 2 <= self.max_width:
                            
                            new_pattern[item_name_2up] = new_pattern.get(item_name_2up, 0) + 2
                            current_total_width += item_width_2up * 2
                            current_total_pieces += 2
                            remaining_base_count -= 4
                            continue # Skip the normal loop for this base_width

                for i in range(4, 0, -1):
                    if remaining_base_count < i:
                        continue

                    item_name = f"{base_width}x{i}"
                    if item_name in self.item_info:
                        num_to_use = remaining_base_count // i
                        item_width = self.item_info[item_name]
                        
                        if num_to_use > 0 and \
                           current_total_pieces + num_to_use <= self.max_pieces and \
                           current_total_width + item_width * num_to_use <= self.max_width:
                            
                            # 최대 롤지폭 제약조건 체크 (New)
                            max_allowed_width = self.width_max_constraints.get(base_width, self.original_max_width)
                            if item_width > max_allowed_width:
                                continue

                            new_pattern[item_name] = new_pattern.get(item_name, 0) + num_to_use
                            current_total_width += item_width * num_to_use
                            current_total_pieces += num_to_use
                            remaining_base_count -= num_to_use * i
            
            # 3. 재구성된 패턴을 사용할지 결정
            is_new_pattern_valid = (self.min_width <= current_total_width and self.min_pieces <= current_total_pieces)
            
            chosen_pattern = pattern # 기본적으로 원본 유지
            if is_new_pattern_valid and new_pattern and frozenset(new_pattern.items()) != frozenset(pattern.items()):
                chosen_pattern = new_pattern # 유효하고 변경되었으면 새 패턴 선택

            pattern_key = frozenset(chosen_pattern.items())
            if pattern_key not in seen_patterns:
                processed_patterns.append(chosen_pattern)
                seen_patterns.add(pattern_key)

        original_count = len(self.patterns)
        self.patterns = processed_patterns
        logging.info(f"--- 패턴 통합 완료: {original_count}개 -> {len(self.patterns)}개 패턴으로 정리됨 ---")

    def _solve_master_problem(self, is_final_mip=False):
        """
        마스터 문제(Master Problem)를 선형계획법(LP) 또는 정수계획법(MIP)으로 해결합니다.
        
        목적 함수:
        1. 총 생산 롤 수 최소화
        2. 과생산/부족생산 페널티 최소화
        3. 패턴 복잡도 및 교체 비용 페널티 최소화
        4. 패턴 내 총 롤(piece) 개수에 대한 페널티 (Quadratic)
        
        Args:
            is_final_mip (bool): True이면 정수해(Integer Solution)를 구하고, 
                               False이면 열 생성을 위한 실수해(Relaxed LP)와 Dual Value를 구합니다.
        """
        # 1. [Final MIP] Try Gurobi Direct Solver
        if is_final_mip:
            logging.info(f"--- [DEBUG] Entering _solve_master_problem(is_final_mip={is_final_mip}, Patterns={len(self.patterns)})")
            # [DEBUG]
            logging.debug(f"--- [DEBUG] Exact Match Widths: {sorted(list(self.exact_match_widths))}")

            # [DEBUG] 생성된 패턴 목록 출력
            # logging.info("--- [DEBUG] 생성된 패턴 목록 ---")
            # for idx, pattern in enumerate(self.patterns):
            #     widths = [self.item_info[item] for item in pattern for _ in range(pattern[item])]
            #     total_width = sum(widths)
            #     widths_str = ', '.join(map(str, sorted(widths, reverse=True)))
            #     logging.info(f"  패턴 {idx}: [{widths_str}] = {total_width}mm")
            # logging.info("--- [DEBUG] 패턴 목록 끝 ---")
            
            try:
                logging.info("Trying Gurobi Direct Solver SheetOptimize (gurobipy)...")
                model = gp.Model("SheetOptimization")
                model.setParam("OutputFlag", 0)  # Silence console output
                model.setParam("LogToConsole", 0)
                if hasattr(self, 'num_threads'):
                    model.setParam("Threads", self.num_threads)
                
                model.setParam("TimeLimit", self.solver_time_limit_ms / 1000.0)
                model.setParam("MIPFocus", 0) # 1: Find valid solution (Feasibility) first

                # Variables
                x = {}
                for j in range(len(self.patterns)):
                    x[j] = model.addVar(vtype=GRB.INTEGER, name=f'P_{j}')

                over_prod_vars = {}
                for width in self.demands_in_rolls:
                    over_prod_vars[width] = model.addVar(vtype=GRB.CONTINUOUS, name=f'Over_{width}')
                
                under_prod_vars = {}
                for width, required_rolls in self.demands_in_rolls.items():
                    allowed_under_prod = max(1, math.ceil(required_rolls))
                    under_prod_vars[width] = model.addVar(lb=0, ub=allowed_under_prod, vtype=GRB.CONTINUOUS, name=f'Under_{width}')

                # Constraints: Demand
                # item_composition의 키가 vwidth이므로 직접 매칭
                constraints = {}
                for vwidth, required_rolls in self.demands_in_rolls.items():
                    production_expr = gp.quicksum(
                        x[j] * sum(self.item_composition[item_name].get(vwidth, 0) * count for item_name, count in self.patterns[j].items())
                        for j in range(len(self.patterns))
                    )
                    model.addConstr(production_expr + under_prod_vars[vwidth] == required_rolls + over_prod_vars[vwidth], name=f'demand_{vwidth}')

                    # Exact Match Logic (width >= 900mm)
                    if vwidth in self.exact_match_widths:
                        over_prod_vars[vwidth].ub = 0.0

                # Objective
                total_rolls = gp.quicksum(x[j] for j in range(len(self.patterns)))
                total_over_prod_penalty = gp.quicksum(OVER_PROD_PENALTY * var for var in over_prod_vars.values())
                total_under_prod_penalty = gp.quicksum(UNDER_PROD_PENALTY * var for var in under_prod_vars.values())
                total_complexity_penalty = gp.quicksum(PATTERN_COMPLEXITY_PENALTY * len(self.patterns[j]) * x[j] for j in range(len(self.patterns)))
                
                # Quadratic Piece Count Penalty
                # Note: Gurobi supports quadratic objectives directly
                total_piece_penalty = gp.quicksum(
                PIECE_COUNT_PENALTY * (sum(count for item, count in self.patterns[j].items()) ** 2) * x[j]
                    for j in range(len(self.patterns))
                )

                model.setObjective(total_rolls + total_over_prod_penalty + total_under_prod_penalty + total_complexity_penalty + total_piece_penalty, GRB.MINIMIZE)
                
                # Solve
                model.optimize()

                if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                    status_msg = "Optimal" if model.Status == GRB.OPTIMAL else "Feasible (TimeLimit)"
                    logging.info(f"Using solver: GUROBI for Final MIP (Success: {status_msg}, Obj={model.ObjVal})")
                    solution = {
                        'objective': model.ObjVal,
                        'pattern_counts': {j: x[j].X for j in range(len(self.patterns))},
                        'over_production': {w: over_prod_vars[w].X for w in over_prod_vars},
                        'under_production': {w: under_prod_vars[w].X for w in under_prod_vars}
                    }
                    
                    return solution
                else:
                     logging.warning(f"Gurobi failed to find optimal solution (Status={model.Status}). Fallback to SCIP.")

            except Exception as e:
                logging.warning(f"Gurobi direct execution failed: {e}. Fallback to SCIP.")

        # 2. [Fallback/Default] OR-Tools Solver (SCIP or GLOP)
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        
        # Enable multi-threading
        if hasattr(solver, 'SetNumThreads'):
            solver.SetNumThreads(self.num_threads)

        if is_final_mip:
            solver.SetTimeLimit(self.solver_time_limit_ms)
        else:
            solver.SetTimeLimit(30000) # 30 seconds for LP

        # 변수 정의
        x = {j: solver.IntVar(0, solver.infinity(), f'P_{j}') if is_final_mip else solver.NumVar(0, solver.infinity(), f'P_{j}') for j in range(len(self.patterns))}
        over_prod_vars = {w: solver.NumVar(0, solver.infinity(), f'Over_{w}') for w in self.demands_in_rolls}
        
        under_prod_vars = {}
        for width, required_rolls in self.demands_in_rolls.items():
            allowed_under_prod = max(1, math.ceil(required_rolls))
            under_prod_vars[width] = solver.NumVar(0, allowed_under_prod, f'Under_{width}')

        # 제약조건: 생산량 + 부족량 = 수요량 + 과생산량
        # item_composition의 키가 vwidth이므로 직접 매칭
        constraints = {}
        for vwidth, required_rolls in self.demands_in_rolls.items():
            production_for_width = solver.Sum(
                x[j] * sum(self.item_composition[item_name].get(vwidth, 0) * count for item_name, count in self.patterns[j].items())
                for j in range(len(self.patterns))
            )
            constraints[vwidth] = solver.Add(production_for_width + under_prod_vars[vwidth] == required_rolls + over_prod_vars[vwidth], f'demand_{vwidth}')

            # Exact Match 제약조건 적용 (지폭 >= 900mm)
            if vwidth in self.exact_match_widths:
                # 과생산 금지 (Upper Bound = 0)
                over_prod_vars[vwidth].SetBounds(0.0, 0.0)

        # 목적함수: 총 롤 수 + 페널티 최소화
        total_rolls = solver.Sum(x.values())
        total_over_prod_penalty = solver.Sum(OVER_PROD_PENALTY * var for var in over_prod_vars.values())
        total_under_prod_penalty = solver.Sum(UNDER_PROD_PENALTY * var for var in under_prod_vars.values())
        total_complexity_penalty = solver.Sum(PATTERN_COMPLEXITY_PENALTY * len(self.patterns[j]) * x[j] for j in range(len(self.patterns)))
        
        # 패턴 내 총 롤(piece) 개수에 대한 페널티 추가 (Quadratic: 제곱 비례)
        # 롤 개수가 많을수록 페널티가 급격히 증가하여, 적은 롤 개수의 패턴을 선호하게 함
        # 패턴 내 총 롤(piece) 개수에 대한 페널티 추가 (Quadratic: 제곱 비례)
        # 롤 개수가 많을수록 페널티가 급격히 증가하여, 적은 롤 개수의 패턴을 선호하게 함
        # 수정: item_composition의 values sum(쉬트 수)이 아니라, item 자체의 수(부모 롤 수)를 기준으로 페널티 부과
        total_piece_penalty = solver.Sum(
            PIECE_COUNT_PENALTY * (sum(count for item, count in self.patterns[j].items()) ** 2) * x[j] 
            for j in range(len(self.patterns))
        )

        solver.Minimize(total_rolls + total_over_prod_penalty + total_under_prod_penalty + total_complexity_penalty + total_piece_penalty)
        
        if not is_final_mip:
             logging.debug("--- [DEBUG] Calling OR-Tools solver.Solve() for LP Relaxation...")
        
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

    def _solve_subproblem(self, duals):
        """
        서브 문제(Sub-problem)를 동적 계획법(Dynamic Programming, Knapsack-like)으로 해결합니다.
        
        Master Problem에서 얻은 Dual Value(잠재 가격)를 활용하여, 
        현재 해를 개선할 수 있는(Reduced Cost가 음수인) 새로운 유망 패턴 후보를 탐색합니다.
        """
        width_limit = self.max_width
        piece_limit = self.max_pieces

        # item_details 리스트 초기화: (item_name, item_width, item_value)를 저장
        item_details = []
        for item_name in self.items:
            item_width = self.item_info[item_name]
            # duals 값을 이용하여 item_value 계산
            item_value = sum(count * duals.get(width, 0) for width, count in self.item_composition[item_name].items())
            
            # item_value가 양수인 경우에만 item_details에 추가
            # DP 탐색 시에는 순수 Dual Sum만 사용하고, 페널티는 나중에 적용
            if item_value <= 0:
                continue
            if item_value <= 0:
                continue
            item_details.append((item_name, item_width, item_value))

        # item_details가 비어있으면 빈 리스트 반환
        # 동적 프로그래밍(DP) 테이블 초기화
        if not item_details:
            return []

        dp_value = [[float('-inf')] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_parent = [[None] * (width_limit + 1) for _ in range(piece_limit + 1)]
        dp_value[0][0] = 0.0

        # DP 테이블 채우기: pieces와 width에 대한 모든 조합 고려
        for pieces in range(piece_limit + 1):
            for width in range(width_limit + 1):
                # 현재 상태의 값 가져오기
                current_value = dp_value[pieces][width]
                # 현재 상태에 도달할 수 없는 경우, 다음 상태로 건너뛰기
                if current_value == float('-inf'):
                    continue
                # 모든 아이템을 반복하여 추가할지 고려
                for item_name, item_width, item_value in item_details:
                    # 다음 상태 계산
                    next_pieces = pieces + 1
                    next_width = width + item_width
                    # 다음 상태가 범위를 벗어나면 건너뛰기
                    if next_pieces > piece_limit or next_width > width_limit:
                        continue
                    # 새로운 값 계산 및 DP 테이블 업데이트
                    new_value = current_value + item_value
                    if new_value > dp_value[next_pieces][next_width] + 1e-9:
                        dp_value[next_pieces][next_width] = new_value
                        dp_parent[next_pieces][next_width] = (pieces, width, item_name)

        candidate_patterns = []
        seen_patterns = set()

        # 최적의 패턴 후보 추출
        for pieces in range(self.min_pieces, piece_limit + 1):
            for width in range(self.min_width, width_limit + 1):
                # 현재 값과 부모 정보 가져오기
                value = dp_value[pieces][width]
                # 값이 특정 임계값보다 작으면 건너뛰기
                if value <= 1.0 + 1e-6:
                    continue
                parent = dp_parent[pieces][width]
                # 부모가 없으면 건너뛰기
                if not parent:
                    continue

                # 패턴 재구성
                pattern = {}
                cur_pieces, cur_width = pieces, width
                # 부모를 따라 현재 패턴 재구성
                while cur_pieces > 0:
                    parent_info = dp_parent[cur_pieces][cur_width]
                    if not parent_info:
                        pattern = None
                        break
                    prev_pieces, prev_width, item_name = parent_info
                    pattern[item_name] = pattern.get(item_name, 0) + 1
                    cur_pieces, cur_width = prev_pieces, prev_width

                # 패턴이 유효하지 않으면 건너뛰기
                if not pattern or cur_pieces != 0 or cur_width != 0:
                    continue

                # 패턴이 이미 seen_patterns에 있으면 건너뛰기
                pattern_key = frozenset(pattern.items())
                if pattern_key in seen_patterns:
                    continue

                total_width = sum(self.item_info[name] * count for name, count in pattern.items())
                if total_width < self.min_width or total_width > self.max_width:
                    # 패턴이 너비 제약 조건을 충족하지 못하면 건너뛰기
                    continue

                # Reduced Cost 계산 (Quadratic Penalty 적용)
                # Reduced Cost = Dual_Sum - (1 + Penalty * pieces^2)
                # DP value는 Dual Sum을 담고 있음
                reduced_cost = value - (1.0 + PIECE_COUNT_PENALTY * (pieces ** 2))

                seen_patterns.add(pattern_key)
                # value를 reduced_cost로 업데이트하여 정렬 시 사용
                candidate_patterns.append({'pattern': pattern, 'value': reduced_cost, 'width': total_width, 'pieces': pieces})

        if not candidate_patterns:
            return []
        
        # 가치에 따라 후보 정렬
        candidate_patterns.sort(key=lambda x: x['value'], reverse=True)

        # 상위 N개 패턴 반환
        return [cand['pattern'] for cand in candidate_patterns[:CG_SUBPROBLEM_TOP_N]]

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

    def _add_fallback_patterns(self):
        """
        초기 패턴 생성(Brute-force) 후에 호출되어, 
        각 주문 지폭에 대해 min_width 제약을 무시하고 생성 가능한 가장 넓은 '순수 패턴'을 강제로 추가합니다.
        
        목적: 
        - min_width 제약으로 인해 소량 주문 지폭을 처리할 패턴이 아예 생성되지 않거나,
        - 다른 대량 주문 지폭과 섞인 '혼합 패턴'만 존재하여, 소량 주문 처리 시 막대한 과생산이 발생하는 것을 방지합니다.
        - Trim Loss가 발생하더라도(롤 수 증가), 과생산 페널티(5억)를 피할 수 있는 '비상 탈출구'를 솔버에게 제공합니다.
        """
        logging.info("--- [Fallback] 과생산 방지를 위한 순수/효율 패턴 추가 시작 ---")
        added_count = 0
        seen_patterns = {frozenset(p.items()) for p in self.patterns}
        
        for width in self.order_widths:
            # 1. 해당 지폭으로 만들 수 있는 가장 긴 '순수 아이템 패턴' 찾기
            best_pure_pattern = None
            max_width_found = 0
            
            # 1-1. 복합폭 아이템 확인
            pass_pure = False
            for item in self.items:
                # 해당 아이템이 이 지폭으로만 구성되어 있는지 확인
                if item in self.item_composition and len(self.item_composition[item]) == 1 and width in self.item_composition[item]:
                     # 예: "420x4"(1704) -> 420mm
                     item_width = self.item_info[item]
                     
                     # 이 아이템을 최대 몇 개 넣을 수 있는지 계산
                     max_count = min(int(self.max_width / item_width), self.max_pieces)
                     if max_count > 0:
                         current_width = item_width * max_count
                         if current_width > max_width_found:
                             max_width_found = current_width
                             best_pure_pattern = {item: max_count}
            
            # 1-2. 만약 복합폭 아이템으로 찾지 못했거나 더 좋은 게 있을 수 있으므로 직접 계산
            # (사실 위의 루프에서 다 커버되지만, 혹시 items에 없는 조합이 있을까봐)
            
            if best_pure_pattern:
                # min_width 제약조건 체크 추가
                if max_width_found < self.min_width:
                    logging.debug(f"    - [스킵] 지폭 {width}mm 순수 패턴: {best_pure_pattern} (폭: {max_width_found}mm < min_width: {self.min_width}mm)")
                    continue
                pattern_key = frozenset(best_pure_pattern.items())
                if pattern_key not in seen_patterns:
                    self.patterns.append(best_pure_pattern)
                    seen_patterns.add(pattern_key)
                    added_count += 1
                    logging.info(f"    -> [추가] 지폭 {width}mm 순수 패턴: {best_pure_pattern} (폭: {max_width_found}mm)")
        
        logging.info(f"--- [Fallback] 총 {added_count}개의 순수/효율 패턴이 추가되었습니다. ---")

    def run_optimize(self, start_prod_seq=0):
        """최적화 실행 메인 함수"""

        logging.info(f"OVER_PROD_PENALTY: {OVER_PROD_PENALTY}")
        logging.info(f"UNDER_PROD_PENALTY: {UNDER_PROD_PENALTY}")
        logging.info(f"PATTERN_COMPLEXITY_PENALTY: {PATTERN_COMPLEXITY_PENALTY}")
        logging.info(f"PIECE_COUNT_PENALTY: {PIECE_COUNT_PENALTY}")
        logging.info(f"SOLVER_TIME_LIMIT_MS: {self.solver_time_limit_ms}")


        if len(self.order_widths) <= SMALL_PROBLEM_THRESHOLD:
            logging.info(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 모든 패턴을 탐색합니다 (Small-scale) ---")
            self._generate_all_patterns()
            
            self._generate_initial_patterns_db()  # DB에서 사용자 편집 패턴 추가 (Small-scale에도 적용)
        else:
            logging.info(f"\n--- 주문 종류가 {len(self.order_widths)}개 이므로, 열 생성 기법을 시작합니다 (Large-scale) ---")
            self._generate_initial_patterns_db()  # DB에서 사용자 편집 패턴을 먼저 추가
            self._generate_initial_patterns()
            
            initial_pattern_count = len(self.patterns)
            # min_width, max_width, max_pieces 제약조건 모두 적용하여 필터링
            self.patterns = [p for p in self.patterns 
                             if self.min_width <= sum(self.item_info[i] * c for i, c in p.items()) <= self.max_width
                             and sum(p.values()) <= self.max_pieces]
            logging.info(f"--- 초기 패턴 필터링: {initial_pattern_count}개 -> {len(self.patterns)}개 (너비 범위 {self.min_width}~{self.max_width}mm, 최대 {self.max_pieces}폭 적용)")

            if not self.patterns:
                return {"error": "초기 유효 패턴을 생성할 수 없습니다. 제약조건이 너무 엄격할 수 있습니다."}

            # # 열 생성 루프
            # no_improvement_count = 0
            # for iteration in range(CG_MAX_ITERATIONS):
            #     master_solution = self._solve_master_problem()
            #     if not master_solution or 'duals' not in master_solution:
            #         break

            #     new_patterns = self._solve_subproblem(master_solution['duals'])
                
            #     patterns_added = 0
            #     if new_patterns:
            #         current_pattern_keys = {frozenset(p.items()) for p in self.patterns}
            #         for new_pattern in new_patterns:
            #             if frozenset(new_pattern.items()) not in current_pattern_keys:
            #                 pattern_width = sum(self.item_info[item] * count for item, count in new_pattern.items())
            #                 # min_width와 max_width 모두 검증
            #                 if self.min_width <= pattern_width <= self.max_width:
            #                     self.patterns.append(new_pattern)
            #                     patterns_added += 1
                
            #     if patterns_added > 0:
            #         logging.debug(f"--- [CG] Iteration {iteration}: Added {patterns_added} new patterns. Obj={master_solution.get('objective', 'N/A'):.2f}")
            #         no_improvement_count = 0
            #     else:
            #         logging.debug(f"--- 더 이상 유효한 신규 패턴이 생성되지 않아 조기 종료합니다 (반복 {iteration}). ---")
            #         break
                
            #     if no_improvement_count >= CG_NO_IMPROVEMENT_LIMIT:
            #         logging.debug(f"--- {CG_NO_IMPROVEMENT_LIMIT}번의 반복 동안 개선이 없어 수렴으로 간주하고 종료합니다 (반복 {iteration}). ---")
            #         break

        if not self.patterns:
            return {"error": "유효한 패턴을 생성할 수 없습니다."}

        logging.info(f"--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        
        # 최종 최적화 전에 min_width/max_width 제약조건 최종 검증 (안전장치)
        pre_filter_count = len(self.patterns)
        self.patterns = [p for p in self.patterns 
                         if self.min_width <= sum(self.item_info.get(i, 0) * c for i, c in p.items()) <= self.max_width]
        if len(self.patterns) < pre_filter_count:
            logging.info(f"--- [최종 필터링] 너비 범위({self.min_width}~{self.max_width}mm) 벗어난 패턴 {pre_filter_count - len(self.patterns)}개 제거됨 ---")
        
        # 최종 최적화 전에 패턴 통합(Consolidation) 한 번 더 수행
        # self._consolidate_patterns()

        final_solution = self._solve_master_problem(is_final_mip=True)        
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
        return self._format_results(final_solution, start_prod_seq=start_prod_seq)

    def _format_results(self, final_solution, start_prod_seq=0):
        """최종 결과를 데이터프레임 형식으로 포매팅합니다."""
        
        # 결과 데이터프레임 생성
        (
            result_patterns,
            pattern_details_for_db,
            pattern_roll_details_for_db,
            pattern_roll_cut_details_for_db,
            demand_tracker,
            last_prod_seq,
        ) = self._build_pattern_details(final_solution, start_prod_seq=start_prod_seq)
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['pattern', 'pattern_width', 'count', 'loss_per_roll', 'pattern_length', 'wd_width']]

        # 주문 이행 요약 생성 (수정된 _build_fulfillment_summary 호출)
        fulfillment_summary = self._build_fulfillment_summary(demand_tracker)

        logging.info("\n[주문 이행 요약 (그룹오더별)]")
        # logging.info(fulfillment_summary.to_string())

        return {
            "pattern_result": df_patterns,  # TSP 정렬 순서 유지 (count 정렬 제거)
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "last_prod_seq": last_prod_seq,
        }

    def _build_pattern_details(self, final_solution, start_prod_seq=0):
        """
        패턴 사용 결과와 DB 저장을 위한 상세 정보를 생성합니다.
        roll_optimize.py와 유사하게, 패턴별로 하나의 레코드를 생성하고 Count를 사용합니다.
        """
        demand_tracker = self.df_orders.copy()
        demand_tracker['original_order_idx'] = demand_tracker.index
        demand_tracker = demand_tracker[['original_order_idx', 'group_order_no', '지폭', 'rolls']].copy()
        demand_tracker['fulfilled'] = 0
        demand_tracker = demand_tracker.sort_values(by=['지폭', 'group_order_no']).reset_index(drop=True)

        result_patterns = []
        pattern_details_for_db = []
        pattern_roll_details_for_db = []
        pattern_roll_cut_details_for_db = []
        
        prod_seq_counter = start_prod_seq
        total_cut_seq_counter = 0

        pattern_summary_map = {}
        for j, pattern_dict in enumerate(self.patterns):
            sorted_pattern_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)
            pattern_item_strs = []
            total_width_for_pattern = 0
            for item_name, num_of_composite in sorted_pattern_items:
                composite_width = self.item_info[item_name]
                total_width_for_pattern += composite_width * num_of_composite
                base_width_dict = self.item_composition[item_name]
                base_width, num_of_base = list(base_width_dict.items())[0]
                formatted_name = f"{composite_width}({base_width}*{num_of_base})"
                pattern_item_strs.extend([formatted_name] * num_of_composite)
            pattern_summary_map[j] = {
                'pattern': ' + '.join(pattern_item_strs),
                'wd_width': total_width_for_pattern,
                'loss_per_roll': self.original_max_width - total_width_for_pattern
            }

        # Helper for safe int conversion
        def safe_int(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        # Extract common properties from the first row of the dataframe
        first_row = self.df_orders.iloc[0]
        common_props = {
            'diameter': 0, # Sheet orders don't use diameter
            'color': first_row.get('color', ''),
            'luster': safe_int(first_row.get('luster', 0)),
            'p_lot': self.lot_no,
            'core': 0, # Sheet orders don't use core
            'order_pattern': first_row.get('order_pattern', '')
        }
        logging.info(f"--- [DEBUG] Building Pattern Details. Start Prod Seq: {start_prod_seq}")
        
        # ====== 패턴 정렬: 칼날 교체 최소화를 위한 TSP-like 정렬 ======
        def get_pattern_widths(pattern_idx):
            """패턴에서 사용되는 복합폭 집합 반환 (칼날 위치 기준)"""
            pattern_dict = self.patterns[pattern_idx]
            composite_widths = set()
            for item_name in pattern_dict.keys():
                # item_info에서 복합폭 추출 (예: "879x1" -> 903mm)
                if item_name in self.item_info:
                    composite_widths.add(self.item_info[item_name])
            return composite_widths
        
        def calculate_knife_change_cost(widths1, widths2):
            """두 패턴 간 칼날 교체 비용 계산
            
            - 기본 비용: 변경되는 지폭 수 (대칭 차집합)
            - Tie-breaker: 공통 지폭이 많을수록 좋음 (음수로 반환하여 우선순위 높임)
            
            반환값: (변경 비용, -공통 지폭 수) 튜플로 정렬에 사용
            """
            changes = len(widths1 ^ widths2)  # 대칭 차집합: 변경되는 지폭
            common = len(widths1 & widths2)   # 교집합: 공통 지폭
            return (changes, -common)  # 변경 적고, 공통 많을수록 좋음
        
        def sort_patterns_by_similarity(pattern_indices):
            """Greedy TSP: 현재 패턴과 가장 유사한 다음 패턴 선택"""
            if len(pattern_indices) <= 1:
                return pattern_indices
            
            # 각 패턴의 지폭 집합 미리 계산
            pattern_widths_map = {idx: get_pattern_widths(idx) for idx in pattern_indices}
            
            # 가장 큰 지폭을 가진 패턴을 시작점으로 선택 (안정적인 시작)
            sorted_patterns = []
            remaining = list(pattern_indices)
            
            # 첫 패턴 선택: 가장 큰 최대 지폭을 가진 패턴
            first_pattern = max(remaining, key=lambda idx: max(pattern_widths_map[idx]) if pattern_widths_map[idx] else 0)
            sorted_patterns.append(first_pattern)
            remaining.remove(first_pattern)
            
            # Greedy 탐색
            while remaining:
                current_widths = pattern_widths_map[sorted_patterns[-1]]
                # 칼날 교체 비용이 가장 적은 다음 패턴 선택
                next_pattern = min(remaining, 
                                   key=lambda idx: calculate_knife_change_cost(current_widths, pattern_widths_map[idx]))
                sorted_patterns.append(next_pattern)
                remaining.remove(next_pattern)
            
            return sorted_patterns
        
        # 사용할 패턴 인덱스 수집 (count > 0인 것만)
        used_pattern_indices = [j for j, count in final_solution['pattern_counts'].items() if count > 0.001]
        
        # 패턴 정렬
        sorted_pattern_indices = sort_patterns_by_similarity(used_pattern_indices)
        
        # [DEBUG] 정렬 결과 로그
        logging.info(f"--- [TSP 정렬] 패턴 수: {len(sorted_pattern_indices)}, 정렬 순서: {sorted_pattern_indices}")
        for idx, j in enumerate(sorted_pattern_indices):
            pat_widths = get_pattern_widths(j)
            logging.info(f"    {idx+1}. 패턴 {j}: 지폭={sorted(pat_widths, reverse=True)}")
        
        # 1. Expand patterns into individual rolls
        all_rolls_data = [] # List of dictionaries, each dict represents one roll

        # 정렬된 순서로 패턴 처리 (칼날 교체 최소화)
        for j in sorted_pattern_indices:
            count = final_solution['pattern_counts'][j]
            if count <= 0.001:
                continue
            


            batch_count = int(round(count))
            roll_count = batch_count # Restore variable name for downstream logic

            pattern_dict = self.patterns[j]
            


            # prod_seq_counter += 1 # Original line, moved to inside grouped_batches loop
            
            # --- [Modified Logic] Assign orders per roll and group into batches ---
            
            # 1. Generate assignment for EACH roll individually
            roll_assignments = [] # List of (composite_widths, composite_group_nos) for each roll
            
            # We need to iterate 'roll_count' times. 
            # But we also have 'num_of_composite' for each item in the pattern.
            # The original logic iterated pattern items, then num_of_composite, then roll_count (implicitly via multiplication).
            # To assign per roll, we must iterate roll_count first, then the pattern structure.
            
            sorted_pattern_items = sorted(pattern_dict.items(), key=lambda item: self.item_info[item[0]], reverse=True)

            all_rolls_data = []
            
            for _ in range(roll_count):
                roll_data = {
                    'composite_widths': [],
                    'composite_group_nos': [],
                    'roll_details': [] # List of dicts for pattern_roll_details
                }
                
                for item_name, num_of_composite in sorted_pattern_items:
                    composite_width = self.item_info[item_name]
                    base_width_dict = self.item_composition[item_name]

                    for _ in range(num_of_composite):
                        base_widths_for_item = []
                        base_group_nos_for_item = []
                        assigned_group_no_for_composite = None

                        for g_no_key, num_of_base in base_width_dict.items():
                            # group_order_no 기반이므로 해당 그룹오더의 지폭 조회
                            info = self.group_order_info.get(g_no_key, {})
                            orig_width = info.get('width', 0)
                            
                            for _ in range(num_of_base):
                                # 해당 group_order_no에 직접 할당
                                target_indices = demand_tracker[
                                    (demand_tracker['group_order_no'] == g_no_key) &
                                    (demand_tracker['fulfilled'] < demand_tracker['rolls'])
                                ].index
                                
                                assigned_group_no = g_no_key  # 기본적으로 아이템의 그룹오더에 할당
                                if not target_indices.empty:
                                    target_idx = target_indices[0]
                                    assigned_group_no = demand_tracker.loc[target_idx, 'group_order_no']
                                    demand_tracker.loc[target_idx, 'fulfilled'] += 1
                                else:
                                    # 해당 그룹오더가 이미 충족되었으면 과생산으로 처리
                                    fallback_indices = demand_tracker[demand_tracker['group_order_no'] == g_no_key].index
                                    if not fallback_indices.empty:
                                        target_idx = fallback_indices[0]
                                        assigned_group_no = demand_tracker.loc[target_idx, 'group_order_no']
                                        demand_tracker.loc[target_idx, 'fulfilled'] += 1
                                    else:
                                        assigned_group_no = "OVERPROD"
                                
                                base_widths_for_item.append(orig_width)
                                base_group_nos_for_item.append(assigned_group_no)
                                
                                if assigned_group_no_for_composite is None:
                                    assigned_group_no_for_composite = assigned_group_no
                        
                        roll_data['composite_widths'].append(composite_width)
                        roll_data['composite_group_nos'].append(assigned_group_no_for_composite if assigned_group_no_for_composite is not None else "")
                        
                        roll_data['roll_details'].append({
                            'rollwidth': composite_width,
                            'widths': (base_widths_for_item + [0] * 7)[:7],
                            'group_nos': (base_group_nos_for_item + [''] * 7)[:7]
                        })
                
                all_rolls_data.append(roll_data)



            # 2. Group identical rolls into batches
            import itertools
            
            # Helper to create a key for grouping (tuple of tuples)
            def get_group_key(r_data):
                return (
                    tuple(r_data['composite_widths']),
                    tuple(r_data['composite_group_nos']),
                    # We must also check if the *internal* assignments (base items) are identical
                    tuple(
                        (d['rollwidth'], tuple(d['widths']), tuple(d['group_nos'])) 
                        for d in r_data['roll_details']
                    )
                )

            # Group consecutive identical rolls
            grouped_batches = []
            for key, group in itertools.groupby(all_rolls_data, key=get_group_key):
                batch_rolls = list(group)
                
                grouped_batches.append({
                    'count': len(batch_rolls),
                    'data': batch_rolls[0] # Representative data
                })
            
            # 3. Generate DB records for each batch
            for batch in grouped_batches:
                prod_seq_counter += 1
                batch_count = batch['count']
                r_data = batch['data']
                
                # pattern_details_for_db
                pattern_details_for_db.append({
                    'pattern_length': self.sheet_roll_length,
                    'count': batch_count,
                    'widths': (r_data['composite_widths'] + [0] * 8)[:8],
                    'group_nos': (r_data['composite_group_nos'] + [''] * 8)[:8],
                    'prod_seq': prod_seq_counter,
                    'rs_gubun': 'S',
                    **common_props
                })

                # Add to result_patterns (CSV)
                # Reconstruct pattern string from composite widths
                # Note: The original pattern_str might have been sorted or formatted differently.
                # We'll use the composite widths to build a string.
                # Or better, we can just use the widths from r_data.
                # But we need to match the format "w1, w2, ..."
                
                # Filter out 0s
                valid_widths = [w for w in r_data['composite_widths'] if w > 0]
                # Sort to match typical pattern representation if needed, but keeping order is fine.
                # valid_widths.sort(reverse=True) 
                batch_pattern_str = ", ".join(map(str, valid_widths))
                
                # Calculate loss
                batch_roll_width = sum(valid_widths)
                batch_loss = self.max_width - batch_roll_width # Assuming max_width is the roll width constraint
                
                result_patterns.append({
                    'pattern': batch_pattern_str,
                    'pattern_width': batch_roll_width,
                    'count': batch_count,
                    'loss_per_roll': batch_loss,
                    'pattern_length': self.sheet_roll_length,
                    'wd_width': batch_roll_width
                })
                
                # pattern_roll_details_for_db & pattern_roll_cut_details_for_db
                roll_seq_counter = 0
                for detail in r_data['roll_details']:
                    roll_seq_counter += 1
                    
                    pattern_roll_details_for_db.append({
                        'rollwidth': detail['rollwidth'],
                        'pattern_length': self.sheet_roll_length,
                        'widths': detail['widths'],
                        'group_nos': detail['group_nos'],
                        'count': batch_count,
                        'prod_seq': prod_seq_counter,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'S',
                        'sheet_trim': self.sheet_trim,
                        **common_props
                    })
                    
                    cut_seq_counter = 0
                    for i in range(len(detail['widths'])):
                        width = detail['widths'][i]
                        if width > 0:
                            cut_seq_counter += 1
                            total_cut_seq_counter += 1
                            group_no = detail['group_nos'][i]
                            weight = (self.b_wgt * (width / 1000) * self.sheet_roll_length)
                            
                            pattern_roll_cut_details_for_db.append({
                                'prod_seq': prod_seq_counter,
                                'unit_no': prod_seq_counter,
                                'seq': total_cut_seq_counter, # Global sequence? Or per prod_seq? Keeping global for now.
                                'roll_seq': roll_seq_counter,
                                'cut_seq': cut_seq_counter,
                                'width': width,
                                'group_no': group_no,
                                'weight': weight,
                                'pattern_length': self.sheet_roll_length,
                                'count': batch_count,
                                'rs_gubun': 'S',
                                **common_props
                            })

        return (
            result_patterns,
            pattern_details_for_db,
            pattern_roll_details_for_db,
            pattern_roll_cut_details_for_db,
            demand_tracker,
            prod_seq_counter,
        )

    def _build_fulfillment_summary(self, demand_tracker):
        """주문 이행 요약 데이터프레임을 생성합니다. (개별 주문별)"""
        
        summary_df = self.df_orders[['group_order_no', '가로', '세로', '수출내수', '등급', '주문톤', 'rolls']].copy()
        summary_df.rename(columns={'rolls': '필요롤수', '주문톤': '주문량(톤)'}, inplace=True)
        
        # Merge fulfilled_rolls directly using original_order_idx
        # demand_tracker has 'original_order_idx' from _build_pattern_details
        summary_df = pd.merge(summary_df, demand_tracker[['original_order_idx', 'fulfilled']], 
                              left_index=True, right_on='original_order_idx', how='left')
        summary_df.rename(columns={'fulfilled': '생산롤수'}, inplace=True)
        summary_df.drop(columns=['original_order_idx'], inplace=True) # Drop the temporary merge key

        summary_df['생산롤수'] = summary_df['생산롤수'].fillna(0).astype(int)
        
        # 4. 과부족 및 생산톤 계산
        summary_df['과부족(롤)'] = summary_df['생산롤수'].astype(int) - summary_df['필요롤수'].astype(int)
        
        # 실제 생산량 계산: (가로(mm) × 세로(mm) × b_wgt(g/m²) × 생산롤수) / 10^9
        # 세로 = sheet_roll_length (롤 길이, mm 단위로 변환 필요시 확인)
        # 쉬트지의 세로는 개별 쉬트 길이가 아니라 롤 길이(sheet_roll_length)를 사용
        summary_df['생산량(톤)'] = (
            summary_df['가로'] * self.sheet_roll_length * self.b_wgt * summary_df['생산롤수'] / 1e9
        ).round(3)
        summary_df['과부족(톤)'] = (summary_df['생산량(톤)'] - summary_df['주문량(톤)']).round(3)

        # 최종 컬럼 순서 정리
        return summary_df[[
            'group_order_no', '가로', '세로', '수출내수', '등급', '주문량(톤)', '생산량(톤)', '과부족(톤)',
            '필요롤수', '생산롤수', '과부족(롤)'
        ]]
    
    def _generate_initial_patterns_test(self):
        """초기 패턴 생성을 위해 First-Fit-Decreasing 휴리스틱을 사용합니다."""
        logging.info("\n--- 유효한 초기 패턴을 생성합니다 ---")
        
        # frozenset으로 패턴 중복 체크를 효율적으로 관리
        seen_patterns = set()

        # # 기존: 너비가 넓은 아이템부터 순서대로 처리 (First-Fit-Decreasing)
        # # sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)

        # 변경: 아이템 순서를 랜덤으로 섞어 다양한 패턴 생성 시도 (First-Fit)
        # 이렇게 하면 매번 실행할 때마다 다른 순서로 패턴 생성을 시도하여
        # 더 다양한 초기 패턴을 탐색할 수 있습니다.
        # randomized_items = list(self.items)
        # random.shuffle(randomized_items)
        
        

        # for item in randomized_items:
        #     item_width = self.item_info[item]
            
        #     current_pattern = {item: 1}
        #     current_width = item_width
        #     current_pieces = 1

        #     while current_pieces < self.max_pieces:
        #         remaining_width = self.max_width - current_width
                
        #         # 남은 공간에 맞는 '첫 번째' 아이템을 찾음 (First-Fit)
        #         # randomized_items가 정렬되어 있지 않으므로, 가장 큰 아이템이 아니라 랜덤 순서에서 처음으로 발견되는 아이템이 선택됩니다.
        #         best_fit_item = next((i for i in randomized_items if self.item_info[i] <= remaining_width), None)
                
        #         if not best_fit_item:
        #             break 

        #         current_pattern[best_fit_item] = current_pattern.get(best_fit_item, 0) + 1
        #         current_width += self.item_info[best_fit_item]
        #         current_pieces += 1

        #     while current_width < self.min_width and current_pieces < self.max_pieces:
        #         # 너비가 min_width보다 작은 경우, 추가 아이템을 탐색하여 보정합니다.
        #         item_to_add = next((i for i in reversed(randomized_items) if current_width + self.item_info[i] <= self.max_width), None)
                
        #         if item_to_add:
        #             current_pattern[item_to_add] = current_pattern.get(item_to_add, 0) + 1
        #             current_width += self.item_info[item_to_add]
        #             current_pieces += 1
        #         else:
        #             break # 더 이상 추가할 아이템이 없으면 종료

        #     if self.min_width <= current_width and self.min_pieces <= current_pieces:
        #         pattern_key = frozenset(current_pattern.items())
        #         if pattern_key not in seen_patterns:
        #             self.patterns.append(current_pattern)
        #             seen_patterns.add(pattern_key)
        # print(f"--- {len(self.patterns)}개의 혼합 패턴 생성됨 ---")

        # 너비가 넓은 아이템부터 순서대로 처리 (First-Fit-Decreasing)
        sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=True)

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
        logging.info(f"--- {len(self.patterns)}개의 혼합 패턴 생성됨 ---")


        # 너비가 작은 아이템부터 순서대로 처리 (First-Fit-Decreasing)
        sorted_items = sorted(self.items, key=lambda i: self.item_info[i], reverse=False)

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
        logging.info(f"--- {len(self.patterns)}개의 혼합 패턴 생성됨 ---")

        # --- 2. 모든 복합폭에 대해 '순수 품목 패턴' 생성 ---
        pure_patterns_added = 0
        for item in sorted_items:
            item_width = self.item_info.get(item, 0)
            if item_width <= 0: continue

            # 해당 아이템으로만 구성된 패턴 생성 시도
            num_items = min(int(self.max_width / item_width), self.max_pieces)
            
            # 너비가 큰 조합부터 작은 조합까지 순차적으로 확인
            while num_items > 0:
                new_pattern = {item: num_items}
                total_width = item_width * num_items
                
                if self.min_width <= total_width and self.min_pieces <= num_items:
                    pattern_key = frozenset(new_pattern.items())
                    if pattern_key not in seen_patterns:
                        self.patterns.append(new_pattern)
                        seen_patterns.add(pattern_key)
                        pure_patterns_added += 1
                        break # 이 아이템으로 만들 수 있는 가장 좋은 순수패턴을 찾았으므로 종료
                
                num_items -= 1

        if pure_patterns_added > 0:
            logging.info(f"--- {pure_patterns_added}개의 순수 품목 패턴 추가됨 ---")

        # --- 폴백 로직: 초기 패턴으로 커버되지 않는 주문이 있는지 최종 확인 ---
        # item_composition의 키가 vwidth이므로 직접 비교 가능
        covered_vwidths = {w for p in self.patterns for item_name in p for w in self.item_composition.get(item_name, {})}
        uncovered_widths = set(self.order_widths) - covered_vwidths

        if uncovered_widths:
            logging.info(f"--- 경고: 초기 패턴에 포함되지 않은 그룹오더 발견: {uncovered_widths} ---")
            logging.info("--- 해당 주문에 대한 폴백 패턴을 추가 생성합니다. ---")
            
            for g_no in uncovered_widths:
                info = self.group_order_info.get(g_no, {})
                orig_width = info.get('width', 0)
                
                logging.info(f"  - 그룹오더 {g_no} ({orig_width}mm)에 대한 순수 품목 패턴 생성 시도...")

                # 1. 이 그룹오더로 만들 수 있는 유효한 복합폭 아이템 목록을 찾습니다.
                valid_components = []
                for i in range(1, 5): # 1~4폭 고려
                    item_name = f"G{g_no}x{i}"
                    # 아이템이 이미 생성되었는지 확인
                    if item_name in self.item_info:
                        valid_components.append(item_name)
                    else:
                        # 동적으로 생성 및 유효성 검사
                        composite_width = orig_width * i + self.sheet_trim
                        if (self.min_sc_width <= composite_width <= self.max_sc_width) and \
                           (composite_width <= self.original_max_width):
                            # 유효하면 아이템 정보에 추가
                            if item_name not in self.items: self.items.append(item_name)
                            self.item_info[item_name] = composite_width
                            self.item_composition[item_name] = {g_no: i}  # group_order_no 저장
                            valid_components.append(item_name)

                if not valid_components:
                    logging.info(f"    - 경고: 그룹오더 {g_no} ({orig_width}mm)로 만들 수 있는 유효한 복합폭 아이템이 없습니다. 폴백 패턴을 생성할 수 없습니다.")
                    continue

                # 2. 너비가 넓은 순으로 정렬하여 Greedy 알고리즘 준비
                sorted_components = sorted(valid_components, key=lambda i: self.item_info[i], reverse=True)
                
                # 3. Greedy 방식으로 최적의 단일 품목 패턴 구성
                new_pattern = {}
                current_width = 0
                current_pieces = 0
                
                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    
                    # 남은 공간에 들어갈 수 있는 가장 큰 구성요소 찾기
                    best_fit = next((item for item in sorted_components if self.item_info[item] <= remaining_width), None)
                    
                    if not best_fit:
                        break # 더 이상 추가할 수 있는 구성요소가 없음
                    
                    new_pattern[best_fit] = new_pattern.get(best_fit, 0) + 1
                    current_width += self.item_info[best_fit]
                    current_pieces += 1

                # 4. 생성된 패턴의 유효성 검사 및 추가
                if new_pattern:
                    total_width = sum(self.item_info[name] * count for name, count in new_pattern.items())
                    total_pieces = sum(new_pattern.values())

                    if self.min_width -200 <= total_width and self.min_pieces <= total_pieces:
                        pattern_key = frozenset(new_pattern.items())
                        if pattern_key not in seen_patterns:
                            self.patterns.append(new_pattern)
                            seen_patterns.add(pattern_key)
                            logging.info(f"    -> 생성된 순수 패턴: {new_pattern} (너비: {total_width}mm, 폭 수: {total_pieces}) -> 폴백 패턴으로 추가됨.")
                        else:
                            logging.info(f"    - 생성된 순수 패턴 {new_pattern}은 이미 존재합니다.")
                    else:
                        logging.info(f"    - 생성된 순수 패턴 {new_pattern}이 최종 제약조건(최소너비/폭수)을 만족하지 못합니다. (너비: {total_width}, 폭 수: {total_pieces})")
                else:
                    logging.info(f"    - 가상지폭 {vwidth} (원본: {orig_width}mm)에 대한 순수 패턴을 구성하지 못했습니다.")

        logging.info(f"--- 총 {len(self.patterns)}개의 초기 패턴 생성됨 ---")
        logging.info(self.patterns)
        logging.info("--------------------------\n")

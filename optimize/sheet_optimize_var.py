"""
[쉬트 가변 길이 최적화 모듈]
이 파일은 쉬트지(Sheet) 주문에 대해 최적의 재단 패턴을 찾는 로직을 담고 있습니다.
주요 특징:
1.  **가변 길이 대응**: 쉬트의 길이가 Min/Max 범위 내에서 가변적일 수 있는 상황을 고려(하려고 했으나 현재 코드는 표준 길이 기반으로 보임)하거나, 
    표준 길이를 기준으로 하되 여러 장(Sheet)을 합쳐서 복합 폭을 구성하는 방식을 사용합니다.
2.  **톤 -> 미터 변환**: 주문량(톤)을 생산 설비 기준인 길이(미터)로 변환하여 최적화를 수행합니다.
3.  **복합 폭(Composite Item) 활용**: 나이프(Knife) 개수 제약 등을 고려하여, 동일한 폭의 주문을 1~4장 묶어서 하나의 'Item'으로 취급해 최적화 효율을 높입니다.
4.  **DB 연동 데이터 생성**: 최적화 결과를 DB 테이블(TH_SHEET_PATTERN, TH_SHEET_ROLL_SEQ 등)에 저장할 수 있는 형태로 상세하게 변환합니다.

주의: 현재 코드에는 핵심 최적화 엔진인 `_solve_master_problem_ilp` 메서드가 누락되어 있어 실행 시 오류가 발생할 수 있습니다.
"""
import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import math
import random
import time

# --- 최적화 설정 상수 ---
# 페널티 값
OVER_PROD_PENALTY = 200.0    # 과생산에 대한 페널티
UNDER_PROD_PENALTY = 500.0  # 부족생산에 대한 페널티
PATTERN_COMPLEXITY_PENALTY = 0.01  # 패턴 복잡성에 대한 페널티
ITEM_SINGLE_STRIP_PENALTIES = {}
DEFAULT_SINGLE_STRIP_PENALTY = 100.0  # 지정되지 않은 단일폭은 기본적으로 패널티 없음
DISALLOWED_SINGLE_BASE_WIDTHS = {}  # 단일 사용을 금지할 주문 폭 집합

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 2      # 패턴에 포함될 수 있는 최소 폭(piece)의 수
SMALL_PROBLEM_THRESHOLD = 8     # 전체 탐색을 수행할 최대 주문 지폭 종류 수
SOLVER_TIME_LIMIT_MS = 300000    # 최종 MIP 솔버의 최대 실행 시간 (밀리초)
CG_MAX_ITERATIONS = 100000         # 열 생성(Column Generation) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 100    # 개선 없는 경우, 열 생성 조기 종료 조건
CG_SUBPROBLEM_TOP_N = 10         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴
# 나이프 로드 제약: 패턴 생산 횟수는 k1*a + k2*b 형태여야 함 (a,b>=0 정수)
KNIFE_LOAD_K1 = 3
KNIFE_LOAD_K2 = 4

NUM_THREADS = 4


class SheetOptimizeVar:
    def __init__(
            self,
            df_spec_pre,
            max_width,
            min_width,
            max_pieces,
            b_wgt,
            min_sheet_roll_length,
            max_sheet_roll_length,
            sheet_trim,
            min_sc_width,
            max_sc_width,
            db=None,
            lot_no=None,
            version=None
    ):
        """
        초기화 메서드
        :param df_spec_pre: 주문 데이터 프레임
        :param max_width: 설비 최대 폭 (Mother Roll Width)
        :param min_width: 설비 최소 폭
        :param max_pieces: 패턴 당 허용 최대 조각(Knife) 수
        :param b_wgt: 평량 (Basis Weight)
        :param min_sheet_roll_length: (사용 안함/예비) 최소 롤 길이
        :param max_sheet_roll_length: (사용 안함/예비) 최대 롤 길이
        :param sheet_trim: 변제(Trim) 폭
        :param min_sc_width: 최소 스크롤(Scroll) 폭 (설비 제약)
        :param max_sc_width: 최대 스크롤(Scroll) 폭 (설비 제약)
        """
        df_spec_pre['지폭'] = df_spec_pre['가로']

        self.b_wgt = b_wgt
        self.min_sheet_roll_length = min_sheet_roll_length
        self.max_sheet_roll_length = max_sheet_roll_length
        self.sheet_trim = sheet_trim
        self.original_max_width = max_width
        
        self.df_orders, self.demands_in_meters, self.order_sheet_lengths = self._calculate_demand_meters(df_spec_pre)
        self.order_widths = list(self.demands_in_meters.keys())

        width_summary = {}
        tons_per_width = self.df_orders.groupby('지폭')['주문톤'].sum()
        for width, required_meters in self.demands_in_meters.items():
            order_tons = tons_per_width.get(width, 0)
            width_summary[width] = {'order_tons': order_tons}
        self.width_summary = width_summary

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
        print(f"--- 패턴 제약조건: 최소 {self.min_pieces}폭, 최대 {self.max_pieces}폭 ---")

        self.patterns = []

    def _prepare_items(self, min_sc_width, max_sc_width):
        """
        [최적화 대상 아이템 준비]
        단순 지폭뿐만 아니라, 동일 지폭을 여러 장(1~4장) 합친 '복합 폭(Composite Width)'을 하나의 아이템으로 생성합니다.
        이는 나이프 수를 줄이고 생산 효율을 높이기 위함입니다.
        
        로직:
        1. 각 주문 지폭에 대해 1배~4배까지 확장을 시도합니다.
        2. 확장된 폭(base_width)이 설비의 스크롤 폭 제약(min_sc_width ~ max_sc_width)을 만족하는지 확인합니다.
        3. 만족한다면 최적화 후보 아이템으로 등록합니다.
        
        Returns:
            items: 아이템 이름 리스트 (예: "800x2")
            item_info: 아이템 이름 -> 실제 폭 매핑
            item_composition: 아이템 이름 -> 구성 정보 ({원지폭: 장수})
        """
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

    def _calculate_demand_meters(self, df_orders):
        """
        [주문량 변환: 톤 -> 미터]
        최적화 알고리즘은 '길이(Meter)' 기준으로 동작하므로, 중량 단위 주문을 길이 단위로 변환해야 합니다.
        
        공식:
        1. 장당 무게(g) = (평량 * 가로 * 세로) / 1,000,000
        2. 필요 장수 = (주문톤 * 1,000,000) / 장당 무게
        3. 필요 길이(m) = 필요 장수 * (세로 / 1000)
        
        Returns:
            df_copy: 미터 환산 컬럼이 추가된 주문 데이터프레임
            demand_meters: 지폭별 총 필요 길이 (Dictionary)
            order_sheet_lengths: 지폭별 쉬트 길이 (Dictionary)
        """
        df_copy = df_orders.copy()

        def calculate_meters(row):
            width_mm = row['가로']
            length_mm = row['세로']
            order_ton = row['주문톤']

            if self.b_wgt <= 0 or width_mm <= 0 or length_mm <= 0 or order_ton <= 0:
                return 0

            sheet_weight_g = (self.b_wgt * width_mm * length_mm) / 1000000
            if sheet_weight_g <= 0:
                return 0

            total_sheets_needed = (order_ton * 1000000) / sheet_weight_g
            total_meters_needed = total_sheets_needed * (length_mm / 1000)
            return total_meters_needed

        df_copy['meters'] = df_copy.apply(calculate_meters, axis=1)
        demand_meters = df_copy.groupby('지폭')['meters'].sum().to_dict()
        order_sheet_lengths = df_copy.groupby('지폭')['세로'].first().to_dict()

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        print("--- 지폭별 필요 총 길이 ---")
        print("--------------------------")

        return df_copy, demand_meters, order_sheet_lengths

    def run_optimize(self, start_prod_seq=0):
        """
        [최적화 실행 메인 함수]
        1. 초기 패턴이 존재하는지 확인합니다.
        2. 마스터 문제(Master Problem)를 풀어서 최적의 패턴 조합과 생산 횟수를 구합니다.
           (주의: _solve_master_problem_ilp 메서드가 현재 파일에 누락되어 있음)
        3. 결과를 포맷팅하여 반환합니다.
        """
        # 1. 초기 패턴 생성
        if not self.patterns:
            self._generate_initial_patterns()
        
        if not self.patterns:
             return {"error": "초기 패턴 생성 실패. 제약조건 확인 필요."}

        print(f"--- Column Generation 시작 (초기 패턴 {len(self.patterns)}개) ---")
        
        # 2. Column Generation Loop
        start_time = time.time()
        for i in range(CG_MAX_ITERATIONS):
            # 2.1 Solve Master Problem (Relaxed LP)
            solution = self._solve_master_problem_ilp(is_final_mip=False)
            if not solution:
                print("Master Problem(LP) 해를 찾을 수 없음.")
                break
                
            duals = solution.get('duals', {})
            
            # 2.2 Solve Subproblem
            new_patterns = self._solve_subproblem_dp(duals)
            
            if not new_patterns:
                print(f"Iter {i}: 더 이상 개선 가능한 패턴 없음.")
                break
                
            # 2.3 Add new patterns
            added = 0
            seen_patterns = {frozenset(p['composition'].items()) for p in self.patterns}
            for np in new_patterns:
                p_key = frozenset(np['composition'].items())
                if p_key not in seen_patterns:
                    self.patterns.append(np)
                    seen_patterns.add(p_key)
                    added += 1
            
            if added == 0:
                print(f"Iter {i}: 중복된 패턴만 생성됨. 종료.")
                break
                
            # if i % 10 == 0:
            #     print(f"Iter {i}: 패턴 {added}개 추가됨 (Total {len(self.patterns)}). VP: {solution['objective']:.2f}")

        print(f"--- CG 종료. 총 {len(self.patterns)}개 패턴으로 최종 MIP 수행 ---")
        
        # 3. Final MIP Solve
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
        return self._format_results(final_solution, start_prod_seq)

    def _format_results(self, final_solution, start_prod_seq=0):
        result_patterns, pattern_details_for_db, pattern_roll_details_for_db, pattern_roll_cut_details_for_db, demand_tracker, last_prod_seq = self._build_pattern_details(final_solution, start_prod_seq)
        
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['pattern', 'wd_width', 'roll_length', 'count', 'loss_per_roll']]


        fulfillment_summary = self._build_fulfillment_summary(demand_tracker)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]") 
        print(f"--- 최적화 완료: 최종 패턴 {len(result_patterns)}개 사용 ---")
        print("[주문 이행 요약 (그룹오더별)]")

        
        return {
            "pattern_result": df_patterns.sort_values('count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "last_prod_seq": last_prod_seq
        }

    def _build_pattern_details(self, final_solution, start_prod_seq=0):
        """
        [상세 결과 생성 및 DB 매핑]
        솔버가 도출한 최적 패턴(추상적 결과)을 실제 생산 가능한 형태(구체적 결과)로 상세화합니다.
        주문 할당(Assignment) 로직을 포함합니다.

        주요 로직:
        1. **패턴 분해**: 솔버 결과인 '패턴'과 '횟수'를 가져와서, 실제 어떤 지폭들이 포함되었는지 분해합니다.
        2. **주문 매핑(Greedy)**: 
           - 해당 지폭을 필요로 하는 주문(Group Order No)을 찾아 할당합니다.
           - 아직 생산량이 부족한 주문을 우선적으로 찾습니다.
           - 만약 모든 주문이 충족되었다면 'OVERPROD'(과생산)로 처리하거나, 임의의 주문에 할당하여 잔여를 처리합니다.
        3. **DB 데이터 생성**:
           - pattern_details_for_db: 패턴별 개요
           - pattern_roll_details_for_db: 롤 단위 생산 정보 (복합 폭 기준)
           - pattern_roll_cut_details_for_db: 컷 단위 상세 정보 (복합 폭을 다시 낱장 폭으로 분해)
        4. **진척도 갱신**: 할당된 만큼 남은 주문량을 차감하여(fulfilled_meters) 다음 롤 할당 시 반영합니다.

        Returns:
            상세 데이터 리스트들 및 최종 주문 이행 현황(demand_tracker)
        """
        demand_tracker = self.df_orders.copy()
        demand_tracker['original_order_idx'] = demand_tracker.index
        demand_tracker = demand_tracker[['original_order_idx', 'group_order_no', '지폭', 'meters']].copy()
        demand_tracker['fulfilled_meters'] = 0.0
        demand_tracker = demand_tracker.sort_values(by=['지폭', 'group_order_no']).reset_index(drop=True)

        result_patterns = []
        pattern_details_for_db = []
        pattern_roll_details_for_db = []
        pattern_roll_cut_details_for_db = []
        prod_seq_counter = start_prod_seq
        total_cut_seq_counter = 0

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

            for item_name, num in sorted_pattern_items:
                width = self.item_info[item_name]
                total_width += width * num
                
                base_width_dict = self.item_composition[item_name]
                for base_width, num_base in base_width_dict.items():
                    all_base_pieces_in_roll.extend([base_width] * (num_base * num))

                base_width, multiplier = map(int, item_name.split('x'))
                formatted_name = f"{self.item_info[item_name]}({base_width}*{multiplier})"
                pattern_item_strs.extend([formatted_name] * num)
            
            result_patterns.append({
                'pattern': ' + '.join(pattern_item_strs),
                'wd_width': total_width,
                'roll_length': round(pattern_length, 2),
                'count': roll_count,
                'loss_per_roll': self.original_max_width - total_width
            })

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
                    assigned_group_no_for_composite = None

                    for base_width, num_of_base in base_width_dict.items():
                        for _ in range(num_of_base):
                            target_indices = demand_tracker[
                                (demand_tracker['지폭'] == base_width) &
                                (demand_tracker['fulfilled_meters'] < demand_tracker['meters'])
                            ].index
                            
                            assigned_group_no = "OVERPROD"
                            if not target_indices.empty:
                                target_idx = target_indices.min()
                                assigned_group_no = demand_tracker.loc[target_idx, 'group_order_no']
                            else:
                                fallback_indices = demand_tracker[demand_tracker['지폭'] == base_width].index
                                if not fallback_indices.empty:
                                    assigned_group_no = demand_tracker.loc[fallback_indices.min(), 'group_order_no']
                            
                            base_widths_for_item.append(base_width)
                            base_group_nos_for_item.append(assigned_group_no)

                            if assigned_group_no_for_composite is None:
                                assigned_group_no_for_composite = assigned_group_no
                    
                    composite_widths_for_db.append(composite_width)
                    composite_group_nos_for_db.append(assigned_group_no_for_composite if assigned_group_no_for_composite is not None else "")

                    pattern_roll_details_for_db.append({
                        'rollwidth': composite_width,
                        'pattern_length': pattern_length,
                        'widths': (base_widths_for_item + [0] * 7)[:7],
                        'group_nos': (base_group_nos_for_item + [''] * 7)[:7],
                        'count': roll_count,
                        'prod_seq': prod_seq_counter,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'S',
                        **common_props
                    })

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

            pattern_details_for_db.append({
                'pattern_length': pattern_length,
                'count': roll_count,
                'widths': (composite_widths_for_db + [0] * 8)[:8],
                'group_nos': (composite_group_nos_for_db + [''] * 8)[:8],
                'prod_seq': prod_seq_counter,
                'rs_gubun': 'S',
            })

            # Batch update demand tracker
            base_counts_in_roll = Counter(all_base_pieces_in_roll)
            for base_width, num_in_roll in base_counts_in_roll.items():
                produced_meters = num_in_roll * pattern_length * roll_count
                
                relevant_orders = demand_tracker[demand_tracker['지폭'] == base_width].index
                
                for order_idx in relevant_orders:
                    if produced_meters <= 0:
                        break
                    
                    needed = demand_tracker.loc[order_idx, 'meters'] - demand_tracker.loc[order_idx, 'fulfilled_meters']
                    if needed > 0:
                        fulfill_amount = min(needed, produced_meters)
                        demand_tracker.loc[order_idx, 'fulfilled_meters'] += fulfill_amount
                        produced_meters -= fulfill_amount

        return result_patterns, pattern_details_for_db, pattern_roll_details_for_db, pattern_roll_cut_details_for_db, demand_tracker, prod_seq_counter

    def _build_fulfillment_summary(self, demand_tracker):
        """
        [주문 이행 결과 요약]
        각 주문별로 목표량(주문량) 대비 실제 생산량(생산길이, 톤)을 비교하여 과부족을 계산합니다.
        """
        summary_df = self.df_orders[['group_order_no', '가로', '세로', '수출내수', '등급', '주문톤', 'meters']].copy()
        summary_df.rename(columns={'meters': '필요길이(m)', '주문톤': '주문량(톤)'}, inplace=True)
        
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

    def _generate_initial_patterns(self):
        """
        초기 패턴 생성 (휴리스틱 + First Fit)
        sheet_optimize.py의 로직을 차용하되, 패턴 구조를 {'composition': {}, 'length': max_length} 형태로 저장합니다.
        """
        print("--- 초기 패턴 생성 시작 ---")
        seen_patterns = {frozenset(p['composition'].items()) for p in self.patterns}

        # 정렬 전략
        heuristics = [
            sorted(self.items, key=lambda i: self.demands_in_meters.get(list(self.item_composition[i].keys())[0], 0), reverse=True), # 수요 내림차순
            sorted(self.items, key=lambda i: self.item_info[i], reverse=True), # 지폭 내림차순
            sorted(self.items, key=lambda i: self.item_info[i], reverse=False), # 지폭 오름차순
        ]
        
        # Random Shuffles
        random.seed(41)
        for _ in range(5):
            items_copy = list(self.items)
            random.shuffle(items_copy)
            heuristics.append(items_copy)

        for sorted_items in heuristics:
            # First Fit
            for item in sorted_items:
                item_width = self.item_info[item]
                
                current_pattern = {item: 1}
                current_width = item_width
                current_pieces = 1

                while current_pieces < self.max_pieces:
                    remaining_width = self.max_width - current_width
                    best_fit = next((i for i in sorted_items if self.item_info[i] <= remaining_width), None)
                    
                    if not best_fit: break

                    current_pattern[best_fit] = current_pattern.get(best_fit, 0) + 1
                    current_width += self.item_info[best_fit]
                    current_pieces += 1

                if self.min_width <= current_width: # min_pieces 조건은 초기 생성 시 완화 가능
                    pattern_comp_key = frozenset(current_pattern.items())
                    if pattern_comp_key not in seen_patterns:
                        self.patterns.append({
                            'composition': current_pattern,
                            'length': self.max_sheet_roll_length, # 가변 길이는 추후 확장, 현재는 최대로 고정
                            'width': current_width
                        })
                        seen_patterns.add(pattern_comp_key)

        # 단일 품목 패턴 추가 (필수)
        for item in self.items:
            item_width = self.item_info[item]
            max_count = min(int(self.max_width / item_width), self.max_pieces)
            for count in range(max_count, 0, -1):
                current_pattern = {item: count}
                current_width = item_width * count
                if current_width >= self.min_width:
                     pattern_comp_key = frozenset(current_pattern.items())
                     if pattern_comp_key not in seen_patterns:
                        self.patterns.append({
                            'composition': current_pattern,
                            'length': self.max_sheet_roll_length,
                            'width': current_width
                        })
                        seen_patterns.add(pattern_comp_key)
                        break 

        print(f"--- {len(self.patterns)}개의 초기 패턴 생성됨 ---")

    def _solve_master_problem_ilp(self, is_final_mip=False):
        """
        마스터 문제 해결 (Column Generation)
        목적: 최소 롤 수(사실상 최소 길이 생산) 로 모든 주문 길이(Meters) 만족
        
        Variables:
            x[j]: j번째 패턴의 사용 횟수 (Roll Count)
        
        Constraints:
            For each width w:
               Sum(x[j] * PatternLength[j] * CountOfDoesWidthInPattern[j]) >= DemandMeters[w]
        """
        # 초기화: 반복 횟수 제어 등을 위해 필요한 경우 _generate_initial_patterns 호출
        if not self.patterns:
            self._generate_initial_patterns()
        
        solver = pywraplp.Solver.CreateSolver('SCIP' if is_final_mip else 'GLOP')
        if not solver: return None

        if hasattr(solver, 'SetNumThreads'):
            solver.SetNumThreads(4)
        
        if is_final_mip:
            solver.SetTimeLimit(SOLVER_TIME_LIMIT_MS)

        # Variables
        # x[j] represents number of ROLLS of pattern j
        x = {}
        for j in range(len(self.patterns)):
            var_name = f'P_{j}'
            if is_final_mip:
                x[j] = solver.IntVar(0, solver.infinity(), var_name)
            else:
                x[j] = solver.NumVar(0, solver.infinity(), var_name)

        over_prod_vars = {w: solver.NumVar(0, solver.infinity(), f'Over_{w}') for w in self.demands_in_meters}
        under_prod_vars = {w: solver.NumVar(0, solver.infinity(), f'Under_{w}') for w in self.demands_in_meters}

        # Constraints
        constraints = {}
        for width, required_meters in self.demands_in_meters.items():
            # 생산된 총 길이 (Meters)
            # 패턴 j의 길이 * 패턴 j 내 width의 개수 * 패턴 j 사용 횟수(x[j])
            production_expr = solver.Sum(
                x[j] * self.patterns[j]['length'] * 
                sum(self.item_composition[item].get(width, 0) * count 
                    for item, count in self.patterns[j]['composition'].items())
                for j in range(len(self.patterns))
            )
            
            # Constraint: Production + Under = Demand + Over
            constraints[width] = solver.Add(
                production_expr + under_prod_vars[width] == required_meters + over_prod_vars[width], 
                f'demand_{width}'
            )

        # Objective Function
        # Minimize Total Rolls (which minimizes total length since length is roughly constant/max)
        # + Penalties
        total_rolls = solver.Sum(x.values())
        
        over_penalty_cost = solver.Sum(OVER_PROD_PENALTY * v for v in over_prod_vars.values())
        # 부족 생산 페널티는 매우 크게 (METER 단위이므로 페널티 스케일 조절 필요할 수 있음)
        # 기존 로직 유지: UNDER_PROD_PENALTY 사용 
        under_penalty_cost = solver.Sum(UNDER_PROD_PENALTY * v for v in under_prod_vars.values())
        
        # 패턴 개수/복잡도 페널티 (선택적)
        complexity_penalty = solver.Sum(PATTERN_COMPLEXITY_PENALTY * len(self.patterns[j]['composition']) * x[j] for j in range(len(self.patterns)))

        solver.Minimize(total_rolls + over_penalty_cost + under_penalty_cost + complexity_penalty)

        # Solve
        status = solver.Solve()
        
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            solution = {
                'objective': solver.Objective().Value(),
                'pattern_counts': {j: x[j].solution_value() for j in x},
                'over_production': {w: over_prod_vars[w].solution_value() for w in over_prod_vars},
                'under_production': {w: under_prod_vars[w].solution_value() for w in under_prod_vars}
            }
            if not is_final_mip:
                solution['duals'] = {w: constraints[w].dual_value() for w in self.demands_in_meters}
                
                # Column Generation Loop (if not final)
                # 여기서는 재귀적으로 호출하거나, 외부에서 루프를 돌려야 하는데
                # 구조상 외부(run_optimize)에서 루프를 돌리는 것이 맞으나,
                # 현재 코드 구조상 이 함수 내에서 처리하거나 run_optimize를 수정해야 함.
                # 편의상 여기서 Subproblem 호출 및 루프 처리를 하지 않고,
                # run_optimize를 수정하여 루프를 돌리도록 유도해야 함.
                # 하지만 기존 sheet_optimize 구조를 보면 run_optimize 내에 루프 로직이 있었을 것.
                # 일단 여기서는 Dual 값 반환까지만 수행.
            else:
                 pass # Final MIP
            
            return solution
        else:
            print("Solver Failed to find solution")
            return None

    def _solve_subproblem_dp(self, duals):
        """
        Subproblem: Find a new pattern with negative Reduced Cost
        Reduced Cost = 1 - Sum(Dual_w * Count_w * Length) (Roll Count 최소화 기준)
        Here, Duals are per METER. 
        Value of a pattern = Sum(Dual_w * Count_w * PatternLength)
        We want to MAXIMIZE Value to find Reduced Cost < 0 (i.e., Value > 1)
        
        Since PatternLength is fixed to self.max_sheet_roll_length (for now),
        Effective Item Value for DP = Sum(Dual_w * Count_w) * self.max_sheet_roll_length
        """
        items_val = []
        for item_name in self.items:
            # item 하나가 기여하는 Value 계산
            # item_info[item_name] is width. Not used for value.
            # item_composition[item_name] = {width: count}
            
            unit_val_sum = sum(duals.get(w, 0) * cnt for w, cnt in self.item_composition[item_name].items())
            item_val = unit_val_sum * self.max_sheet_roll_length
            
            if item_val > 0.0001:
                items_val.append({
                    'name': item_name,
                    'width': self.item_info[item_name],
                    'value': item_val
                })

        # Solving Knapsack-like problem to Maximize Value under Width constraint
        # DP: dp[w] = max value with width w
        W = int(self.max_width)
        dp = [-1.0] * (W + 1)
        dp[0] = 0.0
        parent = {} # Reconstruct path: dp_idx -> (prev_idx, item_name)

        # Simple 1D Knapsack (Unbounded: can use multiple items, but we have max_pieces constraint)
        # To handle max_pieces, we need 2D DP: dp[pieces][width]
        
        P = self.max_pieces
        dp2 = [[-1.0] * (W + 1) for _ in range(P + 1)]
        dp2[0][0] = 0.0
        parent2 = {} # (p, w) -> (prev_p, prev_w, item_name)

        for p in range(P):
            for w in range(W + 1):
                if dp2[p][w] < -0.5: continue
                
                current_val = dp2[p][w]
                
                for item in items_val:
                    nw = w + int(item['width'])
                    if nw <= W:
                        nval = current_val + item['value']
                        np = p + 1
                        if nval > dp2[np][nw]:
                            dp2[np][nw] = nval
                            parent2[(np, nw)] = (p, w, item['name'])

        # Find best solution
        best_val = 1.0 + 1e-5 # We need Reduced Cost < 0 <=> Value > Cost (Cost=1 roll)
        best_state = None

        for p in range(self.min_pieces, P + 1):
            for w in range(int(self.min_width), W + 1):
                if dp2[p][w] > best_val:
                    best_val = dp2[p][w]
                    best_state = (p, w)

        if best_state:
            # Reconstruct
            new_pattern_comp = {}
            curr = best_state
            while curr != (0, 0):
                prev_p, prev_w, item_name = parent2[curr]
                new_pattern_comp[item_name] = new_pattern_comp.get(item_name, 0) + 1
                curr = (prev_p, prev_w)
            
            return [{
                'composition': new_pattern_comp,
                'length': self.max_sheet_roll_length,
                'width': best_state[1]
            }]
        
        return []

    def _generate_all_patterns(self): # Fallback (Not implemented fully for brevity, can rely on heuristics)
        pass


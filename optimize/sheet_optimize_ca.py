import pandas as pd
from ortools.linear_solver import pywraplp
from collections import Counter
import math
import random
import time

# --- 최적화 설정 상수 ---
# 비용 상수 (모든 목적 함수 항을 '비용'으로 통일하기 위해 사용)
COST_PER_ROLL = 5000.0          # 롤 1개 교체/사용에 대한 비용 (예시)
COST_PER_METER_MATERIAL = 0.8  # 원자재 1미터당 비용 (예시)

# 페널티 값
OVER_PROD_PENALTY = 200.0    # 과생산에 대한 페널티
UNDER_PROD_PENALTY = 100000.0  # 부족생산에 대한 페널티
PATTERN_COMPLEXITY_PENALTY = 0.01  # 패턴 복잡성에 대한 페널티
TRIM_PENALTY = 0          # 트림(loss) 면적(mm^2)당 페널티. 폐기물 비용.
ITEM_SINGLE_STRIP_PENALTIES = {}
DEFAULT_SINGLE_STRIP_PENALTY = 1000  # 지정되지 않은 단일폭은 기본적으로 패널티 없음
DISALLOWED_SINGLE_BASE_WIDTHS = {}  # 단일 사용을 금지할 주문 폭 집합

# 알고리즘 파라미터
MIN_PIECES_PER_PATTERN = 1      # 패턴에 포함될 수 있는 최소 폭(piece)의 수
SMALL_PROBLEM_THRESHOLD = 8     # 전체 탐색을 수행할 최대 주문 지폭 종류 수
SOLVER_TIME_LIMIT_MS = 180000    # 최종 MIP 솔버의 최대 실행 시간 (밀리초)
CG_MAX_ITERATIONS = 1000         # 열 생성(Column Generation) 최대 반복 횟수
CG_NO_IMPROVEMENT_LIMIT = 100    # 개선 없는 경우, 열 생성 조기 종료 조건
CG_SUBPROBLEM_TOP_N = 1         # 열 생성 시, 각 반복에서 추가할 상위 N개 신규 패턴
# 나이프 로드 제약: 패턴 생산 횟수는 k1*a + k2*b 형태여야 함 (a,b>=0 정수)
KNIFE_LOAD_K1 = 1
KNIFE_LOAD_K2 = 1

class SheetOptimizeCa:
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
        from itertools import combinations_with_replacement
        
        items = []
        item_info = {}  # item_name -> width
        item_composition = {}  # composite_item_name -> {original_width: count}
        
        max_pieces_in_composite = 4 

        for width in self.order_widths:
            for i in range(1, max_pieces_in_composite + 1):
                base_width = width * i + self.sheet_trim
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                item_name = f"{width}x{i}"
                if base_width <= self.original_max_width:
                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = base_width
                        item_composition[item_name] = {width: i}

        for i in range(2, max_pieces_in_composite + 1):
            for combo in combinations_with_replacement(self.order_widths, i):
                if len(set(combo)) == 1:
                    continue

                base_width = sum(combo) + self.sheet_trim
                if not (min_sc_width <= base_width <= max_sc_width):
                    continue

                if base_width <= self.original_max_width:
                    comp_counts = Counter(combo)
                    item_name = "+".join(sorted([f"{w}x{c}" for w, c in comp_counts.items()]))

                    if item_name not in items:
                        items.append(item_name)
                        item_info[item_name] = base_width
                        item_composition[item_name] = dict(comp_counts)

        return items, item_info, item_composition

    def _calculate_demand_meters(self, df_orders):
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
        if not self.patterns:
            return {"error": "유효한 패턴을 생성할 수 없습니다."}

        print(f"--- 총 {len(self.patterns)}개의 패턴으로 최종 최적화를 수행합니다. ---")
        final_solution = self._solve_master_problem_ilp(is_final_mip=True)
        if not final_solution:
            return {"error": "최종 해를 찾을 수 없습니다."}
        
        return self._format_results(final_solution, start_prod_seq)

    def _format_results(self, final_solution, start_prod_seq=0):
        result_patterns, pattern_details_for_db, pattern_roll_details_for_db, pattern_roll_cut_details_for_db, demand_tracker, last_prod_seq = self._build_pattern_details(final_solution, start_prod_seq)
        
        df_patterns = pd.DataFrame(result_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns[['Pattern', 'wd_width', 'Roll_Length', 'Count', 'Loss_per_Roll']]

        fulfillment_summary = self._build_fulfillment_summary(demand_tracker)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]") 
        print("[주문 이행 요약 (그룹오더별)]")
        
        return {
            "pattern_result": df_patterns.sort_values('Count', ascending=False) if not df_patterns.empty else df_patterns,
            "pattern_details_for_db": pattern_details_for_db,
            "pattern_roll_details_for_db": pattern_roll_details_for_db,
            "pattern_roll_cut_details_for_db": pattern_roll_cut_details_for_db,
            "fulfillment_summary": fulfillment_summary,
            "last_prod_seq": last_prod_seq
        }

    def _build_pattern_details(self, final_solution, start_prod_seq=0):
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

        for j, count in final_solution['pattern_counts'].items():
            if count < 0.99:
                continue
            
            roll_count = int(round(count))
            pattern = self.patterns[j]
            pattern_comp = pattern['composition']
            pattern_len = pattern['length']

            prod_seq_counter += 1

            sorted_pattern_items = sorted(pattern_comp.items(), key=lambda item: self.item_info[item[0]], reverse=True)
            pattern_item_strs = []
            total_width = 0
            for item_name, num in sorted_pattern_items:
                width = self.item_info[item_name]
                total_width += width * num
                
                sub_items = item_name.split('+')
                if len(sub_items) > 1 or 'x' not in item_name:
                     formatted_name = f"{width}({item_name})"
                else:
                    try:
                        base_width, multiplier = map(int, item_name.split('x'))
                        formatted_name = f"{width}({base_width}*{multiplier})"
                    except ValueError:
                        formatted_name = f"{width}({item_name})"
                pattern_item_strs.extend([formatted_name] * num)
            
            result_patterns.append({
                'Pattern': ' + '.join(pattern_item_strs),
                'wd_width': total_width,
                'Roll_Length': round(pattern_len, 2),
                'Count': roll_count,
                'Loss_per_Roll': pattern['loss_per_roll']
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
                            piece_width = base_width
                            
                            target_indices = demand_tracker[
                                (demand_tracker['지폭'] == piece_width) &
                                (demand_tracker['fulfilled_meters'] < demand_tracker['meters'])
                            ].index
                            
                            assigned_group_no = None
                            if not target_indices.empty:
                                target_idx = target_indices.min()
                                demand_tracker.loc[target_idx, 'fulfilled_meters'] += pattern_len
                                assigned_group_no = demand_tracker.loc[target_idx, 'group_order_no']
                            else:
                                fallback_indices = demand_tracker[demand_tracker['지폭'] == piece_width].index
                                if not fallback_indices.empty:
                                    assigned_group_no = demand_tracker.loc[fallback_indices.min(), 'group_order_no']
                                else:
                                    assigned_group_no = "ERROR"
                            
                            base_widths_for_item.append(base_width)
                            base_group_nos_for_item.append(assigned_group_no)

                            if assigned_group_no_for_composite is None:
                                assigned_group_no_for_composite = assigned_group_no
                    
                    composite_widths_for_db.append(composite_width)
                    composite_group_nos_for_db.append(assigned_group_no_for_composite if assigned_group_no_for_composite is not None else "")

                    pattern_roll_details_for_db.append({
                        'rollwidth': composite_width,
                        'roll_production_length': pattern_len,
                        'widths': (base_widths_for_item + [0] * 7)[:7],
                        'group_nos': (base_group_nos_for_item + [''] * 7)[:7],
                        'Count': roll_count,
                        'Prod_seq': prod_seq_counter,
                        'Roll_seq': roll_seq_counter
                    })

                    cut_seq_counter = 0
                    for i in range(len(base_widths_for_item)):
                        width = base_widths_for_item[i]
                        if width > 0:
                            cut_seq_counter += 1
                            total_cut_seq_counter += 1
                            group_no = base_group_nos_for_item[i]
                            
                            weight = (self.b_wgt * (width / 1000) * pattern_len)

                            pattern_roll_cut_details_for_db.append({
                                'PROD_SEQ': prod_seq_counter,
                                'UNIT_NO': prod_seq_counter,
                                'SEQ': total_cut_seq_counter,
                                'ROLL_SEQ': roll_seq_counter,
                                'CUT_SEQ': cut_seq_counter,
                                'WIDTH': width,
                                'GROUP_NO': group_no,
                                'WEIGHT': weight,
                                'TOTAL_LENGTH': pattern_len,
                                'Count': len([w for w in base_widths_for_item if w > 0]),
                            })

            pattern_details_for_db.append({
                'roll_production_length': pattern_len,
                'Count': roll_count,
                'widths': (composite_widths_for_db + [0] * 8)[:8],
                'group_nos': (composite_group_nos_for_db + [''] * 8)[:8],
                'Prod_seq': prod_seq_counter
            })
        
        return result_patterns, pattern_details_for_db, pattern_roll_details_for_db, pattern_roll_cut_details_for_db, demand_tracker, prod_seq_counter

    def _build_fulfillment_summary(self, demand_tracker):
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

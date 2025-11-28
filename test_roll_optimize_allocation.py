import pandas as pd
import numpy as np
from optimize.roll_optimize import RollOptimize
from collections import Counter

def test_roll_optimization_with_allocation():
    print("--- Test Case: Grouping Optimization with Individual Result Allocation ---")
    
    # 1. 원본 데이터 (개별 오더)
    # group_order_no는 '지폭', '롤길이', 'order_no' 별로 유니크하게 생성 (여기서는 order_no를 사용)
    data = {
        'order_no': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', '10'],
        '지폭': [606, 606, 667, 706, 706, 727, 788, 909, 970, 1091],
        '주문수량': [6, 8, 7, 7, 14, 7, 8, 5, 10, 4],
        '롤길이': [2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050]
    }
    df_original = pd.DataFrame(data)
    
    # 'group_order_no' 컬럼 생성 (개별 오더 식별용)
    df_original['group_order_no'] = df_original['order_no']
    
    print("\n[1] 원본 오더 데이터 (Group Order = Order No):")
    print(df_original)

    # 2. 지폭 그룹핑 (Width Grouping for Engine)
    # 엔진 수행을 위한 '지폭그룹오더' 생성
    width_group_cols = ['지폭', '롤길이']
    
    # 지폭 그룹 ID 생성 (WG1, WG2...)
    df_width_groups = df_original.groupby(width_group_cols).agg(
        total_qty=('주문수량', 'sum')
    ).reset_index()
    
    df_width_groups['width_group_no'] = [f'WG{i+1}' for i in range(len(df_width_groups))]
    
    # 원본 데이터에 지폭 그룹 ID 매핑
    df_merged = pd.merge(df_original, df_width_groups, on=width_group_cols, how='left')
    
    print("\n[2] 지폭 그룹핑 결과 (엔진 입력용):")
    print(df_merged[['group_order_no', 'width_group_no', '지폭', '주문수량', 'total_qty']])

    # 3. 최적화 수행 (Optimization)
    # 엔진에는 'width_group_no'를 'group_order_no'로 속여서 전달합니다.
    df_for_engine = df_width_groups.rename(columns={'width_group_no': 'group_order_no', 'total_qty': '주문수량'})
    
    optimizer = RollOptimize(
        df_spec_pre=df_for_engine,
        max_width=4880,
        min_width=4500,
        max_pieces=8
    )
    
    results = optimizer.run_optimize()
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n[3] 최적화 결과 (지폭 그룹별):")
    print(results['pattern_result'][['pattern', 'count']])

    # 4. 결과 배분 및 DB 데이터 생성 (Allocation & DB Generation)
    # 엔진 결과(지폭 그룹 기준)를 개별 오더(group_order_no)로 변환하여 pattern_details_for_db 등을 생성
    
    print("\n[4] 결과 배분 및 DB 데이터 생성 시작...")
    
    # 개별 오더의 잔여 수요 추적
    remaining_demands = df_merged.set_index('group_order_no')['주문수량'].to_dict()
    # 지폭 그룹 -> 개별 오더 리스트 매핑
    width_group_to_orders = df_merged.groupby('width_group_no')['group_order_no'].apply(list).to_dict()
    
    final_pattern_details = []
    final_pattern_roll_details = []
    final_pattern_roll_cut_details = []
    
    prod_seq = 0
    
    # 엔진이 생성한 패턴들을 순회
    # results['pattern_result']는 DataFrame이므로 iterrows 사용
    for _, row in results['pattern_result'].iterrows():
        pattern_str = row['pattern'] # 예: "606, 606, 606" (문자열일 수 있음)
        pattern_count = int(row['count'])
        
        # 패턴 내의 지폭 그룹 ID들을 파싱해야 함.
        # 하지만 RollOptimize 결과의 'pattern' 컬럼은 지폭 숫자만 나열되어 있어서 정확한 ID를 알기 어려울 수 있음.
        # 다행히 optimizer.patterns 에 구조화된 정보가 있음.
        # 하지만 여기서는 results['pattern_result']만으로는 부족할 수 있으니, 
        # optimizer 내부의 patterns를 직접 참조하거나, pattern_str을 파싱해서 width_group_no를 찾아야 함.
        # 가장 확실한 건 results['pattern_details_for_db']를 참조하는 것인데, 여기엔 group_nos가 WG로 되어있음.
        pass

    # 더 정확한 방법: results['pattern_details_for_db']를 기반으로 재가공
    # 이 리스트는 (WG ID 리스트, 반복 횟수) 정보를 담고 있음.
    
    raw_db_details = results['pattern_details_for_db']
    
    for entry in raw_db_details:
        # entry는 하나의 패턴 설정에 대한 정보 (WG ID들 포함)
        wg_ids = entry['group_nos'] # 예: ['WG1', 'WG1', 'WG2', '', ...]
        widths = entry['widths']
        total_run_count = entry['count']
        
        # 이 패턴(WG 조합)을 total_run_count만큼 반복 생산해야 함.
        # 각 반복(run)마다 구체적인 오더(Order No)를 할당.
        
        # 할당된 결과들을 모아서 (Order 조합, 횟수) 형태로 요약
        allocated_runs = [] # list of tuple (tuple of order_ids, count)
        
        current_run_orders = [] # 현재 run에 할당될 오더 ID들
        
        # 1회 생산 시 필요한 오더들을 결정하는 로직
        # 매 run마다 잔여 수요가 가장 급한 오더부터 할당 (Greedy)
        
        for _ in range(total_run_count):
            run_assignment = []
            for i, wg_id in enumerate(wg_ids):
                if not wg_id: # 빈 슬롯
                    run_assignment.append('')
                    continue
                
                # 해당 WG에 속하는 오더들 중 잔여 수요가 있는 것 찾기
                candidate_orders = width_group_to_orders.get(wg_id, [])
                assigned_order = None
                
                # 1. 잔여 수요가 있는 오더 우선
                for order_id in candidate_orders:
                    if remaining_demands.get(order_id, 0) > 0:
                        assigned_order = order_id
                        remaining_demands[order_id] -= 1
                        break
                
                # 2. 잔여 수요가 없으면 (과잉 생산), 아무거나 선택 (보통 마지막 오더나 큰 오더)
                if not assigned_order and candidate_orders:
                    assigned_order = candidate_orders[-1] # 마지막 오더에 몰아주기
                    # 과잉 생산 카운팅은 별도로 하거나 무시 (여기서는 단순 할당)
                
                run_assignment.append(assigned_order if assigned_order else '')
            
            allocated_runs.append(tuple(run_assignment))
            
        # 할당된 run들을 그룹핑하여 DB entry 생성
        run_counts = Counter(allocated_runs)
        
        for order_combo, count in run_counts.items():
            prod_seq += 1
            
            # 새로운 DB entry 생성
            new_entry = entry.copy()
            new_entry['group_nos'] = list(order_combo)
            new_entry['count'] = count
            new_entry['prod_seq'] = prod_seq
            
            final_pattern_details.append(new_entry)
            
            # Roll Details, Cut Details 도 유사하게 생성 가능 (생략 또는 간략화)
            # 실제로는 여기서 pattern_roll_details_for_db 등도 같이 만들어야 함
            
            # Roll Details 생성 예시
            roll_seq_counter = 0
            for i, width in enumerate(widths):
                if width <= 0: continue
                roll_seq_counter += 1
                group_no = list(order_combo)[i]
                
                final_pattern_roll_details.append({
                    'rollwidth': width,
                    'pattern_length': entry.get('pattern_length', 0), # 기존 정보 유지
                    'widths': [width] + [0]*7, # 단순화
                    'group_nos': [group_no] + ['']*7,
                    'count': count,
                    'prod_seq': prod_seq,
                    'roll_seq': roll_seq_counter,
                    'rs_gubun': 'R',
                    'p_lot': entry.get('p_lot'),
                    'diameter': entry.get('diameter'),
                    'core': entry.get('core')
                })

    print(f"\n[4] 최종 변환된 DB 데이터 (총 {len(final_pattern_details)}개 패턴 엔트리):")
    df_final_db = pd.DataFrame(final_pattern_details)
    if not df_final_db.empty:
        print(df_final_db[['prod_seq', 'count', 'widths', 'group_nos']])
    
    print("\n[5] 잔여 수요 확인 (0이어야 함, 음수면 과잉생산):")
    print(remaining_demands)
    print(final_pattern_details)
    print(final_pattern_roll_details)
    print(final_pattern_roll_cut_details)

if __name__ == "__main__":
    test_roll_optimization_with_allocation()

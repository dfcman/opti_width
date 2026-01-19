import pandas as pd
from collections import Counter
import sys
import time
import configparser
import os
import logging
import argparse
import pprint
from optimize.roll_optimize import RollOptimize
from optimize.roll_optimize_ca import RollOptimizeCa
# from optimize.roll_optimize_cpsat import RollOptimizeCpsat
from optimize.roll_sl_optimize import RollSLOptimize
from optimize.sheet_optimize import SheetOptimize
from optimize.sheet_optimize_var import SheetOptimizeVar
from optimize.sheet_optimize_ca import SheetOptimizeCa
from db.db_connector import Database

# Load configuration
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf', 'config.ini')
NUM_THREADS = 4
if os.path.exists(config_path):
    config.read(config_path)
    NUM_THREADS = config.getint('optimization', 'num_threads', fallback=4)
else:
    # logging is not setup yet, so just print or simple log
    print(f"Config file not found at {config_path}. Using default NUM_THREADS={NUM_THREADS}")



def process_sheet_lot(
        db, plant, pm_no, schedule_unit, lot_no, version,  
        paper_type, b_wgt, color, time_limit,
        re_min_width, re_max_width, re_max_pieces,
        min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
        start_prod_seq=0, start_group_order_no=0
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""

    if re_max_pieces > 4:
        re_max_pieces = 4
        logging.warning(f"max_pieces가 4보다 큽니다. max_pieces를 4로 설정합니다.")
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}, min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, sheet_length_re={sheet_length_re}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_sheet_orders_from_db(paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 쉬트지 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
    # df_orders.rename(columns={'오더번호': 'order_no'}, inplace=True)
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    # --- [New Grouping Logic] ---
    df_orders, df_groups, last_group_order_no = apply_sheet_grouping(df_orders, start_group_order_no, lot_no)
    
    logging.info(f"--- Lot {df_groups.to_string()} 그룹마스터 정보 ---")

    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "pattern_roll_details_for_db": [],
        "pattern_roll_cut_details_for_db": [],
        "fulfillment_summary": []
    }
    
    grouping_cols = ['등급']
    unique_groups = df_orders[grouping_cols].drop_duplicates()
    prod_seq_counter = start_prod_seq
    
    for _, row in unique_groups.iterrows():
        grade = row['등급']
        logging.info(f"\n--- 등급 {grade}에 대한 쉬트지 최적화 시작 ---")
        df_subset = df_orders[df_orders['등급'] == grade].copy()
        
        if df_subset.empty:
            continue

        # [Mod] Color 값 주입
        df_subset['color'] = color
        
        if df_subset.empty:
            continue

        optimizer = SheetOptimize(
            db=db,
            plant=plant,
            pm_no=pm_no,
            schedule_unit=schedule_unit,
            lot_no=lot_no,
            version=version,
            paper_type=paper_type,
            b_wgt=float(b_wgt),
            df_spec_pre=df_subset,
            min_width=int(re_min_width),
            max_width=int(re_max_width),
            max_pieces=int(re_max_pieces),
            time_limit=time_limit,
            sheet_roll_length=sheet_length_re,
            sheet_trim=sheet_trim_size,
            min_sc_width=min_sc_width,
            max_sc_width=max_sc_width,
            num_threads=NUM_THREADS
        )
        
        try:
            results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)
            
            if not results or "error" in results:
                error_msg = results['error'] if results and 'error' in results else "No solution found"
                logging.error(f"[에러] Lot {lot_no}, 등급 {grade} 쉬트지 최적화 실패: {error_msg}")
                continue
                
            prod_seq_counter = results.get('last_prod_seq', prod_seq_counter)
            
            if "pattern_details_for_db" in results:
                for detail in results["pattern_details_for_db"]:
                    detail['max_width'] = int(re_max_width)
            
            all_results["pattern_result"].append(results["pattern_result"])
            all_results["pattern_details_for_db"].extend(results["pattern_details_for_db"])
            all_results["pattern_roll_details_for_db"].extend(results.get("pattern_roll_details_for_db", []))
            all_results["pattern_roll_cut_details_for_db"].extend(results.get("pattern_roll_cut_details_for_db", []))
            all_results["fulfillment_summary"].append(results["fulfillment_summary"])
            
            logging.info(f"--- 등급 {grade} 쉬트지 최적화 성공 ---")
            
        except Exception as e:
            import traceback
            logging.error(f"[에러] Lot {lot_no}, 등급 {grade} 처리 중 예외 발생")
            logging.error(traceback.format_exc())
            continue

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 쉬트지 최적화 결과가 없습니다 (모든 등급 실패).")
        return None, None, start_prod_seq, start_group_order_no

    # --- [New] Sheet Sequence Allocation & Data Generation ---


    
    # NEW Allocation Logic using helper function
    pattern_sheet_details_for_db = generate_allocated_sheet_details(
        df_orders, 
        all_results["pattern_roll_cut_details_for_db"], 
        b_wgt
    )

    # Call DB Insert
    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True) if all_results["pattern_result"] else pd.DataFrame(),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": pd.concat(all_results["fulfillment_summary"], ignore_index=True) if all_results["fulfillment_summary"] else pd.DataFrame()
    }

    logging.info("쉬트지 최적화 성공 (전체 등급 완료).")
    return final_results, df_orders, prod_seq_counter, last_group_order_no


def process_roll_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit, re_min_width, re_max_width, re_max_pieces,
        start_prod_seq=0, start_group_order_no=0
):
    """롤지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Roll Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_roll_orders_from_db(paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 롤지 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    # 1. 개별 오더 그룹핑 (DB 저장용) - order_no 포함
    group_cols = ['지폭', '롤길이', '등급', 'core', 'dia', 'order_no']
    for col in ['지폭', '롤길이']:
        df_orders[col] = pd.to_numeric(df_orders[col])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    logging.info(f"--- Lot {lot_no} 원본 주문 정보 (그룹오더 포함) ---")
    logging.info(df_orders.to_string())
    logging.info("\n")

    # 2. 지폭 그룹핑 (엔진 최적화용) - order_no 제외
    # 엔진 효율을 위해 동일 규격은 하나로 묶음
    # 2. 지폭 그룹핑 (엔진 최적화용) - order_no 제외
    # 엔진 효율을 위해 동일 규격은 하나로 묶음
    width_group_cols = ['지폭', '롤길이', '등급', 'core', 'dia', '수출내수', 'sep_qt']
    df_width_groups = df_orders.groupby(width_group_cols).agg(
        total_qty=('주문수량', 'sum')
    ).reset_index()
    
    # 임시 그룹 ID 생성 (WG...)
    df_width_groups['width_group_no'] = [f'WG{i+1}' for i in range(len(df_width_groups))]
    
    # 원본 데이터에 지폭 그룹 ID 매핑
    df_orders = pd.merge(df_orders, df_width_groups[['지폭', '롤길이', '등급', 'core', 'dia', '수출내수', 'sep_qt', 'width_group_no']], 
                         on=['지폭', '롤길이', '등급', 'core', 'dia', '수출내수', 'sep_qt'], how='left')

    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "pattern_roll_details_for_db": [],
        "pattern_roll_cut_details_for_db": [],
        "fulfillment_summary": []
    }
    
    # 엔진 수행 그룹핑 기준 컬럼 설정
    grouping_cols = ['롤길이', 'core', 'dia', '등급']
    unique_groups = df_orders[grouping_cols].drop_duplicates()
    prod_seq_counter = start_prod_seq

    # 결과 배분을 위한 준비
    remaining_demands = df_orders.set_index('group_order_no')['주문수량'].to_dict()
    width_group_to_orders = df_orders.groupby('width_group_no')['group_order_no'].apply(list).to_dict()

    for _, row in unique_groups.iterrows():
        roll_length = row['롤길이']
        core = row['core']
        dia = row['dia']
        quality_grade = row['등급']
        logging.info(f"\n--- 롤길이 그룹 {roll_length}, Core {core}, Dia {dia}에 대한 최적화 시작 ---")
        
        # 엔진에는 지폭 그룹 데이터를 전달 (width_group_no를 group_order_no로 위장)
        df_subset_engine = df_width_groups[
            (df_width_groups['롤길이'] == roll_length) & 
            (df_width_groups['core'] == core) & 
            (df_width_groups['dia'] == dia) &
            (df_width_groups['등급'] == quality_grade)
        ].copy()

        if df_subset_engine.empty:
            continue
            
        # [Mod] Color 값 주입 (Passed from caller)
        df_subset_engine['color'] = color
            
        # 컬럼명 변경 (엔진 호환성)
        df_subset_engine = df_subset_engine.rename(columns={'width_group_no': 'group_order_no', 'total_qty': '주문수량'})

        logging.info(f"--- 롤길이 그룹 {roll_length}, Core {core}, Dia {dia}에 대한 주문 정보 (지폭 그룹핑) ---")
        logging.info(df_subset_engine.to_string())
        
        optimizer = RollOptimize(
            db=db,
            plant=plant,
            pm_no=pm_no,
            schedule_unit=schedule_unit,
            lot_no=lot_no,
            version=version,
            paper_type=paper_type,
            b_wgt=float(b_wgt),
            color=color,
            p_type=paper_type,
            p_wgt=float(b_wgt),
            p_color=color,
            p_machine=pm_no,
            df_spec_pre=df_subset_engine,
            min_width=int(re_min_width),
            max_width=int(re_max_width),
            max_pieces=int(re_max_pieces),
            time_limit=time_limit,
            num_threads=NUM_THREADS
        )
        results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)

        if "error" in results:
            logging.error(f"[에러] Lot {lot_no}, 롤길이 {roll_length}, Core {core}, Dia {dia} 최적화 실패: {results['error']}")
            continue
        
        # 3. 결과 배분 (Allocation)
        # 엔진 결과(WG 기준)를 개별 오더(group_order_no)로 변환
        
        allocated_pattern_details = []
        allocated_roll_details = []
        allocated_cut_details = []
        
        # results['pattern_details_for_db']는 WG ID를 포함하고 있음
        for entry in results['pattern_details_for_db']:
            wg_ids = entry['group_nos'] # 예: ['WG1', 'WG1', 'WG2', '', ...]
            widths = entry['widths']
            total_run_count = entry['count']
            
            allocated_runs = []
            
            for _ in range(total_run_count):
                run_assignment = []
                for i, wg_id in enumerate(wg_ids):
                    if not wg_id:
                        run_assignment.append('')
                        continue
                    
                    candidate_orders = width_group_to_orders.get(wg_id, [])
                    assigned_order = None
                    
                    # 잔여 수요 우선 할당
                    for order_id in candidate_orders:
                        if remaining_demands.get(order_id, 0) > 0:
                            assigned_order = order_id
                            remaining_demands[order_id] -= 1
                            break
                    
                    # 과잉 생산 시 마지막 오더 할당
                    if not assigned_order and candidate_orders:
                        assigned_order = candidate_orders[-1]
                    
                    run_assignment.append(assigned_order if assigned_order else '')
                allocated_runs.append(tuple(run_assignment))
            
            run_counts = Counter(allocated_runs)
            
            for order_combo, count in run_counts.items():
                prod_seq_counter += 1
                
                # Pattern Details
                new_entry = entry.copy()
                new_entry['group_nos'] = list(order_combo)
                new_entry['count'] = count
                new_entry['prod_seq'] = prod_seq_counter
                allocated_pattern_details.append(new_entry)
                
                # Roll Details
                roll_seq_counter = 0
                for i, width in enumerate(widths):
                    if width <= 0: continue
                    roll_seq_counter += 1
                    group_no = list(order_combo)[i]
                    
                    allocated_roll_details.append({
                        'rollwidth': width,
                        'pattern_length': entry.get('pattern_length', 0),
                        'widths': [width] + [0]*7,
                        'group_nos': [group_no] + ['']*7,
                        'count': count,
                        'prod_seq': prod_seq_counter,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'R',
                        'p_lot': entry.get('p_lot'),
                        'diameter': entry.get('diameter'),
                        'core': entry.get('core'),
                        'color': entry.get('color'),
                        'luster': entry.get('luster')
                    })
                    
                    # Cut Details (Roll Optimize에서는 보통 Roll Detail과 1:1 매핑되거나 생략될 수 있으나, 기존 로직 따름)
                    # 기존 로직: pattern_roll_cut_details_for_db 생성
                    # 여기서는 Roll Detail 하나당 Cut Detail 하나로 가정 (단순화)
                    allocated_cut_details.append({
                        'prod_seq': prod_seq_counter,
                        'unit_no': prod_seq_counter,
                        'seq': roll_seq_counter, # 임시
                        'roll_seq': roll_seq_counter,
                        'cut_seq': 1,
                        'rs_gubun': 'R',
                        'width': width,
                        'group_no': group_no,
                        'weight': 0,
                        'pattern_length': entry.get('pattern_length', 0),
                        'count': count,
                        'p_lot': entry.get('p_lot'),
                        'diameter': entry.get('diameter'),
                        'core': entry.get('core'),
                        'color': entry.get('color'),
                        'luster': entry.get('luster')
                    })

        logging.info(f"--- 롤길이 그룹 {roll_length}, Core {core}, Dia {dia} 최적화 성공 (배분 완료) ---")
        all_results["pattern_result"].append(results["pattern_result"]) # 요약용 (WG 기준일 수 있음, 주의)
        for detail in allocated_pattern_details:
            detail['max_width'] = int(re_max_width)
        all_results["pattern_details_for_db"].extend(allocated_pattern_details)
        all_results["pattern_roll_details_for_db"].extend(allocated_roll_details)
        all_results["pattern_roll_cut_details_for_db"].extend(allocated_cut_details)
        
        # Fulfillment Summary 재계산 필요 (개별 오더 기준)
        # 하지만 여기서는 일단 results['fulfillment_summary'] (WG 기준)를 넣고, 나중에 전체 집계 시 다시 계산하거나
        # 혹은 여기서 개별 오더 기준으로 다시 만들어야 함.
        # 시간 관계상, 그리고 save_results에서 fulfillment_summary는 로깅용으로 주로 쓰이므로
        # 정확한 개별 오더 충족 현황을 보려면 remaining_demands를 역산해야 함.
        # 일단은 WG 기준 Summary를 유지하되, 로그에 개별 오더 현황을 찍어주는 것이 좋음.
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 롤지 최적화 결과가 없습니다.")
        return None, None, start_prod_seq, start_group_order_no

    # Fulfillment Summary를 개별 오더 기준으로 재생성 (정확한 리포팅을 위해)
    # 생산된 수량 집계
    production_counts = Counter()
    for detail in all_results["pattern_details_for_db"]:
        for group_no in detail['group_nos']:
            if group_no:
                production_counts[group_no] += detail['count']
    
    df_prod = pd.DataFrame.from_dict(production_counts, orient='index', columns=['생산롤수'])
    df_prod.index.name = 'group_order_no'
    
    # 원본 오더 정보와 결합
    df_summary = df_orders.set_index('group_order_no')[['지폭', '롤길이', '등급', '주문수량']].copy()
    df_summary = df_summary.join(df_prod).fillna(0)
    df_summary['과부족(롤)'] = df_summary['생산롤수'] - df_summary['주문수량']
    df_summary = df_summary.reset_index()
    
    # --- [New] Sheet Sequence Data Generation for Roll Orders ---
    pattern_sheet_details_for_db = generate_allocated_sheet_details(df_orders, all_results["pattern_roll_cut_details_for_db"], b_wgt)

    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True), # 여전히 WG 기준 패턴일 수 있음
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": df_summary # 개별 오더 기준 Summary로 교체
    }

    logging.info("\n--- 롤지 최적화 성공. ---")
    return final_results, df_orders, prod_seq_counter, last_group_order_no

def process_sheet_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, coating_yn, 
        paper_type, b_wgt, color,
        p_type, p_wgt, p_color, p_machine,
        min_width, max_width, min_piece, max_piece, 
        time_limit, sheet_length_re, std_roll_cnt,
        min_sc_width, max_sc_width, sheet_trim_size, 
        min_cm_width, max_cm_width, max_sl_count, ww_trim_size, ww_trim_size_sheet,
        double_cutter='N', # [New]
        start_prod_seq=0, start_group_order_no=0
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot (CA): {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: coating_yn={coating_yn}, min_width={min_width}, max_width={max_width}, max_pieces={max_piece}, sheet_length_re={sheet_length_re}")
    logging.info(f"std_roll_cnt={std_roll_cnt}, min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, sheet_trim_size={sheet_trim_size}")
    logging.info(f"min_cm_width={min_cm_width}, max_cm_width={max_cm_width}, max_sl_count={max_sl_count}, ww_trim_size={ww_trim_size}, ww_trim_size_sheet={ww_trim_size_sheet}, max_sl_count={max_sl_count}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_sheet_orders_from_db_ca(paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 쉬트지(CA) 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
    # df_orders.rename(columns={'오더번호': 'order_no'}, inplace=True)
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['가로', '세로', '등급', 'pattern']
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    df_orders['pattern'] = df_orders['pattern'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    df_orders['color'] = color
    
    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "pattern_roll_details_for_db": [],
        "pattern_roll_cut_details_for_db": [],
        "fulfillment_summary": []
    }
    
    # 등급 + pattern 조합으로 그룹핑 (각 조합별로 별도 최적화)
    grouping_cols = ['등급', 'pattern']
    unique_groups = df_orders[grouping_cols].drop_duplicates()
    prod_seq_counter = start_prod_seq
    
    for _, grp_row in unique_groups.iterrows():
        grade = grp_row['등급']
        pattern = grp_row['pattern']
        logging.info(f"\n--- 등급 {grade}, pattern {pattern}에 대한 쉬트지(CA) 최적화 시작 ---")
        df_subset = df_orders[(df_orders['등급'] == grade) & (df_orders['pattern'] == pattern)].copy()
        
        if df_subset.empty:
            continue

        optimizer = SheetOptimizeCa(
            db=db,
            plant=plant,
            pm_no=pm_no,
            schedule_unit=schedule_unit,
            lot_no=lot_no,
            version=version,
            paper_type=paper_type,
            b_wgt=float(b_wgt),
            color=color,
            p_type=p_type,
            p_wgt=float(p_wgt),
            p_color=p_color,
            p_machine=p_machine,
            coating_yn=coating_yn,
            df_spec_pre=df_subset,
            min_width=int(min_width),
            max_width=int(max_width),
            max_pieces=int(max_piece),
            min_sheet_roll_length=int(sheet_length_re) // 10 * 10,
            max_sheet_roll_length=int(sheet_length_re) // 10 * 10,
            std_roll_cnt=std_roll_cnt,
            sheet_trim=sheet_trim_size,
            min_sc_width=min_sc_width,
            max_sc_width=max_sc_width,
            min_cm_width=min_cm_width,
            max_cm_width=max_cm_width,
            max_sl_count=max_sl_count,
            ww_trim_size=ww_trim_size,
            ww_trim_size_sheet=ww_trim_size_sheet,
            num_threads=NUM_THREADS,
            double_cutter=double_cutter
        )
        try:
            results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)
            
            if not results or "error" in results:
                error_msg = results['error'] if results and 'error' in results else "No solution found"
                logging.error(f"[에러] Lot {lot_no}, 등급 {grade}, pattern {pattern} 쉬트지(CA) 최적화 실패: {error_msg}")
                continue
                
            prod_seq_counter = results.get('last_prod_seq', prod_seq_counter)
            
            if "pattern_details_for_db" in results:
                for detail in results["pattern_details_for_db"]:
                    detail['max_width'] = int(max_width)
            
            all_results["pattern_result"].append(results["pattern_result"])
            all_results["pattern_details_for_db"].extend(results["pattern_details_for_db"])
            all_results["pattern_roll_details_for_db"].extend(results.get("pattern_roll_details_for_db", []))
            all_results["pattern_roll_cut_details_for_db"].extend(results.get("pattern_roll_cut_details_for_db", []))
            all_results["fulfillment_summary"].append(results["fulfillment_summary"])
            
            logging.info(f"--- 등급 {grade}, pattern {pattern} 쉬트지(CA) 최적화 성공 ---")
            
        except Exception as e:
            import traceback
            logging.error(f"[에러] Lot {lot_no}, 등급 {grade}, pattern {pattern} 처리 중 예외 발생")
            logging.error(traceback.format_exc())
            continue

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 쉬트지(CA) 최적화 결과가 없습니다 (모든 등급/패턴 실패).")
        return None, None, start_prod_seq, start_group_order_no

    # --- [New] Sheet Sequence Allocation & Data Generation ---
    # CA(5000) 공장은 unroll 하지 않고 count 값에 입력 (User Request)
    pattern_sheet_details_for_db = generate_allocated_sheet_details(
        df_orders, all_results["pattern_roll_cut_details_for_db"], b_wgt, unroll_by_count=False
    )

    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True) if all_results["pattern_result"] else pd.DataFrame(),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": pd.concat(all_results["fulfillment_summary"], ignore_index=True) if all_results["fulfillment_summary"] else pd.DataFrame()
    }

    logging.info("쉬트지(CA) 최적화 성공 (전체 등급/패턴 완료).")
    return final_results, df_orders, prod_seq_counter, last_group_order_no

def process_roll_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        paper_type, b_wgt, color, 
        p_type, p_wgt, p_color, p_machine,
        re_min_width, re_max_width, re_max_pieces, time_limit, 
        start_prod_seq=0, start_group_order_no=0
):
    """롤지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Roll Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_roll_orders_from_db_ca(plant=plant, pm_no=pm_no, schedule_unit=schedule_unit, paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 롤지 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    # 1. 개별 오더 그룹핑 (DB 저장용) - order_no 포함
    group_cols = ['지폭', '롤길이', '등급']
    for col in ['지폭', '롤길이']:
        df_orders[col] = pd.to_numeric(df_orders[col])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    logging.info(f"--- Lot {lot_no} 원본 주문 정보 (그룹오더 포함) ---")
    logging.info(df_orders.to_string())
    logging.info("\n")

    # 2. 지폭 그룹핑 (엔진 최적화용) - order_no 제외
    # 엔진 효율을 위해 동일 규격은 하나로 묶음
    # 2. 지폭 그룹핑 (엔진 최적화용) - order_no 제외
    # 엔진 효율을 위해 동일 규격은 하나로 묶음
    width_group_cols = ['지폭', '롤길이', '등급']
    df_width_groups = df_orders.groupby(width_group_cols).agg(
        total_qty=('주문수량', 'sum')
    ).reset_index()
    
    # 임시 그룹 ID 생성 (WG...)
    df_width_groups['width_group_no'] = [f'WG{i+1}' for i in range(len(df_width_groups))]
    
    # 원본 데이터에 지폭 그룹 ID 매핑
    df_orders = pd.merge(df_orders, df_width_groups[['지폭', '롤길이', '등급', 'width_group_no']], 
                         on=['지폭', '롤길이', '등급'], how='left')

    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "pattern_roll_details_for_db": [],
        "pattern_roll_cut_details_for_db": [],
        "fulfillment_summary": []
    }
    
    # 엔진 수행 그룹핑 기준 컬럼 설정
    grouping_cols = ['롤길이', '등급']
    unique_groups = df_orders[grouping_cols].drop_duplicates()
    prod_seq_counter = start_prod_seq

    # 결과 배분을 위한 준비
    remaining_demands = df_orders.set_index('group_order_no')['주문수량'].to_dict()
    width_group_to_orders = df_orders.groupby('width_group_no')['group_order_no'].apply(list).to_dict()

    for _, row in unique_groups.iterrows():
        roll_length = row['롤길이']
        quality_grade = row['등급']
        logging.info(f"\n--- 롤길이 그룹 {roll_length}, 등급 {quality_grade}에 대한 최적화 시작 ---")
        
        # 엔진에는 지폭 그룹 데이터를 전달 (width_group_no를 group_order_no로 위장)
        df_subset_engine = df_width_groups[
            (df_width_groups['롤길이'] == roll_length) & 
            (df_width_groups['등급'] == quality_grade)
        ].copy()

        if df_subset_engine.empty:
            continue
            
        # [Mod] Color 값 주입 (Passed from caller)
        df_subset_engine['color'] = color
            
        # 컬럼명 변경 (엔진 호환성)
        df_subset_engine = df_subset_engine.rename(columns={'width_group_no': 'group_order_no', 'total_qty': '주문수량'})

        logging.info(f"--- 롤길이 그룹 {roll_length}, 등급 {quality_grade}에 대한 주문 정보 (지폭 그룹핑) ---")
        logging.info(df_subset_engine.to_string())
        
        optimizer = RollOptimize(
            db=db,
            plant=plant,
            pm_no=pm_no,
            schedule_unit=schedule_unit,
            lot_no=lot_no,
            version=version,
            paper_type=paper_type,
            b_wgt=float(b_wgt),
            color=color,
            p_type=paper_type,
            p_wgt=float(b_wgt),
            p_color=color,
            p_machine=pm_no,
            df_spec_pre=df_subset_engine,
            min_width=int(re_min_width),
            max_width=int(re_max_width),
            max_pieces=int(re_max_pieces),
            time_limit=time_limit,
            num_threads=NUM_THREADS
        )
        results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)

        if "error" in results:
            logging.error(f"[에러] Lot {lot_no}, 롤길이 {roll_length}, 최적화 실패: {results['error']}")
            continue
        
        # 3. 결과 배분 (Allocation)
        # 엔진 결과(WG 기준)를 개별 오더(group_order_no)로 변환
        
        allocated_pattern_details = []
        allocated_roll_details = []
        allocated_cut_details = []
        
        # results['pattern_details_for_db']는 WG ID를 포함하고 있음
        for entry in results['pattern_details_for_db']:
            wg_ids = entry['group_nos'] # 예: ['WG1', 'WG1', 'WG2', '', ...]
            widths = entry['widths']
            total_run_count = entry['count']
            
            allocated_runs = []
            
            for _ in range(total_run_count):
                run_assignment = []
                for i, wg_id in enumerate(wg_ids):
                    if not wg_id:
                        run_assignment.append('')
                        continue
                    
                    candidate_orders = width_group_to_orders.get(wg_id, [])
                    assigned_order = None
                    
                    # 잔여 수요 우선 할당
                    for order_id in candidate_orders:
                        if remaining_demands.get(order_id, 0) > 0:
                            assigned_order = order_id
                            remaining_demands[order_id] -= 1
                            break
                    
                    # 과잉 생산 시 마지막 오더 할당
                    if not assigned_order and candidate_orders:
                        assigned_order = candidate_orders[-1]
                    
                    run_assignment.append(assigned_order if assigned_order else '')
                allocated_runs.append(tuple(run_assignment))
            
            run_counts = Counter(allocated_runs)
            
            for order_combo, count in run_counts.items():
                prod_seq_counter += 1
                
                # Pattern Details
                new_entry = entry.copy()
                new_entry['group_nos'] = list(order_combo)
                new_entry['count'] = count
                new_entry['prod_seq'] = prod_seq_counter
                allocated_pattern_details.append(new_entry)
                
                # Roll Details
                roll_seq_counter = 0
                for i, width in enumerate(widths):
                    if width <= 0: continue
                    roll_seq_counter += 1
                    group_no = list(order_combo)[i]
                    
                    allocated_roll_details.append({
                        'rollwidth': width,
                        'pattern_length': entry.get('pattern_length', 0),
                        'widths': [width] + [0]*7,
                        'group_nos': [group_no] + ['']*7,
                        'count': count,
                        'prod_seq': prod_seq_counter,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'R',
                        'p_lot': entry.get('p_lot'),
                        'diameter': entry.get('diameter'),
                        'core': entry.get('core'),
                        'color': entry.get('color'),
                        'luster': entry.get('luster')
                    })
                    
                    # Cut Details (Roll Optimize에서는 보통 Roll Detail과 1:1 매핑되거나 생략될 수 있으나, 기존 로직 따름)
                    # 기존 로직: pattern_roll_cut_details_for_db 생성
                    # 여기서는 Roll Detail 하나당 Cut Detail 하나로 가정 (단순화)
                    allocated_cut_details.append({
                        'prod_seq': prod_seq_counter,
                        'unit_no': prod_seq_counter,
                        'seq': roll_seq_counter, # 임시
                        'roll_seq': roll_seq_counter,
                        'cut_seq': 1,
                        'rs_gubun': 'R',
                        'width': width,
                        'group_no': group_no,
                        'weight': 0,
                        'pattern_length': entry.get('pattern_length', 0),
                        'count': count,
                        'p_lot': entry.get('p_lot'),
                        'diameter': entry.get('diameter'),
                        'core': entry.get('core'),
                        'color': entry.get('color'),
                        'luster': entry.get('luster')
                    })

        logging.info(f"--- 롤길이 그룹 {roll_length}, 최적화 성공 (배분 완료) ---")
        all_results["pattern_result"].append(results["pattern_result"]) # 요약용 (WG 기준일 수 있음, 주의)
        for detail in allocated_pattern_details:
            detail['max_width'] = int(re_max_width)
        all_results["pattern_details_for_db"].extend(allocated_pattern_details)
        all_results["pattern_roll_details_for_db"].extend(allocated_roll_details)
        all_results["pattern_roll_cut_details_for_db"].extend(allocated_cut_details)
        
        # Fulfillment Summary 재계산 필요 (개별 오더 기준)
        # 하지만 여기서는 일단 results['fulfillment_summary'] (WG 기준)를 넣고, 나중에 전체 집계 시 다시 계산하거나
        # 혹은 여기서 개별 오더 기준으로 다시 만들어야 함.
        # 시간 관계상, 그리고 save_results에서 fulfillment_summary는 로깅용으로 주로 쓰이므로
        # 정확한 개별 오더 충족 현황을 보려면 remaining_demands를 역산해야 함.
        # 일단은 WG 기준 Summary를 유지하되, 로그에 개별 오더 현황을 찍어주는 것이 좋음.
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 롤지 최적화 결과가 없습니다.")
        return None, None, start_prod_seq, start_group_order_no

    # Fulfillment Summary를 개별 오더 기준으로 재생성 (정확한 리포팅을 위해)
    # 생산된 수량 집계
    production_counts = Counter()
    for detail in all_results["pattern_details_for_db"]:
        for group_no in detail['group_nos']:
            if group_no:
                production_counts[group_no] += detail['count']
    
    df_prod = pd.DataFrame.from_dict(production_counts, orient='index', columns=['생산롤수'])
    df_prod.index.name = 'group_order_no'
    
    # 원본 오더 정보와 결합
    df_summary = df_orders.set_index('group_order_no')[['지폭', '롤길이', '등급', '주문수량']].copy()
    df_summary = df_summary.join(df_prod).fillna(0)
    df_summary['과부족(롤)'] = df_summary['생산롤수'] - df_summary['주문수량']
    df_summary = df_summary.reset_index()
    
    # --- [New] Sheet Sequence Data Generation for Roll Orders ---
    pattern_sheet_details_for_db = generate_allocated_sheet_details(df_orders, all_results["pattern_roll_cut_details_for_db"], b_wgt)

    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True), # 여전히 WG 기준 패턴일 수 있음
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": df_summary # 개별 오더 기준 Summary로 교체
    }

    logging.info("\n--- 롤지 최적화 성공. ---")
    return final_results, df_orders, prod_seq_counter, last_group_order_no

def process_coating_roll_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, time_limit, coating_yn, 
        paper_type, b_wgt, color,
        p_type, p_wgt, p_color, p_machine,
        min_width, max_width, max_pieces, 
        min_cm_width, max_cm_width, max_sl_count, ww_trim_size, 
        start_prod_seq=0, start_group_order_no=0
):
    """롤지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info("process_coating_roll_lot_ca 함수 호출")
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Roll Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={min_width}, max_width={max_width}, max_pieces={max_pieces}")
    logging.info(f"min_cm_width={min_cm_width}, max_cm_width={max_cm_width}, max_sl_count={max_sl_count}")
    logging.info(f"coating_yn={coating_yn}, ww_trim_size={ww_trim_size}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_roll_orders_from_db_ca(plant=plant, pm_no=pm_no, schedule_unit=schedule_unit, paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 롤지 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['지폭', '롤길이', '등급']
    for col in ['지폭', '롤길이']:
        df_orders[col] = pd.to_numeric(df_orders[col])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    
    df_groups['group_order_no'] = [f"50{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    logging.info(f"--- Lot {lot_no} 원본 주문 정보 (그룹오더 포함) ---")
    logging.info(df_orders.to_string())
    logging.info("\n")

    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "pattern_roll_details_for_db": [],
        "pattern_roll_cut_details_for_db": [],
        "fulfillment_summary": []
    }
    
    # std_length + 등급 조합으로 그룹핑 (각 조합별로 별도 최적화)
    grouping_cols = ['std_length', '등급']
    unique_groups = df_orders[grouping_cols].drop_duplicates()
    prod_seq_counter = start_prod_seq

    for _, grp_row in unique_groups.iterrows():
        std_length = grp_row['std_length']
        grade = grp_row['등급']
        logging.info(f"\n--- 표준길이 {std_length}, 등급 {grade}에 대한 최적화 시작 ---")
        df_subset = df_orders[(df_orders['std_length'] == std_length) & (df_orders['등급'] == grade)].copy()

        if df_subset.empty:
            continue

        optimizer = RollOptimizeCa(
            db=db,
            plant=plant,
            pm_no=pm_no,
            schedule_unit=schedule_unit,
            lot_no=lot_no,
            version=version,
            paper_type=paper_type,
            b_wgt=float(b_wgt),
            color=color,
            p_type=p_type,
            p_wgt=float(p_wgt),
            p_color=p_color,
            p_machine=p_machine,
            df_spec_pre=df_subset,
            coating_yn=coating_yn,
            min_width=int(min_width),
            max_width=int(max_width),
            max_pieces=int(max_pieces),
            min_cm_width=min_cm_width,
            max_cm_width=max_cm_width,
            max_sl_count=max_sl_count,
            ww_trim_size=ww_trim_size,
            num_threads=NUM_THREADS
        )
        results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)

        if "error" in results:
            logging.error(f"[에러] Lot {lot_no}, 표준길이 {std_length}, 등급 {grade} 최적화 실패: {results['error']}")
            continue
        
        prod_seq_counter = results.get('last_prod_seq', prod_seq_counter)

        logging.info(f"--- 표준길이 {std_length}, 등급 {grade} 최적화 성공 ---")
        all_results["pattern_result"].append(results["pattern_result"])
        for detail in results["pattern_details_for_db"]:
            detail['max_width'] = int(max_width)
        all_results["pattern_details_for_db"].extend(results["pattern_details_for_db"])
        all_results["pattern_roll_details_for_db"].extend(results.get("pattern_roll_details_for_db", []))
        all_results["pattern_roll_cut_details_for_db"].extend(results.get("pattern_roll_cut_details_for_db", []))
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 롤지 최적화 결과가 없습니다.")
        return None, None, start_prod_seq, start_group_order_no

    # CA(5000) 공장은 unroll 하지 않고 count 값에 입력 (User Request)
    pattern_sheet_details_for_db = generate_allocated_sheet_details(
        df_orders, all_results["pattern_roll_cut_details_for_db"], b_wgt, unroll_by_count=False
    )

    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": pd.concat(all_results["fulfillment_summary"], ignore_index=True)
    }

    logging.info("\n--- 롤지 최적화 성공. ---")
    return final_results, df_orders, prod_seq_counter, last_group_order_no

def process_sheet_lot_var(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt, color,
        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re,
        start_prod_seq=0, start_group_order_no=0
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot (Var): {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    logging.info(f"min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, min_sheet_length_re={min_sheet_length_re}, max_sheet_length_re={max_sheet_length_re}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_sheet_orders_from_db_var(paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 쉬트지(Var) 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
    # df_orders.rename(columns={'오더번호': 'order_no'}, inplace=True)
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['가로', '세로', '등급']
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    df_orders['color'] = color
    logging.info("--- 쉬트지(Var) 최적화 시작 ---")
    optimizer = SheetOptimizeVar(
        df_spec_pre=df_orders,
        max_width=int(re_max_width),
        min_width=int(re_min_width),
        max_pieces=int(re_max_pieces),
        b_wgt=float(b_wgt),
        min_sheet_roll_length=int(min_sheet_length_re) // 10 * 10,
        max_sheet_roll_length=int(max_sheet_length_re) // 10 * 10,
        sheet_trim=sheet_trim_size,
        min_sc_width=min_sc_width,
        max_sc_width=max_sc_width
    )
    try:
        results = optimizer.run_optimize(start_prod_seq=start_prod_seq)
        prod_seq_counter = results.get('last_prod_seq', start_prod_seq)
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        logging.error(f"[에러] Lot {lot_no} 쉬트지(Var) 최적화 실패: {error_msg}.")
        return None, None, start_prod_seq, start_group_order_no
    
    if results and "pattern_details_for_db" in results:
        for detail in results["pattern_details_for_db"]:
            detail['max_width'] = int(re_max_width)

    # --- [New] Sheet Sequence Allocation & Data Generation ---
    pattern_sheet_details_for_db = generate_allocated_sheet_details(df_orders, results["pattern_roll_cut_details_for_db"], b_wgt)

    results["pattern_sheet_details_for_db"] = pattern_sheet_details_for_db
    logging.info("쉬트지(Var) 최적화 성공.")
    return results, df_orders, prod_seq_counter, last_group_order_no


def process_roll_sl_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt, color,
        min_sl_width, max_sl_width, sl_trim_size,
        start_prod_seq=0, start_group_order_no=0
):
    """롤-슬리터 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Roll-SL Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}, min_sl_width={min_sl_width}, max_sl_width={max_sl_width}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_roll_sl_orders_from_db(paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 롤-슬리터 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
    # df_orders.rename(columns={'오더번호': 'order_no'}, inplace=True)
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['지폭', '롤길이', '등급', 'core', 'dia', 'luster', 'color', 'order_pattern']
    df_orders['지폭'] = pd.to_numeric(df_orders['지폭'])
    df_orders['롤길이'] = pd.to_numeric(df_orders['롤길이'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)

    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    df_orders['color'] = color
    
    optimizer = RollSLOptimize(
        df_spec_pre=df_orders,
        max_width=int(re_max_width),
        min_width=int(re_min_width),
        max_pieces=int(re_max_pieces),
        b_wgt=float(b_wgt),
        sl_trim=sl_trim_size,
        min_sl_width=min_sl_width,

        max_sl_width=max_sl_width,
        lot_no=lot_no
    )
    try:
        results = optimizer.run_optimize(start_prod_seq=start_prod_seq)
        prod_seq_counter = results.get('last_prod_seq', start_prod_seq)
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        logging.error(f"[에러] Lot {lot_no} 롤-슬리터 최적화 실패: {error_msg}.")
        return None, None, start_prod_seq, start_group_order_no
    
    if results and "pattern_details_for_db" in results:
        for detail in results["pattern_details_for_db"]:
            detail['max_width'] = int(re_max_width)
    
    # --- [New] Sheet Sequence Data Generation for Roll-SL Orders ---
    pattern_sheet_details_for_db = generate_allocated_sheet_details(df_orders, results["pattern_roll_cut_details_for_db"], b_wgt)

    final_results = {
        "pattern_result": results["pattern_result"],
        "pattern_details_for_db": results["pattern_details_for_db"],
        "pattern_roll_details_for_db": results.get("pattern_roll_details_for_db", []),
        "pattern_roll_cut_details_for_db": results.get("pattern_roll_cut_details_for_db", []),
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": results["fulfillment_summary"]
    }

    logging.info("롤-슬리터 최적화 성공.")
    return final_results, df_orders, prod_seq_counter, last_group_order_no

def apply_sheet_grouping(df_orders, start_group_order_no, lot_no):
    """
    쉬트지 오더 그룹핑 로직을 적용하고 group_order_no를 생성합니다.
    """
    def get_group_key(row):
        # 1. 내수
        if row['수출내수'] == '내수':
            # 내수이고 export_yn이 N이면 오더번호별로 그룹오더생성
            return f"DOM_{row['order_no']}"
        
        # 2. 수출
        else: # 수출내수 == '수출'
            # 수출이고 export_yn이 Y이면 order_gubun 값이 A 이면 오더번호 별로 그룹오더 생성
            if row.get('order_gubun') == 'A':
                return f"EXP_A_{row['order_no']}"
            else:
                # 그 외에는 오더정보의 length 가 450~549, 550~699, 700 이상, 이 그룹으로 그룹오더 생성
                length = row['세로']
                if 450 <= length <= 549:
                    return "EXP_L_450_549"
                elif 550 <= length <= 699:
                    return "EXP_L_550_699"
                elif length >= 700:
                    return "EXP_L_700_PLUS"
                else:
                    return f"EXP_L_OTHER_{length}"

    df_orders['_group_key'] = df_orders.apply(get_group_key, axis=1)
    
    # 그룹핑 기준: 가로, 등급, _group_key
    group_cols = ['가로', '등급', '_group_key']

    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()
    
    # 정렬 (가로, 등급 순)
    df_groups = df_groups.sort_values(by=['가로', '등급']).reset_index(drop=True)
    
    # Group Order No 생성
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    # 원본 데이터에 병합
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')
    
    # 임시 컬럼 제거
    if '_group_key' in df_orders.columns:
        df_orders = df_orders.drop(columns=['_group_key'])
        
    return df_orders, df_groups, last_group_order_no


def generate_allocated_sheet_details(df_orders, source_details, b_wgt=None, unroll_by_count=True):
    """
    Generates sheet sequence details by allocation grouped orders based on quantity.
    Sorts orders within a group by quantity (ASC) and distributes production.
    
    Args:
        df_orders (pd.DataFrame): Order dataframe
        source_details (list): List of pattern roll cut details.
        b_wgt (float, optional): Basis Weight.
        unroll_by_count (bool): If True, explodes 'count' into individual rows (1 per count).
                                If False, keeps 'count' grouped in one row (e.g. for Plant 5000).
    """
    # 1. Build Group -> Orders Map (Sorted by Quantity ASC)
    group_orders_map = {}
    
    # Ensure quantitative columns are numeric
    if '주문톤' in df_orders.columns:
        df_orders['주문톤'] = pd.to_numeric(df_orders['주문톤'], errors='coerce').fillna(0)
    
    # Get unique group_order_nos
    unique_groups = df_orders['group_order_no'].unique()
    
    for group_no in unique_groups:
        group_df = df_orders[df_orders['group_order_no'] == group_no].copy()
        # Sort by Order Amount (Smallest First)
        # Prefer '주문톤', fallback to '주문수량' if needed
        sort_col = '주문톤' if '주문톤' in group_df.columns else '주문수량'
        if sort_col not in group_df.columns:
             # Fallback: just use order_no if no qty info
            group_df['temp_sort'] = 0
            sort_col = 'temp_sort'
            
        group_df = group_df.sort_values(by=sort_col, ascending=True)
        
        orders_list = []
        for _, row in group_df.iterrows():
            # Determine correct length and Rs Gubun
            rs_gubun = row.get('rs_gubun', 'S')  # Default to Sheet if missing
            
            # For Sheet: Length is '세로' (Cutoff Length, usually in mm)
            # For Roll: Length is '롤길이' (Roll Length, usually in m)
            if rs_gubun == 'R':
                order_len = row.get('롤길이', 0)
            else:
                order_len = row.get('세로', 0)
                
            orders_list.append({
                'order_no': row['order_no'],
                'pack_type': row.get('pack_type', ''),
                'length': order_len, 
                'rs_gubun': rs_gubun,
                'target_ton': row.get('주문톤', 0),
                'fulfilled_ton': 0.0
            })
        group_orders_map[group_no] = orders_list

    allocated_details = []
    
    for detail in source_details:
        group_no = detail.get('group_no') # Cut Detail has group_no
        if not group_no or group_no not in group_orders_map:
            continue
            
        candidate_orders = group_orders_map[group_no]
        count = detail.get('count', 1)
        
        pattern_length = detail.get('pattern_length', 0)
        width = detail.get('width', 0)
        
        # Determine Sheet Count per 1 Cut based on type
        # All orders in same group MUST have same length/type
        if not candidate_orders:
            continue
            
        first_order = candidate_orders[0]
        order_length = first_order['length']
        rs_gubun = first_order['rs_gubun']
        
        sheet_cnt = 0
        if rs_gubun == 'R':
            # For Rolls, we produce 1 roll per cut (implied by exploded 1)
            # Or if count > 1, this loop handles it.
            # Sheet Count col for rolls might mean "Length" or just "1"? 
            # Usually strict sheet count is invalid for rolls. 
            # But DB might expect 1.
            sheet_cnt = 1
        else:
            # For Sheets
            if order_length > 0:
                # pattern_length (m) / order_length (mm) * 1000 = Count
                sheet_cnt = int((pattern_length / order_length) * 1000)
            else:
                 sheet_cnt = 0
        
        # Skip if sheet_cnt is 0 for sheets? 
        # For rolls, sheet_cnt=1 is fine.
        if rs_gubun == 'S' and sheet_cnt <= 0:
             continue
        
        # Determine iteration range
        iter_range = range(count) if unroll_by_count else range(1)
        
        # We process 'count' batches (or 1 batch if not unrolling).
        for _ in iter_range:
            # Find target order
            # Strategy: Fill order until Fulfilled >= Target.
            target_order = None
            
            # Simple Greedy: First order that is NOT full
            for order in candidate_orders:
                if order['fulfilled_ton'] < order['target_ton']:
                    target_order = order
                    break
            
            # If all full, overflow to the last one (Largest)
            if not target_order:
                target_order = candidate_orders[-1]
                
            # Create Entry
            new_entry = detail.copy()
            new_entry.update({
                'count': 1 if unroll_by_count else count, # Exploded to 1 if unrolling, else keep count
                'order_no': target_order['order_no'],
                'pack_type': target_order['pack_type'],
                'sheet_cnt': sheet_cnt,
                'sheet_seq': 1,
                'width': width,
                'length': pattern_length, # Pattern Length (m) stored in DB 'length' col usually?
                # DB insert_sheet_sequence uses :length -> pattern_length.
                # So here we keep pattern_length as 'length'.
                # But note: target_order['length'] is cut-off/roll length. 
                # new_entry['length'] is usually Production Length.
                'override_seq': 1 
            })
            
            # Update Fulfillment
            # production weight for this batch
            if b_wgt:
                 try:
                     weight_ton = (pattern_length * width * float(b_wgt)) / (10**9)
                 except (ValueError, TypeError):
                     logging.warning(f"[Alloc] Invalid b_wgt: {b_wgt}. Using 0.")
                     weight_ton = 0
                     
                 target_order['fulfilled_ton'] += weight_ton
            else:
                 # fallback if no b_wgt, just increment arbitrary or assume 1 unit
                 target_order['fulfilled_ton'] += 1
                 
            allocated_details.append(new_entry)
            
    # Post-process to fix PK uniqueness
    # Let's adjust sheet_seq grouping by parent keys
    # Keys identifying a Cut: (prod_seq, roll_seq, cut_seq)
    
    from collections import defaultdict
    seq_tracker = defaultdict(int)
    
    for entry in allocated_details:
        key = (entry.get('prod_seq'), entry.get('roll_seq'), entry.get('cut_seq'))
        seq_tracker[key] += 1
        entry['sheet_seq'] = seq_tracker[key]
        entry['override_seq'] = seq_tracker[key] 

    return allocated_details


def save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, all_results, all_df_orders,
                 p_machine=None, p_type=None, p_wgt=None, p_color=None):
    """
    최적화 결과를 DB에 저장하고 CSV파일로 출력합니다.
    
    Returns:
        int: 상태 코드 (0: 모든 오더 충족, 1: 일부 오더 부족, 2: 에러)
    """
    if not all_results:
        logging.warning(f"Lot {lot_no}에 대해 저장할 결과가 없습니다.")
        return 2  # 결과가 없으면 에러

    final_pattern_result = pd.concat([res["pattern_result"] for res in all_results], ignore_index=True)
    final_fulfillment_summary = pd.concat([res["fulfillment_summary"] for res in all_results], ignore_index=True)
    final_pattern_details_for_db = [item for res in all_results for item in res["pattern_details_for_db"]]
    final_pattern_roll_details_for_db = [item for res in all_results for item in res.get("pattern_roll_details_for_db", [])]
    final_pattern_roll_cut_details_for_db = [item for res in all_results for item in res.get("pattern_roll_cut_details_for_db", [])]
    final_pattern_sheet_details_for_db = [item for res in all_results for item in res.get("pattern_sheet_details_for_db", [])]
    
    final_df_orders = pd.concat(all_df_orders, ignore_index=True)

    logging.info("최적화 결과 (패턴별 생산량):")
    logging.info("\n" + final_pattern_result.to_string())

    # logging.info("패턴 상세 정보 (final_pattern_details_for_db):")
    # # Ensure full output without truncation
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.max_colwidth', None)
    # logging.info("\n" + pd.DataFrame(final_pattern_details_for_db).to_string())
    
    # Fill NaNs for better logging
    final_fulfillment_summary = final_fulfillment_summary.fillna(0)
    
    logging.info("\n# ================= 주문 충족 현황 ================== #\n")
    logging.info("\n" + final_fulfillment_summary.to_string())
    logging.info("\n")
    logging.info("최적화 성공. 이제 결과를 DB에 저장합니다.")
    
    # [DEBUG] Check for NaNs in fulfillment summary
    if final_fulfillment_summary.isnull().values.any():
        logging.warning("[경고] Fulfillment Summary에 NaN 값이 포함되어 있습니다!")
        nan_rows = final_fulfillment_summary[final_fulfillment_summary.isnull().any(axis=1)]
        logging.warning(f"NaN Rows:\n{nan_rows.to_string()}")
    
    # [DEBUG] Log pattern details count
    logging.info(f"[DEBUG] Saving {len(final_pattern_details_for_db)} pattern details to DB.")
    # if len(final_pattern_details_for_db) > 0:
    #     logging.info(f"[DEBUG] First pattern detail sample: {final_pattern_details_for_db[0]}")

    connection = None
    try:
        connection = db.pool.acquire()
        
        # 기존 최적화 결과 삭제 (동일 트랜잭션에서 처리)
        cursor = connection.cursor()
        tables_to_delete = [
            "th_pattern_sequence",
            "th_roll_sequence",
            "th_cut_sequence",
            "th_sheet_sequence",
            "th_order_group",
            "th_group_master"
        ]
        for table in tables_to_delete:
            delete_query = f"DELETE FROM {table} WHERE lot_no = :lot_no AND version = :version"
            cursor.execute(delete_query, lot_no=lot_no, version=version)
            logging.info(f"Deleted data from {table} for lot {lot_no}, version {version}")

        if not final_df_orders.empty:
            # [Update] prod_wgt 계산 및 반영
            prod_wgt_map = {}
            try:
                b_wgt_val = float(b_wgt)
            except (ValueError, TypeError):
                b_wgt_val = 0.0

            if final_pattern_sheet_details_for_db:
                for detail in final_pattern_sheet_details_for_db:
                    order_no = detail.get('order_no')
                    if not order_no: continue
                    
                    width = detail.get('width', 0)
                    # User requested formula: (pattern_length * width * b_wgt / 10^9)
                    # pattern_length covers both sheet total length and roll length
                    pattern_length = detail.get('pattern_length', 0)
                    
                    # Weight (kg) = (pattern_length * width * b_wgt) / 10^6
                    wgt_kg = (pattern_length * width * b_wgt_val) / 10**6
                    
                    prod_wgt_map[order_no] = prod_wgt_map.get(order_no, 0) + wgt_kg
            
            final_df_orders['prod_wgt'] = final_df_orders['order_no'].map(prod_wgt_map).fillna(0).round(1)

            # [Update] th_group_master 입력
            # group_order_no 별로 대표 오더 하나를 선정하여 insert
            # 필요한 컬럼들만 select해서 중복 제거하거나 groupby first 사용
            group_cols = ['group_order_no']
            # 추가 정보 컬럼들 (모두 동일하다고 가정하거나 첫번째 값 사용)
            agg_dict = {
                'order_no': 'first',
                'plant': 'first',
                'schedule_unit': 'first',
                'paper_type': 'first',
                'b_wgt': 'first',
                '가로': 'first', # 'width'
                '세로': 'first', # 'length'
                'rs_gubun': 'first',
                'export_yn': 'first', # 'export'
                'nation_code': 'first',
                'customer_name': 'first', # 'customer'
                'pt_gubun': 'first',
                'skid_yn': 'first',
                'dia': 'first',
                'core': 'first'
            }
            
            # agg_dict에 있는 컬럼이 실제로 존재하는지 확인 후 필터링
            valid_agg_dict = {k: v for k, v in agg_dict.items() if k in final_df_orders.columns}
            
            df_groups = final_df_orders.groupby(group_cols).agg(valid_agg_dict).reset_index()
            
            db.insert_group_master(
                connection, lot_no, version, plant, pm_no, schedule_unit, df_groups
            )

            db.insert_order_group(
                connection, lot_no, version, plant, pm_no, schedule_unit, final_df_orders
            )



        if plant == 5000:
            
            logging.info("\n\n# ================= 패턴 상세 정보 (final_pattern_details_for_db) ================== #\n")
            logging.info(f"롤 재단 상세 정보 개수: {len(final_pattern_details_for_db)}")
            db.insert_pattern_sequence(
                connection, lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
                paper_type, b_wgt, final_pattern_details_for_db,
                p_machine=p_machine, p_type=p_type, p_wgt=p_wgt, p_color=p_color
            )

            if final_pattern_roll_details_for_db:
                logging.info("\n\n# ================= 패턴롤 정보 (final_pattern_roll_details_for_db) ================== #\n")
                logging.info(f"롤 재단 상세 정보 개수: {len(final_pattern_roll_details_for_db)}")
                db.insert_roll_sequence(
                    connection, lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
                    paper_type, b_wgt, final_pattern_roll_details_for_db,
                    p_machine=p_machine, p_type=p_type, p_wgt=p_wgt, p_color=p_color
                )

            if final_pattern_roll_cut_details_for_db:
                logging.info("\n\n# ================= 롤 cut 재단 상세 정보 (final_pattern_roll_cut_details_for_db) ================== #\n")
                logging.info(f"롤 재단 상세 정보 개수: {len(final_pattern_roll_cut_details_for_db)}")
                db.insert_cut_sequence(
                    connection, lot_no, version, plant, pm_no, schedule_unit, 
                    paper_type, b_wgt, final_pattern_roll_cut_details_for_db,
                    p_machine=p_machine, p_type=p_type, p_wgt=p_wgt, p_color=p_color
                )

            if final_pattern_sheet_details_for_db:
                logging.info("\n\n# ================= 쉬트 재단 상세 정보 (final_pattern_sheet_details_for_db) ================== #\n")
                logging.info(f"쉬트 재단 상세 정보 개수: {len(final_pattern_sheet_details_for_db)}")
                # logging.info(f"\n\n# ================= 쉬트 재단 상세 정보 {final_pattern_sheet_details_for_db} ================== #\n")
                db.insert_sheet_sequence(
                    connection, lot_no, version, plant, pm_no, schedule_unit, 
                    paper_type, b_wgt, final_pattern_sheet_details_for_db,
                    p_machine=p_machine, p_type=p_type, p_wgt=p_wgt, p_color=p_color
                )

        else:
            logging.info("\n\n# ================= 패턴 상세 정보 (final_pattern_details_for_db) ================== #\n")
            logging.info(f"롤 재단 상세 정보 개수: {len(final_pattern_details_for_db)}")
            db.insert_pattern_sequence(
                connection, lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
                paper_type, b_wgt, final_pattern_details_for_db
            )

            if final_pattern_roll_details_for_db:
                logging.info("\n\n# ================= 패턴롤 정보 (final_pattern_roll_details_for_db) ================== #\n")
                logging.info(f"롤 재단 상세 정보 개수: {len(final_pattern_roll_details_for_db)}")
                db.insert_roll_sequence(
                    connection, lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
                    paper_type, b_wgt, final_pattern_roll_details_for_db
                )

            if final_pattern_roll_cut_details_for_db:
                logging.info("\n\n# ================= 롤 cut 재단 상세 정보 (final_pattern_roll_cut_details_for_db) ================== #\n")
                logging.info(f"롤 재단 상세 정보 개수: {len(final_pattern_roll_cut_details_for_db)}")
                db.insert_cut_sequence(
                    connection, lot_no, version, plant, pm_no, schedule_unit, 
                    paper_type, b_wgt, final_pattern_roll_cut_details_for_db
                )

            if final_pattern_sheet_details_for_db:
                logging.info("\n\n# ================= 쉬트 재단 상세 정보 (final_pattern_sheet_details_for_db) ================== #\n")
                logging.info(f"쉬트 재단 상세 정보 개수: {len(final_pattern_sheet_details_for_db)}")
                # logging.info(f"\n\n# ================= 쉬트 재단 상세 정보 {final_pattern_sheet_details_for_db} ================== #\n")
                db.insert_sheet_sequence(
                    connection, lot_no, version, plant, pm_no, schedule_unit, 
                    paper_type, b_wgt, final_pattern_sheet_details_for_db
                )




        connection.commit()
        logging.info("DB 트랜잭션이 성공적으로 커밋되었습니다.")

        date_folder = time.strftime('%Y%m%d')
        output_dir = os.path.join('results', date_folder)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime('%y%m%d%H%M%S')
        output_filename = f"{lot_no}_{version}_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_filename)
        final_pattern_result.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"\n[성공] 요약 결과가 다음 파일에 저장되었습니다: {output_path}")
        
        # 최적화 상태 결정: fulfillment_summary의 과부족 확인
        # 과부족 컨럼이 없으면 기본적으로 성공(0)으로 간주
        final_status = 0  # 기본값: 모든 오더 충족

        # '롤길이' 컬럼이 없는 경우 (예: 쉬트지만 최적화 시) 0으로 초기화하여 에러 방지
        if '롤길이' not in final_fulfillment_summary.columns:
             final_fulfillment_summary['롤길이'] = 0
        
        # 롤 최적화 결과 확인 (과부족(롤) 컨럼이 있는 경우)
        if '과부족(롤)' in final_fulfillment_summary.columns:
            under_production_rolls = final_fulfillment_summary[
                (final_fulfillment_summary['과부족(롤)'] != 0) & 
                (final_fulfillment_summary['롤길이'] > 0)
            ]
            if not under_production_rolls.empty:
                final_status = 0  # 일부 오더 초과(부족)
                logging.warning(f"[경고] 초과(부족) 생산된 롤 오더가 있습니다:\n{under_production_rolls.to_string()}")

        
        # 쉬트 최적화 결과 확인 (과부족(톤) 컨럼이 있는 경우)
        if '과부족(톤)' in final_fulfillment_summary.columns:
            under_production_sheets = final_fulfillment_summary[
                (final_fulfillment_summary['과부족(톤)'] < -2) & 
                (final_fulfillment_summary['롤길이'] == 0)
            ]  # 소수점 오차 고려
            over_production_sheets = final_fulfillment_summary[
                (final_fulfillment_summary['과부족(톤)'] >= 2) & 
                (final_fulfillment_summary['롤길이'] == 0)
            ]  # 소수점 오차 고려
            if not under_production_sheets.empty:
                final_status = 0  # 일부 오더 초과(부족)
                logging.warning(f"[경고] 부족 생산된 쉬트 오더가 있습니다:\n{under_production_sheets.to_string()}")
            if not over_production_sheets.empty:
                final_status = 0  # 일부 오더 초과(부족)
                logging.warning(f"[경고] 초과 생산된 쉬트 오더가 있습니다:\n{over_production_sheets.to_string()}")
        
        return final_status

    except Exception as e:
        logging.error(f"[에러] 데이터 저장 중 오류 발생: {e}")
        if connection:
            connection.rollback()
            logging.info("DB 트랜잭션이 롤백되었습니다.")
            db.pool.release(connection)
            connection = None
        
        db.update_lot_status(lot_no=lot_no, version=version, status=2)
        return 2  # 에러 상태

    finally:
        if connection:
            db.pool.release(connection)
        logging.info(f"\n{'='*60}")
        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Lot: {lot_no} 처리 완료")
        logging.info(f"{'='*60}")

def setup_logging(lot_no, version):
    """로그 설정을 초기화합니다. 각 lot마다 새로운 로그 파일을 생성합니다."""
    # 날짜별 폴더 생성 (results/YYYYMMDD/)
    date_folder = time.strftime('%Y%m%d')
    log_dir = os.path.join('results', date_folder)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%y%m%d%H%M%S')
    log_filename = f"{lot_no}_{version}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # 기존 핸들러 모두 제거 (이전 lot의 핸들러)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # 새로운 핸들러 추가
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def main():
    """메인 실행 함수"""
    db = None
    lot_no = None
    version = None
    parser = argparse.ArgumentParser(description='Optimization Executor')
    parser.add_argument('--plant', type=str, default='3000', help='Plant Code (3000, 5000, 8000)')
    args = parser.parse_args()
    plant_arg = args.plant

    db_section_map = {
        '3000': 'database_dj',
        '5000': 'database_ca',
        '8000': 'database_st'
    }
    db_section = db_section_map.get(plant_arg, 'database_dj')
    print(f"Connecting to DB Section: {db_section} for Plant: {plant_arg}")

    try:
        config = configparser.ConfigParser()
        config_path = os.path.join('conf', 'config.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 파일을 찾을 수 없습니다.")
        config.read(config_path, encoding='utf-8')
        
        if db_section not in config:
            raise KeyError(f"Config file does not contain section: {db_section}")
        
        db_config = config[db_section]
        
        db = Database(user=db_config['user'], password=db_config['password'], dsn=db_config['dsn'])

        # Dynamic Patching of DataInserters based on Plant
        # update_lot_status는 공통 모듈에서 가져오고, 나머지 insert 함수들은 공장별 모듈에서 가져옴
        from db import db_insert_data as db_common_module
        
        if plant_arg == '5000':
            from db import db_insert_data_ca as db_module
        elif plant_arg == '8000':
            from db import db_insert_data_st as db_module
        else: # 3000 or default
            from db import db_insert_data_dj as db_module

        # update_lot_status는 공통 모듈에서 바인딩 (3개 공장 공통)
        db.update_lot_status = db_common_module.DataInserters.update_lot_status.__get__(db, Database)
        
        # 나머지 insert 함수들은 공장별 모듈에서 바인딩
        db.insert_pattern_sequence = db_module.DataInserters.insert_pattern_sequence.__get__(db, Database)
        db.insert_roll_sequence = db_module.DataInserters.insert_roll_sequence.__get__(db, Database)
        db.insert_cut_sequence = db_module.DataInserters.insert_cut_sequence.__get__(db, Database)
        db.insert_sheet_sequence = db_module.DataInserters.insert_sheet_sequence.__get__(db, Database)
        db.insert_order_group = db_module.DataInserters.insert_order_group.__get__(db, Database)
        db.insert_group_master = db_module.DataInserters.insert_group_master.__get__(db, Database)

        print(f"Applied common DataInserters from {db_common_module.__name__}")
        print(f"Applied plant-specific DataInserters from {db_module.__name__}")


        while True:
            # Plant에 따라 대상 Lot 조회 함수 분기
            if plant_arg == '5000':
                (                     
                    plant, pm_no, schedule_unit, lot_no, version, time_limit, paper_type, b_wgt, color, 
                    min_width, roll_max_width, min_sc_width, max_sc_width, coating_yn, 
                    sheet_trim_size, ww_trim_size,
                    min_cm_width, max_cm_width, max_sl_count, p_type, p_wgt, ww_trim_size_sheet,
                    sheet_order_cnt, roll_order_cnt
                ) = db.get_target_lot_ca()
            elif plant_arg == '8000':
                ( 
                    plant, pm_no, schedule_unit, lot_no, version, time_limit, min_width, 
                    max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
                    paper_type, b_wgt, color,
                    min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
                    sheet_order_cnt, roll_order_cnt
                ) = db.get_target_lot_st()
            else:  # 3000 or default
                ( 
                    plant, pm_no, schedule_unit, lot_no, version, min_width, 
                    max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
                    paper_type, b_wgt, color,
                    min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
                    sheet_order_cnt, roll_order_cnt, time_limit
                ) = db.get_target_lot()

            if not lot_no:
                # print("처리할 Lot이 없습니다. 10초 후 다시 시도합니다.")
                time.sleep(10)
                continue

            setup_logging(lot_no, version)
            db.update_lot_status(lot_no=lot_no, version=version, status=8)
            

            prod_seq_counter = 0
            group_order_no_counter = 0
            all_results = []
            all_df_orders = []

            if roll_order_cnt > 0:
                logging.info(f"롤지 오더 {roll_order_cnt}건 처리 시작.")
                if plant == '3000':
                    roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_roll_lot(
                        db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, 
                        time_limit, min_width, max_width, max_pieces, 
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                elif plant == '5000':
                    ( 
                        plant, pm_no, schedule_unit, lot_no, version, time_limit, coating_yn, 
                        paper_type, b_wgt, color, 
                        p_type, p_wgt, p_color, p_machine,
                        min_width, max_width, min_pieces, max_pieces,
                        min_cm_width, max_cm_width, max_sl_count, ww_trim_size
                    ) = db.get_lot_param_roll_ca(lot_no=lot_no, version=version)
                    
                    if coating_yn == 'Y':
                        roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_coating_roll_lot_ca(
                            db, plant, pm_no, schedule_unit, lot_no, version, time_limit, coating_yn, 
                            paper_type, b_wgt, color,
                            p_type, p_wgt, p_color, p_machine,
                            min_width, max_width, max_pieces,
                            min_cm_width, max_cm_width, max_sl_count, ww_trim_size,
                            start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                        )
                    else:
                        roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_roll_lot_ca(
                            db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit, 
                            min_width, max_width, max_pieces, 
                            p_type, p_wgt, p_color, p_machine,
                            start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                        )
                else:
                    ( 
                        plant, pm_no, schedule_unit, lot_no, version, time_limit, min_width, 
                        max_width, _, max_pieces, _, 
                        paper_type, b_wgt, color,
                        min_sl_width, max_sl_width, sl_trim_size
                    ) = db.get_lot_param_roll_sl(lot_no=lot_no, version=version)
                    
                    roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_roll_sl_lot(
                        db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit, 
                        min_width, max_width, max_pieces, 
                        min_sl_width, max_sl_width, sl_trim_size,
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                if roll_results:
                    all_results.append(roll_results)
                    all_df_orders.append(roll_df_orders)

            if sheet_order_cnt > 0: 
                logging.info(f"쉬트지 오더 {sheet_order_cnt}건 처리 시작.")
                if plant == '3000':
                    sheet_results, sheet_df_orders, prod_seq_counter, group_order_no_counter = process_sheet_lot(
                        db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit, 
                        min_width, sheet_max_width, sheet_max_pieces, 
                        min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                elif plant == '5000':
                    ( 
                        plant, pm_no, schedule_unit, lot_no, version, time_limit, coating_yn, 
                        paper_type, b_wgt, color,
                        p_type, p_wgt, p_color, p_machine,
                        min_width, max_width, min_piece, max_piece, sheet_length_re, std_roll_cnt,
                        min_sc_width, max_sc_width, sheet_trim_size, 
                        min_cm_width, max_cm_width, max_sl_count, ww_trim_size, ww_trim_size_sheet
                    ) = db.get_lot_param_sheet_ca(lot_no=lot_no, version=version)

                    sheet_results, sheet_df_orders, prod_seq_counter, group_order_no_counter = process_sheet_lot_ca(
                        db, plant, pm_no, schedule_unit, lot_no, version, coating_yn,
                        paper_type, b_wgt, color, 
                        p_type, p_wgt, p_color, p_machine,
                        min_width, max_width, min_piece, max_piece, 
                        time_limit, sheet_length_re, std_roll_cnt,
                        min_sc_width, max_sc_width, sheet_trim_size, 
                        min_cm_width, max_cm_width, max_sl_count, ww_trim_size, ww_trim_size_sheet,
                        double_cutter='Y',
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                else:
                    ( 
                        plant, pm_no, schedule_unit, lot_no, version, time_limit, min_width, 
                        _, sheet_max_width, _, sheet_max_pieces, 
                        paper_type, b_wgt,
                        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
                    ) = db.get_target_lot_st(lot_no=lot_no)

                    sheet_results, sheet_df_orders, prod_seq_counter, group_order_no_counter = process_sheet_lot_st(
                        db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit,
                        min_width, sheet_max_width, sheet_max_pieces, 
                        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re,
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                
                if sheet_results:
                    all_results.append(sheet_results)
                    all_df_orders.append(sheet_df_orders)

            if all_results:
                # p_machine, p_type, p_wgt, p_color는 5000 공장에서만 사용되므로 조건부 전달
                _locals = locals()
                _p_machine = _locals.get('p_machine')
                _p_type = _locals.get('p_type')
                _p_wgt = _locals.get('p_wgt')
                _p_color = _locals.get('p_color')
                final_status = save_results(
                    db, lot_no, version, plant, pm_no, schedule_unit, max_width, paper_type, b_wgt, 
                    all_results, all_df_orders,
                    p_machine=_p_machine, p_type=_p_type, p_wgt=_p_wgt, p_color=_p_color
                )
                db.update_lot_status(lot_no=lot_no, version=version, status=final_status)
                status_desc = {0: "모든 오더 충족", 1: "일부 오더 부족", 2: "에러"}.get(final_status, "알 수 없음")
                logging.info(f"[상태 업데이트] Lot {lot_no} Version {version} -> status={final_status} ({status_desc})")
            else:
                logging.error(f"[에러] Lot {lot_no}에 대한 최적화 결과가 없습니다. 상태를 2(에러)로 변경합니다.")
                db.update_lot_status(lot_no=lot_no, version=version, status=2)
            
            logging.info(f"{'='*60}")
            logging.info(f"{'='*60}")
            logging.info(f"{'='*60}")
            time.sleep(10)

    except FileNotFoundError as e:        
        logging.error(f"[치명적 에러] 설정 파일을 찾을 수 없습니다: {e}")
    except KeyboardInterrupt:
        logging.info("\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        import traceback
        logging.error(f"\n[치명적 에러] 실행 중 예외 발생: {e}")
        logging.error(traceback.format_exc())
        if db and lot_no and version:
            db.update_lot_status(lot_no=lot_no, version=version, status=2)
    finally:
        if db:
            db.close_pool()
        logging.info("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()

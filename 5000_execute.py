"""
5000 천안공장 전용 execute 모듈.
- process_roll_lot_ca: 롤지 Lot 최적화 (천안)
- process_coating_roll_lot_ca: 코팅 롤지 Lot 최적화
- process_sheet_lot_ca: 쉬트지 Lot 최적화 (천안)
- main: 5000 공장 메인 루프
"""
import pandas as pd
from collections import Counter
import time
import logging
from optimize.roll_optimize import RollOptimize
from optimize.roll_optimize_ca import RollOptimizeCa
from optimize.sheet_optimize_ca import SheetOptimizeCa
from execute_common import (
    NUM_THREADS, init_db, setup_logging, save_results, generate_allocated_sheet_details
)


def process_sheet_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, coating_yn, 
        paper_type, b_wgt, color,
        p_type, p_wgt, p_color, p_machine,
        min_width, max_width, min_piece, max_piece, 
        time_limit, sheet_length_re, std_roll_cnt,
        min_sc_width, max_sc_width, sheet_trim_size, 
        min_cm_width, max_cm_width, max_sl_count, ww_trim_size, ww_trim_size_sheet,
        double_cutter,
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
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['가로', '세로', '등급', 'order_pattern']
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    df_orders['order_pattern'] = df_orders['order_pattern'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"50{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
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
    
    # 등급 + order_pattern 조합으로 그룹핑 (각 조합별로 별도 최적화)
    grouping_cols = ['등급', 'order_pattern']
    unique_groups = df_orders[grouping_cols].drop_duplicates()
    prod_seq_counter = start_prod_seq
    
    for _, grp_row in unique_groups.iterrows():
        grade = grp_row['등급']
        pattern = grp_row['order_pattern']
        logging.info(f"\n--- 등급 {grade}, pattern {pattern}에 대한 쉬트지(CA) 최적화 시작 ---")
        df_subset = df_orders[(df_orders['등급'] == grade) & (df_orders['order_pattern'] == pattern)].copy()
        
        if df_subset.empty:
            continue
        
        # 가로, 세로 기준 고유 규격 건수 체크
        unique_specs = df_subset[['가로', '세로']].drop_duplicates()
        unique_spec_count = len(unique_specs)
        logging.info(f"[규격 체크] 가로x세로 고유 규격 수: {unique_spec_count}건")
        logging.info(f"[규격 목록] {unique_specs.values.tolist()}")

        if unique_spec_count > 3:
            double_cutter = 'N'
        else:
            double_cutter = 'Y'

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
        db, plant, pm_no, schedule_unit, lot_no, version, time_limit, 
        paper_type, b_wgt, color, 
        p_type, p_wgt, p_color, p_machine,
        re_min_width, re_max_width, re_max_pieces, 
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
    group_cols = ['지폭', '롤길이', '등급', 'order_pattern']
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

    # 2. 지폭 그룹핑 (엔진 최적화용) - order_no 제외
    width_group_cols = ['지폭', '롤길이', '등급', 'order_pattern']
    df_width_groups = df_orders.groupby(width_group_cols).agg(
        total_qty=('주문수량', 'sum')
    ).reset_index()
    
    df_width_groups['width_group_no'] = [f'WG{i+1}' for i in range(len(df_width_groups))]
    
    df_orders = pd.merge(df_orders, df_width_groups[['지폭', '롤길이', '등급', 'width_group_no']], 
                         on=['지폭', '롤길이', '등급'], how='left')

    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "pattern_roll_details_for_db": [],
        "pattern_roll_cut_details_for_db": [],
        "fulfillment_summary": []
    }
    
    grouping_cols = ['롤길이', '등급', 'order_pattern']
    unique_groups = df_orders[grouping_cols].drop_duplicates()
    prod_seq_counter = start_prod_seq

    remaining_demands = df_orders.set_index('group_order_no')['주문수량'].to_dict()
    width_group_to_orders = df_orders.groupby('width_group_no')['group_order_no'].apply(list).to_dict()

    for _, row in unique_groups.iterrows():
        roll_length = row['롤길이']
        quality_grade = row['등급']
        order_pattern = row['order_pattern']
        logging.info(f"\n--- 롤길이 그룹 {roll_length}, 등급 {quality_grade}, order_pattern {order_pattern}에 대한 최적화 시작 ---")
        
        df_subset_engine = df_width_groups[
            (df_width_groups['롤길이'] == roll_length) & 
            (df_width_groups['등급'] == quality_grade) & 
            (df_width_groups['order_pattern'] == row['order_pattern'])
        ].copy()

        if df_subset_engine.empty:
            continue
            
        df_subset_engine['color'] = color
        df_subset_engine = df_subset_engine.rename(columns={'width_group_no': 'group_order_no', 'total_qty': '주문수량'})

        logging.info(f"--- 롤길이 그룹 {roll_length}, 등급 {quality_grade}, order_pattern {order_pattern}에 대한 주문 정보 (지폭 그룹핑) ---")
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
        
        allocated_pattern_details = []
        allocated_roll_details = []
        allocated_cut_details = []
        
        for entry in results['pattern_details_for_db']:
            wg_ids = entry['group_nos']
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
                    
                    for order_id in candidate_orders:
                        if remaining_demands.get(order_id, 0) > 0:
                            assigned_order = order_id
                            remaining_demands[order_id] -= 1
                            break
                    
                    if not assigned_order and candidate_orders:
                        assigned_order = candidate_orders[-1]
                    
                    run_assignment.append(assigned_order if assigned_order else '')
                allocated_runs.append(tuple(run_assignment))
            
            run_counts = Counter(allocated_runs)
            
            for order_combo, count in run_counts.items():
                prod_seq_counter += 1
                
                new_entry = entry.copy()
                new_entry['group_nos'] = list(order_combo)
                new_entry['count'] = count
                new_entry['prod_seq'] = prod_seq_counter
                allocated_pattern_details.append(new_entry)
                
                roll_seq_counter = 0
                for i, width in enumerate(widths):
                    if width <= 0: continue
                    roll_seq_counter += 1
                    group_no = list(order_combo)[i]
                    
                    allocated_roll_details.append({
                        'rollwidth': width,
                        'pattern_length': entry.get('pattern_length', 0),
                        'widths': [width] + [0]*7,
                        'roll_widths': ([0] * 7)[:7],
                        'group_nos': [group_no] + ['']*7,
                        'rs_gubuns': ['R'] + ['']*7,
                        'count': count,
                        'prod_seq': prod_seq_counter,
                        'roll_seq': roll_seq_counter,
                        'rs_gubun': 'R',
                        'sc_trim': 0,
                        'sl_trim': 0,
                        'p_lot': entry.get('p_lot'),
                        'diameter': entry.get('diameter'),
                        'core': entry.get('core'),
                        'color': entry.get('color'),
                        'luster': entry.get('luster')
                    })
                    
                    allocated_cut_details.append({
                        'prod_seq': prod_seq_counter,
                        'unit_no': prod_seq_counter,
                        'seq': roll_seq_counter,
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
        all_results["pattern_result"].append(results["pattern_result"])
        for detail in allocated_pattern_details:
            detail['max_width'] = int(re_max_width)
        all_results["pattern_details_for_db"].extend(allocated_pattern_details)
        all_results["pattern_roll_details_for_db"].extend(allocated_roll_details)
        all_results["pattern_roll_cut_details_for_db"].extend(allocated_cut_details)
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 롤지 최적화 결과가 없습니다.")
        return None, None, start_prod_seq, start_group_order_no

    production_counts = Counter()
    for detail in all_results["pattern_details_for_db"]:
        for group_no in detail['group_nos']:
            if group_no:
                production_counts[group_no] += detail['count']
    
    df_prod = pd.DataFrame.from_dict(production_counts, orient='index', columns=['생산롤수'])
    df_prod.index.name = 'group_order_no'
    
    df_summary = df_orders.set_index('group_order_no')[['지폭', '롤길이', '등급', '주문수량']].copy()
    df_summary = df_summary.join(df_prod).fillna(0)
    df_summary['과부족(롤)'] = df_summary['생산롤수'] - df_summary['주문수량']
    df_summary = df_summary.reset_index()
    
    pattern_sheet_details_for_db = generate_allocated_sheet_details(df_orders, all_results["pattern_roll_cut_details_for_db"], b_wgt)

    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": df_summary
    }

    logging.info("\n--- 롤지 최적화 성공. ---")
    return final_results, df_orders, prod_seq_counter, last_group_order_no


def process_coating_roll_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, time_limit, coating_yn, 
        paper_type, b_wgt, color,
        p_type, p_wgt, p_color, p_machine,
        min_width, max_width, max_pieces, 
        min_cm_width, max_cm_width, max_sl_count, ww_trim_size_sheet, ww_trim_size, 
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
            ww_trim_size_sheet=ww_trim_size_sheet,
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


def main():
    """5000 창원공장 메인 실행 함수"""
    db = None
    lot_no = None
    version = None

    try:
        db = init_db('5000')

        while True:
            (                     
                plant, pm_no, schedule_unit, lot_no, version, time_limit, paper_type, b_wgt, color, 
                min_width, roll_max_width, min_sc_width, max_sc_width, coating_yn, 
                sheet_trim_size, ww_trim_size,
                min_cm_width, max_cm_width, max_sl_count, p_type, p_wgt, ww_trim_size_sheet,
                sheet_order_cnt, roll_order_cnt
            ) = db.get_target_lot_ca()

            if not lot_no:
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
                ( 
                    plant, pm_no, schedule_unit, lot_no, version, time_limit, coating_yn, 
                    paper_type, b_wgt, color, 
                    p_type, p_wgt, p_color, p_machine,
                    min_width, max_width, min_pieces, max_pieces,
                    min_cm_width, max_cm_width, max_sl_count, ww_trim_size_sheet, ww_trim_size
                ) = db.get_lot_param_roll_ca(lot_no=lot_no, version=version)
                
                if coating_yn == 'Y':
                    roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_coating_roll_lot_ca(
                        db, plant, pm_no, schedule_unit, lot_no, version, time_limit, coating_yn, 
                        paper_type, b_wgt, color,
                        p_type, p_wgt, p_color, p_machine,
                        min_width, max_width, max_pieces,
                        min_cm_width, max_cm_width, max_sl_count, ww_trim_size_sheet, ww_trim_size,
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                else:
                    roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_roll_lot_ca(
                        db, plant, pm_no, schedule_unit, lot_no, version, time_limit, 
                        paper_type, b_wgt, color, 
                        p_type, p_wgt, p_color, p_machine,
                        min_width, max_width, max_pieces, 
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                if roll_results:
                    all_results.append(roll_results)
                    all_df_orders.append(roll_df_orders)

            if sheet_order_cnt > 0: 
                logging.info(f"쉬트지 오더 {sheet_order_cnt}건 처리 시작.")
                ( 
                    plant, pm_no, schedule_unit, lot_no, version, time_limit, coating_yn, 
                    paper_type, b_wgt, color,
                    p_type, p_wgt, p_color, p_machine,
                    min_width, max_width, min_piece, max_piece, sheet_length_re, std_roll_cnt,
                    min_sc_width, max_sc_width, sheet_trim_size, 
                    min_cm_width, max_cm_width, max_sl_count, ww_trim_size, ww_trim_size_sheet,
                    double_cutter
                ) = db.get_lot_param_sheet_ca(lot_no=lot_no, version=version)
                
                logging.info(f"[DEBUG] p_type={p_type}, p_wgt={p_wgt}, p_color={p_color}, p_machine={p_machine}")

                sheet_results, sheet_df_orders, prod_seq_counter, group_order_no_counter = process_sheet_lot_ca(
                    db, plant, pm_no, schedule_unit, lot_no, version, coating_yn,
                    paper_type, b_wgt, color, 
                    p_type, p_wgt, p_color, p_machine,
                    min_width, max_width, min_piece, max_piece, 
                    time_limit, sheet_length_re, std_roll_cnt,
                    min_sc_width, max_sc_width, sheet_trim_size, 
                    min_cm_width, max_cm_width, max_sl_count, ww_trim_size, ww_trim_size_sheet,
                    double_cutter,
                    start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                )
                
                if sheet_results:
                    all_results.append(sheet_results)
                    all_df_orders.append(sheet_df_orders)

            if all_results:
                _locals = locals()
                _p_machine = _locals.get('p_machine')
                _p_type = _locals.get('p_type')
                _p_wgt = _locals.get('p_wgt')
                _p_color = _locals.get('p_color')
                
                logging.info(f"[DEBUG] save_results 호출 전: _p_type={_p_type}, _p_wgt={_p_wgt}, _p_color={_p_color}, _p_machine={_p_machine}")
                
                final_status = save_results(
                    db, lot_no, version, plant, pm_no, schedule_unit, max_width if 'max_width' in dir() else roll_max_width, paper_type, b_wgt, 
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

import pandas as pd
import sys
import time
import configparser
import os
import logging
import argparse
import pprint
from optimize.roll_optimize import RollOptimize
from optimize.roll_sl_optimize import RollSLOptimize
from optimize.sheet_optimize import SheetOptimize
from optimize.sheet_optimize_var import SheetOptimizeVar
from optimize.sheet_optimize_ca import SheetOptimizeCa
from db.db_connector import Database

def process_roll_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, re_min_width, re_max_width, re_max_pieces, paper_type, b_wgt,
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

    group_cols = ['지폭', '롤길이', '등급', 'core', 'dia']
    for col in ['지폭', '롤길이']:
        df_orders[col] = pd.to_numeric(df_orders[col])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
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
    
    # 엔진 수행 그룹핑 기준 컬럼 설정
    grouping_cols = ['롤길이', 'core', 'dia']
    unique_groups = df_orders[grouping_cols].drop_duplicates()
    prod_seq_counter = start_prod_seq

    for _, row in unique_groups.iterrows():
        roll_length = row['롤길이']
        core = row['core']
        dia = row['dia']
        
        logging.info(f"\n--- 롤길이 그룹 {roll_length}, Core {core}, Dia {dia}에 대한 최적화 시작 ---")
        
        df_subset = df_orders[
            (df_orders['롤길이'] == roll_length) & 
            (df_orders['core'] == core) & 
            (df_orders['dia'] == dia)
        ].copy()

        if df_subset.empty:
            continue

        optimizer = RollOptimize(
            df_spec_pre=df_subset,
            max_width=int(re_max_width),
            min_width=int(re_min_width),

            max_pieces=int(re_max_pieces),
            lot_no=lot_no
        )
        results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)

        if "error" in results:
            logging.error(f"[에러] Lot {lot_no}, 롤길이 {roll_length}, Core {core}, Dia {dia} 최적화 실패: {results['error']}")
            continue
        
        prod_seq_counter = results.get('last_prod_seq', prod_seq_counter)

        logging.info(f"--- 롤길이 그룹 {roll_length}, Core {core}, Dia {dia} 최적화 성공 ---")
        all_results["pattern_result"].append(results["pattern_result"])
        all_results["pattern_details_for_db"].extend(results["pattern_details_for_db"])
        all_results["pattern_roll_details_for_db"].extend(results.get("pattern_roll_details_for_db", []))
        all_results["pattern_roll_cut_details_for_db"].extend(results.get("pattern_roll_cut_details_for_db", []))
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 롤지 최적화 결과가 없습니다.")
        return None, None, start_prod_seq, start_group_order_no

    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "fulfillment_summary": pd.concat(all_results["fulfillment_summary"], ignore_index=True)
    }

    logging.info("\n--- 롤지 최적화 성공. ---")
    return final_results, df_orders, prod_seq_counter, last_group_order_no


def process_roll_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, re_min_width, re_max_width, re_max_pieces, paper_type, b_wgt,
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

    group_cols = ['지폭', '롤길이', '등급', '오더번호']
    for col in ['지폭', '롤길이']:
        df_orders[col] = pd.to_numeric(df_orders[col])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
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
    
    unique_roll_lengths = df_orders['롤길이'].unique()
    prod_seq_counter = start_prod_seq

    for roll_length in unique_roll_lengths:
        logging.info(f"\n--- 롤길이 그룹 {roll_length}에 대한 최적화 시작 ---")
        df_subset = df_orders[df_orders['롤길이'] == roll_length].copy()

        if df_subset.empty:
            continue

        optimizer = RollOptimize(
            df_spec_pre=df_subset,
            max_width=int(re_max_width),
            min_width=int(re_min_width),
            max_pieces=int(re_max_pieces)
        )
        results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)

        if "error" in results:
            logging.error(f"[에러] Lot {lot_no}, 롤길이 {roll_length} 최적화 실패: {results['error']}")
            continue
        
        prod_seq_counter = results.get('last_prod_seq', prod_seq_counter)

        logging.info(f"--- 롤길이 그룹 {roll_length} 최적화 성공 ---")
        all_results["pattern_result"].append(results["pattern_result"])
        all_results["pattern_details_for_db"].extend(results["pattern_details_for_db"])
        all_results["pattern_roll_details_for_db"].extend(results.get("pattern_roll_details_for_db", []))
        all_results["pattern_roll_cut_details_for_db"].extend(results.get("pattern_roll_cut_details_for_db", []))
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 롤지 최적화 결과가 없습니다.")
        return None, None, start_prod_seq, start_group_order_no

    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "fulfillment_summary": pd.concat(all_results["fulfillment_summary"], ignore_index=True)
    }

    logging.info("\n--- 롤지 최적화 성공. ---")
    return final_results, df_orders, prod_seq_counter, last_group_order_no

def process_roll_sl_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
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
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['지폭', '롤길이', '등급', 'core', 'dia', 'luster', 'color', 'order_pattern']
    df_orders['지폭'] = pd.to_numeric(df_orders['지폭'])
    df_orders['롤길이'] = pd.to_numeric(df_orders['롤길이'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)

    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

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
    
    logging.info("롤-슬리터 최적화 성공.")
    return results, df_orders, prod_seq_counter, last_group_order_no

def process_sheet_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
        start_prod_seq=0, start_group_order_no=0
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
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
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['가로', '세로', '등급']
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')
    logging.info(f"--- Lot {df_groups.to_string()} 그룹마스터 정보 ---")

    logging.info("--- 쉬트지 최적화 시작 ---")
    optimizer = SheetOptimize(
        df_spec_pre=df_orders,
        max_width=int(re_max_width),
        min_width=int(re_min_width),
        max_pieces=int(re_max_pieces),
        b_wgt=float(b_wgt),
        sheet_roll_length=sheet_length_re,
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
        logging.error(f"[에러] Lot {lot_no} 쉬트지 최적화 실패: {error_msg}.")
        return None, None, start_prod_seq, start_group_order_no
    
    logging.info("쉬트지 최적화 성공.")
    return results, df_orders, prod_seq_counter, last_group_order_no

def process_sheet_lot_var(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
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
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['가로', '세로', '등급']
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

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
    
    logging.info("쉬트지(Var) 최적화 성공.")
    return results, df_orders, prod_seq_counter, last_group_order_no

def process_sheet_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re,
        start_prod_seq=0, start_group_order_no=0
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot (CA): {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    logging.info(f"min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, min_sheet_length_re={min_sheet_length_re}, max_sheet_length_re={max_sheet_length_re}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_sheet_orders_from_db_ca(paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 쉬트지(CA) 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
    df_orders['lot_no'] = lot_no
    df_orders['version'] = version

    group_cols = ['가로', '세로', '등급']
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    logging.info("--- 쉬트지(CA) 최적화 시작 ---")
    optimizer = SheetOptimizeCa(
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
        logging.error(f"[에러] Lot {lot_no} 쉬트지(CA) 최적화 실패: {error_msg}.")
        return None, None, start_prod_seq, start_group_order_no
    
    logging.info("쉬트지(CA) 최적화 성공.")
    return results, df_orders, prod_seq_counter, last_group_order_no

def save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, all_results, all_df_orders):
    """최적화 결과를 DB에 저장하고 CSV파일로 출력합니다."""
    if not all_results:
        logging.warning(f"Lot {lot_no}에 대해 저장할 결과가 없습니다.")
        return

    final_pattern_result = pd.concat([res["pattern_result"] for res in all_results], ignore_index=True)
    final_fulfillment_summary = pd.concat([res["fulfillment_summary"] for res in all_results], ignore_index=True)
    final_pattern_details_for_db = [item for res in all_results for item in res["pattern_details_for_db"]]
    final_pattern_roll_details_for_db = [item for res in all_results for item in res.get("pattern_roll_details_for_db", [])]
    final_pattern_roll_cut_details_for_db = [item for res in all_results for item in res.get("pattern_roll_cut_details_for_db", [])]
    
    final_df_orders = pd.concat(all_df_orders, ignore_index=True)

    logging.info("최적화 결과 (패턴별 생산량):")
    logging.info("\n" + final_pattern_result.to_string())
    logging.info("\n# ================= 주문 충족 현황 ================== #\n")
    logging.info("\n" + final_fulfillment_summary.to_string())
    logging.info("\n")
    logging.info("최적화 성공. 이제 결과를 DB에 저장합니다.")

    connection = None
    try:
        connection = db.pool.acquire()

        if not final_df_orders.empty:
            db.insert_order_group(
                connection, lot_no, version, plant, pm_no, schedule_unit, final_df_orders
            )

        logging.info("\n\n# ================= 패턴 상세 정보 (final_pattern_details_for_db) ================== #\n")
        logging.info(f"롤 재단 상세 정보 개수: {final_pattern_details_for_db}")
        db.insert_pattern_sequence(
            connection, lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
            paper_type, b_wgt, final_pattern_details_for_db
        )

        if final_pattern_roll_details_for_db:
            logging.info("\n\n# ================= 패턴롤 정보 (final_pattern_roll_details_for_db) ================== #\n")
            logging.info(f"롤 재단 상세 정보 개수: {final_pattern_roll_details_for_db}")
            db.insert_roll_sequence(
                connection, lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
                paper_type, b_wgt, final_pattern_roll_details_for_db
            )

        if final_pattern_roll_cut_details_for_db:
            logging.info("\n\n# ================= 롤 cut 재단 상세 정보 (final_pattern_roll_cut_details_for_db) ================== #\n")
            logging.info(f"롤 재단 상세 정보 개수: {final_pattern_roll_cut_details_for_db}")
            db.insert_cut_sequence(
                connection, lot_no, version, plant, pm_no, schedule_unit, 
                paper_type, b_wgt, final_pattern_roll_cut_details_for_db
            )

        logging.info("\n\n# ================= 쉬트 재단 상세 정보 ================== #\n")
        db.insert_sheet_sequence(
            connection, lot_no, version, plant, schedule_unit
        )

        connection.commit()
        logging.info("DB 트랜잭션이 성공적으로 커밋되었습니다.")

        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{lot_no}_{version}.csv"
        output_path = os.path.join(output_dir, output_filename)
        final_pattern_result.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"\n[성공] 요약 결과가 다음 파일에 저장되었습니다: {output_path}")
        db.update_lot_status(lot_no=lot_no, version=version, status=0)

    except Exception as e:
        logging.error(f"[에러] 데이터 저장 중 오류 발생: {e}")
        if connection:
            connection.rollback()
            logging.info("DB 트랜잭션이 롤백되었습니다.")
            db.pool.release(connection)
            connection = None
        
        db.update_lot_status(lot_no=lot_no, version=version, status=99)

    finally:
        if connection:
            db.pool.release(connection)
        logging.info(f"\n{'='*60}")
        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Lot: {lot_no} 처리 완료")
        logging.info(f"{'='*60}")

def setup_logging(lot_no, version):
    """로그 설정을 초기화합니다."""
    log_dir = 'results'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{lot_no}_{version}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main():
    """메인 실행 함수"""
    db = None
    lot_no = None
    version = None
    try:
        config = configparser.ConfigParser()
        config_path = os.path.join('conf', 'config.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 파일을 찾을 수 없습니다.")
        config.read(config_path, encoding='utf-8')
        db_config = config['database']
        
        db = Database(user=db_config['user'], password=db_config['password'], dsn=db_config['dsn'])

        while True:
            ( 
                plant, pm_no, schedule_unit, lot_no, version, min_width, 
                max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
                paper_type, b_wgt,
                min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
                sheet_order_cnt, roll_order_cnt
            ) = db.get_target_lot()

            if not lot_no:
                print("처리할 Lot이 없습니다. 10초 후 다시 시도합니다.")
                time.sleep(10)
                continue

            setup_logging(lot_no, version)
            
            db.delete_optimization_results(lot_no, version)
            db.update_lot_status(lot_no=lot_no, version=version, status=1)

            prod_seq_counter = 0
            group_order_no_counter = 0
            all_results = []
            all_df_orders = []

            if roll_order_cnt > 0:
                logging.info(f"롤지 오더 {roll_order_cnt}건 처리 시작.")
                if plant == '3000':
                    roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_roll_lot(
                        db, plant, pm_no, schedule_unit, lot_no, version, 
                        min_width, max_width, max_pieces, paper_type, b_wgt,
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                else:
                    ( 
                        plant_sl, pm_no_sl, schedule_unit_sl, lot_no_sl, version_sl, min_width_sl, 
                        max_width_sl, _, max_pieces_sl, _, 
                        paper_type_sl, b_wgt_sl,
                        min_sl_width, max_sl_width, sl_trim_size
                    ) = db.get_target_lot_sl(lot_no=lot_no)
                    
                    roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_roll_sl_lot(
                        db, plant_sl, pm_no_sl, schedule_unit_sl, lot_no_sl, version_sl, 
                        min_width_sl, max_width_sl, max_pieces_sl, paper_type_sl, b_wgt_sl,
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
                        db, plant, pm_no, schedule_unit, lot_no, version, 
                        min_width, sheet_max_width, sheet_max_pieces, paper_type, b_wgt,
                        min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                elif plant == '5000':
                    ( 
                        plant_ca, pm_no_ca, schedule_unit_ca, lot_no_ca, version_ca, min_width_ca, 
                        _, sheet_max_width_ca, _, sheet_max_pieces_ca, 
                        paper_type_ca, b_wgt_ca,
                        min_sc_width_ca, max_sc_width_ca, sheet_trim_size_ca, min_sheet_length_re_ca, max_sheet_length_re_ca
                    ) = db.get_target_lot_ca(lot_no=lot_no)

                    sheet_results, sheet_df_orders, prod_seq_counter, group_order_no_counter = process_sheet_lot_ca(
                        db, plant_ca, pm_no_ca, schedule_unit_ca, lot_no_ca, version_ca, 
                        min_width_ca, sheet_max_width_ca, sheet_max_pieces_ca, paper_type_ca, b_wgt_ca,
                        min_sc_width_ca, max_sc_width_ca, sheet_trim_size_ca, min_sheet_length_re_ca, max_sheet_length_re_ca,
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                else:
                    ( 
                        plant_var, pm_no_var, schedule_unit_var, lot_no_var, version_var, min_width_var, 
                        _, sheet_max_width_var, _, sheet_max_pieces_var, 
                        paper_type_var, b_wgt_var,
                        min_sc_width_var, max_sc_width_var, sheet_trim_size_var, min_sheet_length_re_var, max_sheet_length_re_var
                    ) = db.get_target_lot_var(lot_no=lot_no)

                    sheet_results, sheet_df_orders, prod_seq_counter, group_order_no_counter = process_sheet_lot_var(
                        db, plant_var, pm_no_var, schedule_unit_var, lot_no_var, version_var, 
                        min_width_var, sheet_max_width_var, sheet_max_pieces_var, paper_type_var, b_wgt_var,
                        min_sc_width_var, max_sc_width_var, sheet_trim_size_var, min_sheet_length_re_var, max_sheet_length_re_var,
                        start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                    )
                
                if sheet_results:
                    all_results.append(sheet_results)
                    all_df_orders.append(sheet_df_orders)

            if all_results:
                save_results(db, lot_no, version, plant, pm_no, schedule_unit, max_width, paper_type, b_wgt, all_results, all_df_orders)
            else:
                logging.error(f"[에러] Lot {lot_no}에 대한 최적화 결과가 없습니다. 상태를 99(에러)로 변경합니다.")
                db.update_lot_status(lot_no=lot_no, version=version, status=99)
            
            logging.info(f"{'='*60}")
            logging.info(f"{'='*60}")
            logging.info(f"{'='*60}")
            time.sleep(100)

    except FileNotFoundError as e:        
        logging.error(f"[치명적 에러] 설정 파일을 찾을 수 없습니다: {e}")
    except KeyboardInterrupt:
        logging.info("\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        import traceback
        logging.error(f"\n[치명적 에러] 실행 중 예외 발생: {e}")
        logging.error(traceback.format_exc())
        if db and lot_no and version:
            db.update_lot_status(lot_no=lot_no, version=version, status=99)
    finally:
        if db:
            db.close_pool()
        logging.info("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()

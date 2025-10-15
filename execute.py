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
        db, plant, pm_no, schedule_unit, lot_no, version, re_min_width, re_max_width, re_max_pieces, paper_type, b_wgt
):
    """롤지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Roll Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    logging.info(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_roll_orders_from_db(paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성
    group_cols = ['지폭', '롤길이', '등급']
    for col in ['지폭', '롤길이']:
        df_orders[col] = pd.to_numeric(df_orders[col])
    df_orders['등급'] = df_orders['등급'].astype(str)
    # group_cols로 그룹화하고 각 그룹의 첫 번째 '오더번호'를 대표로 선택합니다.
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first'),
        plant=('plant', 'first'),
        pm_no=('pm_no', 'first'),
        schedule_unit=('schedule_unit', 'first'),
        lot_no=('lot_no', 'first'),
        version=('version', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    logging.info(f"--- Lot {lot_no} 원본 주문 정보 (그룹오더 포함) ---")
    logging.info(df_orders.to_string())
    logging.info("\n")

    # --- 롤길이별 최적화 실행 ---
    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "pattern_roll_details_for_db": [],
        "fulfillment_summary": []
    }
    
    unique_roll_lengths = df_orders['롤길이'].unique()
    prod_seq_counter = 0  # prod_seq 카운터 초기화

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
        # 카운터를 전달하여 최적화 실행
        results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)

        if "error" in results:
            logging.error(f"[에러] Lot {lot_no}, 롤길이 {roll_length} 최적화 실패: {results['error']}")
            continue
        
        # 다음 루프를 위해 마지막 prod_seq 값으로 카운터 업데이트
        prod_seq_counter = results.get('last_prod_seq', prod_seq_counter)

        logging.info(f"--- 롤길이 그룹 {roll_length} 최적화 성공 ---")
        all_results["pattern_result"].append(results["pattern_result"])
        all_results["pattern_details_for_db"].extend(results["pattern_details_for_db"])
        all_results["pattern_roll_details_for_db"].extend(results.get("pattern_roll_details_for_db", []))
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no}에 대한 최적화 결과가 없습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    # 모든 롤길이 그룹의 결과 취합
    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "fulfillment_summary": pd.concat(all_results["fulfillment_summary"], ignore_index=True)
    }

    logging.info("\n--- 전체 최적화 성공. 최종 결과를 처리합니다. ---")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, final_results, df_orders)

def process_roll_sl_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sl_width, max_sl_width, sl_trim_size
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}, min_sc_width={min_sl_width}, max_sc_width={max_sl_width}")
    logging.info(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_roll_sl_orders_from_db(paper_prod_seq=lot_no)
    logging.info(f"--- Lot {lot_no} 원본 주문 정보 ---")

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성 (쉬트지 기준)
    group_cols = ['지폭', '롤길이', '등급'] # 쉬트지는 '가로'(width)가 중요
    df_orders['지폭'] = pd.to_numeric(df_orders['지폭'])
    df_orders['롤길이'] = pd.to_numeric(df_orders['롤길이'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    # group_cols로 그룹화하고 각 그룹의 첫 번째 '오더번호'를 대표로 선택합니다.
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first'),
        plant=('plant', 'first'),
        pm_no=('pm_no', 'first'),
        schedule_unit=('schedule_unit', 'first'),
        lot_no=('lot_no', 'first'),
        version=('version', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]

    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    # 최적화 실행
    logging.info("--- 쉬트지 최적화 시작 ---")
    # b_wgt, 롤길이(6330), 트림(20) 등 쉬트지 사양 전달
    optimizer = RollSLOptimize(
        df_spec_pre=df_orders,
        max_width=int(re_max_width),
        min_width=int(re_min_width),
        max_pieces=int(re_max_pieces),
        b_wgt=float(b_wgt),
        sl_trim=sl_trim_size,
        min_sl_width=min_sl_width,
        max_sl_width=max_sl_width
    )
    try:
        results = optimizer.run_optimize()
        logging.info("--- Optimizer results ---\n")
        # logging.info(pprint.pformat(results))
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        logging.error(f"[에러] Lot {lot_no} 최적화 실패: {error_msg}. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return
    
    logging.info("최적화 성공. 결과를 처리합니다.")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results, df_orders)

def process_sheet_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}, min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, sheet_length_re={sheet_length_re}")
    logging.info(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_sheet_orders_from_db(paper_prod_seq=lot_no)
    logging.info(f"--- Lot {lot_no} 원본 주문 정보 ---")

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성 (쉬트지 기준)
    group_cols = ['가로', '세로', '등급'] # 쉬트지는 '가로'(width)가 중요
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    # df_groups = df_orders[group_cols].drop_duplicates().sort_values(by=group_cols).reset_index(drop=True)# 그룹핑하는 로직.

    
    # group_cols로 그룹화하고 각 그룹의 첫 번째 '오더번호'를 대표로 선택합니다.
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first'),
        plant=('plant', 'first'),
        pm_no=('pm_no', 'first'),
        schedule_unit=('schedule_unit', 'first'),
        lot_no=('lot_no', 'first'),
        version=('version', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')
    logging.info(f"--- Lot {df_groups.to_string()} 그룹마스터 정보 ---")

    # 최적화 실행
    logging.info("--- 쉬트지 최적화 시작 ---")
    # b_wgt, 롤길이(6330), 트림(20) 등 쉬트지 사양 전달
    optimizer = SheetOptimize(
        df_spec_pre=df_orders,
        max_width=int(re_max_width),
        min_width=int(re_min_width),
        max_pieces=int(re_max_pieces),
        b_wgt=float(b_wgt),
        sheet_roll_length=sheet_length_re, # 하드코딩 6330, 14740
        sheet_trim=sheet_trim_size,
        min_sc_width=min_sc_width,
        max_sc_width=max_sc_width
    )
    try:
        results = optimizer.run_optimize()
        logging.info("--- Optimizer results ---\n")
        logging.info(pprint.pformat(results))
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        logging.error(f"[에러] Lot {lot_no} 최적화 실패: {error_msg}. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return
    
    logging.info("최적화 성공. 결과를 처리합니다.")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results, df_orders)


def process_sheet_lot_var(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    logging.info(f"min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, min_sheet_length_re={min_sheet_length_re}, max_sheet_length_re={max_sheet_length_re}")
    logging.info(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_sheet_orders_from_db_var(paper_prod_seq=lot_no)
    logging.info(f"--- Lot {lot_no} 원본 주문 정보 ---")

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성 (쉬트지 기준)
    group_cols = ['가로', '세로', '등급'] # 쉬트지는 '가로'(width)가 중요
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    # group_cols로 그룹화하고 각 그룹의 첫 번째 '오더번호'를 대표로 선택합니다.
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first'),
        plant=('plant', 'first'),
        pm_no=('pm_no', 'first'),
        schedule_unit=('schedule_unit', 'first'),
        lot_no=('lot_no', 'first'),
        version=('version', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    # 최적화 실행
    logging.info("--- 쉬트지 최적화 시작 ---")
    # b_wgt, 롤길이(6330), 트림(20) 등 쉬트지 사양 전달
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
        results = optimizer.run_optimize()
        logging.info("--- Optimizer results ---\n")
        # logging.info(pprint.pformat(results))
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        logging.error(f"[에러] Lot {lot_no} 최적화 실패: {error_msg}. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return
    
    logging.info("최적화 성공. 결과를 처리합니다.")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results, df_orders)


def process_sheet_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    logging.info(f"min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, min_sheet_length_re={min_sheet_length_re}, max_sheet_length_re={max_sheet_length_re}")
    logging.info(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_sheet_orders_from_db_ca(paper_prod_seq=lot_no)
    logging.info(f"--- Lot {lot_no} 원본 주문 정보 ---")

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성 (쉬트지 기준)
    group_cols = ['가로', '세로', '등급'] # 쉬트지는 '가로'(width)가 중요
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    # df_groups = df_orders[group_cols].drop_duplicates().sort_values(by=group_cols).reset_index(drop=True) # 그룹핑하는 로직.
    
    # group_cols로 그룹화하고 각 그룹의 첫 번째 '오더번호'를 대표로 선택합니다.
    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('오더번호', 'first'),
        plant=('plant', 'first'),
        pm_no=('pm_no', 'first'),
        schedule_unit=('schedule_unit', 'first'),
        lot_no=('lot_no', 'first'),
        version=('version', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    # 최적화 실행
    logging.info("--- 쉬트지 최적화 시작 ---")
    # b_wgt, 롤길이(6330), 트림(20) 등 쉬트지 사양 전달
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
        results = optimizer.run_optimize()
        logging.info("--- Optimizer results ---\n")
        logging.info(pprint.pformat(results))
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        logging.error(f"[에러] Lot {lot_no} 최적화 실패: {error_msg}. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return
    
    logging.info("최적화 성공. 결과를 처리합니다.")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results, df_orders)

def save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results, df_orders=None):
    """최적화 결과를 DB에 저장하고 CSV파일로 출력합니다."""
    logging.info("최적화 결과 (패턴별 생산량):")
    logging.info("\n" + results["pattern_result"].to_string())
    logging.info("\n# ================= 주문 충족 현황 ================== #\n")
    logging.info("\n" + results["fulfillment_summary"].to_string())
    logging.info("\n")
    logging.info("최적화 성공. 이제 결과를 DB에 저장합니다.")

    connection = None
    try:
        # 트랜잭션 시작
        connection = db.pool.acquire()

        # 1. 그룹 오더 정보 저장
        if df_orders is not None and not df_orders.empty:
            db.insert_order_group(
                connection, lot_no, version, plant, pm_no, schedule_unit, df_orders
            )

        # 2. DB에 패턴 저장
        db.insert_pattern_sequence(
            connection, lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
            paper_type, b_wgt, results['pattern_details_for_db']
        )

        # 3. 롤 상세 정보 저장
        if 'pattern_roll_details_for_db' in results and results['pattern_roll_details_for_db']:
            logging.info("\n\n# ================= 롤 상세 정보 (pattern_roll_details_for_db) ================== #\n")
            db.insert_roll_sequence(
                connection, lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
                paper_type, b_wgt, results['pattern_roll_details_for_db']
            )

        # 4. 롤 재단 상세 정보 저장
        if 'pattern_roll_cut_details_for_db' in results and results['pattern_roll_cut_details_for_db']:
            logging.info("\n\n# ================= 롤 재단 상세 정보 (pattern_roll_cut_details_for_db) ================== #\n")
            db.insert_cut_sequence(
                connection, lot_no, version, plant, pm_no, schedule_unit, 
                paper_type, b_wgt, results['pattern_roll_cut_details_for_db']
            )

        # 5. 쉬트 재단 상세 정보 저장        
        logging.info("\n\n# ================= 쉬트 재단 상세 정보 ================== #\n")
        db.insert_sheet_sequence(
            connection, lot_no, version, plant, schedule_unit
        )

        # 모든 DB 저장이 성공한 경우 커밋
        connection.commit()
        logging.info("DB 트랜잭션이 성공적으로 커밋되었습니다.")

        # CSV 파일로 결과 저장 및 상태 업데이트
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{lot_no}_{version}.csv"
        output_path = os.path.join(output_dir, output_filename)
        results['pattern_result'].to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"\n[성공] 요약 결과가 다음 파일에 저장되었습니다: {output_path}")
        db.update_lot_status(lot_no=lot_no, version=version, status=0)

    except Exception as e:
        logging.error(f"[에러] 데이터 저장 중 오류 발생: {e}")
        if connection:
            connection.rollback()
            logging.info("DB 트랜잭션이 롤백되었습니다.")
            # 교착 상태를 방지하기 위해 update_lot_status 호출 전에 커넥션을 명시적으로 해제합니다.
            db.pool.release(connection)
            connection = None  # finally 블록에서 다시 해제하지 않도록 None으로 설정
        
        # 이제 update_lot_status는 풀에서 새 커넥션을 얻을 수 있습니다.
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

    # 루트 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler() # 콘솔 출력
        ]
    )

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="롤지 또는 쉬트지 최적화를 실행합니다.")
    parser.add_argument("--order-type", required=True, choices=['roll', 'roll_sl', 'sheet', 'sheet_var', 'sheet_ca'], help="오더 유형 ('roll' 또는 'sheet')")
    args = parser.parse_args()

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

        ( 
            plant, pm_no, schedule_unit, lot_no, version, min_width, 
            max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
            paper_type, b_wgt,
            min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re
        ) = db.get_target_lot()

        if not lot_no:
            print("처리할 Lot이 없습니다.") # 이 부분은 로거 설정 전이므로 print 유지
            return

        # 로깅 설정
        setup_logging(lot_no, version)

        if args.order_type == 'roll':
            process_roll_lot(
                db, plant, pm_no, schedule_unit, lot_no, version, 
                min_width, max_width, max_pieces, paper_type, b_wgt
            )
        elif args.order_type == 'roll_sl':
            ( 
                plant, pm_no, schedule_unit, lot_no, version, min_width, 
                max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
                paper_type, b_wgt,
                min_sl_width, max_sl_width, sl_trim_size
            ) = db.get_target_lot_sl()

            process_roll_sl_lot(
                db, plant, pm_no, schedule_unit, lot_no, version, 
                min_width, max_width, max_pieces, paper_type, b_wgt,
                min_sl_width, max_sl_width, sl_trim_size
            )
        elif args.order_type == 'sheet':
            process_sheet_lot(
                db, plant, pm_no, schedule_unit, lot_no, version, 
                min_width, sheet_max_width, sheet_max_pieces, paper_type, b_wgt,
                min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re
            )
        elif args.order_type == 'sheet_var':
            # 데몬 방식 대신, get_target_lot()을 한 번만 호출하여 테스트합니다.
            ( 
                plant, pm_no, schedule_unit, lot_no, version, min_width, 
                max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
                paper_type, b_wgt,
                min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
            ) = db.get_target_lot_var()

            process_sheet_lot_var(
                db, plant, pm_no, schedule_unit, lot_no, version, 
                min_width, sheet_max_width, sheet_max_pieces, paper_type, b_wgt,
                min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
            )
        elif args.order_type == 'sheet_ca':
            # 데몬 방식 대신, get_target_lot()을 한 번만 호출하여 테스트합니다.
            ( 
                plant, pm_no, schedule_unit, lot_no, version, min_width, 
                max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
                paper_type, b_wgt,
                min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
            ) = db.get_target_lot_ca()

            process_sheet_lot_ca(
                db, plant, pm_no, schedule_unit, lot_no, version, 
                min_width, sheet_max_width, sheet_max_pieces, paper_type, b_wgt,
                min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
            )

    except FileNotFoundError as e:
        logging.error(f"[치명적 에러] 설정 파일을 찾을 수 없습니다: {e}")
    except KeyboardInterrupt:
        logging.info("\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        logging.error(f"\n[치명적 에러] 실행 중 예외 발생: {e}")
        if lot_no and version:
            db.update_lot_status(lot_no=lot_no, version=version, status=99)
    finally:
        if db:
            db.close_pool()
        logging.info("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()

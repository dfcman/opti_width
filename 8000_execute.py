"""
8000 신탄진공장 전용 execute 모듈.
- process_roll_lot_st: 롤-슬리터 Lot 최적화
- process_sheet_lot_var: 쉬트지 Lot 최적화 (가변길이)
- main: 8000 공장 메인 루프
"""
import pandas as pd
import time
import logging
from optimize.roll_optimize_st import RollOptimizeSt
from optimize.sheet_optimize_st import SheetOptimizeSt
from execute_common import (
    NUM_THREADS, init_db, setup_logging, save_results, generate_allocated_sheet_details
)


def group_compatible_roll_lengths(roll_lengths):
    """
    호환 가능한 롤길이끼리 그룹핑합니다.
    
    규칙:
    1. 100 단위 절사값이 같으면 같은 그룹 (예: 14000, 14080 → 둘 다 14000으로 절사)
    2. 절사값이 배수 관계이면 같은 그룹 (예: 6000, 12000 → 12000/6000=2)
    
    Args:
        roll_lengths: 고유 롤길이 배열
        
    Returns:
        dict: {롤길이: 그룹ID} 매핑
    """
    lengths = sorted(set(int(l) for l in roll_lengths))
    if not lengths:
        return {}
    
    # 1단계: 100 단위 절사값 계산
    truncated = {l: (l // 100) * 100 for l in lengths}
    
    # 2단계: Union-Find로 그룹 병합
    parent = {l: l for l in lengths}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # 큰 값을 루트로 (그룹 대표 = 최대 롤길이)
            if ra < rb:
                parent[ra] = rb
            else:
                parent[rb] = ra
    
    # 2-1: 절사값이 같은 롤길이 병합
    by_truncated = {}
    for l in lengths:
        t = truncated[l]
        if t not in by_truncated:
            by_truncated[t] = []
        by_truncated[t].append(l)
    
    for group in by_truncated.values():
        for i in range(1, len(group)):
            union(group[0], group[i])
    
    # 2-2: 절사값이 배수 관계인 그룹 병합
    truncated_keys = sorted(by_truncated.keys())
    for i in range(len(truncated_keys)):
        for j in range(i + 1, len(truncated_keys)):
            small, big = truncated_keys[i], truncated_keys[j]
            if small > 0 and big % small == 0:
                # 각 그룹의 대표 아이템으로 union
                union(by_truncated[small][0], by_truncated[big][0])
    
    # 결과: {롤길이: 그룹ID}
    result = {l: find(l) for l in lengths}
    
    # 로깅
    groups_summary = {}
    for l, gid in result.items():
        if gid not in groups_summary:
            groups_summary[gid] = []
        groups_summary[gid].append(l)
    for gid, members in groups_summary.items():
        if len(members) > 1:
            logging.info(f"[롤길이 그룹핑] {members} → 그룹 (std_length={max(members)})")
    
    return result

def process_roll_lot_st(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        paper_type, b_wgt, color, time_limit,
        re_min_width, re_max_width, re_max_pieces, 
        min_sl_width, max_sl_width, sl_trim_size,
        start_prod_seq=0, start_group_order_no=0
):
    """롤-슬리터 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Roll-SL Lot: {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}, min_sl_width={min_sl_width}, max_sl_width={max_sl_width}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_roll_orders_from_db_st(paper_prod_seq=lot_no)

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
        대표오더번호=('order_no', 'first')
    ).reset_index()
    df_groups = df_groups.sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"80{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)

    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    df_orders['color'] = color

    # --- 호환 롤길이 그룹핑 (배수/유사 길이) ---
    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "pattern_roll_details_for_db": [],
        "pattern_roll_cut_details_for_db": [],
        "fulfillment_summary": []
    }

    # 호환 롤길이 그룹 생성
    roll_length_groups = group_compatible_roll_lengths(df_orders['롤길이'].unique())
    df_orders['롤길이그룹'] = df_orders['롤길이'].map(roll_length_groups)
    
    unique_group_ids = df_orders['롤길이그룹'].unique()
    prod_seq_counter = start_prod_seq

    for group_id in sorted(unique_group_ids):
        group_roll_lengths = df_orders[df_orders['롤길이그룹'] == group_id]['롤길이'].unique()
        group_std_length = int(max(group_roll_lengths))
        logging.info(f"\n--- 롤길이 그룹 {sorted(group_roll_lengths)} (std_length={group_std_length})에 대한 최적화 시작 ---")

        df_subset = df_orders[df_orders['롤길이그룹'] == group_id].copy()
        df_subset['std_length'] = group_std_length

        if df_subset.empty:
            continue
    
        optimizer = RollOptimizeSt(
            df_spec_pre=df_subset,
            max_width=int(re_max_width),
            min_width=int(re_min_width),
            max_pieces=int(re_max_pieces),
            b_wgt=float(b_wgt),
            ww_trim_size=sl_trim_size,
            min_sl_width=min_sl_width,
            max_sl_width=max_sl_width,
            lot_no=lot_no
        )
        try:
            results = optimizer.run_optimize(start_prod_seq=prod_seq_counter)
            prod_seq_counter = results.get('last_prod_seq', prod_seq_counter)
        except Exception as e:
            import traceback
            logging.error(f"[에러] 롤길이 그룹 {sorted(group_roll_lengths)} 최적화 중 예외 발생")
            logging.error(traceback.format_exc())
            continue

        if not results or "error" in results:
            error_msg = results['error'] if results and 'error' in results else "No solution found"
            logging.error(f"[에러] Lot {lot_no}, 롤길이 그룹 {sorted(group_roll_lengths)} 최적화 실패: {error_msg}")
            continue
    
        if results and "pattern_details_for_db" in results:
            for detail in results["pattern_details_for_db"]:
                detail['max_width'] = int(re_max_width)

        all_results["pattern_result"].append(results["pattern_result"])
        all_results["pattern_details_for_db"].extend(results["pattern_details_for_db"])
        all_results["pattern_roll_details_for_db"].extend(results.get("pattern_roll_details_for_db", []))
        all_results["pattern_roll_cut_details_for_db"].extend(results.get("pattern_roll_cut_details_for_db", []))
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

        logging.info(f"--- 롤길이 그룹 {sorted(group_roll_lengths)} 최적화 성공 ---")

    if not all_results["pattern_details_for_db"]:
        logging.error(f"[에러] Lot {lot_no} 롤-슬리터 최적화 결과가 없습니다 (모든 그룹 실패).")
        return None, None, start_prod_seq, start_group_order_no
    
    # --- [New] Sheet Sequence Data Generation for Roll-SL Orders ---
    pattern_sheet_details_for_db = generate_allocated_sheet_details(df_orders, all_results["pattern_roll_cut_details_for_db"], b_wgt)

    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True) if all_results["pattern_result"] else pd.DataFrame(),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": pd.concat(all_results["fulfillment_summary"], ignore_index=True) if all_results["fulfillment_summary"] else pd.DataFrame()
    }

    logging.info("롤-슬리터 최적화 성공 (전체 그룹 완료).")
    return final_results, df_orders, prod_seq_counter, last_group_order_no


def process_sheet_lot_st(
        db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit,
        re_min_width, re_max_width, re_max_pieces, 
        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re,
        start_prod_seq=0, start_group_order_no=0
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리하고 결과를 반환합니다."""
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot (ST): {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    logging.info(f"min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, min_sheet_length_re={min_sheet_length_re}, max_sheet_length_re={max_sheet_length_re}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    raw_orders = db.get_sheet_orders_from_db_st(paper_prod_seq=lot_no)

    if not raw_orders:
        logging.error(f"[에러] Lot {lot_no}의 쉬트지(ST) 오더를 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    df_orders = pd.DataFrame(raw_orders)
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
    df_groups['group_order_no'] = [f"80{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)
    
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    df_orders['color'] = color
    logging.info("--- 쉬트지(ST) 최적화 시작 ---")
    optimizer = SheetOptimizeSt(
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
        logging.error(f"[에러] Lot {lot_no} 쉬트지(ST) 최적화 실패: {error_msg}.")
        return None, None, start_prod_seq, start_group_order_no
    
    if results and "pattern_details_for_db" in results:
        for detail in results["pattern_details_for_db"]:
            detail['max_width'] = int(re_max_width)

    # --- [New] Sheet Sequence Allocation & Data Generation ---
    pattern_sheet_details_for_db = generate_allocated_sheet_details(df_orders, results["pattern_roll_cut_details_for_db"], b_wgt)

    results["pattern_sheet_details_for_db"] = pattern_sheet_details_for_db
    logging.info("쉬트지(Var) 최적화 성공.")
    return results, df_orders, prod_seq_counter, last_group_order_no


def main():
    """8000 세종공장 메인 실행 함수"""
    db = None
    lot_no = None
    version = None

    try:
        db = init_db('8000')

        while True:
            ( 
                module, plant, pm_no, schedule_unit, lot_no, version, 
                sheet_order_cnt, roll_order_cnt
            ) = db.get_target_lot_st()

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
                    module,
                    plant,
                    pm_no,
                    schedule_unit,
                    lot_no,
                    version,
                    version_id,
                    time_limit,
                    paper_type, b_wgt, color,
                    sheet_subject_order,
                    sheet_subject_width,
                    sheet_subject_ptn_cnt,
                    sheet_subject_balance,
                    sheet_subject_roll_cnt,
                    min_width,
                    max_width,
                    min_re_count,
                    max_re_count,
                    sheet_min_re_pok_cnt,
                    sheet_max_re_pok_cnt,
                    min_length,
                    max_length,
                    length_version_cnt,
                    rs_mix_flag,
                    length_similar,
                    allow_multiple,
                    allow_multiple_width,
                    yeopok_yn,
                    rs_mix_maxpok,
                    sl_min_width,
                    sl_max_width,
                    sl_min_pok_cnt,
                    sl_max_pok_cnt,
                    sl_trim_size,
                    sl_increase_trim,
                    sl_max_trim,
                    sl_mix_yn,
                    sl_sroll_mix_yn,
                    sl_sroll_width,
                    sl_sroll_widthmin,
                    sl_sroll_widthmax,
                    sl_single_yn,
                    sl_single_trim_yn,
                    sc_unwinder_min,
                    sc_unwinder_max,
                    sc_min_pok_cnt,
                    sc_max_pok_cnt,
                    max_unit_weight,
                    sheet_trim_size,
                    sc_increase_trim,
                    sc_max_trim,
                    knife_min,
                    knife_max,
                    mix_type
                ) = db.get_lot_param_roll_st(lot_no=lot_no, version=version)
                
                roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_roll_lot_st(
                    db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit, 
                    min_width, max_width, max_re_count, 
                    sl_min_width, sl_max_width, sl_trim_size,
                    start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                )
                if roll_results:
                    all_results.append(roll_results)
                    all_df_orders.append(roll_df_orders)

            if sheet_order_cnt > 0: 
                logging.info(f"쉬트지 오더 {sheet_order_cnt}건 처리 시작.")
                ( 
                    module,
                    plant,
                    pm_no,
                    schedule_unit,
                    lot_no,
                    version,
                    version_id,
                    time_limit,
                    paper_type, b_wgt, color,
                    sheet_subject_order,
                    sheet_subject_width,
                    sheet_subject_ptn_cnt,
                    sheet_subject_balance,
                    sheet_subject_roll_cnt,
                    min_width,
                    max_width,
                    min_re_count,
                    max_re_count,
                    sheet_min_re_pok_cnt,
                    sheet_max_re_pok_cnt,
                    min_length,
                    max_length,
                    length_version_cnt,
                    rs_mix_flag,
                    length_similar,
                    allow_multiple,
                    allow_multiple_width,
                    yeopok_yn,
                    rs_mix_maxpok,
                    sl_min_width,
                    sl_max_width,
                    sl_min_pok_cnt,
                    sl_max_pok_cnt,
                    sl_trim_size,
                    sl_increase_trim,
                    sl_max_trim,
                    sl_mix_yn,
                    sl_sroll_mix_yn,
                    sl_sroll_width,
                    sl_sroll_widthmin,
                    sl_sroll_widthmax,
                    sl_single_yn,
                    sl_single_trim_yn,
                    sc_min_width,
                    sc_max_width,
                    sc_min_pok_cnt,
                    sc_max_pok_cnt,
                    max_unit_weight,
                    sheet_trim_size,
                    sc_increase_trim,
                    sc_max_trim,
                    knife_min,
                    knife_max,
                    mix_type
                ) = db.get_lot_param_sheet_st(lot_no=lot_no, version=version)

                sheet_results, sheet_df_orders, prod_seq_counter, group_order_no_counter = process_sheet_lot_st(
                    db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit,
                    min_width, max_width, sheet_max_re_pok_cnt, 
                    sc_min_width, sc_max_width, sheet_trim_size, min_length, max_length,
                    start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                )
                
                if sheet_results:
                    all_results.append(sheet_results)
                    all_df_orders.append(sheet_df_orders)

            if all_results:
                final_status = save_results(
                    db, lot_no, version, plant, pm_no, schedule_unit, max_width, paper_type, b_wgt, 
                    all_results, all_df_orders
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

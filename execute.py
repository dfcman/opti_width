import pandas as pd
import sys
import time
import configparser
import os
from roll_optimize import RollOptimize
from sheet_optimize import SheetOptimize
from sheet_optimize_var import SheetOptimizeVar
from sheet_optimize_ca import SheetOptimizeCa
from db_connector import Database
import argparse

def process_roll_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, re_min_width, re_max_width, re_max_pieces, paper_type, b_wgt
):
    """롤지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Roll Lot: {lot_no} (Version: {version}) 처리 시작")
    print(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    print(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_roll_orders_from_db(paper_prod_seq=lot_no)

    if not raw_orders:
        print(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성
    group_cols = ['지폭', '롤길이', '등급']
    for col in ['지폭', '롤길이']:
        df_orders[col] = pd.to_numeric(df_orders[col])
    df_orders['등급'] = df_orders['등급'].astype(str)
    df_groups = df_orders[group_cols].drop_duplicates().sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    print(f"--- Lot {lot_no} 원본 주문 정보 (그룹오더 포함) ---")
    print(df_orders.to_string())
    print("\n")

    # --- 롤길이별 최적화 실행 ---
    all_results = {
        "pattern_result": [],
        "pattern_details_for_db": [],
        "fulfillment_summary": []
    }
    
    unique_roll_lengths = df_orders['롤길이'].unique()

    for roll_length in unique_roll_lengths:
        print(f"\n--- 롤길이 그룹 {roll_length}에 대한 최적화 시작 ---")
        df_subset = df_orders[df_orders['롤길이'] == roll_length].copy()

        if df_subset.empty:
            continue

        optimizer = RollOptimize(
            df_spec_pre=df_subset,
            max_width=int(re_max_width),
            min_width=int(re_min_width),
            max_pieces=int(re_max_pieces)
        )
        results = optimizer.run_optimize()

        if "error" in results:
            print(f"[에러] Lot {lot_no}, 롤길이 {roll_length} 최적화 실패: {results['error']}")
            # Optionally, handle this case, maybe skip this group or stop the whole process
            continue
        
        print(f"--- 롤길이 그룹 {roll_length} 최적화 성공 ---")
        all_results["pattern_result"].append(results["pattern_result"])
        all_results["pattern_details_for_db"].extend(results["pattern_details_for_db"])
        all_results["fulfillment_summary"].append(results["fulfillment_summary"])

    if not all_results["pattern_details_for_db"]:
        print(f"[에러] Lot {lot_no}에 대한 최적화 결과가 없습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    # 모든 롤길이 그룹의 결과 취합
    final_results = {
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "fulfillment_summary": pd.concat(all_results["fulfillment_summary"], ignore_index=True)
    }

    print("\n--- 전체 최적화 성공. 최종 결과를 처리합니다. ---")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, final_results)

def process_sheet_lot(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot: {lot_no} (Version: {version}) 처리 시작")
    print(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}, min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, sheet_length_re={sheet_length_re}")
    print(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_sheet_orders_from_db(paper_prod_seq=lot_no)
    print(f"--- Lot {lot_no} 원본 주문 정보 ---")
    # # raw_orders가 리스트 안에 딕셔너리 형태로 되어 있다고 가정
    # for order in raw_orders:
    #     print(order)
    # print("\n")


    if not raw_orders:
        print(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성 (쉬트지 기준)
    group_cols = ['가로', '세로', '등급'] # 쉬트지는 '가로'(width)가 중요
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    df_groups = df_orders[group_cols].drop_duplicates().sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    # 최적화 실행
    print("--- 쉬트지 최적화 시작 ---")
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
        print("--- Optimizer results ---")
        print(results)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        print(f"[에러] Lot {lot_no} 최적화 실패: {error_msg}. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return
    
    print("최적화 성공. 결과를 처리합니다.")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results)


def process_sheet_lot_var(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot: {lot_no} (Version: {version}) 처리 시작")
    print(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    print(f"min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, min_sheet_length_re={min_sheet_length_re}, max_sheet_length_re={max_sheet_length_re}")
    print(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_sheet_orders_from_db_var(paper_prod_seq=lot_no)
    print(f"--- Lot {lot_no} 원본 주문 정보 ---")
    # # raw_orders가 리스트 안에 딕셔너리 형태로 되어 있다고 가정
    # for order in raw_orders:
    #     print(order)
    # print("\n")


    if not raw_orders:
        print(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성 (쉬트지 기준)
    group_cols = ['가로', '세로', '등급'] # 쉬트지는 '가로'(width)가 중요
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    df_groups = df_orders[group_cols].drop_duplicates().sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    # 최적화 실행
    print("--- 쉬트지 최적화 시작 ---")
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
        print("--- Optimizer results ---")
        print(results)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        print(f"[에러] Lot {lot_no} 최적화 실패: {error_msg}. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return
    
    print("최적화 성공. 결과를 처리합니다.")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results)


def process_sheet_lot_ca(
        db, plant, pm_no, schedule_unit, lot_no, version, 
        re_min_width, re_max_width, re_max_pieces, 
        paper_type, b_wgt,
        min_sc_width, max_sc_width, sheet_trim_size, min_sheet_length_re, max_sheet_length_re
):
    """쉬트지 lot에 대한 전체 최적화 프로세스를 처리합니다."""
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sheet Lot: {lot_no} (Version: {version}) 처리 시작")
    print(f"적용 파라미터: min_width={re_min_width}, max_width={re_max_width}, max_pieces={re_max_pieces}")
    print(f"min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, min_sheet_length_re={min_sheet_length_re}, max_sheet_length_re={max_sheet_length_re}")
    print(f"{'='*60}")

    db.update_lot_status(lot_no=lot_no, version=version, status=1)
    raw_orders = db.get_sheet_orders_from_db_ca(paper_prod_seq=lot_no)
    print(f"--- Lot {lot_no} 원본 주문 정보 ---")
    # # raw_orders가 리스트 안에 딕셔너리 형태로 되어 있다고 가정
    # for order in raw_orders:
    #     print(order)
    # print("\n")


    if not raw_orders:
        print(f"[에러] Lot {lot_no}의 오더를 가져오지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    df_orders = pd.DataFrame(raw_orders)

    # 그룹오더 번호 생성 (쉬트지 기준)
    group_cols = ['가로', '세로', '등급'] # 쉬트지는 '가로'(width)가 중요
    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    df_groups = df_orders[group_cols].drop_duplicates().sort_values(by=group_cols).reset_index(drop=True)
    df_groups['group_order_no'] = [f"30{lot_no}{i+1:03d}" for i in df_groups.index]
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    # 최적화 실행
    print("--- 쉬트지 최적화 시작 ---")
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
        print("--- Optimizer results ---")
        print(results)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise e

    if not results or "error" in results:
        error_msg = results['error'] if results and 'error' in results else "No solution found"
        print(f"[에러] Lot {lot_no} 최적화 실패: {error_msg}. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return
    
    print("최적화 성공. 결과를 처리합니다.")
    save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results)

def save_results(db, lot_no, version, plant, pm_no, schedule_unit, re_max_width, paper_type, b_wgt, results):
    """최적화 결과를 DB에 저장하고 CSV파일로 출력합니다."""
    print("최적화 결과 (패턴별 생산량):")
    print(results["pattern_result"].to_string())
    print("\n\n# ================= 주문 충족 현황 ================== #\n")
    print(results["fulfillment_summary"].to_string())
    print("\n")
    print("최적화 성공. 이제 결과를 DB에 저장합니다.")

    # DB에 패턴 저장
    success_db = db.insert_pattern_sequence(
        lot_no, version, plant, pm_no, schedule_unit, re_max_width, 
        paper_type, b_wgt, results['pattern_details_for_db']
    )

    if not success_db:
        print(f"[에러] Lot {lot_no}의 패턴을 DB에 저장하지 못했습니다. 상태를 99(에러)로 변경합니다.")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)
        return

    # CSV 파일로 결과 저장
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{lot_no}_{version}.csv"
    output_path = os.path.join(output_dir, output_filename)
    try:
        results['pattern_result'].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[성공] 요약 결과가 다음 파일에 저장되었습니다: {output_path}")
        db.update_lot_status(lot_no=lot_no, version=version, status=0)
    except Exception as e:
        print(f"[에러] 결과를 CSV 파일에 저장하는 중 오류 발생: {e}")
        db.update_lot_status(lot_no=lot_no, version=version, status=99)

    print(f"\n{'='*60}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Lot: {lot_no} 처리 완료")
    print(f"{'='*60}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="롤지 또는 쉬트지 최적화를 실행합니다.")
    parser.add_argument("--order-type", required=True, choices=['roll', 'sheet', 'sheet_var', 'sheet_ca'], help="오더 유형 ('roll' 또는 'sheet')")
    # parser.add_argument("--lot-no", required=True, help="처리할 Lot 번호")
    args = parser.parse_args()

    db = None
    try:
        config = configparser.ConfigParser()
        if not os.path.exists('config.ini'):
            raise FileNotFoundError("config.ini 파일을 찾을 수 없습니다.")
        config.read('config.ini')
        db_config = config['database']
        
        db = Database(user=db_config['user'], password=db_config['password'], dsn=db_config['dsn'])

        # 데몬 방식 대신, get_target_lot()을 한 번만 호출하여 테스트합니다.
        ( 
            plant, pm_no, schedule_unit, lot_no, version, min_width, 
            max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
            paper_type, b_wgt,
            min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re
        ) = db.get_target_lot()

        if not lot_no:
            print("처리할 Lot이 없습니다.")
            return

        if args.order_type == 'roll':
            process_roll_lot(
                db, plant, pm_no, schedule_unit, lot_no, version, 
                min_width, max_width, max_pieces, paper_type, b_wgt
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
        print(f"[치명적 에러] 설정 파일을 찾을 수 없습니다: {e}")
    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[치명적 에러] 실행 중 예외 발생: {e}")
    finally:
        if db:
            db.close_pool()
        print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()

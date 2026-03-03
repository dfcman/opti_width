"""
3000 대전공장 전용 execute 모듈.
- process_roll_lot: 롤지 Lot 최적화
- process_sheet_lot: 쉬트지 Lot 최적화 (표준길이)
- apply_sheet_grouping: 쉬트 그룹핑
- main: 3000 공장 메인 루프
"""
import pandas as pd
from collections import Counter
import time
import logging
from optimize.roll_optimize import RollOptimize
from optimize.sheet_optimize import SheetOptimize
from execute_common import (
    NUM_THREADS, init_db, setup_logging, save_results, generate_allocated_sheet_details
)


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


def process_sheet_lot(
        db, plant, pm_no, schedule_unit, lot_no, version,  
        paper_type, b_wgt, color, time_limit,
        re_min_width, re_max_width, re_max_pieces,
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

    df_orders['가로'] = pd.to_numeric(df_orders['가로'])
    df_orders['세로'] = pd.to_numeric(df_orders['세로'])
    df_orders['등급'] = df_orders['등급'].astype(str)
    
    # --- [New Grouping Logic] ---
    df_orders, df_groups, last_group_order_no = apply_sheet_grouping(df_orders, start_group_order_no, lot_no)
    
    logging.info(f"--- Lot {df_groups.to_string()} \n그룹마스터 정보 ---")

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

        logging.info(f"--- 롤길이 그룹 {roll_length}, Core {core}, Dia {dia} 최적화 성공 (배분 완료) ---")
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

    # Fulfillment Summary를 개별 오더 기준으로 재생성
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
        "pattern_result": pd.concat(all_results["pattern_result"], ignore_index=True),
        "pattern_details_for_db": all_results["pattern_details_for_db"],
        "pattern_roll_details_for_db": all_results["pattern_roll_details_for_db"],
        "pattern_roll_cut_details_for_db": all_results["pattern_roll_cut_details_for_db"],
        "pattern_sheet_details_for_db": pattern_sheet_details_for_db,
        "fulfillment_summary": df_summary
    }

    logging.info("\n--- 롤지 최적화 성공. ---")
    return final_results, df_orders, prod_seq_counter, last_group_order_no


def main():
    """3000 대전공장 메인 실행 함수"""
    db = None
    lot_no = None
    version = None

    try:
        db = init_db('3000')

        while True:
            ( 
                plant, pm_no, schedule_unit, lot_no, version, min_width, 
                max_width, sheet_max_width, max_pieces, sheet_max_pieces, 
                paper_type, b_wgt, color,
                min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
                sheet_order_cnt, roll_order_cnt, time_limit
            ) = db.get_target_lot()

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
                roll_results, roll_df_orders, prod_seq_counter, group_order_no_counter = process_roll_lot(
                    db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, 
                    time_limit, min_width, max_width, max_pieces, 
                    start_prod_seq=prod_seq_counter, start_group_order_no=group_order_no_counter
                )
                if roll_results:
                    all_results.append(roll_results)
                    all_df_orders.append(roll_df_orders)

            if sheet_order_cnt > 0: 
                logging.info(f"쉬트지 오더 {sheet_order_cnt}건 처리 시작.")
                # 표준길이 검증
                if not sheet_length_re:
                    logging.error(f"[에러] Lot {lot_no}에 대한 표준길이(sheet_length_re)가 없어 최적화를 수행할 수 없습니다. 상태를 2(에러)로 변경합니다.")
                    db.update_lot_status(lot_no=lot_no, version=version, status=2)
                    continue
                
                sheet_results, sheet_df_orders, prod_seq_counter, group_order_no_counter = process_sheet_lot(
                    db, plant, pm_no, schedule_unit, lot_no, version, paper_type, b_wgt, color, time_limit, 
                    min_width, sheet_max_width, sheet_max_pieces, 
                    min_sc_width, max_sc_width, sheet_trim_size, sheet_length_re,
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

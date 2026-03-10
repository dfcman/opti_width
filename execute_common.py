"""
공장별 execute 파일에서 공통으로 사용하는 유틸리티 함수 모듈.
- setup_logging: 로그 설정
- save_results: 최적화 결과 DB 저장 및 CSV 출력
- generate_allocated_sheet_details: 쉬트 시퀀스 할당 상세 생성
- init_db: DB 연결 초기화 및 DataInserters 바인딩
"""
import pandas as pd
import time
import configparser
import os
import logging
from collections import defaultdict
from db.db_connector import Database


# Load configuration
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf', 'config.ini')
NUM_THREADS = 4
if os.path.exists(config_path):
    config.read(config_path)
    NUM_THREADS = config.getint('optimization', 'num_threads', fallback=4)
else:
    print(f"Config file not found at {config_path}. Using default NUM_THREADS={NUM_THREADS}")


def init_db(plant_arg):
    """
    공장 코드에 따라 DB 연결을 초기화하고 DataInserters를 바인딩합니다.
    
    Args:
        plant_arg (str): 공장 코드 ('2000', '3000', '5000', '8000')
    
    Returns:
        Database: 초기화된 Database 인스턴스
    """
    config = configparser.ConfigParser()
    config_path = os.path.join('conf', 'config.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} 파일을 찾을 수 없습니다.")
    config.read(config_path, encoding='utf-8')

    db_section_map = {
        '2000': 'database_jh',
        '3000': 'database_dj',
        '5000': 'database_ca',
        '8000': 'database_st'
    }
    db_section = db_section_map.get(plant_arg, 'database_dj')
    print(f"Connecting to DB Section: {db_section} for Plant: {plant_arg}")

    if db_section not in config:
        raise KeyError(f"Config file does not contain section: {db_section}")
    
    db_config = config[db_section]
    db = Database(user=db_config['user'], password=db_config['password'], dsn=db_config['dsn'])

    # Dynamic Patching of DataInserters based on Plant
    if plant_arg == '5000':
        from db import db_insert_data_ca as db_module
    elif plant_arg == '8000':
        from db import db_insert_data_st as db_module
    elif plant_arg == '2000':
        from db import db_insert_data_jh as db_module
    else:  # 3000 or default
        from db import db_insert_data_dj as db_module

    # insert 함수들은 공장별 모듈에서 바인딩
    db.insert_pattern_sequence = db_module.DataInserters.insert_pattern_sequence.__get__(db, Database)
    db.insert_roll_sequence = db_module.DataInserters.insert_roll_sequence.__get__(db, Database)
    db.insert_cut_sequence = db_module.DataInserters.insert_cut_sequence.__get__(db, Database)
    db.insert_sheet_sequence = db_module.DataInserters.insert_sheet_sequence.__get__(db, Database)
    db.insert_order_group = db_module.DataInserters.insert_order_group.__get__(db, Database)
    db.insert_group_master = db_module.DataInserters.insert_group_master.__get__(db, Database)

    print(f"Applied plant-specific DataInserters from {db_module.__name__}")

    return db


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
                'length': pattern_length,
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
    # Keys identifying a Cut: (prod_seq, unit_no, seq, roll_seq, cut_seq)
    
        
    seq_tracker = defaultdict(int)
    
    for entry in allocated_details:
        key = (entry.get('prod_seq'), entry.get('unit_no'), entry.get('seq'), entry.get('roll_seq'), entry.get('cut_seq'))
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
                    pattern_length = detail.get('pattern_length', 0)
                    
                    # [Fix] count 값 반영 (Plant 5000의 경우 unroll_by_count=False로 count가 1이 아닌 원본 값 유지)
                    count = detail.get('count', 1)
                    wgt_kg = (pattern_length * width * b_wgt_val * count) / 10**6
                    
                    prod_wgt_map[order_no] = prod_wgt_map.get(order_no, 0) + wgt_kg
            
            final_df_orders['prod_wgt'] = final_df_orders['order_no'].map(prod_wgt_map).fillna(0).round(1)

            # [Update] th_group_master 입력
            group_cols = ['group_order_no']
            agg_dict = {
                'order_no': 'first',
                'plant': 'first',
                'schedule_unit': 'first',
                'paper_type': 'first',
                'b_wgt': 'first',
                '가로': 'first',
                '세로': 'first',
                'rs_gubun': 'first',
                'export_yn': 'first',
                'nation_code': 'first',
                'customer_name': 'first',
                'pt_gubun': 'first',
                'skid_yn': 'first',
                'dia': 'first',
                'core': 'first'
            }
            
            logging.info(f"\n\n# ================= {final_df_orders} ================== #\n")
            # agg_dict에 있는 컬럼이 실제로 존재하는지 확인 후 필터링
            valid_agg_dict = {k: v for k, v in agg_dict.items() if k in final_df_orders.columns}
            
            df_groups = final_df_orders.groupby(group_cols).agg(valid_agg_dict).reset_index()
            
            db.insert_group_master(
                connection, lot_no, version, plant, pm_no, schedule_unit, df_groups
            )

            db.insert_order_group(
                connection, lot_no, version, plant, pm_no, schedule_unit, final_df_orders
            )

        if plant == '5000':
            
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
        final_status = 0  # 기본값: 모든 오더 충족

        # '롤길이' 컬럼이 없는 경우 0으로 초기화하여 에러 방지
        if '롤길이' not in final_fulfillment_summary.columns:
             final_fulfillment_summary['롤길이'] = 0
        
        # 롤 최적화 결과 확인 (과부족(롤) 컬럼이 있는 경우)
        if '과부족(롤)' in final_fulfillment_summary.columns:
            under_production_rolls = final_fulfillment_summary[
                (final_fulfillment_summary['과부족(롤)'] != 0) & 
                (final_fulfillment_summary['롤길이'] > 0)
            ]
            if not under_production_rolls.empty:
                final_status = 0  # 일부 오더 초과(부족)
                logging.warning(f"[경고] 초과(부족) 생산된 롤 오더가 있습니다:\n{under_production_rolls.to_string()}")

        
        # 쉬트 최적화 결과 확인 (과부족(톤) 컬럼이 있는 경우)
        if '과부족(톤)' in final_fulfillment_summary.columns:
            under_production_sheets = final_fulfillment_summary[
                (final_fulfillment_summary['과부족(톤)'] < -1) & 
                (final_fulfillment_summary['롤길이'] == 0)
            ]  # 소수점 오차 고려
            over_production_sheets = final_fulfillment_summary[
                (final_fulfillment_summary['과부족(톤)'] >= 1) & 
                (final_fulfillment_summary['롤길이'] == 0)
            ]  # 소수점 오차 고려
            if not under_production_sheets.empty:
                final_status = 1  # 일부 오더 초과(부족)
                logging.warning(f"[경고] 부족 생산된 쉬트 오더가 있습니다:\n{under_production_sheets.to_string()}")
            if not over_production_sheets.empty:
                final_status = 1  # 일부 오더 초과(부족)
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

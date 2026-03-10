"""
2000 장항공장 전용 execute 모듈.
- process_lot_jh: 롤지+쉬트지 통합 Lot 최적화 (OptimizeJh 사용)
- apply_sheet_grouping_jh: 인접규격 그룹핑
- main: 2000 공장 메인 루프
"""
import pandas as pd
import numpy as np
import time
import logging
from optimize.optimize_jh import OptimizeJh
from execute_common import (
    NUM_THREADS, init_db, setup_logging, save_results, generate_allocated_sheet_details
)


def apply_sheet_grouping_jh(df_orders, start_group_order_no, lot_no, adj_cnt, adj_trim, max_wgt, min_wgt):
    """
    장항공장 인접규격 그룹핑 로직을 적용하고 group_order_no를 생성합니다.
    Java의 AdjacentWidth.updateAdjacentWidth를 Python으로 변환한 함수입니다.

    인접 규격 알고리즘 (4단계):
      STEP 1: 변수 초기화 - 규격별 폭수(pok), 최대/최소 무게 제약 check
      STEP 2: 인접규격 대상 식별 - 트림 차이 조건으로 인접규격 대상 판별 (2개/3개 묶음)
      STEP 3: 주문량 합계 조건 - 연속 변경 시 주문량 합계 비교하여 불리한 쪽 롤백
      STEP 4: 최종 조정 - 상/하위 동시 해당 시 조정 + 3개 규격 묶음 처리

    Parameters:
        df_orders (DataFrame): 오더 데이터 (가로, 주문톤, max_sc_width, digital, 등급, order_no 컬럼 필요)
        start_group_order_no (int): 시작 그룹오더 번호
        lot_no (str): Lot 번호
        adj_cnt (int): 인접규격 적용 최대 개수 (2: 2개 묶음만, 3 이상: 3개 묶음 가능)
        adj_trim (int): 인접규격 트림 허용치 (폭 차이 × 폭수 <= adj_trim 이면 적용)
        max_wgt (int): 최대 무게 제약 (미만인 규격만 인접규격 적용)
        min_wgt (int): 최소 무게 제약 (미만인 규격이 있으면 인접규격 강제 적용)

    Returns:
        tuple: (df_orders, df_groups, last_group_order_no)
    """
    # --- 1. 디지털 오더 분리 ---
    if 'digital' in df_orders.columns:
        df_digital = df_orders[df_orders['digital'] == 'Y'].copy()
        df_non_digital = df_orders[df_orders['digital'] != 'Y'].copy()
        if not df_digital.empty:
            for _, row in df_digital.iterrows():
                logging.info(f"Exclude Digital Order : {lot_no}, {row['가로']}")
    else:
        df_digital = pd.DataFrame()
        df_non_digital = df_orders.copy()

    # adj_trim이 0 이하이면 인접규격 미적용, 기본 그룹핑만 수행
    if adj_trim <= 0 or df_non_digital.empty:
        logging.info("adj_trim <= 0 또는 비-디지털 오더 없음. 인접규격 미적용, 기본 그룹핑만 수행.")
    else:
        # --- 2. 규격별 그룹 데이터 생성 (가로 기준으로 집계) ---
        group_data = df_non_digital.groupby('가로').agg(
            order_ton_cnt=('주문톤', 'sum'),
            max_sc_width=('max_sc_width', 'first')
        ).reset_index()

        # 가로 내림차순 정렬 (Java와 동일: width[i] > width[i+1])
        group_data = group_data.sort_values('가로', ascending=False).reset_index(drop=True)

        lst_size = len(group_data)

        if lst_size > 1:
            # --- STEP 1: 변수 초기화 ---
            width = group_data['가로'].values.astype(int)
            order_cnt = group_data['order_ton_cnt'].values.astype(float)
            max_width_arr = group_data['max_sc_width'].values.astype(int)

            adj_width_result = width.copy()
            adj_tmp = width.copy()
            order_sum = np.zeros(lst_size, dtype=float)
            triple_sum = np.zeros(lst_size, dtype=float)

            chk_above = np.zeros(lst_size, dtype=bool)   # 상위 규격과 그룹 가능한지
            chk_under = np.zeros(lst_size, dtype=bool)   # 하위 규격과 그룹 가능한지
            chk_change = np.zeros(lst_size, dtype=bool)  # 상위 규격으로 변경되는지
            chk_max_wgt = order_cnt < max_wgt             # 최대 무게 제약 check
            chk_min_wgt = order_cnt < min_wgt             # 최소 무게 제약 check
            chk_triple = np.zeros(lst_size, dtype=bool)   # 3개 규격 그룹 가능한지

            # pok 계산: 각 규격이 SC 최대 폭에 몇 폭 들어가는지
            pok = np.zeros(lst_size, dtype=int)
            for i in range(lst_size):
                poki = 1
                while width[i] * poki + 30 <= max_width_arr[i]:
                    pok[i] = poki
                    poki += 1

            logging.info(f"인접규격 초기 데이터 (내림차순): {list(zip(width, order_cnt, max_width_arr, pok))}")

            # --- STEP 2: 인접규격 대상 규격 변경 ---
            for i in range(lst_size):
                # 3개 규격 묶음 체크 (adjCnt > 2 일 때만)
                if adj_cnt > 2 and i < lst_size - 2:
                    diff_trim = (width[i] - width[i + 2]) * pok[i]
                    if diff_trim <= adj_trim:
                        if (chk_max_wgt[i] and chk_max_wgt[i + 1] and chk_max_wgt[i + 2]):
                            chk_triple[i] = True
                            triple_sum[i] = order_cnt[i] + order_cnt[i + 1] + order_cnt[i + 2]
                        elif (chk_min_wgt[i] or chk_min_wgt[i + 1] or chk_min_wgt[i + 2]):
                            chk_triple[i] = True
                            triple_sum[i] = order_cnt[i] + order_cnt[i + 1] + order_cnt[i + 2]

                # 2개 규격 인접규격 체크
                if i < lst_size - 1:
                    diff_trim = (width[i] - width[i + 1]) * pok[i]
                    if diff_trim <= adj_trim:
                        if (chk_max_wgt[i] and chk_max_wgt[i + 1]):
                            # 최대 무게 제약 미만인 규격들만 인접규격 적용
                            adj_tmp[i + 1] = width[i]
                            order_sum[i + 1] = order_cnt[i] + order_cnt[i + 1]
                            chk_under[i] = True
                            chk_above[i + 1] = True
                            chk_change[i + 1] = True
                        elif (chk_min_wgt[i] or chk_min_wgt[i + 1]):
                            # 최소 무게 미만인 경우 인접규격 강제 적용
                            adj_tmp[i + 1] = width[i]
                            order_sum[i + 1] = order_cnt[i] + order_cnt[i + 1]
                            chk_under[i] = True
                            chk_above[i + 1] = True
                            chk_change[i + 1] = True

                # 인접규격 adjWidth에 임시 규격 adjTmp 저장
                adj_width_result[i] = adj_tmp[i]

            # --- STEP 3: 주문량 합계 조건 적용하여 인접규격 수정 ---
            for i in range(lst_size - 1):
                if chk_change[i] and chk_change[i + 1]:
                    if i > 0 and chk_change[i - 1] and chk_change[i] and chk_change[i + 1]:
                        # 3개 규격이 연속으로 변경된 경우
                        if order_sum[i] < order_sum[i - 1] and order_sum[i] < order_sum[i + 1]:
                            # i번째 주문합이 가장 작으면 (i-1), (i+1) 롤백
                            adj_width_result[i - 1] = width[i - 1]
                            adj_width_result[i + 1] = width[i + 1]
                            chk_change[i - 1] = False
                            chk_change[i + 1] = False
                    elif order_sum[i] > order_sum[i + 1]:
                        # 상위 규격 주문량 합계가 더 크면 상위 규격 롤백
                        adj_width_result[i] = width[i]
                        chk_change[i] = False
                    elif order_sum[i] < order_sum[i + 1]:
                        # 하위 규격 주문량 합계가 더 크면 하위 규격 롤백
                        adj_width_result[i + 1] = width[i + 1]
                        chk_change[i + 1] = False

            # --- STEP 4: 인접규격 최종 조정 ---
            for i in range(lst_size - 1):
                # 상위/하위 규격과 모두 인접 규격으로 묶일 경우
                if chk_above[i] and chk_under[i]:
                    if i < lst_size - 1 and not chk_change[i + 1]:
                        # 하위 규격이 변경 안 됐으면 현 규격을 상위 규격으로 변경
                        adj_width_result[i] = adj_tmp[i]
                        chk_change[i] = True
                    elif i < lst_size - 1 and chk_change[i + 1]:
                        # 하위 규격이 변경 됐으면 현 규격은 자신의 규격 유지
                        adj_width_result[i] = width[i]
                        chk_change[i] = False

            # 3개 규격 묶음 최종 처리 (adjCnt > 2)
            if adj_cnt > 2:
                for i in range(lst_size - 1):
                    if i < lst_size - 2:
                        # 3개 규격이 유일하게 인접그룹으로 묶일 경우
                        if chk_triple[i] and not chk_triple[i + 1]:
                            if not chk_above[i] and not chk_under[i + 2]:
                                # 상위/하위 연결 없으면 3개 규격을 하나로
                                adj_width_result[i + 1] = width[i]
                                adj_width_result[i + 2] = width[i]
                                chk_change[i + 1] = True
                                chk_change[i + 2] = True
                            elif chk_above[i] and triple_sum[i] < order_sum[i]:
                                # 3개 규격의 주문량 합이 상위2개 주문량 합보다 작은 경우
                                adj_width_result[i + 1] = width[i]
                                adj_width_result[i + 2] = width[i]
                                chk_change[i + 1] = True
                                chk_change[i + 2] = True
                                if chk_change[i]:
                                    adj_width_result[i] = width[i]
                                    chk_change[i] = False
                            elif (i < lst_size - 3 and chk_under[i + 2]
                                  and triple_sum[i] < order_sum[i + 3]):
                                # 3개 규격의 주문량 합이 하위2개 주문량 합보다 작은 경우
                                adj_width_result[i + 1] = width[i]
                                adj_width_result[i + 2] = width[i]
                                chk_change[i + 1] = True
                                chk_change[i + 2] = True
                                if chk_change[i]:
                                    adj_width_result[i] = width[i]
                                    chk_change[i] = False
                                if chk_change[i + 3]:
                                    adj_width_result[i + 3] = width[i + 3]
                                    chk_change[i + 3] = False

            # --- 3. 인접규격 매핑 생성 및 df_orders에 적용 ---
            width_mapping = dict(zip(width, adj_width_result))

            # 변경된 규격 로깅
            for orig, adj in width_mapping.items():
                if orig != adj:
                    logging.info(f"인접규격 적용: {orig} -> {adj}")

            # df_non_digital에 인접규격 적용 (원본 보존)
            df_non_digital['가로_원본'] = df_non_digital['가로']
            df_non_digital['가로'] = df_non_digital['가로'].map(width_mapping)
        else:
            logging.info("규격이 1개 이하이므로 인접규격 미적용.")

    # --- 4. 디지털 오더와 합침 ---
    if not df_digital.empty:
        df_orders = pd.concat([df_non_digital, df_digital], ignore_index=True)
    else:
        df_orders = df_non_digital.copy()

    # --- 5. 그룹오더 생성 (가로 + 등급 기준) ---
    group_cols = ['rs_gubun','가로', 'roll_length', '등급', 'short_sheet', 'semi']

    df_groups = df_orders.groupby(group_cols).agg(
        대표오더번호=('order_no', 'first')
    ).reset_index()

    # 정렬 (가로, 등급 순)
    df_groups = df_groups.sort_values(by=['가로', '등급']).reset_index(drop=True)

    # Group Order No 생성 (장항 = 20 접두사)
    df_groups['group_order_no'] = [f"20{lot_no}{start_group_order_no + i + 1:03d}" for i in df_groups.index]
    last_group_order_no = start_group_order_no + len(df_groups)

    # 원본 데이터에 병합
    df_orders = pd.merge(df_orders, df_groups, on=group_cols, how='left')

    logging.info(f"// Adjacent Width Grouping Complete!! (총 {len(df_groups)}개 그룹)")
    return df_orders, df_groups, last_group_order_no


def process_lot_jh(db, lot_params, start_prod_seq=0, start_group_order_no=0):
    """
    장항공장 롤지+쉬트지 통합 Lot 최적화.
    OptimizeJh를 사용하여 롤/쉬트를 구분하지 않고 한번에 최적화합니다.

    Parameters:
        db: DB 연결 객체
        lot_params (dict): get_target_lot_jh에서 반환된 딕셔너리 (쿼리 alias가 키)
        start_prod_seq (int): 시작 생산 시퀀스 번호
        start_group_order_no (int): 시작 그룹 오더 번호
    """
    # 딕셔너리에서 필요한 값 추출
    plant = lot_params['plant']
    pm_no = lot_params['pm_no']
    schedule_unit = lot_params['schedule_unit']
    lot_no = lot_params['lot_no']
    version = lot_params['version']
    paper_type = lot_params['paper_type']
    b_wgt = lot_params['b_wgt']
    color = lot_params['color']
    min_width = int(lot_params['re_minwidth'])
    max_width = int(lot_params['re_maxwidth'])
    max_pieces = int(lot_params['re_max_pieces'])
    re_sheet_min_pieces = int(lot_params['re_sheet_min_pieces'])
    re_sheet_max_pieces = int(lot_params['re_sheet_max_pieces'])

    min_sc_width = lot_params['sc_minwidth']
    max_sc_width = lot_params['sc_maxwidth']
    sheet_trim_size = lot_params['sc_basetrim']
    min_sheet_length_re = lot_params['min_re_stdlength']
    max_sheet_length_re = lot_params['max_re_stdlength']    
    yn_stdlength = lot_params['yn_stdlength']      # 표준길이 적용여부
    rs_mix_flag = str(lot_params['yn_rstypemix'])  # 롤/시트 혼합여부

    mx_maxsheetroll = lot_params['mx_maxsheetroll']
    mx_maxsheetwidth = lot_params['mx_maxsheetwidth']
    mx_rsmixwidth = lot_params['mx_rsmixwidth']
    
    # 롤지(SL) 파라미터
    sl_rollmixtype = lot_params['sl_rollmixtype']
    sl_trim = int(lot_params['sl_trim'])
    sl_minwidth = int(lot_params['sl_minwidth'])
    sl_maxwidth = int(lot_params['sl_maxwidth'])
    min_sl_count = int(lot_params['min_sl_count'])
    max_sl_count = int(lot_params['max_sl_count'])

    # 2026-03-09 추가: 디지털 오더 파라미터
    sc_digital_trim = lot_params['sc_digital_trim']
    sc_digital_minwidth = lot_params['sc_digital_minwidth']
    sc_digital_maxwidth = lot_params['sc_digital_maxwidth']
    sc_digital_minsheet = lot_params['sc_digital_minsheet']
    sc_digital_maxsheet = lot_params['sc_digital_maxsheet']

    re_minton = lot_params['re_minton']
    sc_minton = lot_params['sc_minton']
    sheet_order_moq = lot_params['sheet_order_moq']
    moq_yn = lot_params['moq_yn']
    moq_ton = lot_params['moq_ton']
    moq_sc_width = lot_params['moq_sc_width']

    adj_yn = 'Y' if int(lot_params['adj_width']) > 0 else 'N'  # 인접그룹 조합여부
    adj_width = lot_params['adj_width']   # 인접그룹시 가로 규격 차이
    adj_min_wgt = lot_params['adj_min_wgt']     # 인접그룹시 최소오더톤
    adj_max_wgt = lot_params['adj_max_wgt']     # 인접그룹시 최대오더톤

    
    
    

    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Lot (JH 통합): {lot_no} (Version: {version}) 처리 시작")
    logging.info(f"적용 파라미터: min_width={min_width}, max_width={max_width}, max_pieces={max_pieces}")
    logging.info(f"min_sc_width={min_sc_width}, max_sc_width={max_sc_width}, sheet_trim_size={sheet_trim_size}")
    logging.info(f"sl_minwidth={sl_minwidth}, sl_maxwidth={sl_maxwidth}, sl_trim={sl_trim}, max_sl_count={max_sl_count}")
    logging.info(f"min_sheet_length_re={min_sheet_length_re}, max_sheet_length_re={max_sheet_length_re}")
    logging.info(f"시작 시퀀스 번호: prod_seq={start_prod_seq}, group_order_no={start_group_order_no}")
    logging.info(f"{'='*60}")

    # --- 1. 롤지 오더 가져오기 ---
    df_roll_orders = None
    raw_roll_orders = db.get_roll_orders_from_db_jh(in_lot_no=lot_no, in_version=version)
    if raw_roll_orders:
        df_roll_orders = pd.DataFrame(raw_roll_orders)
        df_roll_orders['lot_no'] = lot_no
        df_roll_orders['version'] = version
        df_roll_orders['지폭'] = pd.to_numeric(df_roll_orders['지폭'])
        df_roll_orders['롤길이'] = pd.to_numeric(df_roll_orders['롤길이'])
        df_roll_orders['등급'] = df_roll_orders['등급'].astype(str)

        configured_roll_std_length = 0
        if str(yn_stdlength).upper() == 'Y':
            configured_roll_std_length = int(pd.to_numeric(max_sheet_length_re, errors='coerce') or 0)
            if configured_roll_std_length <= 0:
                configured_roll_std_length = int(pd.to_numeric(min_sheet_length_re, errors='coerce') or 0)

        if 'std_length' in df_roll_orders.columns:
            std_length_series = pd.to_numeric(df_roll_orders['std_length'], errors='coerce').fillna(0)
        else:
            std_length_series = pd.Series(0, index=df_roll_orders.index, dtype='float64')

        if configured_roll_std_length > 0:
            std_length_series = std_length_series.where(std_length_series > 0, configured_roll_std_length)
        std_length_series = std_length_series.where(std_length_series > 0, df_roll_orders['롤길이'])
        df_roll_orders['std_length'] = np.maximum(std_length_series, df_roll_orders['롤길이']).astype(int)

        logging.info(
            "롤지 std_length 보정 완료: %s",
            df_roll_orders[['롤길이', 'std_length']].drop_duplicates().sort_values(['std_length', '롤길이']).to_dict('records')
        )

        # 롤지 그룹오더 생성
        roll_group_cols = ['지폭', '롤길이', '등급', 'core', 'dia', 'rs_gubun', 'semi']
        df_roll_groups = df_roll_orders.groupby(roll_group_cols).agg(
            대표오더번호=('order_no', 'first')
        ).reset_index()
        df_roll_groups = df_roll_groups.sort_values(by=roll_group_cols).reset_index(drop=True)
        df_roll_groups['group_order_no'] = [f"20{lot_no}{start_group_order_no + i + 1:03d}" for i in df_roll_groups.index]
        start_group_order_no += len(df_roll_groups)
        df_roll_orders = pd.merge(df_roll_orders, df_roll_groups, on=roll_group_cols, how='left')

        logging.info(f"롤지 오더 {len(df_roll_orders)}건 로드 완료.")
    else:
        logging.info("롤지 오더 없음.")

    # --- 2. 쉬트지 오더 가져오기 ---
    df_sheet_orders = None
    raw_sheet_orders = db.get_sheet_orders_from_db_jh(in_lot_no=lot_no, in_version=version)
    if raw_sheet_orders:
        df_sheet_orders = pd.DataFrame(raw_sheet_orders)
        df_sheet_orders['lot_no'] = lot_no
        df_sheet_orders['version'] = version
        df_sheet_orders['가로'] = pd.to_numeric(df_sheet_orders['가로'])
        df_sheet_orders['세로'] = pd.to_numeric(df_sheet_orders['세로'])
        df_sheet_orders['등급'] = df_sheet_orders['등급'].astype(str)

        # 쉬트지 그룹오더 생성
        sheet_group_cols = ['가로', '세로', '등급', 'semi', 'short_sheet']
        df_sheet_groups = df_sheet_orders.groupby(sheet_group_cols).agg(
            대표오더번호=('order_no', 'first')
        ).reset_index()
        df_sheet_groups = df_sheet_groups.sort_values(by=sheet_group_cols).reset_index(drop=True)
        df_sheet_groups['group_order_no'] = [f"20{lot_no}{start_group_order_no + i + 1:03d}" for i in df_sheet_groups.index]
        start_group_order_no += len(df_sheet_groups)
        df_sheet_orders = pd.merge(df_sheet_orders, df_sheet_groups, on=sheet_group_cols, how='left')

        logging.info(f"쉬트지 오더 {len(df_sheet_orders)}건 로드 완료.")
    else:
        logging.info("쉬트지 오더 없음.")

    # --- 3. 오더가 없으면 종료 ---
    if df_roll_orders is None and df_sheet_orders is None:
        logging.error(f"[에러] Lot {lot_no}의 롤지/쉬트지 오더를 모두 가져오지 못했습니다.")
        return None, None, start_prod_seq, start_group_order_no

    # --- 4. OptimizeJh 통합 최적화 실행 ---
    optimizer = OptimizeJh(
        db=db,
        plant=plant,
        pm_no=pm_no,
        schedule_unit=schedule_unit,
        lot_no=lot_no,
        version=version,
        paper_type=paper_type,
        b_wgt=float(b_wgt),
        color=color,
        # 롤지 데이터
        df_roll_orders=df_roll_orders,
        # 쉬트지 데이터
        df_sheet_orders=df_sheet_orders,
        min_sheet_roll_length=int(min_sheet_length_re) // 10 * 10 if min_sheet_length_re else None,
        max_sheet_roll_length=int(max_sheet_length_re) // 10 * 10 if max_sheet_length_re else None,
        sheet_trim=sheet_trim_size,
        min_sc_width=min_sc_width,
        max_sc_width=max_sc_width,


        yn_stdlength=yn_stdlength,

        # 롤지(SL) 제약
        min_sl_width=sl_minwidth,
        max_sl_width=sl_maxwidth,
        max_sl_count=max_sl_count,
        ww_trim_size=sl_trim,
        rs_mix_flag=rs_mix_flag,
        # 공통 제약
        min_width=min_width,
        max_width=max_width,
        max_pieces=max_pieces,
        re_sheet_min_pieces=re_sheet_min_pieces,
        re_sheet_max_pieces=re_sheet_max_pieces,

        sc_digital_trim=sc_digital_trim,
        sc_digital_minwidth=sc_digital_minwidth,
        sc_digital_maxwidth=sc_digital_maxwidth,
        sc_digital_minsheet=sc_digital_minsheet,
        sc_digital_maxsheet=sc_digital_maxsheet,
        
        re_minton=re_minton,
        sc_minton=sc_minton,
        sheet_order_moq=sheet_order_moq,
        moq_yn=moq_yn,
        moq_ton=moq_ton,
        moq_sc_width=moq_sc_width,

        adj_yn=adj_yn,
        adj_width=adj_width,
        adj_min_wgt=adj_min_wgt,
        adj_max_wgt=adj_max_wgt,

        num_threads=NUM_THREADS,
        time_limit=lot_params['time_order'] * -1000  # time_order는 음수로 저장됨
    )

    try:
        results = optimizer.run_optimize(start_prod_seq=start_prod_seq)
        prod_seq_counter = results.get('last_prod_seq', start_prod_seq)
    except Exception as e:
        import traceback
        logging.error(f"[에러] Lot {lot_no} 통합 최적화 중 예외 발생")
        logging.error(traceback.format_exc())
        return None, None, start_prod_seq, start_group_order_no

    if not results or "error" in results:
        error_msg = results.get('error', 'No solution found') if results else 'No solution found'
        logging.error(f"[에러] Lot {lot_no} 통합 최적화 실패: {error_msg}")
        return None, None, start_prod_seq, start_group_order_no

    if "pattern_details_for_db" in results:
        for detail in results["pattern_details_for_db"]:
            detail['max_width'] = max_width

    # --- 5. Sheet Sequence 생성 ---
    # 롤+쉬트 합친 df_orders 생성 (save_results에서 사용)
    all_order_frames = []
    if df_roll_orders is not None:
        all_order_frames.append(df_roll_orders)
    if df_sheet_orders is not None:
        all_order_frames.append(df_sheet_orders)
    df_all_orders = pd.concat(all_order_frames, ignore_index=True) if all_order_frames else pd.DataFrame()

    pattern_sheet_details_for_db = generate_allocated_sheet_details(
        df_all_orders, results.get("pattern_roll_cut_details_for_db", []), b_wgt
    )
    results["pattern_sheet_details_for_db"] = pattern_sheet_details_for_db

    logging.info("통합 최적화 성공.")
    return results, df_all_orders, prod_seq_counter, start_group_order_no


def main():
    """2000 장항공장 메인 실행 함수"""
    db = None
    lot_no = None
    version = None

    try:
        db = init_db('2000')

        while True:
            # --- 대상 Lot 파라미터 조회 (딕셔너리로 반환) ---
            # calc_successful = '9' 인 대상을 자동으로 조회
            lot_params = db.get_target_lot_jh()

            if not lot_params or not lot_params.get('lot_no'):
                time.sleep(10)
                continue

            lot_no = lot_params['lot_no']
            version = lot_params['version']

            setup_logging(lot_no, version)
            db.update_lot_status(lot_no=lot_no, version=version, status=8)

            # --- 롤/쉬트 구분 없이 통합 최적화 ---
            results, df_all_orders, prod_seq_counter, group_order_no_counter = process_lot_jh(
                db, lot_params,
                start_prod_seq=0, start_group_order_no=0
            )

            if results:
                all_results = [results]
                all_df_orders = [df_all_orders]

                final_status = save_results(
                    db, lot_no, version,
                    lot_params['plant'], lot_params['pm_no'], lot_params['schedule_unit'],
                    int(lot_params['re_maxwidth']), lot_params['paper_type'], lot_params['b_wgt'],
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

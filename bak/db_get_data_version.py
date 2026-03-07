import oracledb

class VersionGetters:
    def get_target_lot(self):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """

            # query = """
            #     select 
            #         plant, pm_no, schedule_unit, lot_no, '05' as version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     from th_versions_manager 
            #     where lot_no = '3241100322' and version = '01'   -- 3250900073, 3250900429, 8250900534, 5250900616
            #     and lot_no = '3250900068' and version = '01'
            # """

            query = """
                
                SELECT 
                    a.plant, pm_no, a.schedule_unit, a.lot_no, a.version, a.min_width, a.roll_max_width, 
                    a.sheet_max_width, a.max_re_count as max_pieces, a.max_re_count as sheet_max_pieces,
                    a.paper_type, a.b_wgt, 
                    (select color from sapd12t_tmp s12 where s12.lot_no = a.lot_no and rownum = 1 ) as color, 
                    a.min_sc_width, a.max_sc_width, a.sheet_trim_size, sheet_length_re,
                    ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt,
                    ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type = '1')) as roll_order_cnt,
                    --a.time_limit * 1000 as time_limit
                    case when time_limit < 300 then 300 * 1000 else time_limit * 1000 end as time_limit
                FROM th_versions_manager a, th_tar_std_length b
                WHERE a.plant = b.plant(+)
                AND a.paper_type = b.paper_type(+)
                AND a.b_wgt = b.b_wgt(+)
                --and a.calc_successful = '9'
                and a.lot_no = '3260200110' and a.version = '03'
                and LENGTH(a.version_id) > 0
                and nvl(a.eng_chk, '0') = '1'
                and a.version not in ('98', '99')
                ORDER BY a.plant, a.version_id, a.schedule_unit, a.lot_no, a.version DESC
                FETCH FIRST 1 ROWS ONLY
            """

            # query = """
            #     select 
            #         a.plant, pm_no, a.schedule_unit, a.lot_no, version, a.min_width, a.roll_max_width, 
            #         a.sheet_max_width, a.max_re_count as max_pieces, 4 as sheet_max_pieces,
            #         a.paper_type, a.b_wgt,
            #         750 as min_sc_width, a.max_sc_width, a.sheet_trim_size, sheet_length_re,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type = '1')) as roll_order_cnt
            #     from th_versions_manager a, th_tar_std_length b
            #     where a.plant = b.plant
            #     and a.paper_type = b.paper_type
            #     and a.b_wgt = b.b_wgt 
            #     and lot_no = '3250900073' and version = '01'
            # """

            # query = """
            #     select 
            #         a.plant, pm_no, a.schedule_unit, a.lot_no, '05' as version, a.min_width, a.roll_max_width, 
            #         a.sheet_max_width, a.max_re_count as max_pieces, 4 as sheet_max_pieces,
            #         a.paper_type, a.b_wgt,
            #         a.min_sc_width - 100, a.max_sc_width, a.sheet_trim_size, b.max_length as sheet_length_re,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type = '1')) as roll_order_cnt
            #     from hsfp_st.th_versions_manager@hsfp_st_rlink a, hsfp_st.th_tar_std_length@hsfp_st_rlink b
            #     where a.plant = b.plant(+)
            #     and a.paper_type = b.paper_type(+)
            #     and a.b_wgt = b.b_wgt(+)
            #     and b.operation_code(+) = 'RE'
            #     and b.rs_gubun(+) = 'S'
            #     and lot_no = '8250900534' 
            #     and version = '99'
            # """

            print(f"Executing query to fetch target lot:\n{query}")
            cursor.execute(query)
            result = cursor.fetchone()
            # 반환 값 개수를 20개로 맞춤
            return result if result else (None,) * 20
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 20
        finally:
            if connection:
                self.pool.release(connection)

    def get_target_lot_ca(self):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """
            # 5250900616, 5250900062, 5250900429
            query = """
                SELECT 
                    plant, pm_no, schedule_unit, lot_no, version, time_limit * 1000 as time_limit, paper_type, b_wgt, color, 
                    min_width, roll_max_width, min_sc_width, max_sc_width, coating_yn, 
                    sheet_trim_size, ww_trim_size,
                    min_cm_width, max_cm_width, max_sl_count, p_type, p_wgt, ww_trim_size_sheet,
                    ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt,
                    ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type = '1')) as roll_order_cnt
                FROM th_versions_manager a
                where a.calc_successful = '9'
                -- 5260109085  5260106006  5260104276 5260200180 5260200182  5260200528
                --where lot_no = '5260200180' and version = '01'
                --and nvl(a.eng_chk , '0') = '1'
                and a.version not in ('98', '99')
                ORDER BY a.plant, a.version_id, a.schedule_unit, a.lot_no, a.version
                FETCH FIRST 1 ROWS ONLY
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            # cursor.execute(query)
            cursor.execute(query)
            result = cursor.fetchone()
            # 반환 값 개수를 24개로 맞춤
            return result if result else (None,) * 24
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 24
        finally:
            if connection:
                self.pool.release(connection)

    def get_target_lot_st(self):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """

            query = """
                SELECT MODULE      AS module
				 , PLANT           AS plant
				 , PM_NO           AS pmNo
				 , SCHEDULE_UNIT   AS scheduleUnit
				 , LOT_NO          AS lotNo
				 , VERSION         AS version
                 , ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = t.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt
                 , ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = t.lot_no and fact_status = '3' and pack_type = '1')) as roll_order_cnt
			  FROM 
				   (
				     SELECT T1.MODULE          
					  	  , T1.PLANT           
					  	  , T1.PM_NO          
					  	  , T1.SCHEDULE_UNIT  
					  	  , T1.LOT_NO         
					  	  , T1.VERSION        
					  	  , ROW_NUMBER() OVER (ORDER BY T1.VERSION_ID, T1.LOT_NO, T1.VERSION) AS SEQ
				       FROM TH_VERSIONS_MANAGER T1
					  WHERE 1=1
					    --AND CALC_SUCCESSFUL = '9'		
                        AND T1.LOT_NO = '8241202223' AND T1.VERSION = '01'    -- 8241202223  8241202161
                        AND T1.VERSION NOT IN ('98', '99')
				   ) T
			 WHERE SEQ = 1
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            # cursor.execute(query)
            cursor.execute(query)
            result = cursor.fetchone()
            
            return result if result else (None,) * 8
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 8
        finally:
            if connection:
                self.pool.release(connection)

    def get_target_lot_jh(self):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """

            query = """

                select 
                    v.plant, v.pm_no, v.schedule_unit, v.lot_no, v.version, 
                    v.paper_type, v.b_wgt, v.color,
                    v.min_re_width                              as re_minwidth,                 
                    nvl(v.max_width,4850)                       as re_maxwidth,
                    v.min_re_count                              as re_min_pieces,
                    v.max_re_count                              as re_max_pieces,
                    v.sheet_min_re_pok_cnt                      as re_sheet_min_pieces,
                    nvl(v.sheet_max_re_pok_cnt,4)               as re_sheet_max_pieces,
                    v.min_ptn_wgt                               as re_minton,           -- 패턴 최소중량
                    v.min_cmb_wgt                               as sc_minton,           -- 규격별 최소중량
                    nvl(v.is_std_length, 'N')                   as yn_stdlength,        -- 표준길이 적용여부
                    nvl(l.sheet_length_re,0)                    as min_re_stdlength,    -- 표준길이 최소길이
                    nvl(l.sheet_length_re,0)                    as max_re_stdlength,    -- 표준길이 최대길이
                    v.sheet_order_moq                           as sheet_order_moq,     -- 쉬트 최소주문량
                    v.adjacent_width                            as adj_trim,            -- 인접그룹 조합여부: 값이 0이면 인접그룹 미조합, 0보다 크면 인접그룹 조합
                    v.adjacent_weight_min                       as adj_min_wgt,         -- 인접그룹 조합시 최소 오더톤 제약
                    v.adjacent_weight                           as adj_max_wgt,         -- 인접그룹 조합시 최대 오더톤 제약
                    v.moq_yn                                    as moq_yn,              -- MOQ 미달 지폭여부
                    v.moq_ton                                   as moq_ton,             -- MOQ 오더톤
                    v.moq_sc_width                              as moq_sc_width,        -- MOQ 오더 지폭
                    nvl(v.rs_mix_flag, 'N')                     as yn_rstypemix,        -- RS 혼합 여부

                    v.trim_size                                 as sc_basetrim,
                    v.min_sc_width                              as sc_minwidth,
                    v.max_sc_width                              as sc_maxwidth,
                    1                                           as sc_minsheet,
                    v.max_sc_count                              as sc_maxsheet,
                    nvl(v.ww_trim_size,0)                       as sl_trim,  
                    nvl(v.min_sl_width,765)                     as sl_minwidth,
                    nvl(v.max_sl_width,1800)                    as sl_maxwidth,
                    nvl(v.min_sl_count,1)                       as min_sl_count,
                    nvl(v.max_sl_count,4)                       as max_sl_count,
                    nvl(v.sheet_trim_size_digital, trim_size)   as sc_digital_trim,
                    nvl(v.min_sc_width_digital, 700)            as sc_digital_minwidth,
                    nvl(v.max_sc_width_digital, 1850)           as sc_digital_maxwidth,
                    1                                           as sc_digital_minsheet,
                    nvl(v.max_sc_count_digital, 7)              as sc_digital_maxsheet,  


                    decode(nvl(v.db_cut_rate,0),1,1,0)          as sl_rollmixtype,
                    nvl(v.rs_mix_maxpok, 1)                     as mx_maxsheetroll,
                    case when v.mfc_flag = 'Y'
                            then v.max_non_odd
                        else v.max_sc_width
                    end                                         as mx_maxsheetwidth,
                    nvl(re_rollmixwidth, 4720)                  as mx_rsmixwidth,
                    nvl(v.re_avgcutwidth,v.min_re_width)        as re_avgcutwidth,
                    nvl(v.re_maxpatternratio,1.0)               as re_maxpatternratio,
                    nvl(v.ww_roll_limit_yn, 'N')                as yn_sroll,
                    nvl(v.coating_yn, 'N')                      as yn_coating,
                    r.cutout_width_max                          as re_jumbowidth,


                    nvl(v.min_re_count,2)                       as re_minroll,
                    nvl(v.sheet_max_re_pok_cnt,4)               as re_maxroll,
                    v.max_re_count                              as re_extroll,
                    nvl(v.sheet_std_less,2)                     as less_reg_ton,
                    nvl(v.sheet_std_more,2)                     as more_reg_ton,
                    nvl(v.sheet_nostd_less,0.5)                 as less_var_ton,
                    nvl(v.sheet_nostd_more,0.5)                 as more_var_ton,
                    nvl(v.roll_std_less,2)                      as less_reg_roll,
                    nvl(v.roll_std_more,2)                      as more_reg_roll,
                    nvl(v.roll_nostd_less,0)                    as less_var_roll,
                    nvl(v.roll_nostd_more,0)                    as more_var_roll,
                    -1*nvl(v.time_limit,60)                     as time_order,
                    -1*nvl(v.time_limit,60)                     as time_trim,
                    -1*nvl(v.time_limit,60)                     as time_roll,
                    -1*nvl(v.time_limit,60)                     as time_pattern
                from th_versions_manager v, 
                    th_tar_resource r, 
                    th_tar_std_length l
                where v.calc_successful = '9'
                and v.version not in ('98', '99')
                and r.operation_code = 'PM' 
                and r.resource_code(+) = v.pm_no 
                and l.paper_type(+) = v.paper_type 
                and l.b_wgt(+) = v.b_wgt
                ORDER BY v.plant, v.version_id, v.schedule_unit, v.lot_no, v.version
                FETCH FIRST 1 ROWS ONLY

            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            cursor.execute(query)
            columns = [col[0].lower() for col in cursor.description]
            result = cursor.fetchone()
            if result:
                return dict(zip(columns, result))
            return None
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)
 

    def get_lot_param_sheet_ca(self, lot_no, version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """
            # 5250900616, 5250900062, 5250900429

            query = """
                SELECT 
                    a.plant, a.pm_no, a.schedule_unit, a.lot_no, a.version, a.time_limit * 1000 as time_limit, a.coating_yn, 
                    a.paper_type, a.b_wgt, a.color, 
                    a.p_type, a.p_wgt, a.p_color, a.p_machine,
                    a.min_width, a.roll_max_width, a.min_re_count, a.max_re_count, 
                    d.std_length as sheet_length_re, d.std_roll_cnt, 
                    a.min_sc_width, a.max_sc_width, a.sheet_trim_size,                     
                    a.min_cm_width, a.max_cm_width, a.max_sl_count, a.ww_trim_size, a.ww_trim_size_sheet,
                    'Y' as double_cutter
                FROM th_versions_manager a, th_tar_std_length_ca d 
                where a.plant = d.plant(+) 
                and a.paper_type = d.paper_type 
                and a.b_wgt = d.b_wgt 
                and d.rs_gubun(+) = 'S'
                and a.lot_no = :p_lot_no
                and a.version = :p_version
                ORDER BY a.plant, a.version_id, a.schedule_unit, a.lot_no, a.version
                FETCH FIRST 1 ROWS ONLY
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no, p_version=version)
            result = cursor.fetchone()
            return result if result else (None,) * 26
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 26
        finally:
            if connection:
                self.pool.release(connection)

    def get_lot_param_roll_ca(self, lot_no, version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """
            # 5250900616, 5250900062, 5250900429

            query = """
                SELECT 
                    a.plant, a.pm_no, a.schedule_unit, a.lot_no, a.version, a.time_limit * 1000 as time_limit, a.coating_yn, 
                    a.paper_type, a.b_wgt, a.color, 
                    a.p_type, a.p_wgt, a.p_color, a.p_machine,
                    a.min_width, a.roll_max_width, a.min_re_count, a.max_re_count,
                    a.min_cm_width, a.max_cm_width, a.max_sl_count, a.ww_trim_size_sheet, a.ww_trim_size
                FROM th_versions_manager a
                where a.lot_no = :p_lot_no
                and a.version = :p_version
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no, p_version=version)
            result = cursor.fetchone()
            return result if result else (None,) * 20
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 20
        finally:
            if connection:
                self.pool.release(connection)


    def get_lot_param_roll_st(self, lot_no, version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """
            # 5250900616, 5250900062, 5250900429

            query = """
                    SELECT MODULE, PLANT, PM_NO, SCHEDULE_UNIT, LOT_NO, VERSION 
                    , VERSION_ID        , TIME_LIMIT
                    , PAPER_TYPE, B_WGT, COLOR
                    , DECODE(SHEET_SUBJECT_ORDER, -10,10,1)  as sheet_subject_order
                    , DECODE(SHEET_SUBJECT_WIDTH,10,5,1)     as sheet_subject_width
                    , DECODE(SHEET_SUBJECT_PTN_CNT,10,10,1)  as sheet_subject_ptn_cnt
                    , DECODE(SHEET_SUBJECT_BALANCE, 10,10,1) as sheet_subject_balance
                    , DECODE(SHEET_SUBJECT_ROLL_CNT,10,5,1)  as sheet_subject_roll_cnt
                    , T1.MIN_WIDTH                                    -- RE최소폭
                    , T1.SHEET_MAX_WIDTH                            -- RE최대폭
                    , T1.MIN_RE_COUNT                                -- 롤지 최소폭수
                    , T1.MAX_RE_COUNT                                -- 롤지 최대폭수
                    , T1.SHEET_MIN_RE_POK_CNT                        -- 시트지 최소폭수
                    , T1.SHEET_MAX_RE_POK_CNT                        -- 시트지 최대폭수
                    , CASE WHEN NVL(T1.LENGTH_VERSION_CNT, 0) = 0 THEN T1.LENGTH_INPUT 
                            ELSE 
                                (SELECT MIN_LENGTH 
                                FROM TH_TAR_STD_LENGTH                    
                                WHERE PLANT            = '8000'
                                AND OPERATION_CODE   = 'RE'
                                AND RS_GUBUN         = 'S'
                                AND PAPER_TYPE       = T1.PAPER_TYPE
                                AND B_WGT            = T1.B_WGT
                                )                         
                        END AS MIN_LENGTH        -- 시트지 최소 표준길이
                    , CASE WHEN NVL(T1.LENGTH_VERSION_CNT, 0) = 0 THEN T1.LENGTH_INPUT 
                            ELSE
                                (SELECT MAX_LENGTH
                                FROM TH_TAR_STD_LENGTH
                                WHERE PLANT            = '8000'
                                AND OPERATION_CODE   = 'RE'
                                AND RS_GUBUN         = 'S'
                                AND PAPER_TYPE       = T1.PAPER_TYPE
                                AND B_WGT            = T1.B_WGT
                                )                         
                        END AS MAX_LENGTH        -- 시트지 최대 표준길이
                    , CASE WHEN T1.LENGTH_VERSION_CNT = 0 THEN 1 ELSE T1.LENGTH_VERSION_CNT END LENGTH_VERSION_CNT        -- 시트지 표준길이 갯수
                    , T1.RS_MIX_FLAG                                -- RS혼합패턴 허용 여부 ('Y' = 허용)
                    , NVL(T1.LENGTH_SIMILAR,300) as LENGTH_SIMILAR                   -- 롤지 유사길이 대상 차이 (DEFAULT = 300M)
                    , T1.ALLOW_MULTIPLE                            -- 롤지 가능 배수
                    , T1.ALLOW_MULTIPLE_WIDTH                        -- 롤지 배수 적용 최소 RE지폭
                    , T1.YEOPOK_YN                                    -- 사이드런 적용 여부
                    , NVL(T1.RS_MIX_MAXPOK,3)  as RS_MIX_MAXPOK                  -- RS혼합 패턴 시트 권취수 제한 (DEFALUT : 3폭)
                    , (SELECT MIN(UN_WINDER_MIN)
                        FROM TH_TAR_RESOURCE
                        WHERE PLANT            = '8000'                        
                        AND OPERATION_CODE   = 'SL'
                    )                                AS SL_MIN_WIDTH            -- SL 최소지폭
                    , (SELECT MAX(UN_WINDER_MAX)    
                        FROM TH_TAR_RESOURCE    
                        WHERE PLANT            = '8000'    
                        AND OPERATION_CODE   = 'SL'    
                    )                                          AS SL_MAX_WIDTH            -- SL 최대지폭
                    , 1                                        AS SL_MIN_POK_CNT            -- SL 최소폭수
                    , T1.MAX_SL_COUNT                          AS SL_MAX_POK_CNT            -- SL 최대폭수
                    , T1.WW_TRIM_SIZE                          AS SL_TRIM_SIZE              -- SL 기본트림
                    , 0                                        AS SL_INCREASE_TRIM        -- SL 추가트림(폭수에 따른)
                    , NVL(SL_MAX_TRIM, 80)                     AS SL_MAX_TRIM             -- SL 최대트림
                    , NVL(T1.SL_MIX_YN, 'N')                   AS SL_MIX_YN            -- 권취롤 내 다른폭 혼합 가능여부
                    , NVL(SL_MIX_SHORT_SHORT, 'N')             AS SL_SROLL_MIX_YN      -- 단폭 + 단폭 권취롤 MIX여부  -- HARD
                    , NVL(SL_MIX_SHORT_WIDTH, 500)             AS SL_SROLL_WIDTH       -- 단폭 + 단폭 MIX시 단폭 기준 -- HARD
                    , NVL(SL_MIX_SHORT_LONG_MINWIDTH, 370)     AS SL_SROLL_WIDTHMIN    -- 단폭 + 장폭 MIX시 단폭 기준 -- HARD
                    , NVL(SL_MIX_SHORT_LONG_MAXWIDTH, 1000)    AS SL_SROLL_WIDTHMAX    -- 단폭 + 장폭 MIX시 장폭 기준 --HARD
                    , NVL(SL_WIDTHMIX_YN,'N')                  AS SL_SINGLE_YN          -- 단폭만 허용 여부 
                    , NVL(SL_ONEPOK_TRIM_YN, 'N')              AS SL_SINGLE_TRIM_YN     -- 단폭 트림 적용 여부
                    , T1.SC_UNWINDER_MIN                                        -- SC 최소지폭
                    , T1.SC_UNWINDER_MAX                                        -- SC 최대지폭
                    , 1                                        AS SC_MIN_POK_CNT         -- SC 최소폭수
                    , SC_MAX_POK_CNT                           AS SC_MAX_POK_CNT         -- SC 최대폭수
                    , 4500                             AS MAX_UNIT_WEIGHT      -- SC 단일권취 최대 중량    -- HARD
                    , SHEET_TRIM_SIZE                                            -- SC 기본트림
                    , NVL(SHEET_ADD_TRIM_SIZE, 1)        AS INCREASE_TRIM        -- SC 추가트림(폭수에 따른)
                    , NVL(SC_MAX_TRIM, 80)              as SC_MAX_TRIM                          -- SC 최대트림
                    , (SELECT ROLL_CNT_MIN 
                        FROM TH_TAR_KNIFE_LOAD
                        WHERE PAPER_TYPE    = T1.PAPER_TYPE
                        AND B_WGT         = T1.B_WGT
                    )                                 AS KNIFE_MIN            -- Knife-load 최소
                    , (SELECT ROLL_CNT_MAX
                        FROM TH_TAR_KNIFE_LOAD
                        WHERE PAPER_TYPE    = T1.PAPER_TYPE
                        AND B_WGT         = T1.B_WGT
                    )                                AS KNIFE_MAX            -- Knife-load 최대
                    , CASE WHEN SC_MIX_YN IN ('N','A','W')
                            THEN 'N'
                            ELSE SC_MIX_YN
                    END                              AS MIX_TYPE             -- 권취롤내에 가로규격은 같으나 세로규격이 다른 주문 배정 여부 (H: 세로규격이 달라도 배정 가능)
                FROM TH_VERSIONS_MANAGER T1  
                WHERE T1.LOT_NO            = :p_lot_no
                AND T1.VERSION           = :p_version
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no, p_version=version)
            result = cursor.fetchone()
            return result if result else (None,) * 53
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 53
        finally:
            if connection:
                self.pool.release(connection)



    def get_lot_param_sheet_st(self, lot_no, version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """
            # 5250900616, 5250900062, 5250900429

            query = """
                    SELECT MODULE, PLANT, PM_NO, SCHEDULE_UNIT, LOT_NO, VERSION 
                    , VERSION_ID        , TIME_LIMIT
                    , PAPER_TYPE, B_WGT, COLOR
                    , DECODE(SHEET_SUBJECT_ORDER, -10,10,1)  as sheet_subject_order
                    , DECODE(SHEET_SUBJECT_WIDTH,10,5,1)     as sheet_subject_width
                    , DECODE(SHEET_SUBJECT_PTN_CNT,10,10,1)  as sheet_subject_ptn_cnt
                    , DECODE(SHEET_SUBJECT_BALANCE, 10,10,1) as sheet_subject_balance
                    , DECODE(SHEET_SUBJECT_ROLL_CNT,10,5,1)  as sheet_subject_roll_cnt
                    , T1.MIN_WIDTH                                    -- RE최소폭
                    , T1.SHEET_MAX_WIDTH                            -- RE최대폭
                    , T1.MIN_RE_COUNT                                -- 롤지 최소폭수
                    , T1.MAX_RE_COUNT                                -- 롤지 최대폭수
                    , T1.SHEET_MIN_RE_POK_CNT                        -- 시트지 최소폭수
                    , T1.SHEET_MAX_RE_POK_CNT                        -- 시트지 최대폭수
                    , CASE WHEN NVL(T1.LENGTH_VERSION_CNT, 0) = 0 THEN T1.LENGTH_INPUT 
                            ELSE 
                                (SELECT MIN_LENGTH 
                                FROM TH_TAR_STD_LENGTH                    
                                WHERE PLANT            = '8000'
                                AND OPERATION_CODE   = 'RE'
                                AND RS_GUBUN         = 'S'
                                AND PAPER_TYPE       = T1.PAPER_TYPE
                                AND B_WGT            = T1.B_WGT
                                )                         
                        END AS MIN_LENGTH        -- 시트지 최소 표준길이
                    , CASE WHEN NVL(T1.LENGTH_VERSION_CNT, 0) = 0 THEN T1.LENGTH_INPUT 
                            ELSE
                                (SELECT MAX_LENGTH
                                FROM TH_TAR_STD_LENGTH
                                WHERE PLANT            = '8000'
                                AND OPERATION_CODE   = 'RE'
                                AND RS_GUBUN         = 'S'
                                AND PAPER_TYPE       = T1.PAPER_TYPE
                                AND B_WGT            = T1.B_WGT
                                )                         
                        END AS MAX_LENGTH        -- 시트지 최대 표준길이
                    , CASE WHEN T1.LENGTH_VERSION_CNT = 0 THEN 1 ELSE T1.LENGTH_VERSION_CNT END LENGTH_VERSION_CNT        -- 시트지 표준길이 갯수
                    , T1.RS_MIX_FLAG                                -- RS혼합패턴 허용 여부 ('Y' = 허용)
                    , NVL(T1.LENGTH_SIMILAR,300) as LENGTH_SIMILAR                   -- 롤지 유사길이 대상 차이 (DEFAULT = 300M)
                    , T1.ALLOW_MULTIPLE                            -- 롤지 가능 배수
                    , T1.ALLOW_MULTIPLE_WIDTH                        -- 롤지 배수 적용 최소 RE지폭
                    , T1.YEOPOK_YN                                    -- 사이드런 적용 여부
                    , NVL(T1.RS_MIX_MAXPOK,3)  as RS_MIX_MAXPOK                  -- RS혼합 패턴 시트 권취수 제한 (DEFALUT : 3폭)
                    , (SELECT MIN(UN_WINDER_MIN)
                        FROM TH_TAR_RESOURCE
                        WHERE PLANT            = '8000'                        
                        AND OPERATION_CODE   = 'SL'
                    )                                AS SL_MIN_WIDTH            -- SL 최소지폭
                    , (SELECT MAX(UN_WINDER_MAX)    
                        FROM TH_TAR_RESOURCE    
                        WHERE PLANT            = '8000'    
                        AND OPERATION_CODE   = 'SL'    
                    )                                          AS SL_MAX_WIDTH            -- SL 최대지폭
                    , 1                                        AS SL_MIN_POK_CNT            -- SL 최소폭수
                    , T1.MAX_SL_COUNT                          AS SL_MAX_POK_CNT            -- SL 최대폭수
                    , T1.WW_TRIM_SIZE                          AS SL_TRIM_SIZE              -- SL 기본트림
                    , 0                                        AS SL_INCREASE_TRIM        -- SL 추가트림(폭수에 따른)
                    , NVL(SL_MAX_TRIM, 80)                     AS SL_MAX_TRIM             -- SL 최대트림
                    , NVL(T1.SL_MIX_YN, 'N')                   AS SL_MIX_YN            -- 권취롤 내 다른폭 혼합 가능여부
                    , NVL(SL_MIX_SHORT_SHORT, 'N')             AS SL_SROLL_MIX_YN      -- 단폭 + 단폭 권취롤 MIX여부  -- HARD
                    , NVL(SL_MIX_SHORT_WIDTH, 500)             AS SL_SROLL_WIDTH       -- 단폭 + 단폭 MIX시 단폭 기준 -- HARD
                    , NVL(SL_MIX_SHORT_LONG_MINWIDTH, 370)     AS SL_SROLL_WIDTHMIN    -- 단폭 + 장폭 MIX시 단폭 기준 -- HARD
                    , NVL(SL_MIX_SHORT_LONG_MAXWIDTH, 1000)    AS SL_SROLL_WIDTHMAX    -- 단폭 + 장폭 MIX시 장폭 기준 --HARD
                    , NVL(SL_WIDTHMIX_YN,'N')                  AS SL_SINGLE_YN          -- 단폭만 허용 여부 
                    , NVL(SL_ONEPOK_TRIM_YN, 'N')              AS SL_SINGLE_TRIM_YN     -- 단폭 트림 적용 여부
                    , T1.SC_UNWINDER_MIN                                        -- SC 최소지폭
                    , T1.SC_UNWINDER_MAX                                        -- SC 최대지폭
                    , 1                                        AS SC_MIN_POK_CNT         -- SC 최소폭수
                    , SC_MAX_POK_CNT                           AS SC_MAX_POK_CNT         -- SC 최대폭수
                    , 4500                             AS MAX_UNIT_WEIGHT      -- SC 단일권취 최대 중량    -- HARD
                    , SHEET_TRIM_SIZE                                            -- SC 기본트림
                    , NVL(SHEET_ADD_TRIM_SIZE, 1)        AS INCREASE_TRIM        -- SC 추가트림(폭수에 따른)
                    , NVL(SC_MAX_TRIM, 80)              as SC_MAX_TRIM                          -- SC 최대트림
                    , (SELECT ROLL_CNT_MIN 
                        FROM TH_TAR_KNIFE_LOAD
                        WHERE PAPER_TYPE    = T1.PAPER_TYPE
                        AND B_WGT         = T1.B_WGT
                    )                                 AS KNIFE_MIN            -- Knife-load 최소
                    , (SELECT ROLL_CNT_MAX
                        FROM TH_TAR_KNIFE_LOAD
                        WHERE PAPER_TYPE    = T1.PAPER_TYPE
                        AND B_WGT         = T1.B_WGT
                    )                                AS KNIFE_MAX            -- Knife-load 최대
                    , CASE WHEN SC_MIX_YN IN ('N','A','W')
                            THEN 'N'
                            ELSE SC_MIX_YN
                    END                              AS MIX_TYPE             -- 권취롤내에 가로규격은 같으나 세로규격이 다른 주문 배정 여부 (H: 세로규격이 달라도 배정 가능)
                FROM TH_VERSIONS_MANAGER T1  
                WHERE T1.LOT_NO            = :p_lot_no
                AND T1.VERSION           = :p_version
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no, p_version=version)
            result = cursor.fetchone()
            return result if result else (None,) * 53
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 53
        finally:
            if connection:
                self.pool.release(connection)

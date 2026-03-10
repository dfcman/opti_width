import oracledb

class VersionGettersSt:
    def get_target_lot_st(self):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

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
                        AND T1.LOT_NO = '8260100259' AND T1.VERSION = '01'    -- 8241202223  8241202161  8241200998  8260100259 8260100262
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

    def get_lot_param_roll_st(self, lot_no, version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

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

    def get_roll_orders_from_db_st(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            #
            sql_query = """
                SELECT
                    plant, pm_no, schedule_unit, width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no,
                    core, dia, nvl(pattern, ' ') as pattern, luster, color
                FROM
                    h3t_production_order
                WHERE paper_prod_seq = :p_paper_prod_seq
                  AND rs_gubun = 'R'
                  --AND WIDTH NOT IN ('797', '1020')
                ORDER BY roll_length, width, dia, core
            """
            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no, core, dia, pattern, luster, color = row
                export_type = '수출' if export_yn == 'Y' else '내수'
                raw_orders.append({
                    'plant': plant,
                    'pm_no': pm_no,
                    'schedule_unit': schedule_unit,
                    'order_no': order_no,
                    '지폭': int(width),
                    '가로': int(length),
                    '롤길이': int(roll_length),
                    '주문수량': int(order_roll_cnt),
                    '주문톤': float(order_ton_cnt),
                    '등급': quality_grade,
                    '수출내수': export_type,
                    'core': core,
                    'dia': dia,
                    'luster': luster,
                    'color': color,
                    'order_pattern': pattern
                })
            print(f"Successfully fetched {len(raw_orders)} roll orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting get_roll_orders_from_db orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_orders_from_db_st(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

            sql_query = """
                SELECT
                    a.plant, a.pm_no, a.schedule_unit, a.width, a.length, a.quality_grade, 
                    a.order_ton_cnt, a.export_yn, a.order_no, a.color, a.order_gubun, a.pt_gubun, b.pack_type, nvl(a.pattern, ' ') as pattern
                FROM
                    h3t_production_order a, sapd12t_tmp b
                WHERE a.paper_prod_seq = :p_paper_prod_seq
                  AND a.rs_gubun = 'S'
                  and a.order_no = b.order_no
                ORDER BY width, length
            """

            # print(f"Executing query to fetch sheet orders:\n{sql_query}")
            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color, order_gubun, pt_gubun, pack_type, order_pattern = row
                export_type = '수출' if export_yn == 'Y' else '내수'
                raw_orders.append({
                    'plant': plant,
                    'pm_no': pm_no,
                    'schedule_unit': schedule_unit,
                    'order_no': order_no,
                    '가로': int(width),
                    '세로': int(length),
                    '주문톤': float(order_ton_cnt),
                    '등급': quality_grade,
                    '수출내수': export_type,
                    'color': color,
                    'order_gubun': order_gubun,
                    'pt_gubun': pt_gubun,
                    'pack_type': pack_type,
                    'order_pattern': order_pattern
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet get_sheet_orders_from_db_st from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_patterns_from_db_st(self, lot_no):
        """th_pattern_tot_sheet 테이블에서 현재 오더의 지종/평량에 해당하는 모든 패턴을 가져옵니다.
        
        ST 공장은 단폭 아이템을 사용하므로,
        SQL 단계에서는 지종/평량 일치 여부만 확인하고
        상세한 유효성 검증(현재 오더에 포함된 지폭인지 등)은 Python 로직에서 수행합니다.
        """
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            
            # ST는 단폭 처리가 복잡하므로, 일단 해당 지종/평량의 모든 사용자 패턴을 가져옵니다.
            query = """
                SELECT DISTINCT
                    width1, width2, width3, width4, 
                    width5, width6, width7, width8
                FROM th_pattern_tot_sheet
                WHERE (paper_type, b_wgt) IN (
                    SELECT paper_type, b_wgt
                    FROM h3t_production_order
                    WHERE paper_prod_seq = :lot_no AND ROWNUM = 1
                )
            """
            cursor.execute(query, lot_no=lot_no)
            rows = cursor.fetchall()
            
            db_patterns = []
            for row in rows:
                # None이나 빈 문자열이 아닌 유효한 아이템명만 필터링합니다.
                pattern_items = [item for item in row if item]
                if pattern_items:
                    db_patterns.append(pattern_items)
            
            print(f"Successfully fetched {len(db_patterns)} sheet patterns from ST DB for lot {lot_no}")
            return db_patterns
        except oracledb.Error as error:
            print(f"Error while fetching ST sheet patterns from DB: {error}")
            return []
        finally:
            if connection:
                self.pool.release(connection)

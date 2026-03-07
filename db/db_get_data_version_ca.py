import oracledb

class VersionGettersCa:
    def get_target_lot_ca(self):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
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

    def get_lot_param_sheet_ca(self, lot_no, version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
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

    def get_roll_orders_from_db_ca(self, plant, pm_no, schedule_unit, paper_prod_seq):
        """
        오라클 패키지 PKG_JP_INOUT_MANAGER.SP_JP_PRODUCTION_ORDER의 로직을
        직접 쿼리로 구현하여 롤지 오더 데이터를 가져옵니다.
        """
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            
            # Step 1: V_CA_SL_CNT 값 조회
            ca_sl_cnt_query = """
                SELECT COUNT(*) as cnt FROM TH_MST_COMMONCODE 
                WHERE GEN_TYPE = 'CA_RP_SL' 
                AND GEN_CODE IN (SELECT PAPER_TYPE FROM TH_VERSIONS_MANAGER WHERE LOT_NO = :p_lot_no)
            """
            cursor.execute(ca_sl_cnt_query, p_lot_no=paper_prod_seq)
            v_ca_sl_cnt = cursor.fetchone()[0]
            
            if v_ca_sl_cnt == 0:
                # TH_TAR_ITEM 테이블에서 추가 확인
                ca_sl_cnt_query2 = """
                    SELECT COUNT(*) as cnt FROM TH_TAR_ITEM
                    WHERE PAPER_TYPE IN (SELECT PAPER_TYPE FROM TH_VERSIONS_MANAGER WHERE LOT_NO = :p_lot_no)
                    AND OPERATION_CODE = 'EM'
                """
                cursor.execute(ca_sl_cnt_query2, p_lot_no=paper_prod_seq)
                v_ca_sl_cnt = cursor.fetchone()[0]
            
            # Step 2: 메인 오더 쿼리 (V_CA_SL_CNT 값 적용)
            main_query = """
                SELECT PM_NO, SCHEDULE_UNIT, ORDER_NO, PAPER_PROD_SEQ, RS_GUBUN, 
                       EXPORT_YN, PAPER_TYPE, B_WGT, WIDTH, LENGTH, SKID_YN, DIA, CORE, 
                       order_roll_cnt, order_ton_cnt, QUALITY_GRADE, PT_GUBUN, ROLL_LENGTH, 
                       std_length, DIR_GUBUN, COLOR, LUSTER, CM_NO, PATTERN, ITCHAR
                FROM (       
                    SELECT PM_NO, SCHEDULE_UNIT, ORDER_NO, PAPER_PROD_SEQ, RS_GUBUN, 
                           EXPORT_YN, PAPER_TYPE, B_WGT, WIDTH, LENGTH, SKID_YN, DIA, CORE, 
                           order_roll_cnt, order_ton_cnt, QUALITY_GRADE, PT_GUBUN, ROLL_LENGTH, 
                           std_length, DIR_GUBUN, COLOR, LUSTER, CM_NO, PATTERN, ITCHAR, STD_CNT
                    FROM (       
                        SELECT DISTINCT f.p_machine AS pm_no, 
                               b.schedule_unit, 
                               b.order_no, 
                               b.paper_prod_seq, 
                               b.rs_gubun,
                               b.export_yn, 
                               b.paper_type, 
                               b.b_wgt, 
                               b.width, 
                               b.length,
                               b.skid_yn, 
                               b.dia, 
                               b.core, 
                               b.order_roll_cnt, 
                               b.order_ton_cnt,
                               CASE WHEN b.rs_gubun = 'R' AND b.roll_direction = '2' THEN 'Z6'
                                    WHEN c.dir_gubun = 'ZARL' THEN c.dir_gubun || NVL(e.jipok_group, b.quality_grade)                                     
                                    ELSE NVL(e.jipok_group, b.quality_grade) 
                               END AS quality_grade,
                               '' AS pt_gubun,
                               NVL(b.roll_length, 0) AS roll_length,
                               CASE WHEN :v_ca_sl_cnt > 0 THEN d.std_length
                                    WHEN SUBSTR(b.pm_no, 1, 2) = 'CM' AND b.rs_gubun = 'S' THEN d.std_length
                                    WHEN SUBSTR(b.pm_no, 1, 2) = 'CM' AND b.rs_gubun = 'R' THEN d.std_length
                                    WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' AND b.rs_gubun = 'S' THEN d.std_length
                                    WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' AND b.rs_gubun = 'R' THEN NVL(b.roll_length, 0)
                                    WHEN SUBSTR(b.pm_no, 1, 2) = 'SP' AND b.rs_gubun = 'R' THEN d.std_length
                                    ELSE 0
                               END AS std_length,
                               c.dir_gubun,
                               NVL(b.color, '0') AS color, 
                               NVL(b.luster, '0') AS luster,
                               CASE WHEN :v_ca_sl_cnt > 0 THEN b.pm_no
                                    WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' THEN ''
                                    ELSE b.pm_no
                               END AS cm_no,
                               NVL(b.pattern, ' ') AS pattern, 
                               NVL(b.itchar, '0') AS itchar,
                               NVL(g.std_cnt, 1) AS std_cnt
                        FROM SAPD12T_TMP a, 
                             H3T_PRODUCTION_ORDER b, 
                             SAPD11T_TMP c, 
                             TH_TAR_STD_LENGTH_CA d, 
                             H3T_PRODUCTION_ORDER_PARAM e,
                             (SELECT module, plant, schedule_unit, lot_no, p_machine 
                              FROM TH_VERSIONS_MANAGER
                              WHERE plant = :p_plant
                              AND schedule_unit = :p_schedule_unit
                              AND lot_no = :p_lot_no
                              AND version IN (SELECT MAX(version) 
                                              FROM TH_VERSIONS_MANAGER
                                              WHERE plant = :p_plant
                                              AND schedule_unit = :p_schedule_unit
                                              AND lot_no = :p_lot_no
                                              AND version < '99')
                             ) f,
                             (SELECT plant, paper_type, b_wgt, rs_gubun, order_length, COUNT(*) AS std_cnt 
                              FROM TH_TAR_STD_LENGTH_CA
                              WHERE plant = :p_plant
                              GROUP BY plant, paper_type, b_wgt, rs_gubun, order_length
                             ) g 
                        WHERE b.order_no = a.order_no
                        AND b.plant = :p_plant
                        AND a.fact_status = '3'
                        AND b.pm_no = :p_pm_no
                        AND b.schedule_unit = :p_schedule_unit
                        AND b.paper_prod_seq = :p_lot_no
                        AND a.req_no = c.req_no(+)
                        AND b.paper_type = d.paper_type(+)
                        AND b.b_wgt = d.b_wgt(+)
                        AND b.rs_gubun = d.rs_gubun(+)
                        AND b.roll_length = DECODE(NVL(d.roll_apply_length(+), '0'), 0, d.order_length(+), d.roll_apply_length(+))
                        AND b.plant = e.plant(+)
                        AND b.order_no = e.order_no(+)
                        AND b.plant = f.plant
                        AND b.schedule_unit = f.schedule_unit
                        AND b.paper_prod_seq = f.lot_no
                        AND b.paper_type = g.paper_type(+)
                        AND b.b_wgt = g.b_wgt(+)
                        AND b.rs_gubun = g.rs_gubun(+)
                        AND b.rs_gubun = 'R'
                        AND CASE WHEN :v_ca_sl_cnt > 0 THEN b.roll_length
                                 WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' THEN 0
                                 WHEN SUBSTR(b.pm_no, 1, 2) = 'SP' THEN 0
                                 ELSE b.roll_length
                            END = g.order_length(+)
                    )  
                    WHERE std_cnt < 2
                    ----------------
                    UNION ALL 
                    ----------------
                    SELECT f.p_machine AS pm_no, 
                           b.schedule_unit, 
                           b.order_no, 
                           b.paper_prod_seq, 
                           b.rs_gubun,
                           b.export_yn, 
                           b.paper_type, 
                           b.b_wgt, 
                           b.width, 
                           b.length,
                           b.skid_yn, 
                           b.dia, 
                           b.core, 
                           b.order_roll_cnt, 
                           b.order_ton_cnt,
                           CASE WHEN b.rs_gubun = 'R' AND b.roll_direction = '2' THEN 'Z6'
                                WHEN c.dir_gubun = 'ZARL' THEN c.dir_gubun || NVL(e.jipok_group, b.quality_grade)                                     
                                ELSE NVL(e.jipok_group, b.quality_grade) 
                           END AS quality_grade,
                           '' AS pt_gubun,
                           NVL(b.roll_length, 0) AS roll_length,
                           CASE WHEN :v_ca_sl_cnt > 0 THEN d.std_length
                                WHEN SUBSTR(b.pm_no, 1, 2) = 'CM' AND b.rs_gubun = 'S' THEN d.std_length
                                WHEN SUBSTR(b.pm_no, 1, 2) = 'CM' AND b.rs_gubun = 'R' THEN d.std_length
                                WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' AND b.rs_gubun = 'S' THEN d.std_length
                                WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' AND b.rs_gubun = 'R' THEN NVL(b.roll_length, 0)
                                WHEN SUBSTR(b.pm_no, 1, 2) = 'SP' AND b.rs_gubun = 'R' THEN d.std_length
                                ELSE 0
                           END AS std_length,
                           c.dir_gubun,
                           NVL(b.color, '0') AS color, 
                           NVL(b.luster, '0') AS luster,
                           CASE WHEN :v_ca_sl_cnt > 0 THEN b.pm_no
                                WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' THEN ''
                                ELSE b.pm_no
                           END AS cm_no,
                           NVL(b.pattern, ' ') AS pattern, 
                           NVL(b.itchar, '0') AS itchar,
                           g.std_cnt
                    FROM SAPD12T_TMP a, 
                         H3T_PRODUCTION_ORDER b, 
                         SAPD11T_TMP c, 
                         TH_TAR_STD_LENGTH_CA d, 
                         H3T_PRODUCTION_ORDER_PARAM e,
                         (SELECT module, plant, schedule_unit, lot_no, p_machine 
                          FROM TH_VERSIONS_MANAGER
                          WHERE plant = :p_plant
                          AND schedule_unit = :p_schedule_unit
                          AND lot_no = :p_lot_no
                          AND version IN (SELECT MAX(version) 
                                          FROM TH_VERSIONS_MANAGER
                                          WHERE plant = :p_plant
                                          AND schedule_unit = :p_schedule_unit
                                          AND lot_no = :p_lot_no
                                          AND version < '99')
                         ) f,
                         (SELECT plant, paper_type, b_wgt, rs_gubun, order_length, COUNT(*) AS std_cnt 
                          FROM TH_TAR_STD_LENGTH_CA
                          WHERE plant = :p_plant
                          GROUP BY plant, paper_type, b_wgt, rs_gubun, order_length
                          HAVING COUNT(*) > 1
                         ) g 
                    WHERE b.order_no = a.order_no
                    AND b.plant = :p_plant
                    AND a.fact_status = '3'
                    AND b.pm_no = :p_pm_no
                    AND b.schedule_unit = :p_schedule_unit
                    AND b.paper_prod_seq = :p_lot_no
                    AND a.req_no = c.req_no(+)
                    AND b.paper_type = d.paper_type(+)
                    AND b.b_wgt = d.b_wgt(+)
                    AND b.rs_gubun = d.rs_gubun(+)
                    AND b.rs_gubun = 'R'
                    AND CASE WHEN :v_ca_sl_cnt > 0 THEN b.roll_length
                             WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' THEN 0
                             WHEN SUBSTR(b.pm_no, 1, 2) = 'SP' THEN 0
                             ELSE b.roll_length
                        END = DECODE(NVL(d.roll_apply_length(+), '0'), 0, d.order_length(+), d.roll_apply_length(+))
                    AND b.plant = e.plant(+)
                    AND b.order_no = e.order_no(+)
                    AND b.plant = f.plant
                    AND b.schedule_unit = f.schedule_unit
                    AND b.paper_prod_seq = f.lot_no
                    AND b.paper_type = g.paper_type
                    AND b.b_wgt = g.b_wgt
                    AND b.rs_gubun = g.rs_gubun
                    AND g.order_length(+) = CASE WHEN :v_ca_sl_cnt > 0 THEN b.roll_length
                                                  WHEN SUBSTR(b.pm_no, 1, 2) = 'PM' THEN 0
                                                  WHEN SUBSTR(b.pm_no, 1, 2) = 'SP' THEN 0
                                                  ELSE b.roll_length
                                             END
                    AND g.std_cnt > 1
                    AND DECODE(SUBSTR(b.pm_no, 1, 4), 'CM51', d.STD_LENGTH_CM51, 'CM52', d.STD_LENGTH_CM52, 'CM53', d.STD_LENGTH_CM53, 'CM54', d.STD_LENGTH_CM52) > 0
                )
                ORDER BY pm_no, schedule_unit, paper_prod_seq, export_yn, rs_gubun
            """
            
            cursor.execute(main_query, 
                           v_ca_sl_cnt=v_ca_sl_cnt,
                           p_plant=plant,
                           p_pm_no=pm_no,
                           p_schedule_unit=schedule_unit,
                           p_lot_no=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            
            for row in rows:
                (pm_no_out, schedule_unit_out, order_no, paper_prod_seq_out, rs_gubun, export_yn,
                 paper_type, b_wgt, width, length, skid_yn, dia, core,
                 order_roll_cnt, order_ton_cnt, quality_grade, pt_gubun, roll_length,
                 std_length, dir_gubun, color, luster, cm_no, pattern, itchar) = row
                
                export_type = '수출' if export_yn == 'Y' else '내수'
                
                raw_orders.append({
                    'plant': plant,
                    'pm_no': pm_no_out,
                    'schedule_unit': schedule_unit_out,
                    'order_no': order_no,
                    'paper_type': paper_type,
                    'b_wgt': b_wgt,
                    '지폭': int(width) if width else 0,
                    '가로': int(length) if length else 0,
                    '주문수량': int(order_roll_cnt) if order_roll_cnt else 0,
                    '주문톤': float(order_ton_cnt) if order_ton_cnt else 0,
                    '롤길이': int(roll_length) if roll_length else 0,
                    '등급': quality_grade,
                    '수출내수': export_type,
                    'core': core,
                    'dia': dia,
                    'luster': luster,
                    'color': color,
                    'order_pattern': pattern,
                    'pack_type': pt_gubun,
                    'rs_gubun': rs_gubun,
                    'skid_yn': skid_yn,
                    'cm_no': cm_no,
                    'dir_gubun': dir_gubun,
                    'std_length': int(std_length) if std_length else 0,
                })
            
            print(f"Successfully fetched {len(raw_orders)} roll orders for lot {paper_prod_seq} (V_CA_SL_CNT={v_ca_sl_cnt})")
            return raw_orders
            
        except oracledb.Error as error:
            print(f"Error while getting get_roll_orders_from_db_ca orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    
    def get_sheet_orders_from_db_ca(self, paper_prod_seq):
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

            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color, order_gubun, pt_gubun, pack_type, pattern = row
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
                    'order_pattern': pattern
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_roll_orders_from_db_ca_bak(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # nvl(b.jipok_group, a.quality_grade) as quality_grade
            sql_query = """
                select
                    a.plant, a.pm_no, a.schedule_unit, a.paper_type, a.b_wgt, a.width, a.length, a.roll_length, 
                    a.quality_grade as quality_grade, a.order_roll_cnt, 
                    a.order_ton_cnt, a.export_yn, a.order_no,
                    a.core, a.dia, nvl(a.pattern, ' ') as pattern, a.luster, a.color, 
                    e.pack_type,
                    a.rs_gubun, a.nation_code, a.customer_name, a.skid_yn, e.pte_gubun
                from
                    h3t_production_order a, h3t_production_order_param b, sapd12t_tmp e, sapd11t_tmp f
                where a.order_no = b.order_no(+)
                  and a.paper_prod_seq = :p_paper_prod_seq
                  and a.rs_gubun = 'R'
                  and a.order_no = e.order_no
                  and e.req_no = f.req_no
                order by a.roll_length, a.width, a.dia, a.core
            """
            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, paper_type, b_wgt, width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no, core, dia, pattern, luster, color, pack_type, rs_gubun, nation_code, customer_name, skid_yn, pte_gubun = row
                export_type = '수출' if export_yn == 'Y' else '내수'
                raw_orders.append({
                    'plant': plant,
                    'pm_no': pm_no,
                    'schedule_unit': schedule_unit,
                    'order_no': order_no,
                    'paper_type': paper_type,
                    'b_wgt': b_wgt,
                    '지폭': int(width),
                    '가로': int(length),
                    '주문수량': int(order_roll_cnt),
                    '주문톤': float(order_ton_cnt),
                    '롤길이': int(roll_length),
                    '등급': quality_grade,
                    '수출내수': export_type,
                    'core': core,
                    'dia': dia,
                    'luster': luster,
                    'color': color,
                    'order_pattern': pattern,
                    'pack_type': pack_type,
                    'rs_gubun': rs_gubun,
                    'nation_code': nation_code,
                    'customer_name': customer_name,
                    'skid_yn': skid_yn,
                    'pte_gubun': pte_gubun,
                })
            print(f"Successfully fetched {len(raw_orders)} roll orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting get_roll_orders_from_db_ca orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

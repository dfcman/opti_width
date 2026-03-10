import oracledb

class VersionGettersDj:
    def get_target_lot(self):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
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

    def get_roll_orders_from_db(self, paper_prod_seq):
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
                    c.addchart, nvl(d.gen_hcode, ' ') as sep_qt, e.pack_type,
                    a.rs_gubun, a.nation_code, a.customer_name, a.skid_yn, e.pte_gubun
                from
                    h3t_production_order a, h3t_production_order_param b, batch_master@paper33_link c, th_mst_commoncode d, sapd12t_tmp e, sapd11t_tmp f
                where a.order_no = b.order_no(+)
                and paper_prod_seq = :p_paper_prod_seq
                  and a.rs_gubun = 'R'
                  and a.material_no = c.matnr(+)
                  and a.batch_no = c.batch_no(+)
                  and d.gen_type(+) = 'DJ_NK'
                  and c.addchart = d.gen_code(+)
                  and a.order_no = e.order_no
                  and e.req_no = f.req_no
                order by a.roll_length, a.width, a.dia, a.core
            """
            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, paper_type, b_wgt, width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no, core, dia, pattern, luster, color, addchart, sep_qt, pack_type, rs_gubun, nation_code, customer_name, skid_yn, pte_gubun = row
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
                    'addchart': addchart,
                    'sep_qt': sep_qt,
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
            print(f"Error while get_roll_orders_from_db roll orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_roll_sl_orders_from_db(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            #
            sql_query = """
                SELECT
                    plant, pm_no, schedule_unit, width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no,
                    core, dia, nvl(pattern, ' ') as pattern, luster, color,
                    pack_type
                FROM
                    hsfp_st.h3t_production_order@hsfp_st_rlink
                WHERE paper_prod_seq = :p_paper_prod_seq
                  AND rs_gubun = 'R'
                ORDER BY roll_length, width, dia, core
            """
            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no, core, dia, pattern, luster, color, pack_type = row
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
                    '롤길이': int(roll_length),
                    '등급': quality_grade,
                    '수출내수': export_type,
                    'core': core,
                    'dia': dia,
                    'luster': luster,
                    'color': color,
                    'order_pattern': pattern,
                    'pack_type': pack_type
                })
            print(f"Successfully fetched {len(raw_orders)} roll orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting get_roll_sl_orders_from_db orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_orders_from_db(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

            sql_query = """
                SELECT
                    a.plant, a.pm_no, a.schedule_unit, a.paper_type, a.b_wgt, a.width, a.length, a.quality_grade, a.order_ton_cnt, a.export_yn, a.order_no, a.color, a.order_gubun,
                    NVL(a.pt_gubun,'1') as PT_GUBUN,
                    d.gen_hcode, f.dir_gubun, e.regular_gubun, e.pack_type,
                    a.rs_gubun, a.nation_code, a.customer_name, a.skid_yn, e.pte_gubun, a.dia, a.core, a.nation_code
                FROM
                    h3t_production_order a, h3t_production_order_param b, batch_master@paper33_link c, th_mst_commoncode d, sapd12t_tmp e, sapd11t_tmp f
                WHERE a.order_no = b.order_no(+)
                  and a.paper_prod_seq = :p_paper_prod_seq
                  and a.rs_gubun = 'S'
                  and a.material_no = c.matnr(+)
                  and a.batch_no = c.batch_no(+)
                  and d.gen_type(+) = 'DJ_NK'
                  and c.addchart = d.gen_code(+)
                  and a.order_no = e.order_no
                  and e.req_no = f.req_no
                ORDER BY a.quality_grade, a.width, a.length
            """

            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, paper_type, b_wgt, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color, order_gubun, pt_gubun, gen_hcode, dir_gubun, regular_gubun, pack_type, rs_gubun, nation_code, customer_name, skid_yn, pte_gubun, dia, core, nation_code  = row
                export_type = '수출' if export_yn == 'Y' else '내수'
                raw_orders.append({
                    'plant': plant,
                    'pm_no': pm_no,
                    'schedule_unit': schedule_unit,
                    'order_no': order_no,
                    'paper_type': paper_type,
                    'b_wgt': b_wgt,
                    '가로': int(width),
                    '세로': int(length),
                    '주문톤': float(order_ton_cnt),
                    '등급': quality_grade,
                    '수출내수': export_type,
                    'color': color,
                    'order_gubun': order_gubun, 
                    'pt_gubun': pt_gubun,
                    'gen_hcode': gen_hcode,
                    'dir_gubun': dir_gubun,
                    'regular_gubun': regular_gubun,
                    'pack_type': pack_type,
                    'rs_gubun': rs_gubun,
                    'nation_code': nation_code,
                    'customer_name': customer_name,
                    'skid_yn': skid_yn,
                    'pte_gubun': pte_gubun,
                    'dia': dia,
                    'core': core,
                    'nation_code': nation_code
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_roll_patterns_from_db(self, lot_no, version):
        """지정된 lot_no와 version에 대해 th_pattern_sequence에서 롤 패턴(rollwidth)을 가져옵니다."""
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

            query = """

                WITH Target_Info AS (
                    /* 1. 기준이 되는 paper_type과 b_wgt를 사전에 추출 (1번만 수행) */
                    SELECT paper_type, b_wgt
                    FROM h3t_production_order
                    WHERE paper_prod_seq = :lot_no 
                    AND rownum = 1
                ),
                Allowed_Widths AS (
                    /* 2. 허용되는 지폭(width) 목록을 인덱스 스캔이 가능하도록 구성 */
                    SELECT DISTINCT width 
                    FROM h3t_production_order
                    WHERE paper_prod_seq = :lot_no
                    AND rs_gubun = 'R'
                )
                SELECT 
                    rollwidth1, rollwidth2, rollwidth3, rollwidth4, 
                    rollwidth5, rollwidth6, rollwidth7, rollwidth8
                FROM (
                    SELECT 
                        row_number() over(order by rn desc) as r_num,
                        p.*
                    FROM th_pattern_tot p
                    INNER JOIN Target_Info t 
                        ON p.paper_type = t.paper_type 
                    AND p.b_wgt = t.b_wgt
                    WHERE NOT EXISTS (
                        /* 3. 패턴의 지폭 중, 허용 목록(Allowed_Widths)에 없는 것이 하나라도 있으면 제외 */
                        SELECT 1
                        FROM (
                            SELECT p.rollwidth1 as w FROM dual UNION ALL
                            SELECT p.rollwidth2 FROM dual UNION ALL
                            SELECT p.rollwidth3 FROM dual UNION ALL
                            SELECT p.rollwidth4 FROM dual UNION ALL
                            SELECT p.rollwidth5 FROM dual UNION ALL
                            SELECT p.rollwidth6 FROM dual UNION ALL
                            SELECT p.rollwidth7 FROM dual UNION ALL
                            SELECT p.rollwidth8 FROM dual
                        ) v
                        WHERE v.w > 0 
                        AND v.w NOT IN (SELECT width FROM Allowed_Widths)
                    )
                ) a

            """
            cursor.execute(query, lot_no=lot_no)
            rows = cursor.fetchall()
            
            db_patterns = []
            for row in rows:
                # None이나 0이 아닌 유효한 rollwidth만 필터링합니다.
                pattern_widths = [int(w) for w in row if w and w > 0]
                if pattern_widths:
                    db_patterns.append(pattern_widths)
            
            print(f"Successfully fetched {len(db_patterns)} roll patterns from DB for lot {lot_no} version {version}")
            return db_patterns
        except oracledb.Error as error:
            print(f"Error while fetching roll patterns from DB: {error}")
            return []
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_patterns_from_db(self, lot_no):
        """th_pattern_tot_sheet 테이블에서 현재 오더에 해당하는 지폭이 있는 쉬트 패턴만 가져옵니다.
        
        - width1~8 컬럼에는 아이템명 형식(예: "710x4", "1000x2")으로 저장되어 있습니다.
        - 현재 lot의 지종/평량과 일치하고, 모든 아이템의 지폭이 현재 오더에 존재하는 패턴만 반환합니다.
        """
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            
            query = """
                SELECT DISTINCT
                    width1, width2, width3, width4, 
                    width5, width6, width7, width8
                FROM th_pattern_tot_sheet A
                WHERE (a.rn) IN (
                    SELECT rn
                    FROM (
                        -- 1. UNPIVOT: width 컬럼들을 행으로 변환
                        SELECT rn, item_name
                        FROM th_pattern_tot_sheet
                        UNPIVOT (
                            item_name FOR col_name IN (
                                width1, width2, width3, width4, 
                                width5, width6, width7, width8
                            )
                        )
                    ) P
                    -- 2. 오더 테이블과 조인 (지폭 추출: "710x4" → 710)
                    LEFT JOIN (
                        SELECT DISTINCT width 
                        FROM h3t_production_order
                        WHERE paper_prod_seq = :lot_no
                        AND rs_gubun = 'S'
                    ) O ON TO_NUMBER(SUBSTR(P.item_name, 1, INSTR(P.item_name, 'x') - 1)) = O.width
                    
                    -- 3. 유효 아이템만 있는 패턴 필터링
                    WHERE P.item_name IS NOT NULL
                    GROUP BY rn
                    HAVING 
                        -- 모든 유효 아이템이 오더 목록에 있는지 확인
                        COUNT(P.item_name) = COUNT(O.width)
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
            
            print(f"Successfully fetched {len(db_patterns)} sheet patterns from th_pattern_tot_sheet for lot {lot_no}")
            return db_patterns
        except oracledb.Error as error:
            print(f"Error while fetching sheet patterns from DB: {error}")
            return []
        finally:
            if connection:
                self.pool.release(connection)

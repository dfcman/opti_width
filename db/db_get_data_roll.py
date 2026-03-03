import oracledb

class RollGetters:
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



    def get_roll_sl_orders_from_db_jh(self, in_lot_no, in_version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            #
            sql_query = """
                select plant, pm_no, schedule_unit, paper_prod_seq,
                    width                as width, 
                    min(height)          as height, 
                    roll_length          as roll_length, 
                    rs_gubun             as rs_gubun, 
                    quality_grade        as quality_grade,
                    min(regular_type)    as regular_type,
                    sum(order_roll_cnt) as order_roll_cnt,
                    case when rs_gubun = 'R' 
                            then round(sum(order_ton_cnt * loss_rate_roll), 4)
                            when rs_gubun = 'S'
                            then round(sum(order_ton_cnt * loss_rate_sheet), 4)
                    end                  as order_ton_cnt,
                    doublepack           as doublepack, 
                    digital_yn           as digital, 
                    semi                 as semi, 
                    dia                  as dia, 
                    core                 as core,
                    short_sheet          as short_sheet,
                    min(loss_rate_roll)  as loss_rate_roll,
                    min(loss_rate_sheet) as loss_rate_sheet
                from (
                        select plant, pm_no, schedule_unit, paper_prod_seq,
                            order_no,
                            quality_grade,
                            width,
                            height,  
                            roll_length, 
                            rs_gubun, 
                            regular_type,
                            loss_rate_roll, 
                            loss_rate_sheet,
                            order_roll_cnt,
                            order_ton_cnt,
                            doublepack,
                            (case when nvl(digital_yn, 'N') = 'N'
                                    then 'N'
                                    when sum(digital_cut_ton) over (partition by paper_prod_seq, width, short_sheet, nvl(roll_length,0), semi)  
                                        > sum(digital_ton)     over (partition by paper_prod_seq, width, short_sheet, nvl(roll_length,0), semi)
                                    then 'D' 
                                    else 'Y'
                                end ) as digital_yn,
                            semi,
                            dia, 
                            core,
                            short_sheet
                        from (
                                select p.plant, p.pm_no, p.schedule_unit, p.paper_prod_seq,
                                    p.order_no,
                                    p.quality_grade,
                                    j.adj_width as width,
                                    (
                                        select min(p2.length) as length
                                        from h3t_production_order p2,
                                            th_versions_manager v2
                                        where p2.paper_prod_seq = v2.lot_no
                                        and v2.lot_no = :in_lot_no
                                        and v2.version = :in_version
                                        and p2.width = j.adj_width
                                        and p2.rs_gubun = p.rs_gubun
                                        and (case when (p2.width<=700 or p2.width >810) and p2.length < nvl(v2.short_sheet, 545) and p2.rs_gubun='S' then 'S' else 'L' end) = 
                                            (case when (p.width<=700 or p.width >810) and p.length < nvl(v.short_sheet, 545) and p.rs_gubun='S' then 'S' else 'L' end)
                                        group by p2.width
                                    ) as height,  
                                    p.roll_length, 
                                    p.rs_gubun, 
                                    case when o.regular_gubun = '3' then '1' else o.regular_gubun end regular_type,
                                    p.order_roll_cnt,
                                    p.order_ton_cnt,
                                    nvl(1+0.01*r.loss_rate_roll,1.001) loss_rate_roll,
                                    nvl(1+0.01*(r.loss_rate_sheet + case when fn_get_digi_yn(p.order_no) = 'Y' then nvl(r.loss_rate_digi, 0) else 0 end),1.065) loss_rate_sheet,
                                    (case when p.skid_yn = 'Y' and p.pt_gubun =2 then 'Y' else 'N' end) doublepack,
                                    (case when p.rs_gubun = 'S' and ( o.pack_type in ('5', '6', '7') or o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) then 'Y' else 'N' end) digital_yn,
                                    (case when o2.dir_gubun = 'ZARL' then 'Y' else 'N' end) semi,
                                    p.dia, 
                                    p.core,
                                    (case when (p.width<=700 or p.width >810) and p.length < nvl(v.short_sheet, 545) and p.rs_gubun='S' then 'S' else 'L' end) as short_sheet,
                                    (case when p.rs_gubun = 'S' and o.pack_type in ('5', '7') and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) 
                                                                and (    ( ( p.width  between 210 and 360)  and ( p.length between 279 and 520) )
                                                                        or ( ( p.length between 210 and 360)  and ( p.width  between 279 and 520) )
                                                                    )
                                            then 'Y' else 'N' end) digital_cut_yn,
                                    (case when p.rs_gubun = 'S' and o.pack_type in ('6')      and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) then p.order_ton_cnt else 0 end) digital_ton,
                                    (case when p.rs_gubun = 'S' and o.pack_type in ('5', '7') and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) 
                                                                and (    ( ( p.width  between 210 and 360) and ( p.length between 279 and 520) )
                                                                        or ( ( p.length between 210 and 360) and ( p.width  between 279 and 520) )
                                                                    )
                                            then p.order_ton_cnt else 0 end) digital_cut_ton
                                from h3t_production_order p,
                                    th_tar_loss_rate r,
                                    sapd12t_tmp o, 
                                    sapd11t_tmp o2,
                                    th_order_adj j,
                                    th_versions_manager v 
                                where p.paper_prod_seq = v.lot_no 
                                and p.paper_type = r.paper_type(+)
                                and p.b_wgt = r.cp_b_wgt(+)
                                and p.pm_no = r.resource_code(+)
                                and o.lot_no = v.lot_no 
                                and o2.req_no = o.req_no
                                and p.order_no = o.order_no 
                                and j.order_no = p.order_no
                                and j.lot_no = v.lot_no
                                and j.version = v.version
                                and v.lot_no = :in_lot_no
                                and v.version = :in_version
                            ) t     
                        )
                        group by plant, pm_no, schedule_unit, paper_prod_seq,
                                width,
                                roll_length,
                                rs_gubun, 
                                quality_grade,
                                doublepack,
                                digital_yn,
                                semi,
                                dia, 
                                core,
                                short_sheet   
                order by rs_gubun, width, height, regular_type
            """
            cursor.execute(sql_query, in_lot_no=in_lot_no, in_version=in_version)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                (r_plant, r_pm_no, r_schedule_unit, r_paper_prod_seq,
                 r_width, r_height, r_roll_length, r_rs_gubun, r_quality_grade,
                 r_regular_type, r_order_roll_cnt, r_order_ton_cnt,
                 r_doublepack, r_digital, r_semi, r_dia, r_core,
                 r_short_sheet, r_loss_rate_roll, r_loss_rate_sheet) = row
                raw_orders.append({
                    'plant': r_plant,
                    'pm_no': r_pm_no,
                    'schedule_unit': r_schedule_unit,
                    'paper_prod_seq': r_paper_prod_seq,
                    '지폭': int(r_width),
                    '가로': int(r_height) if r_height else 0,
                    '롤길이': int(r_roll_length) if r_roll_length else 0,
                    'rs_gubun': r_rs_gubun,
                    '등급': r_quality_grade,
                    'regular_type': r_regular_type,
                    '주문수량': int(r_order_roll_cnt) if r_order_roll_cnt else 0,
                    '주문톤': float(r_order_ton_cnt) if r_order_ton_cnt else 0,
                    'doublepack': r_doublepack,
                    'digital': r_digital,
                    'semi': r_semi,
                    'dia': r_dia,
                    'core': r_core,
                    'short_sheet': r_short_sheet,
                    'loss_rate_roll': r_loss_rate_roll,
                    'loss_rate_sheet': r_loss_rate_sheet,
                })
            print(f"Successfully fetched {len(raw_orders)} roll orders for lot {in_lot_no}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting get_roll_sl_orders_from_db orders_jh from DB: {error}")
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


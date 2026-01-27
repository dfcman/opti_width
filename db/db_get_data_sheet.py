import oracledb

class SheetGetters:
    def get_sheet_orders_from_db(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # sql_query = """
            #     SELECT
            #         width, length, quality_grade, order_ton_cnt, export_yn, order_no
            #     FROM
            #         hsfp_st.h3t_production_order
            #     WHERE paper_prod_seq = :p_paper_prod_seq
            #       AND rs_gubun = 'S'
            #     ORDER BY width, length
            # """

            sql_query = """
                SELECT
                    a.plant, a.pm_no, a.schedule_unit, a.paper_type, a.b_wgt, a.width, a.length, a.quality_grade, a.order_ton_cnt, a.export_yn, a.order_no, a.color, a.order_gubun,
                    --CASE WHEN a.export_yn = 'N' AND a.width <= 600
                    --        THEN '2' --내수, SHEET 지폭 600이하는 무조건 SKID_TYPE으로, 2012.10.05, LSY
                    --        ELSE NVL(a.pt_gubun,'1')
                    --END PT_GUBUN,
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

    def get_sheet_orders_from_db_ca(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # sql_query = """ 
            #     SELECT
            #         width, length, quality_grade, order_ton_cnt, export_yn, order_no
            #     FROM
            #         hsfp_st.h3t_production_order
            #     WHERE paper_prod_seq = :p_paper_prod_seq
            #       AND rs_gubun = 'S'
            #     ORDER BY width, length
            # """

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
                    'pattern': pattern
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_orders_from_db_st(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # sql_query = """
            #     SELECT
            #         width, length, quality_grade, order_ton_cnt, export_yn, order_no
            #     FROM
            #         hsfp_st.h3t_production_order
            #     WHERE paper_prod_seq = :p_paper_prod_seq
            #       AND rs_gubun = 'S'
            #     ORDER BY width, length
            # """

            sql_query = """
                SELECT
                    plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, 
                    export_yn, order_no, color, order_gubun, pt_gubun, pack_type
                FROM
                    hsfp_st.h3t_production_order@hsfp_st_rlink
                WHERE paper_prod_seq = :p_paper_prod_seq
                  AND rs_gubun = 'S'
                ORDER BY width, length
            """

            # print(f"Executing query to fetch sheet orders:\n{sql_query}")
            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color, order_gubun, pt_gubun, pack_type = row
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
                    'pack_type': pack_type
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)


    def get_sheet_orders_from_db_jh(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # sql_query = """
            #     SELECT
            #         width, length, quality_grade, order_ton_cnt, export_yn, order_no
            #     FROM
            #         hsfp_st.h3t_production_order
            #     WHERE paper_prod_seq = :p_paper_prod_seq
            #       AND rs_gubun = 'S'
            #     ORDER BY width, length
            # """

            sql_query = """
                SELECT
                    plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, 
                    export_yn, order_no, color, order_gubun, pt_gubun, pack_type
                FROM
                    hsfp_st.h3t_production_order@hsfp_st_rlink
                WHERE paper_prod_seq = :p_paper_prod_seq
                  AND rs_gubun = 'S'
                ORDER BY width, length
            """

            # print(f"Executing query to fetch sheet orders:\n{sql_query}")
            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color, order_gubun, pt_gubun, pack_type = row
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
                    'pack_type': pack_type
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    

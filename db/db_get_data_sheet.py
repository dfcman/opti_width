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
                    plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color
                FROM
                    h3t_production_order
                WHERE paper_prod_seq = :p_paper_prod_seq
                  AND rs_gubun = 'S'
                ORDER BY width, length
            """

            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color = row
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
                    'color': color
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_orders_from_db_var(self, paper_prod_seq):
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
                    plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color
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
                plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color = row
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
                    'color': color
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
                    plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color
                FROM
                    h3t_production_order@hsfp_ca_rlink
                WHERE paper_prod_seq = :p_paper_prod_seq
                  AND rs_gubun = 'S'
                ORDER BY width, length
            """

            cursor.execute(sql_query, p_paper_prod_seq=paper_prod_seq)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                plant, pm_no, schedule_unit, width, length, quality_grade, order_ton_cnt, export_yn, order_no, color = row
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
                    'color': color
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

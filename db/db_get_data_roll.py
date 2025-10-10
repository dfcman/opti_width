import oracledb

class RollGetters:
    def get_roll_orders_from_db(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            #
            sql_query = """
                SELECT
                    width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no 
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
                width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no = row
                export_type = '수출' if export_yn == 'Y' else '내수'
                raw_orders.append({
                    '오더번호': order_no,
                    '지폭': int(width),
                    '가로': int(length),
                    '주문수량': int(order_roll_cnt),
                    '주문톤': float(order_ton_cnt),
                    '롤길이': int(roll_length),
                    '등급': quality_grade,
                    '수출내수': export_type
                })
            print(f"Successfully fetched {len(raw_orders)} roll orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting roll orders from DB: {error}")
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
                    width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no 
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
                width, length, roll_length, quality_grade, order_roll_cnt, order_ton_cnt, export_yn, order_no = row
                export_type = '수출' if export_yn == 'Y' else '내수'
                raw_orders.append({
                    '오더번호': order_no,
                    '지폭': int(width),
                    '롤길이': int(roll_length),
                    '주문수량': int(order_roll_cnt),
                    '주문톤': float(order_ton_cnt),
                    '롤길이': int(roll_length),
                    '등급': quality_grade,
                    '수출내수': export_type
                })
            print(f"Successfully fetched {len(raw_orders)} roll orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting roll orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

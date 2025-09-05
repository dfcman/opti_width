import oracledb
import sys

class Database:
    def __init__(self, user, password, dsn, min_pool=1, max_pool=1, increment=1):
        self.user = user
        self.password = password
        self.dsn = dsn
        self.pool = None
        try:
            self.pool = oracledb.create_pool(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=min_pool,
                max=max_pool,
                increment=increment
            )
            print("Successfully created Oracle connection pool.")
        except oracledb.Error as error:
            print(f"Error while creating connection pool: {error}")
            raise

    def close_pool(self):
        if self.pool:
            self.pool.close()
            print("Oracle connection pool closed.")

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
            #     where lot_no = '3241100322' and version = '01'
            # """

            query = """ 
                select 
                    plant, pm_no, schedule_unit, lot_no, '05' as version, min_width, roll_max_width, sheet_max_width, max_re_count as max_pieces,
                    paper_type, b_wgt
                from th_versions_manager 
                where lot_no = '3250900073' and version = '01'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            # 반환 값 개수를 10개로 맞춤
            return result if result else (None, None, None, None, None, None, None, None, None, None)
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return None, None, None, None, None, None, None, None, None, None
        finally:
            if connection:
                self.pool.release(connection)

    def update_lot_status(self, lot_no, version, status):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원
            query = "UPDATE th_versions_manager SET calc_successful = :status WHERE lot_no = :lot_no and version = :version"
            cursor.execute(query, status=status, lot_no=lot_no, version=version)
            connection.commit()
            print(f"Successfully updated lot {lot_no} to status {status}")
            return True
        except oracledb.Error as error:
            print(f"Error while updating lot status: {error}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.pool.release(connection)

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

    def get_sheet_orders_from_db(self, paper_prod_seq):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            sql_query = """
                SELECT
                    width, length, quality_grade, order_ton_cnt, export_yn, order_no
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
                width, length, quality_grade, order_ton_cnt, export_yn, order_no = row
                export_type = '수출' if export_yn == 'Y' else '내수'
                raw_orders.append({
                    '오더번호': order_no,
                    '가로': int(width),
                    '세로': int(length),
                    '주문톤': float(order_ton_cnt),
                    '등급': quality_grade,
                    '수출내수': export_type
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {paper_prod_seq}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def insert_pattern_sequence(self, lot_no, version, plant, pm_no, schedule_unit, max_width, 
                                paper_type, b_wgt, pattern_details):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

            cursor.execute("DELETE FROM th_pattern_sequence WHERE lot_no = :lot_no AND version = :version", lot_no=lot_no, version=version)
            print(f"Deleted existing patterns for lot {lot_no}, version {version}")

            # pok_cnt 컬럼 추가
            insert_query = """
                INSERT INTO th_pattern_sequence (
                    module, plant, pm_no, schedule_unit, max_width, paper_type, b_wgt,
                    lot_no, version, prod_seq, unit_no, seq, pok_cnt, 
                    wd_width, 
                    rollwidth1, rollwidth2, rollwidth3, rollwidth4, rollwidth5, rollwidth6, rollwidth7, rollwidth8,
                    groupno1, groupno2, groupno3, groupno4, groupno5, groupno6, groupno7, groupno8
                ) VALUES (
                    'C', :plant, :pm_no, :schedule_unit, :max_width, :paper_type, :b_wgt,
                    :lot_no, :version, :prod_seq, :unit_no, 1, :pok_cnt,
                    :w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8,
                    :w1, :w2, :w3, :w4, :w5, :w6, :w7, :w8,
                    :g1, :g2, :g3, :g4, :g5, :g6, :g7, :g8
                )
            """
            
            total_seq = 0
            # print(f"Number of pattern details: {len(pattern_details)}")
            for pattern in pattern_details:
                # print(f"Number of pattern details: {pattern['Count']}")
                for _ in range(pattern['Count']):
                    total_seq += 1
                    
                    # Python에서 pok_cnt 계산
                    pok_cnt_value = len([w for w in pattern['widths'] if w > 0])

                    bind_vars = {
                        'plant': plant,
                        'pm_no': pm_no,
                        'schedule_unit': schedule_unit,
                        'max_width': max_width,
                        'paper_type': paper_type,
                        'b_wgt': b_wgt,
                        'lot_no': lot_no,
                        'version': version,
                        'prod_seq': total_seq,
                        'unit_no': total_seq,
                        'pok_cnt': pok_cnt_value, # 계산된 값 바인딩
                        'w1': pattern['widths'][0], 'w2': pattern['widths'][1],
                        'w3': pattern['widths'][2], 'w4': pattern['widths'][3],
                        'w5': pattern['widths'][4], 'w6': pattern['widths'][5],
                        'w7': pattern['widths'][6], 'w8': pattern['widths'][7],
                        'g1': pattern['group_nos'][0][:15], 'g2': pattern['group_nos'][1][:15],
                        'g3': pattern['group_nos'][2][:15], 'g4': pattern['group_nos'][3][:15],
                        'g5': pattern['group_nos'][4][:15], 'g6': pattern['group_nos'][5][:15],
                        'g7': pattern['group_nos'][6][:15], 'g8': pattern['group_nos'][7][:15],
                    }
                    cursor.execute(insert_query, bind_vars)

            connection.commit()
            print(f"Successfully inserted {total_seq} new pattern sequences.")
            return True

        except oracledb.Error as error:
            print(f"Error while inserting pattern sequence: {error}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.pool.release(connection)

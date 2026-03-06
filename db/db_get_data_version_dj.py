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

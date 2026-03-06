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

import oracledb

class VersionGetters:
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
            #     where lot_no = '3241100322' and version = '01'   -- 3250900073, 3250900429, 8250900534, 5250900616
            #     and lot_no = '3250900068' and version = '01'
            # """

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
                    300 * 1000 as time_limit
                FROM th_versions_manager a, th_tar_std_length b
                WHERE a.plant = b.plant(+)
                AND a.paper_type = b.paper_type(+)
                AND a.b_wgt = b.b_wgt(+)
                and a.calc_successful = '9'
                --and a.lot_no = '3260100412' and a.version = '01'
                and LENGTH(a.version_id) > 0
                ORDER BY a.plant, a.version_id, a.schedule_unit, a.lot_no, a.version DESC
                FETCH FIRST 1 ROWS ONLY
            """

            # query = """
            #     select 
            #         a.plant, pm_no, a.schedule_unit, a.lot_no, version, a.min_width, a.roll_max_width, 
            #         a.sheet_max_width, a.max_re_count as max_pieces, 4 as sheet_max_pieces,
            #         a.paper_type, a.b_wgt,
            #         750 as min_sc_width, a.max_sc_width, a.sheet_trim_size, sheet_length_re,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type = '1')) as roll_order_cnt
            #     from th_versions_manager a, th_tar_std_length b
            #     where a.plant = b.plant
            #     and a.paper_type = b.paper_type
            #     and a.b_wgt = b.b_wgt 
            #     and lot_no = '3250900073' and version = '01'
            # """

            # query = """
            #     select 
            #         a.plant, pm_no, a.schedule_unit, a.lot_no, '05' as version, a.min_width, a.roll_max_width, 
            #         a.sheet_max_width, a.max_re_count as max_pieces, 4 as sheet_max_pieces,
            #         a.paper_type, a.b_wgt,
            #         a.min_sc_width - 100, a.max_sc_width, a.sheet_trim_size, b.max_length as sheet_length_re,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type = '1')) as roll_order_cnt
            #     from hsfp_st.th_versions_manager@hsfp_st_rlink a, hsfp_st.th_tar_std_length@hsfp_st_rlink b
            #     where a.plant = b.plant(+)
            #     and a.paper_type = b.paper_type(+)
            #     and a.b_wgt = b.b_wgt(+)
            #     and b.operation_code(+) = 'RE'
            #     and b.rs_gubun(+) = 'S'
            #     and lot_no = '8250900534' 
            #     and version = '99'
            # """

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

    def get_target_lot_ca(self):
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
                --where a.calc_successful = '9'
                --   5251204230   5251200302   5251200178  5251200510 5251200012  5251201860 5251200705 5251201794 5251203330 5251203142
                where lot_no = '5251203330' and version = '01'
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

    def get_target_lot_st(self, lot_no):
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

            query = """
                select 
                    a.plant, pm_no, a.schedule_unit, a.lot_no, '05' as version, a.min_width, a.roll_max_width, 
                    a.sheet_max_width, a.max_re_count as max_pieces, 4 as sheet_max_pieces,
                    a.paper_type, a.b_wgt, nvl(a.color, ' ') as color, 
                    a.min_cm_width - 100, a.max_cm_width, a.ww_trim_size
                from hsfp_st.th_versions_manager@hsfp_st_rlink a
                where lot_no = :p_lot_no
                and version = '99'
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no)
            result = cursor.fetchone()
            # 반환 값 개수를 16개로 맞춤 (기존 17이라고 주석있지만 코드상으론 15개?)
            # 아니, 위 쿼리 select 컬럼 수: 15개 -> +color = 16개?
            # execute.py SL unpacking: 16개.
            # SL unpacking: 
            # (plant_sl, pm_no_sl, schedule_unit_sl, lot_no_sl, version_sl, min_width_sl, 
            #  max_width_sl, _, max_pieces_sl, _, 
            #  paper_type_sl, b_wgt_sl, color_sl,
            #  min_sl_width, max_sl_width, sl_trim_size) = db.get_lot_param_roll_sl(lot_no=lot_no)
            # Total 16 vars.
            # 쿼리 컬럼 수: 1(plant)+1+1+1+1(version)+1(min)+1(max)+1(sheet_max)+1(pieces)+1(sheet_pieces)+1(pt)+1(bw)+1(color) + 1(min_cm)+1(max_cm)+1(ww_trim) = 16.
            # OK.
            return result if result else (None,) * 16
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 16
        finally:
            if connection:
                self.pool.release(connection)

    def get_target_lot_jh(self, lot_no):
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

            query = """
                select 
                    a.plant, pm_no, a.schedule_unit, a.lot_no, '05' as version, a.min_width, a.roll_max_width, 
                    a.sheet_max_width, a.max_re_count as max_pieces, 4 as sheet_max_pieces,
                    a.paper_type, a.b_wgt, nvl(a.color, ' ') as color, 
                    a.min_sc_width - 100, a.max_sc_width, a.sheet_trim_size, 
                    b.min_length as min_sheet_length_re,
                    b.max_length as max_sheet_length_re
                from hsfp_st.th_versions_manager@hsfp_st_rlink a, hsfp_st.th_tar_std_length@hsfp_st_rlink b
                where a.plant = b.plant
                and a.paper_type = b.paper_type
                and a.b_wgt = b.b_wgt 
                and b.operation_code = 'RE' 
                and b.rs_gubun = 'S'
                and lot_no = :p_lot_no
                and version = '99'
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no)
            result = cursor.fetchone()
            # 반환 값 개수를 18개로 맞춤
            # Query cols: 15 (original) + 1 (color) = 18?
            # Original query: 17 lines of select? No.
            # Original: plant...b_wgt (12 items) + min_sc...max_sheet (5 items) = 17 items.
            # New: +1 = 18 items.
            return result if result else (None,) * 18
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return (None,) * 18
        finally:
            if connection:
                self.pool.release(connection)
 

    def get_lot_param_sheet_ca(self, lot_no, version):
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
            # 5250900616, 5250900062, 5250900429

            query = """
                SELECT 
                    a.plant, a.pm_no, a.schedule_unit, a.lot_no, a.version, a.time_limit * 1000 as time_limit, a.coating_yn, 
                    a.paper_type, a.b_wgt, a.color, 
                    a.p_type, a.p_wgt, a.p_color, a.p_machine,
                    a.min_width, a.roll_max_width, a.min_re_count, a.max_re_count, 
                    d.std_length as sheet_length_re, d.std_roll_cnt, 
                    a.min_sc_width, a.max_sc_width, a.sheet_trim_size,                     
                    a.min_cm_width, a.max_cm_width, a.max_sl_count, a.ww_trim_size, a.ww_trim_size_sheet
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
            # 데몬용 쿼리 복원 (사용자 추가 필드 유지)
            # query = """
            #     SELECT 
            #         plant, pm_no, schedule_unit, lot_no, version, min_width, roll_max_width as max_width, max_re_count as max_pieces,
            #         paper_type, b_wgt
            #     FROM th_versions_manager 
            #     WHERE calc = 9 AND ROWNUM = 1
            # """
            # 5250900616, 5250900062, 5250900429

            query = """
                SELECT 
                    a.plant, a.pm_no, a.schedule_unit, a.lot_no, a.version, a.time_limit * 1000 as time_limit, a.coating_yn, 
                    a.paper_type, a.b_wgt, a.color, 
                    a.p_type, a.p_wgt, a.p_color, a.p_machine,
                    a.min_width, a.roll_max_width, a.min_re_count, a.max_re_count,
                    a.min_cm_width, a.max_cm_width, a.max_sl_count, a.ww_trim_size
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

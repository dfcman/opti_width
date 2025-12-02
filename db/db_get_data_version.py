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

            # query = """
            #     select 
            #         a.plant, pm_no, a.schedule_unit, a.lot_no, version, a.min_width, a.roll_max_width, 
            #         a.sheet_max_width, a.max_re_count as max_pieces, 4 as sheet_max_pieces,
            #         a.paper_type, a.b_wgt,
            #         1000 as min_sc_width, a.max_sc_width, a.sheet_trim_size, sheet_length_re,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt,
            #         ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type = '1')) as roll_order_cnt
            #     from th_versions_manager a, th_tar_std_length b
            #     where a.plant = b.plant
            #     and a.paper_type = b.paper_type
            #     and a.b_wgt = b.b_wgt 
            #     and (lot_no, version) in (
            #             SELECT LOT_NO, VERSION
            #             FROM ( 
            #                 SELECT PLANT, SCHEDULE_UNIT, LOT_NO, VERSION, VERSION_ID, ROWNUM
            #                 FROM TH_CALCULATION_MESSAGES
            #                 WHERE PLANT = '3000'
            #                 AND MESSAGE_SEQ = '9'
            #                 AND LENGTH(VERSION_ID) > 0
            #                 ORDER BY PLANT, VERSION_ID, SCHEDULE_UNIT, LOT_NO, VERSION DESC 
            #             )
            #             WHERE ROWNUM = 1
            #     )
            # """

            query = """
                select 
                    a.plant, pm_no, a.schedule_unit, a.lot_no, version, a.min_width, a.roll_max_width, 
                    a.sheet_max_width, a.max_re_count as max_pieces, 4 as sheet_max_pieces,
                    a.paper_type, a.b_wgt,
                    800 as min_sc_width, a.max_sc_width, a.sheet_trim_size, sheet_length_re,
                    ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '3' and pack_type != '1')) as sheet_order_cnt,
                    ((select count(*) from  sapd12t_tmp s12 where s12.lot_no = a.lot_no and fact_status = '31' and pack_type = '1')) as roll_order_cnt
                from th_versions_manager a, th_tar_std_length b
                where a.plant = b.plant
                and a.paper_type = b.paper_type
                and a.b_wgt = b.b_wgt 
                and lot_no = '3250900073' and version = '01'
            """

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
            # 반환 값 개수를 16개로 맞춤
            return result if result else (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        finally:
            if connection:
                self.pool.release(connection)

    def get_target_lot_sl(self, lot_no):
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
                    a.paper_type, a.b_wgt,
                    a.min_cm_width - 100, a.max_cm_width, a.ww_trim_size
                from hsfp_st.th_versions_manager@hsfp_st_rlink a
                where lot_no = :p_lot_no
                and version = '99'
            """

            print(f"Executing query to fetch target lot:\n{query}")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no)
            result = cursor.fetchone()
            # 반환 값 개수를 17개로 맞춤
            return result if result else (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        finally:
            if connection:
                self.pool.release(connection)

    def get_target_lot_var(self, lot_no):
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
                    a.paper_type, a.b_wgt,
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

            print(f"Executing query to fetch target lot:\n{query}")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no)
            result = cursor.fetchone()
            # 반환 값 개수를 17개로 맞춤
            return result if result else (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        finally:
            if connection:
                self.pool.release(connection)

    def get_target_lot_ca(self, lot_no):
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
                select 
                    a.plant, pm_no, a.schedule_unit, a.lot_no, '05' as version, a.min_width -300, a.roll_max_width, 
                    a.sheet_max_width, a.max_re_count as max_pieces, 2 as sheet_max_pieces,
                    a.paper_type, a.b_wgt,
                    a.min_sc_width - 100, a.max_sc_width, a.sheet_trim_size, 
                    b.std_length as min_sheet_length_re,
                    b.std_length as max_sheet_length_re
                from th_versions_manager@hsfp_ca_rlink a, th_tar_std_length_ca@hsfp_ca_rlink b
                where a.plant = b.plant
                and a.paper_type = b.paper_type
                and a.b_wgt = b.b_wgt 
                and b.rs_gubun = 'S'
                and lot_no = :p_lot_no
                and version = '99'
            """

            # print(f"Executing query to fetch target lot:\n{query}")
            # cursor.execute(query)
            cursor.execute(query, p_lot_no=lot_no)
            result = cursor.fetchone()
            # 반환 값 개수를 17개로 맞춤
            return result if result else (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        finally:
            if connection:
                self.pool.release(connection)

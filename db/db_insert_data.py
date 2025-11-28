import oracledb

class DataInserters:
    def update_lot_status(self, lot_no, version, status):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원
            query = "update th_versions_manager set calc_successful = :status where lot_no = :lot_no and version = :version"
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

    def insert_pattern_sequence(self, connection, lot_no, version, plant, pm_no, schedule_unit, max_width, 
                                paper_type, b_wgt, pattern_details):
        cursor = connection.cursor()

        insert_query = """
            insert into th_pattern_sequence (
                module, plant, pm_no, schedule_unit, max_width, paper_type, b_wgt,
                lot_no, version, prod_seq, unit_no, seq, length, pok_cnt, 
                diameter, cut_cnt, color, luster, p_lot, p_type, p_wgt, core, pattern,
                wd_width, 
                wd_trim, 
                cut_width, 
                cut_trim, 
                prod_amt, 
                prod_wgt, 
                prod_yn, 
                spool_no, spool_seq, rs_gubun, p_machine,
                rollwidth1, rollwidth2, rollwidth3, rollwidth4, rollwidth5, rollwidth6, rollwidth7, rollwidth8,
                groupno1, groupno2, groupno3, groupno4, groupno5, groupno6, groupno7, groupno8
            ) values (
                'C', :plant, :pm_no, :schedule_unit, :max_width, :paper_type, :b_wgt,
                :lot_no, :version, :prod_seq, :unit_no, :seq, :length, :pok_cnt,
                :diameter, :cut_cnt, :color, :luster, :p_lot, :p_type, :p_wgt, :core, :pattern,
                :w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8,
                :max_width - :w1 - :w2 - :w3 - :w4 - :w5 - :w6 - :w7 - :w8, 
                :w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8,
                :max_width - :w1 - :w2 - :w3 - :w4 - :w5 - :w6 - :w7 - :w8, 
                round(:b_wgt * (:w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8) * :length / 1000000, 0),
                round(:b_wgt * (:w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8) * :length / 1000000, 1),
                'N',
                :spool_no, :spool_seq, :rs_gubun, :p_machine,
                :w1, :w2, :w3, :w4, :w5, :w6, :w7, :w8,
                :g1, :g2, :g3, :g4, :g5, :g6, :g7, :g8
            )
        """
        
        total_seq = 0
        bind_vars_list = []
        for pattern in pattern_details:
            for seq in range(pattern['count']):
                total_seq += 1
                prod_seq = pattern['prod_seq']
                pok_cnt_value = len([w for w in pattern['widths'] if w > 0])

                bind_vars = {
                    'plant': plant, 'pm_no': pm_no, 'schedule_unit': schedule_unit,
                    'max_width': max_width, 'paper_type': paper_type, 'b_wgt': b_wgt,
                    'lot_no': lot_no, 'version': version, 'prod_seq': prod_seq,
                    'unit_no': prod_seq, 'seq': seq + 1,
                    'length': pattern.get('pattern_length', 0),
                    'pok_cnt': pok_cnt_value,
                    'diameter': int(pattern.get('diameter', 0)), 
                    'cut_cnt': 1,
                    'color': str(pattern.get('color', '')),
                    'luster': int(pattern.get('luster', 0)),
                    'p_lot': str(pattern.get('p_lot', '')), 
                    'p_type': paper_type,
                    'p_wgt': b_wgt, 
                    'core': int(pattern.get('core', 0)),
                    'pattern': str(pattern.get('order_pattern', '')),
                    'spool_no': prod_seq, 
                    'spool_seq': seq + 1,
                    # 'rs_gubun': self._get_rs_gubun(pattern),
                    'rs_gubun': pattern['rs_gubun'],
                    'p_machine': pm_no, 
                    'w1': pattern['widths'][0], 'w2': pattern['widths'][1],
                    'w3': pattern['widths'][2], 'w4': pattern['widths'][3],
                    'w5': pattern['widths'][4], 'w6': pattern['widths'][5],
                    'w7': pattern['widths'][6], 'w8': pattern['widths'][7],
                    'g1': pattern['group_nos'][0][:15], 'g2': pattern['group_nos'][1][:15],
                    'g3': pattern['group_nos'][2][:15], 'g4': pattern['group_nos'][3][:15],
                    'g5': pattern['group_nos'][4][:15], 'g6': pattern['group_nos'][5][:15],
                    'g7': pattern['group_nos'][6][:15], 'g8': pattern['group_nos'][7][:15],
                }
                bind_vars_list.append(bind_vars)
        
        if bind_vars_list:
            cursor.executemany(insert_query, bind_vars_list)
            
        print(f"Prepared {total_seq} new pattern sequences for transaction.")

    def insert_roll_sequence(self, connection, lot_no, version, plant, pm_no, schedule_unit, max_width, 
                                paper_type, b_wgt, pattern_roll_details):
        cursor = connection.cursor()

        insert_query = """
            insert into  th_roll_sequence (
                module, plant, pm_no, schedule_unit, paper_type, b_wgt,
                lot_no, version, prod_seq, unit_no, seq, roll_seq, pok_cnt,
                rollwidth, length, spool_no, spool_seq, rs_gubun,
                dia, weight, 
                cut_cnt, color, luster, core, pattern, p_lot, p_type, p_wgt, p_machine, 
                width1, width2, width3, width4, width5, width6, width7,
                group1, group2, group3, group4, group5, group6, group7
            ) values (
                'C', :plant, :pm_no, :schedule_unit, :paper_type, :b_wgt,
                :lot_no, :version, :prod_seq, :unit_no, :seq, :roll_seq, :pok_cnt,
                :rollwidth, :length, :spool_no, :spool_seq, :rs_gubun,
                :dia, round(:b_wgt * :rollwidth * :length / 1000000,1), 
                :cut_cnt, :color, :luster, :core, :pattern, :p_lot, :p_type, :p_wgt, :p_machine, 
                :w1, :w2, :w3, :w4, :w5, :w6, :w7,
                :g1, :g2, :g3, :g4, :g5, :g6, :g7
            )
        """

        bind_vars_list = []
        for roll_detail in pattern_roll_details:
            for seq in range(roll_detail['count']):
                prod_seq = roll_detail['prod_seq']
                roll_seq = roll_detail['roll_seq']
                pok_cnt_value = len([w for w in roll_detail['widths'] if w > 0])

                bind_vars = {
                    'plant': plant, 'pm_no': pm_no, 'schedule_unit': schedule_unit,
                    'paper_type': paper_type, 'b_wgt': b_wgt, 'lot_no': lot_no,
                    'version': version, 'prod_seq': prod_seq, 'unit_no': prod_seq,
                    'seq':seq + 1, 'roll_seq': roll_seq, 'pok_cnt': pok_cnt_value,
                    'rollwidth': roll_detail['rollwidth'],
                    'length': roll_detail['pattern_length'],
                    'spool_no': prod_seq, 
                    'spool_seq': seq + 1,
                    'dia': int(roll_detail.get('diameter', 0)), 
                    'cut_cnt': 1,
                    'color': str(roll_detail.get('color', '')),
                    'luster': int(roll_detail.get('luster', 0)),
                    'core': int(roll_detail.get('core', 0)),
                    'pattern': str(roll_detail.get('order_pattern', '')),
                    'p_lot': lot_no,
                    'p_type': paper_type,
                    'p_wgt': b_wgt,  
                    'p_machine': pm_no,
                    # 'rs_gubun': self._get_rs_gubun(roll_detail),
                    'rs_gubun': roll_detail['rs_gubun'],
                    'w1': roll_detail['widths'][0], 'w2': roll_detail['widths'][1],
                    'w3': roll_detail['widths'][2], 'w4': roll_detail['widths'][3],
                    'w5': roll_detail['widths'][4], 'w6': roll_detail['widths'][5],
                    'w7': roll_detail['widths'][6],
                    'g1': roll_detail['group_nos'][0][:15], 'g2': roll_detail['group_nos'][1][:15],
                    'g3': roll_detail['group_nos'][2][:15], 'g4': roll_detail['group_nos'][3][:15],
                    'g5': roll_detail['group_nos'][4][:15], 'g6': roll_detail['group_nos'][5][:15],
                    'g7': roll_detail['group_nos'][6][:15]
                }
                bind_vars_list.append(bind_vars)
        
        if bind_vars_list:
            cursor.executemany(insert_query, bind_vars_list)

        print(f"Prepared {len(bind_vars_list)} new roll sequences for transaction.")

    def insert_cut_sequence(self, connection, lot_no, version, plant, pm_no, schedule_unit, 
                                paper_type, b_wgt, pattern_roll_cut_details):
        cursor = connection.cursor()

        insert_query = """
            insert into th_cut_sequence (
                module, plant, pm_no, schedule_unit, lot_no, version, 
                prod_seq, unit_no, seq, roll_seq, cut_seq, width, group_no, 
                weight,
                spool_no, spool_seq, p_machine,
                total_length, cut_cnt, paper_type, b_wgt,
                color, luster, p_lot, p_type, p_wgt
            ) values (
                'C', :plant, :pm_no, :schedule_unit, :lot_no, :version, 
                :prod_seq, :unit_no, :seq, :roll_seq, :cut_seq, :width, :group_no, 
                round(:b_wgt * :width * :total_length / 1000000,1),
                :spool_no, :spool_seq, :p_machine,
                :total_length, :cut_cnt, :paper_type, :b_wgt,
                :color, :luster, :p_lot, :p_type, :p_wgt
            )
        """

        bind_vars_list = []
        for cut_detail in pattern_roll_cut_details:
            for seq in range(cut_detail['count']):
                bind_vars = {
                    'plant': plant, 'pm_no': pm_no, 'schedule_unit': schedule_unit,
                    'lot_no': lot_no, 'version': version,
                    'prod_seq': cut_detail['prod_seq'], 
                    'unit_no': cut_detail['unit_no'],
                    'seq': seq + 1, 
                    'spool_no': cut_detail['prod_seq'], 
                    'spool_seq': seq + 1,
                    'p_machine': pm_no,
                    'roll_seq': cut_detail['roll_seq'],
                    'cut_seq': cut_detail['cut_seq'], 
                    'width': cut_detail['width'],
                    'group_no': cut_detail['group_no'], 
                    'total_length': cut_detail['pattern_length'],
                    'cut_cnt': 1, 
                    'paper_type': paper_type,
                    'b_wgt': b_wgt,
                    'color': str(cut_detail.get('color', '')),
                    'luster': int(cut_detail.get('luster', 0)),
                    'p_lot': lot_no,
                    'p_type': paper_type,
                    'p_wgt': b_wgt
                }
                bind_vars_list.append(bind_vars)

        if bind_vars_list:
            cursor.executemany(insert_query, bind_vars_list)

        print(f"Prepared {len(bind_vars_list)} new cut sequences for transaction.")


    def insert_sheet_sequence(self, connection, lot_no, version, plant, schedule_unit):
        cursor = connection.cursor()
        
        # As per the user's request, this function will call a stored procedure.
        # The user did not specify if old data should be deleted.

        # Call the stored procedure with named parameters to ensure correctness
        cursor.callproc("SP_JP_GEN_SHEETINFO_BUF", keyword_parameters={
            'a_module': 'C',
            'a_plant': plant,
            'a_pm_no': ' ',
            'a_schedule_unit': schedule_unit,
            'a_lot_no': lot_no,
            'a_version': version
        })

        # Create a variable for the IN OUT cursor parameter
        out_cursor = cursor.var(oracledb.DB_TYPE_CURSOR)

        # Call the stored procedure with named parameters to ensure correctness
        cursor.callproc("PKG_JP_INOUT_MANAGER.SP_JP_GEN_SPOOL_NO", keyword_parameters={
            'P_PLANT': plant,
            'P_SCHEDULE_UNIT': schedule_unit,
            'P_LOT_NO': lot_no,
            'P_VERSION': version,
            'C_SN': out_cursor
        })

        # You can now fetch results from the out_cursor if needed, for example:
        result_cursor = out_cursor.getvalue()
        for row in result_cursor:
            print(row)

        print(f"Prepared sheet sequences for transaction by calling PKG_JP_INOUT_MANAGER.SP_JP_GEN_SPOOL_NO.")

    def insert_order_group(self, connection, lot_no, version, plant, pm_no, schedule_unit, df_orders):
        """
        df_orders DataFrame을 th_order_group 테이블에 저장합니다.
        """
        cursor = connection.cursor()

        insert_query = """
            insert into th_order_group (
                plant, pm_no, schedule_unit, lot_no, version, group_no, order_no
            ) values (
                :plant, :pm_no, :schedule_unit, :lot_no, :version, :group_no, :order_no
            )
        """

        df_copy = df_orders.copy()
        df_copy['lot_no'] = lot_no
        df_copy['version'] = version

        # DataFrame 컬럼 이름을 DB 컬럼에 맞게 매핑합니다.
        rename_map = {
            'group_order_no': 'group_no'
        }
        
        df_to_insert = df_copy.rename(columns={k: v for k, v in rename_map.items() if k in df_copy.columns})
        
        # DB에 저장할 최종 컬럼 목록을 선택합니다.
        final_cols = ['lot_no', 'version', 'plant', 'pm_no', 'schedule_unit', 'group_no', 'order_no']
        
        print(f"Before drop_duplicates: {len(df_to_insert)} rows")
        # 중복 제거 (PK 위반 방지)
        df_to_insert = df_to_insert.drop_duplicates(subset=final_cols)
        print(f"After drop_duplicates: {len(df_to_insert)} rows")
        
        # Check for remaining duplicates (should be 0)
        dups = df_to_insert[df_to_insert.duplicated(subset=final_cols, keep=False)]
        if not dups.empty:
            print(f"WARNING: Duplicates found after drop_duplicates:\n{dups}")

        # 데이터 타입 변환 (DB_TYPE_NUMBER 오류 방지)
        # plant, lot_no, group_no 등이 숫자로 된 문자열일 경우 숫자로 변환 시도
        # version, schedule_unit, order_no 도 추가
        for col in ['plant', 'lot_no', 'group_no', 'version', 'schedule_unit', 'order_no']:
            if col in df_to_insert.columns:
                try:
                    df_to_insert[col] = pd.to_numeric(df_to_insert[col])
                except Exception:
                    pass # 변환 실패 시 원래 값 유지 (문자열이 필요한 컬럼일 수도 있음)

        bind_vars_list = df_to_insert[final_cols].to_dict('records')

        if bind_vars_list:
            print(f"DEBUG: First record to insert: {bind_vars_list[0]}")
            cursor.executemany(insert_query, bind_vars_list)
            print(f"Prepared {len(bind_vars_list)} new order group records for transaction.")

    

import oracledb

class DataInserters:
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
                    lot_no, version, prod_seq, unit_no, seq, length, pok_cnt, 
                    wd_width, 
                    rollwidth1, rollwidth2, rollwidth3, rollwidth4, rollwidth5, rollwidth6, rollwidth7, rollwidth8,
                    groupno1, groupno2, groupno3, groupno4, groupno5, groupno6, groupno7, groupno8
                ) VALUES (
                    'C', :plant, :pm_no, :schedule_unit, :max_width, :paper_type, :b_wgt,
                    :lot_no, :version, :prod_seq, :unit_no, :seq, :length, :pok_cnt,
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
                    prod_seq = pattern['Prod_seq']
                    # print(f"Prod_seq:{prod_seq}")
                    
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
                        'prod_seq': prod_seq,
                        'unit_no': prod_seq,
                        'seq': total_seq,
                        'length': pattern.get('roll_production_length', 0),
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

    def insert_roll_sequence(self, lot_no, version, plant, pm_no, schedule_unit, max_width, 
                                paper_type, b_wgt, pattern_roll_details):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

            cursor.execute("DELETE FROM th_roll_sequence WHERE lot_no = :lot_no AND version = :version", lot_no=lot_no, version=version)
            print(f"Deleted existing roll_sequence for lot {lot_no}, version {version}")

            # This function will insert the detailed roll information. 
            # To avoid conflicts with the main insert, this will NOT delete existing records.
            # It will also start prod_seq from a higher number.
            
            insert_query = """
                INSERT INTO th_roll_sequence (
                    module, plant, pm_no, schedule_unit, paper_type, b_wgt,
                    lot_no, version, prod_seq, unit_no, seq, roll_seq, pok_cnt,
                    rollwidth, 
                    width1, width2, width3, width4, width5, width6, width7,
                    group1, group2, group3, group4, group5, group6, group7
                ) VALUES (
                    'R', :plant, :pm_no, :schedule_unit, :paper_type, :b_wgt,
                    :lot_no, :version, :prod_seq, :unit_no, :seq, :roll_seq, :pok_cnt,
                    :rollwidth,
                    :w1, :w2, :w3, :w4, :w5, :w6, :w7,
                    :g1, :g2, :g3, :g4, :g5, :g6, :g7
                )
            """

            for roll_detail in pattern_roll_details:
                # print(f"Number of pattern details: {pattern['Count']}")
                for seq in range(roll_detail['Count']):
                    prod_seq = roll_detail['Prod_seq']
                    roll_seq = roll_detail['Roll_seq']
                    # roll_detail.get('Roll_seq', 1)

                    pok_cnt_value = len([w for w in roll_detail['widths'] if w > 0])

                    bind_vars = {
                        'plant': plant,
                        'pm_no': pm_no,
                        'schedule_unit': schedule_unit,
                        'paper_type': paper_type,
                        'b_wgt': b_wgt,
                        'lot_no': lot_no,
                        'version': version,
                        'prod_seq': prod_seq,
                        'unit_no': prod_seq,
                        'seq':seq,
                        'roll_seq': roll_seq,
                        'pok_cnt': pok_cnt_value,
                        'rollwidth': roll_detail['rollwidth'],
                        'w1': roll_detail['widths'][0], 'w2': roll_detail['widths'][1],
                        'w3': roll_detail['widths'][2], 'w4': roll_detail['widths'][3],
                        'w5': roll_detail['widths'][4], 'w6': roll_detail['widths'][5],
                        'w7': roll_detail['widths'][6],
                        'g1': roll_detail['group_nos'][0][:15], 'g2': roll_detail['group_nos'][1][:15],
                        'g3': roll_detail['group_nos'][2][:15], 'g4': roll_detail['group_nos'][3][:15],
                        'g5': roll_detail['group_nos'][4][:15], 'g6': roll_detail['group_nos'][5][:15],
                        'g7': roll_detail['group_nos'][6][:15]
                    }
                    cursor.execute(insert_query, bind_vars)

            connection.commit()
            print(f"Successfully inserted {len(pattern_roll_details)} new roll sequences.")
            return True

        except oracledb.Error as error:
            print(f"Error while inserting roll sequence: {error}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.pool.release(connection)

    def insert_cut_sequence(self, lot_no, version, plant, pm_no, schedule_unit, 
                                paper_type, b_wgt, pattern_roll_cut_details):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

            cursor.execute("DELETE FROM th_cut_sequence WHERE lot_no = :lot_no AND version = :version", lot_no=lot_no, version=version)
            print(f"Deleted existing cut_sequence for lot {lot_no}, version {version}")

            insert_query = """
                INSERT INTO th_cut_sequence (
                    MODULE, PLANT, PM_NO, SCHEDULE_UNIT, LOT_NO, VERSION, 
                    PROD_SEQ, UNIT_NO, SEQ, ROLL_SEQ, CUT_SEQ, WIDTH, GROUP_NO, 
                    WEIGHT, TOTAL_LENGTH, CUT_CNT, PAPER_TYPE, B_WGT
                ) VALUES (
                    'C', :plant, :pm_no, :schedule_unit, :lot_no, :version, 
                    :prod_seq, :unit_no, :seq, :roll_seq, :cut_seq, :width, :group_no, 
                    :weight, :total_length, :cut_cnt, :paper_type, :b_wgt
                )
            """

            for cut_detail in pattern_roll_cut_details:
                bind_vars = {
                    'plant': plant,
                    'pm_no': pm_no,
                    'schedule_unit': schedule_unit,
                    'lot_no': lot_no,
                    'version': version,
                    'prod_seq': cut_detail['PROD_SEQ'],
                    'unit_no': cut_detail['UNIT_NO'],
                    'seq': cut_detail['SEQ'],
                    'roll_seq': cut_detail['ROLL_SEQ'],
                    'cut_seq': cut_detail['CUT_SEQ'],
                    'width': cut_detail['WIDTH'],
                    'group_no': cut_detail['GROUP_NO'],
                    'weight': cut_detail['WEIGHT'],
                    'total_length': cut_detail['TOTAL_LENGTH'],
                    'cut_cnt': cut_detail['CUT_CNT'],
                    'paper_type': paper_type,
                    'b_wgt': b_wgt
                }
                cursor.execute(insert_query, bind_vars)

            connection.commit()
            print(f"Successfully inserted {len(pattern_roll_cut_details)} new cut sequences.")
            return True

        except oracledb.Error as error:
            print(f"Error while inserting cut sequence: {error}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.pool.release(connection)

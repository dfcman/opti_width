import oracledb
import pandas as pd

class DataInserters:
    """천안 공장(5000) 전용 데이터 INSERT 함수들"""
    
    def insert_pattern_sequence(self, connection, lot_no, version, plant, pm_no, schedule_unit, max_width, 
                                paper_type, b_wgt, pattern_details, 
                                p_machine=None, p_type=None, p_wgt=None, p_color=None):
        cursor = connection.cursor()

        insert_query = """
            insert into th_pattern_sequence (
                module, plant, pm_no, schedule_unit, max_width, paper_type, b_wgt,
                lot_no, version, prod_seq, unit_no, seq, length, pok_cnt, 
                diameter, cut_cnt, color, luster, p_lot, p_type, p_wgt, p_color, core, pattern,
                wd_width, 
                wd_trim, 
                cut_width, 
                cut_trim, 
                prod_amt, 
                prod_wgt, 
                prod_yn, 
                jaego_yn,
                spool_no, spool_seq, rs_gubun, p_machine,
                rollwidth1, rollwidth2, rollwidth3, rollwidth4, rollwidth5, rollwidth6, rollwidth7, rollwidth8,
                groupno1, groupno2, groupno3, groupno4, groupno5, groupno6, groupno7, groupno8
            ) values (
                'C', :plant, :pm_no, :schedule_unit, :max_width, :paper_type, :b_wgt,
                :lot_no, :version, :prod_seq, :unit_no, :seq, :length, :pok_cnt,
                :diameter, :cut_cnt, :color, :luster, :p_lot, :p_type, :p_wgt, :p_color, :core, :pattern,
                :w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8,
                :max_width - :w1 - :w2 - :w3 - :w4 - :w5 - :w6 - :w7 - :w8, 
                :w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8,
                :max_width - :w1 - :w2 - :w3 - :w4 - :w5 - :w6 - :w7 - :w8, 
                round(:b_wgt * (:w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8) * :length / 1000000, 0),
                round(:b_wgt * (:w1 + :w2 + :w3 + :w4 + :w5 + :w6 + :w7 + :w8) * :length / 1000000, 1),
                'N',
                'N',
                :spool_no, :spool_seq, :rs_gubun, :p_machine,
                :w1, :w2, :w3, :w4, :w5, :w6, :w7, :w8,
                :g1, :g2, :g3, :g4, :g5, :g6, :g7, :g8
            )
        """
        
        
        total_seq = 0
        bind_vars_list = []
        for pattern in pattern_details:
            # for seq in range(pattern['count']):
                total_seq += 1
                prod_seq = pattern['prod_seq']
                pok_cnt_value = len([w for w in pattern['widths'] if w > 0])

                bind_vars = {
                    'plant': plant, 'pm_no': pm_no, 'schedule_unit': schedule_unit,
                    'max_width': pattern.get('max_width', max_width), 'paper_type': paper_type, 'b_wgt': b_wgt,
                    'lot_no': lot_no, 'version': version, 'prod_seq': prod_seq,
                    'unit_no': prod_seq, 'seq': 1,
                    'length': pattern.get('pattern_length', 0),
                    'pok_cnt': pok_cnt_value,
                    'diameter': int(pattern.get('diameter', 0)), 
                    'cut_cnt': pattern['count'],
                    'color': str(pattern.get('color', '')),
                    'luster': int(pattern.get('luster', 0)),
                    'p_lot': str(pattern.get('p_lot', '')), 
                    'p_type': p_type,
                    'p_wgt': p_wgt, 
                    'p_color': p_color,
                    'core': int(pattern.get('core', 0)),
                    'pattern': str(pattern.get('order_pattern', '')),
                    'spool_no': prod_seq, 
                    'spool_seq': 1,
                    'rs_gubun': pattern['rs_gubun'],
                    'p_machine': p_machine, 
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
            print(f"[DEBUG] Inserted {len(bind_vars_list)} rows into th_pattern_sequence.")
            
        print(f"Prepared {total_seq} new pattern sequences for transaction.")

    def insert_roll_sequence(self, connection, lot_no, version, plant, pm_no, schedule_unit, max_width, 
                                paper_type, b_wgt, pattern_roll_details,
                                p_machine=None, p_type=None, p_wgt=None, p_color=None):
        cursor = connection.cursor()

        insert_query = """
            insert into  th_roll_sequence (
                module, plant, pm_no, schedule_unit, paper_type, b_wgt,
                lot_no, version, prod_seq, unit_no, seq, roll_seq, pok_cnt,
                rollwidth, length, spool_no, spool_seq, rs_gubun,
                dia, weight, trim_loss, sc_trim, sl_trim, sl_cut_yn,
                cut_cnt, color, luster, core, pattern, p_lot, p_type, p_wgt, p_color, p_machine, 
                cmin_yn, cmin_trim, cmin_sl_yn, cmin_pok,
                rs_gubun1, rs_gubun2, rs_gubun3, rs_gubun4, rs_gubun5, rs_gubun6, rs_gubun7,
                width1, width2, width3, width4, width5, width6, width7,
                roll_width1, roll_width2, roll_width3, roll_width4, roll_width5, roll_width6, roll_width7,
                group1, group2, group3, group4, group5, group6, group7
            ) values (
                'C', :plant, :pm_no, :schedule_unit, :paper_type, :b_wgt,
                :lot_no, :version, :prod_seq, :unit_no, :seq, :roll_seq, :pok_cnt,
                :rollwidth, :length, :spool_no, :spool_seq, :rs_gubun,
                :dia, round(:b_wgt * :rollwidth * :length / 1000000,1), :trim_loss, :sc_trim, :sl_trim, :sl_cut_yn,
                :cut_cnt, :color, :luster, :core, :pattern, :p_lot, :p_type, :p_wgt, :p_color, :p_machine, 
                :cmin_yn, :cmin_trim, :cmin_sl_yn, :cmin_pok,
                :rs_gubun1, :rs_gubun2, :rs_gubun3, :rs_gubun4, :rs_gubun5, :rs_gubun6, :rs_gubun7, 
                :w1, :w2, :w3, :w4, :w5, :w6, :w7,
                :roll_width1, :roll_width2, :roll_width3, :roll_width4, :roll_width5, :roll_width6, :roll_width7,
                :g1, :g2, :g3, :g4, :g5, :g6, :g7
            )
        """

        bind_vars_list = []
        for roll_detail in pattern_roll_details:
            prod_seq = roll_detail['prod_seq']
            roll_seq = roll_detail['roll_seq']
            pok_cnt_value = len([w for w in roll_detail['widths'] if w > 0])

        bind_vars_list = []
        for roll_detail in pattern_roll_details:
            prod_seq = roll_detail['prod_seq']
            roll_seq = roll_detail['roll_seq']
            pok_cnt_value = len([w for w in roll_detail['widths'] if w > 0])

            bind_vars = {
                'plant': plant, 'pm_no': pm_no, 'schedule_unit': schedule_unit,
                'paper_type': paper_type, 'b_wgt': b_wgt, 'lot_no': lot_no,
                'version': version, 'prod_seq': prod_seq, 'unit_no': prod_seq,
                'seq': 1, 'roll_seq': roll_seq, 'pok_cnt': pok_cnt_value,
                'rollwidth': roll_detail['rollwidth'],
                'length': roll_detail['pattern_length'],
                'spool_no': prod_seq, 
                'spool_seq': 1,
                'dia': int(roll_detail.get('diameter', 0)), 
                'cut_cnt': roll_detail['count'],
                'color': str(roll_detail.get('color', '')),
                'luster': int(roll_detail.get('luster', 0)),
                'core': int(roll_detail.get('core', 0)),
                'pattern': str(roll_detail.get('order_pattern', '')),
                'p_lot': lot_no,
                'p_type': p_type,
                'p_wgt': p_wgt,  
                'p_machine': p_machine,
                'p_color': p_color,
                'trim_loss': roll_detail.get('trim_loss', 0),
                'sc_trim': roll_detail.get('sc_trim', 0),
                'sl_trim': roll_detail.get('sl_trim', 0),
                'sl_cut_yn': 'Y' if roll_detail['rs_gubun'] == "W" else 'N',
                'cmin_yn': 'N',
                'cmin_trim': 0,
                'cmin_sl_yn': 'N',
                'cmin_pok': 0,
                'rs_gubun': roll_detail['rs_gubun'],
                'rs_gubun1': roll_detail['rs_gubuns'][0], 
                'rs_gubun2': roll_detail['rs_gubuns'][1],
                'rs_gubun3': roll_detail['rs_gubuns'][2],
                'rs_gubun4': roll_detail['rs_gubuns'][3],
                'rs_gubun5': roll_detail['rs_gubuns'][4],
                'rs_gubun6': roll_detail['rs_gubuns'][5],
                'rs_gubun7': roll_detail['rs_gubuns'][6],                
                'w1': roll_detail['widths'][0], 
                'w2': roll_detail['widths'][1],
                'w3': roll_detail['widths'][2], 
                'w4': roll_detail['widths'][3],
                'w5': roll_detail['widths'][4], 
                'w6': roll_detail['widths'][5],
                'w7': roll_detail['widths'][6],
                'roll_width1': roll_detail['roll_widths'][0]  if roll_detail['rs_gubun'] == "W" else int(roll_detail['rollwidth']) - int(roll_detail.get('sl_trim', 0)),
                'roll_width2': roll_detail['roll_widths'][1],
                'roll_width3': roll_detail['roll_widths'][2], 
                'roll_width4': roll_detail['roll_widths'][3],
                'roll_width5': roll_detail['roll_widths'][4], 
                'roll_width6': roll_detail['roll_widths'][5],
                'roll_width7': roll_detail['roll_widths'][6],
                'g1': roll_detail['group_nos'][0][:15], 
                'g2': roll_detail['group_nos'][1][:15],
                'g3': roll_detail['group_nos'][2][:15], 
                'g4': roll_detail['group_nos'][3][:15],
                'g5': roll_detail['group_nos'][4][:15], 
                'g6': roll_detail['group_nos'][5][:15],
                'g7': roll_detail['group_nos'][6][:15]
            }
            bind_vars_list.append(bind_vars)
        
        if bind_vars_list:
            cursor.executemany(insert_query, bind_vars_list)
            print(f"[DEBUG] Inserted {len(bind_vars_list)} rows into th_roll_sequence.")

        print(f"Prepared {len(bind_vars_list)} new roll sequences for transaction.")

    def insert_cut_sequence(self, connection, lot_no, version, plant, pm_no, schedule_unit, 
                                paper_type, b_wgt, pattern_roll_cut_details,
                                p_machine=None, p_type=None, p_wgt=None, p_color=None):
        cursor = connection.cursor()

        insert_query = """
            insert into th_cut_sequence (
                module, plant, pm_no, schedule_unit, lot_no, version, 
                prod_seq, unit_no, seq, roll_seq, cut_seq, width, group_no, 
                weight,
                spool_no, spool_seq, p_machine,
                total_length, cut_cnt, paper_type, b_wgt,
                color, luster, p_lot, p_type, p_wgt, p_color
            ) values (
                'C', :plant, :pm_no, :schedule_unit, :lot_no, :version, 
                :prod_seq, :unit_no, :seq, :roll_seq, :cut_seq, :width, :group_no, 
                round(:b_wgt * :width * :total_length / 1000000,1),
                :spool_no, :spool_seq, :p_machine,
                :total_length, :cut_cnt, :paper_type, :b_wgt,
                :color, :luster, :p_lot, :p_type, :p_wgt, :p_color
            )
        """

        bind_vars_list = []
        for cut_detail in pattern_roll_cut_details:
            bind_vars = {
                'plant': plant, 'pm_no': pm_no, 'schedule_unit': schedule_unit,
                'lot_no': lot_no, 'version': version,
                'prod_seq': cut_detail['prod_seq'], 
                'unit_no': cut_detail['unit_no'],
                'seq': 1, 
                'spool_no': cut_detail['prod_seq'], 
                'spool_seq': 1,
                'p_machine': p_machine,
                'roll_seq': cut_detail['roll_seq'],
                'cut_seq': cut_detail['cut_seq'], 
                'width': cut_detail['width'],
                'group_no': cut_detail['group_no'], 
                'total_length': cut_detail['pattern_length'],
                'cut_cnt': cut_detail['count'], 
                'paper_type': paper_type,
                'b_wgt': b_wgt,
                'color': str(cut_detail.get('color', '')),
                'luster': int(cut_detail.get('luster', 0)),
                'p_lot': lot_no,
                'p_type': p_type,
                'p_wgt': p_wgt,
                'p_color': p_color
            }
            bind_vars_list.append(bind_vars)

        if bind_vars_list:
            cursor.executemany(insert_query, bind_vars_list)

        print(f"Prepared {len(bind_vars_list)} new cut sequences for transaction.")


    def insert_sheet_sequence(self, connection, lot_no, version, plant, pm_no, schedule_unit, 
                               paper_type, b_wgt, sheet_details,
                               p_machine=None, p_type=None, p_wgt=None, p_color=None):
        cursor = connection.cursor()

        insert_query = """
            insert into th_sheet_sequence (
                module, plant, pm_no, schedule_unit, lot_no, version, 
                prod_seq, unit_no, seq, roll_seq, cut_seq, sheet_seq, pack_type, width, 
                group_no, order_no,
                weight, sheet_cnt, cut_cnt,
                spool_no, spool_seq, p_machine,
                length, paper_type, b_wgt,
                color, luster, p_lot, p_type, p_wgt, p_color
            ) values (
                'C', :plant, :pm_no, :schedule_unit, :lot_no, :version, 
                :prod_seq, :unit_no, :seq, :roll_seq, :cut_seq, :sheet_seq, :pack_type, :width, 
                :group_no, :order_no,
                round(:b_wgt * :width * :length / 1000000,1), :sheet_cnt, :cut_cnt,
                :spool_no, :spool_seq, :p_machine,
                :length, :paper_type, :b_wgt,
                :color, :luster, :p_lot, :p_type, :p_wgt, :p_color
            )
        """

        bind_vars_list = []
        for sheet_detail in sheet_details:
            current_seq = sheet_detail.get('override_seq', 1)
            bind_vars = {
                    'plant': plant, 'pm_no': pm_no, 'schedule_unit': schedule_unit,
                    'lot_no': lot_no, 'version': version,
                    'prod_seq': sheet_detail['prod_seq'], 
                    'unit_no': sheet_detail['unit_no'],
                    'seq': current_seq, 
                    'spool_no': sheet_detail['prod_seq'], 
                    'spool_seq': current_seq,
                    'p_machine': p_machine,
                    'roll_seq': sheet_detail['roll_seq'],
                    'cut_seq': sheet_detail['cut_seq'], 
                    'sheet_seq': sheet_detail['sheet_seq'],
                    'pack_type': sheet_detail['pack_type'],
                    'width': sheet_detail['width'],
                    'group_no': sheet_detail['group_no'], 
                    'order_no': sheet_detail['order_no'],
                    'sheet_cnt': sheet_detail['sheet_cnt'],
                    'cut_cnt': sheet_detail.get('count', 1),
                    'length': sheet_detail['pattern_length'],
                    'paper_type': paper_type,
                    'b_wgt': b_wgt,
                    'color': str(sheet_detail.get('color', '')),
                    'luster': int(sheet_detail.get('luster', 0)),
                    'p_lot': lot_no,
                    'p_type': p_type,
                    'p_wgt': p_wgt,
                    'p_color': p_color
            }
            bind_vars_list.append(bind_vars)

        if bind_vars_list:
            cursor.executemany(insert_query, bind_vars_list)

        print(f"Prepared {len(bind_vars_list)} sheet sequences for transaction.")
        
        # As per the user's request, this function will call a stored procedure.
        # The user did not specify if old data should be deleted.

        # # Call the stored procedure with named parameters to ensure correctness
        # cursor.callproc("SP_JP_GEN_SHEETINFO_BUF", keyword_parameters={
        #     'a_module': 'C',
        #     'a_plant': plant,
        #     'a_pm_no': pm_no,
        #     'a_schedule_unit': schedule_unit,
        #     'a_lot_no': lot_no,
        #     'a_version': version
        # })

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
        print(f"[DEBUG] Sheet sequence generation procedure called for Lot {lot_no}, Version {version}, PM {pm_no}.")

    def insert_order_group(self, connection, lot_no, version, plant, pm_no, schedule_unit, df_orders):
        """
        df_orders DataFrame을 th_order_group 테이블에 저장합니다.
        """
        cursor = connection.cursor()

        insert_query = """
            insert into th_order_group (
                plant, pm_no, schedule_unit, lot_no, version, group_no, order_no, prod_wgt
            ) values (
                :plant, :pm_no, :schedule_unit, :lot_no, :version, :group_no, :order_no, :prod_wgt
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
        
        # prod_wgt 컬럼이 없으면 0으로 초기화
        if 'prod_wgt' not in df_to_insert.columns:
            df_to_insert['prod_wgt'] = 0
        
        # DB에 저장할 최종 컬럼 목록을 선택합니다.
        final_cols = ['lot_no', 'version', 'plant', 'pm_no', 'schedule_unit', 'group_no', 'order_no', 'prod_wgt']
        
        print(f"Before drop_duplicates: {len(df_to_insert)} rows")
        # 중복 제거 (PK 위반 방지)
        df_to_insert = df_to_insert.drop_duplicates(subset=['lot_no', 'version', 'plant', 'pm_no', 'schedule_unit', 'group_no', 'order_no'])
        print(f"After drop_duplicates: {len(df_to_insert)} rows")
        
        # Check for remaining duplicates (should be 0)
        dups = df_to_insert[df_to_insert.duplicated(subset=['lot_no', 'version', 'plant', 'pm_no', 'schedule_unit', 'group_no', 'order_no'], keep=False)]
        if not dups.empty:
            print(f"WARNING: Duplicates found after drop_duplicates:\\n{dups}")

        bind_vars_list = df_to_insert[final_cols].to_dict('records')

        if bind_vars_list:
            # print(f"DEBUG: First record to insert: {bind_vars_list[0]}")
            cursor.executemany(insert_query, bind_vars_list)
            print(f"Prepared {len(bind_vars_list)} new order group records for transaction.")

    def insert_group_master(self, connection, lot_no, version, plant, pm_no, schedule_unit, df_groups):
        """
        df_groups DataFrame을 th_group_master 테이블에 저장합니다.
        """
        cursor = connection.cursor()

        insert_query = """
            insert into th_group_master (
                plant, schedule_unit, lot_no, version, group_no,
                paper_type, b_wgt, width, length, trim,
                rs_gubun, export, nation_code, customer,
                pt_gubun, skid_yn, dia, core, order_no
            ) values (
                :plant, :schedule_unit, :lot_no, :version, :group_no,
                :paper_type, :b_wgt, :width, :length, :trim,
                :rs_gubun, :export, :nation_code, :customer,
                :pt_gubun, :skid_yn, :dia, :core, :order_no
            )
        """

        df_copy = df_groups.copy()
        df_copy['lot_no'] = lot_no
        df_copy['version'] = version
        df_copy['plant'] = plant
        df_copy['pm_no'] = pm_no
        df_copy['schedule_unit'] = schedule_unit

        # DataFrame 컬럼 이름을 DB 컬럼에 맞게 매핑
        rename_map = {
            'group_order_no': 'group_no',
            '가로': 'width',
            '세로': 'length',
            'customer_name': 'customer',
            # 'export_yn': 'export' (will handle logic below)
        }
        
        df_to_insert = df_copy.rename(columns={k: v for k, v in rename_map.items() if k in df_copy.columns})
        
        # trim column logic?
        # Maybe use 'trim_loss' or 'trim_size' if available, else 0
        if 'trim' not in df_to_insert.columns:
            df_to_insert['trim'] = 0

        if 'export' not in df_to_insert.columns:
            if 'export_yn' in df_to_insert.columns:
                df_to_insert['export'] = df_to_insert['export_yn']
            else:
                df_to_insert['export'] = 'N' # Default value if missing

        required_cols = [
            'plant', 'schedule_unit', 'lot_no', 'version', 'group_no',
            'paper_type', 'b_wgt', 'width', 'length', 
            'rs_gubun', 'nation_code', 'customer',
            'pt_gubun', 'skid_yn', 'dia', 'core', 'order_no'
        ]

        # Ensure all columns exist
        for col in required_cols:
            if col not in df_to_insert.columns:
                # logging.warning(f"Missing column {col} for group master insert. Filling with None/0.")
                if col in ['width', 'length', 'dia', 'core', 'b_wgt']:
                    df_to_insert[col] = 0
                else:
                    df_to_insert[col] = ''

        # Handle NaNs
        df_to_insert = df_to_insert.fillna({
            'width': 0, 'length': 0, 'dia':0, 'core':0, 'b_wgt':0,
            'rs_gubun': '', 'nation_code': '', 'customer': '', 'pt_gubun': '', 'skid_yn': '', 'order_no': ''
        })

        bind_vars_list = df_to_insert[required_cols + ['trim', 'export']].to_dict('records')

        if bind_vars_list:
            # print(f"DEBUG: First group master record to insert: {bind_vars_list[0]}")
            cursor.executemany(insert_query, bind_vars_list)
        
        print(f"Prepared {len(bind_vars_list)} new group master records for transaction.")

    

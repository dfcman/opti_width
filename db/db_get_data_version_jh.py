import oracledb

class VersionGettersJh:
    def get_target_lot_jh(self):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

            query = """

                select 
                    v.plant, v.pm_no, v.schedule_unit, v.lot_no, v.version, 
                    v.paper_type, v.b_wgt, v.color,
                    v.min_re_width                              as re_minwidth,                 
                    nvl(v.max_width,4850)                       as re_maxwidth,
                    v.min_re_count                              as re_min_pieces,
                    v.max_re_count                              as re_max_pieces,
                    v.sheet_min_re_pok_cnt                      as re_sheet_min_pieces,
                    nvl(v.sheet_max_re_pok_cnt,4)               as re_sheet_max_pieces,
                    v.min_ptn_wgt                               as re_minton,           -- 패턴 최소중량
                    v.min_cmb_wgt                               as sc_minton,           -- 규격별 최소중량
                    nvl(v.is_std_length, 'N')                   as yn_stdlength,        -- 표준길이 적용여부
                    nvl(l.sheet_length_re,0)                    as min_re_stdlength,    -- 표준길이 최소길이
                    nvl(l.sheet_length_re,0)                    as max_re_stdlength,    -- 표준길이 최대길이
                    v.sheet_order_moq                           as sheet_order_moq,     -- 쉬트 최소주문량
                    v.adjacent_width                            as adj_trim,            -- 인접그룹 조합여부: 값이 0이면 인접그룹 미조합, 0보다 크면 인접그룹 조합
                    v.adjacent_weight_min                       as adj_min_wgt,         -- 인접그룹 조합시 최소 오더톤 제약
                    v.adjacent_weight                           as adj_max_wgt,         -- 인접그룹 조합시 최대 오더톤 제약
                    v.moq_yn                                    as moq_yn,              -- MOQ 미달 지폭여부
                    v.moq_ton                                   as moq_ton,             -- MOQ 오더톤
                    v.moq_sc_width                              as moq_sc_width,        -- MOQ 오더 지폭
                    nvl(v.rs_mix_flag, 'N')                     as yn_rstypemix,        -- RS 혼합 여부

                    v.trim_size                                 as sc_basetrim,
                    v.min_sc_width                              as sc_minwidth,
                    v.max_sc_width                              as sc_maxwidth,
                    1                                           as sc_minsheet,
                    v.max_sc_count                              as sc_maxsheet,
                    nvl(v.ww_trim_size,0)                       as sl_trim,  
                    nvl(v.min_sl_width,765)                     as sl_minwidth,
                    nvl(v.max_sl_width,1800)                    as sl_maxwidth,
                    nvl(v.min_sl_count,1)                       as min_sl_count,
                    nvl(v.max_sl_count,4)                       as max_sl_count,
                    nvl(v.sheet_trim_size_digital, trim_size)   as sc_digital_trim,
                    nvl(v.min_sc_width_digital, 700)            as sc_digital_minwidth,
                    nvl(v.max_sc_width_digital, 1850)           as sc_digital_maxwidth,
                    1                                           as sc_digital_minsheet,
                    nvl(v.max_sc_count_digital, 7)              as sc_digital_maxsheet,  


                    decode(nvl(v.db_cut_rate,0),1,1,0)          as sl_rollmixtype,
                    nvl(v.rs_mix_maxpok, 1)                     as mx_maxsheetroll,
                    case when v.mfc_flag = 'Y'
                            then v.max_non_odd
                        else v.max_sc_width
                    end                                         as mx_maxsheetwidth,
                    nvl(re_rollmixwidth, 4720)                  as mx_rsmixwidth,
                    nvl(v.re_avgcutwidth,v.min_re_width)        as re_avgcutwidth,
                    nvl(v.re_maxpatternratio,1.0)               as re_maxpatternratio,
                    nvl(v.ww_roll_limit_yn, 'N')                as yn_sroll,
                    nvl(v.coating_yn, 'N')                      as yn_coating,
                    r.cutout_width_max                          as re_jumbowidth,


                    nvl(v.min_re_count,2)                       as re_minroll,
                    nvl(v.sheet_max_re_pok_cnt,4)               as re_maxroll,
                    v.max_re_count                              as re_extroll,
                    nvl(v.sheet_std_less,2)                     as less_reg_ton,
                    nvl(v.sheet_std_more,2)                     as more_reg_ton,
                    nvl(v.sheet_nostd_less,0.5)                 as less_var_ton,
                    nvl(v.sheet_nostd_more,0.5)                 as more_var_ton,
                    nvl(v.roll_std_less,2)                      as less_reg_roll,
                    nvl(v.roll_std_more,2)                      as more_reg_roll,
                    nvl(v.roll_nostd_less,0)                    as less_var_roll,
                    nvl(v.roll_nostd_more,0)                    as more_var_roll,
                    -1*nvl(v.time_limit,60)                     as time_order,
                    -1*nvl(v.time_limit,60)                     as time_trim,
                    -1*nvl(v.time_limit,60)                     as time_roll,
                    -1*nvl(v.time_limit,60)                     as time_pattern
                from th_versions_manager v, 
                    th_tar_resource r, 
                    th_tar_std_length l
                where v.calc_successful = '9'
                and v.version not in ('98', '99')
                and r.operation_code = 'PM' 
                and r.resource_code(+) = v.pm_no 
                and l.paper_type(+) = v.paper_type 
                and l.b_wgt(+) = v.b_wgt
                ORDER BY v.plant, v.version_id, v.schedule_unit, v.lot_no, v.version
                FETCH FIRST 1 ROWS ONLY

            """

            # print(f"Executing query to fetch target lot:\n{query}")
            print(f"Executing query to fetch target lot")
            cursor.execute(query)
            columns = [col[0].lower() for col in cursor.description]
            result = cursor.fetchone()
            if result:
                return dict(zip(columns, result))
            return None
        except oracledb.Error as error:
            print(f"Error while fetching target lot: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

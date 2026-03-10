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
                    v.adjacent_width                            as adj_width,            -- 인접그룹 조합여부: 값이 0이면 인접그룹 미조합, 0보다 크면 인접그룹 조합
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
                --where v.calc_successful = '9'
                where v.lot_no = '2251200311' and v.version = '03'
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

    def get_roll_orders_from_db_jh(self, in_lot_no, in_version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            #
            # sql_query = """
            #     select plant, pm_no, schedule_unit, paper_prod_seq,
            #         width                as width, 
            #         min(height)          as height, 
            #         roll_length          as roll_length, 
            #         rs_gubun             as rs_gubun, 
            #         quality_grade        as quality_grade,
            #         min(regular_type)    as regular_type,
            #         sum(order_roll_cnt) as order_roll_cnt,
            #         case when rs_gubun = 'R' 
            #                 then round(sum(order_ton_cnt * loss_rate_roll), 4)
            #                 when rs_gubun = 'S'
            #                 then round(sum(order_ton_cnt * loss_rate_sheet), 4)
            #         end                  as order_ton_cnt,
            #         doublepack           as doublepack, 
            #         digital_yn           as digital, 
            #         semi                 as semi, 
            #         dia                  as dia, 
            #         core                 as core,
            #         short_sheet          as short_sheet,
            #         min(loss_rate_roll)  as loss_rate_roll,
            #         min(loss_rate_sheet) as loss_rate_sheet
            #     from (
            #             select plant, pm_no, schedule_unit, paper_prod_seq,
            #                 order_no,
            #                 quality_grade,
            #                 width,
            #                 height,  
            #                 roll_length, 
            #                 rs_gubun, 
            #                 regular_type,
            #                 loss_rate_roll, 
            #                 loss_rate_sheet,
            #                 order_roll_cnt,
            #                 order_ton_cnt,
            #                 doublepack,
            #                 (case when nvl(digital_yn, 'N') = 'N'
            #                         then 'N'
            #                         when sum(digital_cut_ton) over (partition by paper_prod_seq, width, short_sheet, nvl(roll_length,0), semi)  
            #                             > sum(digital_ton)     over (partition by paper_prod_seq, width, short_sheet, nvl(roll_length,0), semi)
            #                         then 'D' 
            #                         else 'Y'
            #                     end ) as digital_yn,
            #                 semi,
            #                 dia, 
            #                 core,
            #                 short_sheet
            #             from (
            #                     select p.plant, p.pm_no, p.schedule_unit, p.paper_prod_seq,
            #                         p.order_no,
            #                         p.quality_grade,
            #                         j.adj_width as width,
            #                         (
            #                             select min(p2.length) as length
            #                             from h3t_production_order p2,
            #                                 th_versions_manager v2
            #                             where p2.paper_prod_seq = v2.lot_no
            #                             and v2.lot_no = :in_lot_no
            #                             and v2.version = :in_version
            #                             and p2.width = j.adj_width
            #                             and p2.rs_gubun = p.rs_gubun
            #                             and (case when (p2.width<=700 or p2.width >810) and p2.length < nvl(v2.short_sheet, 545) and p2.rs_gubun='S' then 'S' else 'L' end) = 
            #                                 (case when (p.width<=700 or p.width >810) and p.length < nvl(v.short_sheet, 545) and p.rs_gubun='S' then 'S' else 'L' end)
            #                             group by p2.width
            #                         ) as height,  
            #                         p.roll_length, 
            #                         p.rs_gubun, 
            #                         case when o.regular_gubun = '3' then '1' else o.regular_gubun end regular_type,
            #                         p.order_roll_cnt,
            #                         p.order_ton_cnt,
            #                         nvl(1+0.01*r.loss_rate_roll,1.001) loss_rate_roll,
            #                         nvl(1+0.01*(r.loss_rate_sheet + case when fn_get_digi_yn(p.order_no) = 'Y' then nvl(r.loss_rate_digi, 0) else 0 end),1.065) loss_rate_sheet,
            #                         (case when p.skid_yn = 'Y' and p.pt_gubun =2 then 'Y' else 'N' end) doublepack,
            #                         (case when p.rs_gubun = 'S' and ( o.pack_type in ('5', '6', '7') or o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) then 'Y' else 'N' end) digital_yn,
            #                         (case when o2.dir_gubun = 'ZARL' then 'Y' else 'N' end) semi,
            #                         p.dia, 
            #                         p.core,
            #                         (case when (p.width<=700 or p.width >810) and p.length < nvl(v.short_sheet, 545) and p.rs_gubun='S' then 'S' else 'L' end) as short_sheet,
            #                         (case when p.rs_gubun = 'S' and o.pack_type in ('5', '7') and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) 
            #                                                     and (    ( ( p.width  between 210 and 360)  and ( p.length between 279 and 520) )
            #                                                             or ( ( p.length between 210 and 360)  and ( p.width  between 279 and 520) )
            #                                                         )
            #                                 then 'Y' else 'N' end) digital_cut_yn,
            #                         (case when p.rs_gubun = 'S' and o.pack_type in ('6')      and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) then p.order_ton_cnt else 0 end) digital_ton,
            #                         (case when p.rs_gubun = 'S' and o.pack_type in ('5', '7') and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) 
            #                                                     and (    ( ( p.width  between 210 and 360) and ( p.length between 279 and 520) )
            #                                                             or ( ( p.length between 210 and 360) and ( p.width  between 279 and 520) )
            #                                                         )
            #                                 then p.order_ton_cnt else 0 end) digital_cut_ton
            #                     from h3t_production_order p,
            #                         th_tar_loss_rate r,
            #                         sapd12t_tmp o, 
            #                         sapd11t_tmp o2,
            #                         th_order_adj j,
            #                         th_versions_manager v 
            #                     where p.paper_prod_seq = v.lot_no 
            #                     and p.paper_type = r.paper_type(+)
            #                     and p.b_wgt = r.cp_b_wgt(+)
            #                     and p.pm_no = r.resource_code(+)
            #                     and o.lot_no = v.lot_no 
            #                     and o2.req_no = o.req_no
            #                     and p.order_no = o.order_no 
            #                     and j.order_no = p.order_no
            #                     and j.lot_no = v.lot_no
            #                     and j.version = v.version
            #                     and v.lot_no = :in_lot_no
            #                     and v.version = :in_version
            #                 ) t     
            #             )
            #             group by plant, pm_no, schedule_unit, paper_prod_seq,
            #                     width,
            #                     roll_length,
            #                     rs_gubun, 
            #                     quality_grade,
            #                     doublepack,
            #                     digital_yn,
            #                     semi,
            #                     dia, 
            #                     core,
            #                     short_sheet   
            #     order by rs_gubun, width, height, regular_type
            # """

            sql_query = """
                    
                select plant, pm_no, schedule_unit, paper_prod_seq,
                    order_no,
                    pack_type,
                    quality_grade,
                    width,
                    height,  
                    roll_length, 
                    rs_gubun, 
                    regular_type,
                    loss_rate_roll, 
                    loss_rate_sheet,
                    order_roll_cnt,
                    order_ton_cnt,
                    doublepack,
                    (case when nvl(digital_yn, 'N') = 'N'
                            then 'N'
                            when sum(digital_cut_ton) over (partition by paper_prod_seq, width, short_sheet, nvl(roll_length,0), semi)  
                                > sum(digital_ton)     over (partition by paper_prod_seq, width, short_sheet, nvl(roll_length,0), semi)
                            then 'D' 
                            else 'Y'
                        end ) as digital_yn,
                    semi,
                    dia, 
                    core,
                    short_sheet
                from (
                        select p.plant, p.pm_no, p.schedule_unit, p.paper_prod_seq,
                            p.order_no, o.pack_type,
                            p.quality_grade,
                            j.adj_width as width,
                            (
                                select min(p2.length) as length
                                from h3t_production_order p2,
                                    th_versions_manager v2
                                where p2.paper_prod_seq = v2.lot_no
                                and v2.lot_no = :in_lot_no
                                and v2.version = :in_version
                                and p2.width = j.adj_width
                                and p2.rs_gubun = p.rs_gubun
                                and (case when (p2.width<=700 or p2.width >810) and p2.length < nvl(v2.short_sheet, 545) and p2.rs_gubun='S' then 'S' else 'L' end) = 
                                    (case when (p.width<=700 or p.width >810) and p.length < nvl(v.short_sheet, 545) and p.rs_gubun='S' then 'S' else 'L' end)
                                group by p2.width
                            ) as height,  
                            p.roll_length, 
                            p.rs_gubun, 
                            case when o.regular_gubun = '3' then '1' else o.regular_gubun end regular_type,
                            p.order_roll_cnt,
                            p.order_ton_cnt,
                            nvl(1+0.01*r.loss_rate_roll,1.001) loss_rate_roll,
                            nvl(1+0.01*(r.loss_rate_sheet + case when fn_get_digi_yn(p.order_no) = 'Y' then nvl(r.loss_rate_digi, 0) else 0 end),1.065) loss_rate_sheet,
                            (case when p.skid_yn = 'Y' and p.pt_gubun =2 then 'Y' else 'N' end) doublepack,
                            (case when p.rs_gubun = 'S' and ( o.pack_type in ('5', '6', '7') or o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) then 'Y' else 'N' end) digital_yn,
                            (case when o2.dir_gubun = 'ZARL' then 'Y' else 'N' end) semi,
                            p.dia, 
                            p.core,
                            (case when (p.width<=700 or p.width >810) and p.length < nvl(v.short_sheet, 545) and p.rs_gubun='S' then 'S' else 'L' end) as short_sheet,
                            (case when p.rs_gubun = 'S' and o.pack_type in ('5', '7') and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) 
                                                        and (    ( ( p.width  between 210 and 360)  and ( p.length between 279 and 520) )
                                                                or ( ( p.length between 210 and 360)  and ( p.width  between 279 and 520) )
                                                            )
                                    then 'Y' else 'N' end) digital_cut_yn,
                            (case when p.rs_gubun = 'S' and o.pack_type in ('6')      and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) then p.order_ton_cnt else 0 end) digital_ton,
                            (case when p.rs_gubun = 'S' and o.pack_type in ('5', '7') and ( o.hogi in ('498','499','541','599') or p.width < 400 or p.length < 400 ) 
                                                        and (    ( ( p.width  between 210 and 360) and ( p.length between 279 and 520) )
                                                                or ( ( p.length between 210 and 360) and ( p.width  between 279 and 520) )
                                                            )
                                    then p.order_ton_cnt else 0 end) digital_cut_ton
                        from h3t_production_order p,
                            th_tar_loss_rate r,
                            sapd12t_tmp o, 
                            sapd11t_tmp o2,
                            th_order_adj j,
                            th_versions_manager v 
                        where p.paper_prod_seq = v.lot_no 
                        and p.paper_type = r.paper_type(+)
                        and p.b_wgt = r.cp_b_wgt(+)
                        and p.pm_no = r.resource_code(+)
                        and p.rs_gubun = 'R'
                        and o.lot_no = v.lot_no 
                        and o2.req_no = o.req_no
                        and p.order_no = o.order_no 
                        and j.order_no = p.order_no
                        and j.lot_no = v.lot_no
                        and j.version = v.version
                        and v.lot_no = :in_lot_no
                        and v.version = :in_version
                    ) t     


            """

            cursor.execute(sql_query, in_lot_no=in_lot_no, in_version=in_version)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                (plant, pm_no, schedule_unit, paper_prod_seq, order_no, pack_type, quality_grade,
                 width, height, roll_length, rs_gubun,                  
                 regular_type, loss_rate_roll, loss_rate_sheet, order_roll_cnt, order_ton_cnt,
                 doublepack, digital, semi, dia, core, short_sheet) = row
                raw_orders.append({
                    'plant': plant,
                    'pm_no': pm_no,
                    'schedule_unit': schedule_unit,
                    'paper_prod_seq': paper_prod_seq,
                    'pack_type': pack_type,
                    'order_no': order_no,
                    '지폭': int(width),
                    '가로': int(height) if height else 0,
                    '롤길이': int(roll_length) if roll_length else 0,
                    'rs_gubun': rs_gubun,
                    '등급': quality_grade,
                    'regular_type': regular_type,
                    '주문수량': int(order_roll_cnt) if order_roll_cnt else 0,
                    '주문톤': float(order_ton_cnt) if order_ton_cnt else 0,
                    'doublepack': doublepack,
                    'digital': digital,
                    'semi': semi,
                    'dia': dia,
                    'core': core,
                    'short_sheet': short_sheet,
                    'loss_rate_roll': loss_rate_roll,
                    'loss_rate_sheet': loss_rate_sheet,
                })
            print(f"Successfully fetched {len(raw_orders)} roll orders for lot {in_lot_no}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting get_roll_sl_orders_from_db orders_jh from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_orders_from_db_jh(self, in_lot_no, in_version):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()

            sql_query = """
                SELECT
                    a.plant, a.pm_no, a.schedule_unit, a.width, a.length, a.quality_grade, a.order_ton_cnt, 
                    nvl(1+0.01*r.loss_rate_roll,1.001) loss_rate_roll,
                    nvl(1+0.01*(r.loss_rate_sheet + case when fn_get_digi_yn(a.order_no) = 'Y' then nvl(r.loss_rate_digi, 0) else 0 end),1.065) loss_rate_sheet,
                    a.export_yn, a.order_no, a.color, a.order_gubun, a.pt_gubun, b.pack_type,
                    (case when c.dir_gubun = 'ZARL' then 'Y' else 'N' end) semi,
                    (case when a.skid_yn = 'Y' and a.pt_gubun =2 then 'Y' else 'N' end) doublepack,
                    (case when a.rs_gubun = 'S' and ( b.pack_type in ('5', '6', '7') or b.hogi in ('498','499','541','599') or a.width < 400 or a.length < 400 ) then 'Y' else 'N' end) digital_yn,
                    a.dia, 
                    a.core,
                    (case when (a.width<=700 or a.width >810) and a.length < nvl(d.short_sheet, 545) and a.rs_gubun='S' then 'S' else 'L' end) as short_sheet,
                    (case when a.rs_gubun = 'S' and b.pack_type in ('5', '7') and ( b.hogi in ('498','499','541','599') or a.width < 400 or a.length < 400 ) 
                                                and (    ( ( a.width  between 210 and 360)  and ( a.length between 279 and 520) )
                                                        or ( ( a.length between 210 and 360)  and ( a.width  between 279 and 520) )
                                                    )
                            then 'Y' else 'N' end) digital_cut_yn,
                    (case when a.rs_gubun = 'S' and b.pack_type in ('6')      and ( b.hogi in ('498','499','541','599') or a.width < 400 or a.length < 400 ) then a.order_ton_cnt else 0 end) digital_ton,
                    (case when a.rs_gubun = 'S' and b.pack_type in ('5', '7') and ( b.hogi in ('498','499','541','599') or a.width < 400 or a.length < 400 ) 
                                                and (    ( ( a.width  between 210 and 360) and ( a.length between 279 and 520) )
                                                        or ( ( a.length between 210 and 360) and ( a.width  between 279 and 520) )
                                                    )
                            then a.order_ton_cnt else 0 end) digital_cut_ton
                FROM
                    h3t_production_order a, sapd12t_tmp b, sapd11t_tmp c, th_versions_manager d, th_tar_loss_rate r
                WHERE a.paper_prod_seq = :in_lot_no
                    and a.paper_type = r.paper_type(+)
                    and a.b_wgt = r.cp_b_wgt(+)
                    and a.pm_no = r.resource_code(+)
                    AND a.order_no = b.order_no
                    and b.req_no = c.req_no
                    and a.paper_prod_seq = d.lot_no
                    and d.version = :in_version
                    AND a.rs_gubun = 'S'
                ORDER BY a.width, a.length
            """

            # print(f"Executing query to fetch sheet orders:\n{sql_query}")
            cursor.execute(sql_query, in_lot_no=in_lot_no, in_version=in_version)
            rows = cursor.fetchall()
            raw_orders = []
            for row in rows:
                (plant, pm_no, schedule_unit, width, length, 
                quality_grade, order_ton_cnt, loss_rate_roll, loss_rate_sheet, export_yn, order_no, 
                color, order_gubun, pt_gubun, pack_type,
                semi, doublepack, digital_yn, dia, core, short_sheet,
                digital_cut_yn, digital_ton, digital_cut_ton
                ) = row
                export_type = '수출' if export_yn == 'Y' else '내수'
                raw_orders.append({
                    'plant': plant,
                    'pm_no': pm_no,
                    'schedule_unit': schedule_unit,
                    'order_no': order_no,
                    '가로': int(width),
                    '세로': int(length),
                    '주문톤': float(order_ton_cnt),
                    '등급': quality_grade,
                    '수출내수': export_type,
                    'color': color,
                    'order_gubun': order_gubun,
                    'pt_gubun': pt_gubun,
                    'pack_type': pack_type,
                    'semi': semi,
                    'doublepack': doublepack,
                    'digital': digital_yn,
                    'dia': dia,
                    'core': core,
                    'short_sheet': short_sheet,
                    'digital_cut': digital_cut_yn,
                    'digital_ton': digital_ton,
                    'digital_cut_ton': digital_cut_ton,
                    'loss_rate_roll': loss_rate_roll,
                    'loss_rate_sheet': loss_rate_sheet
                })
            print(f"Successfully fetched {len(raw_orders)} sheet orders for lot {in_lot_no}")
            return raw_orders
        except oracledb.Error as error:
            print(f"Error while getting sheet orders from DB: {error}")
            return None
        finally:
            if connection:
                self.pool.release(connection)

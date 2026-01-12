import oracledb

class PatternGetters:
    def get_roll_patterns_from_db(self, lot_no, version):
        """지정된 lot_no와 version에 대해 th_pattern_sequence에서 롤 패턴(rollwidth)을 가져옵니다."""
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # rollwidth 필드를 사용하여 지폭 값들을 가져옵니다.
            query = """
                SELECT DISTINCT
                    rollwidth1, rollwidth2, rollwidth3, rollwidth4, 
                    rollwidth5, rollwidth6, rollwidth7, rollwidth8
                FROM th_pattern_tot A
                WHERE (a.rn) IN (
                    SELECT rn
                    FROM (
                        -- 1. UNPIVOT 수행
                        SELECT rn, width_val
                        FROM th_pattern_tot
                        UNPIVOT (
                            width_val FOR col_name IN (
                                rollwidth1, rollwidth2, rollwidth3, rollwidth4, 
                                rollwidth5, rollwidth6, rollwidth7, rollwidth8
                            )
                        )
                        --where ( paper_type, b_wgt ) in (select paper_type, b_wgt from h3t_production_order where paper_prod_seq = :lot_no and rownum = 1 )
                    ) P
                    -- 2. 오더 테이블과 조인
                    LEFT JOIN (
                        SELECT DISTINCT width 
                        FROM h3t_production_order
                        WHERE paper_prod_seq = :lot_no -- 필요시 조건 유지
                        and rs_gubun = 'R'
                    ) O ON P.width_val = O.width
                    
                    -- [핵심 수정] 0인 값은 비교 대상에서 제외합니다.
                    WHERE P.width_val > 0 
                    -- 3. 그룹핑 및 비교
                    GROUP BY rn
                    HAVING 
                        -- 0을 제외한 유효 지폭들이 모두 오더 목록에 있는지 확인
                        COUNT(P.width_val) = COUNT(O.width)
                )
            """
            cursor.execute(query, lot_no=lot_no)
            rows = cursor.fetchall()
            
            db_patterns = []
            for row in rows:
                # None이나 0이 아닌 유효한 rollwidth만 필터링합니다.
                pattern_widths = [int(w) for w in row if w and w > 0]
                if pattern_widths:
                    db_patterns.append(pattern_widths)
            
            print(f"Successfully fetched {len(db_patterns)} roll patterns from DB for lot {lot_no} version {version}")
            return db_patterns
        except oracledb.Error as error:
            print(f"Error while fetching roll patterns from DB: {error}")
            return []
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_patterns_from_db(self, lot_no):
        """th_pattern_tot_sheet 테이블에서 현재 오더에 해당하는 지폭이 있는 쉬트 패턴만 가져옵니다.
        
        - width1~8 컬럼에는 아이템명 형식(예: "710x4", "1000x2")으로 저장되어 있습니다.
        - 현재 lot의 지종/평량과 일치하고, 모든 아이템의 지폭이 현재 오더에 존재하는 패턴만 반환합니다.
        """
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            
            query = """
                SELECT DISTINCT
                    width1, width2, width3, width4, 
                    width5, width6, width7, width8
                FROM th_pattern_tot_sheet A
                WHERE (a.rn) IN (
                    SELECT rn
                    FROM (
                        -- 1. UNPIVOT: width 컬럼들을 행으로 변환
                        SELECT rn, item_name
                        FROM th_pattern_tot_sheet
                        UNPIVOT (
                            item_name FOR col_name IN (
                                width1, width2, width3, width4, 
                                width5, width6, width7, width8
                            )
                        )
                        WHERE (paper_type, b_wgt) IN (
                            SELECT paper_type, b_wgt 
                            FROM h3t_production_order 
                            WHERE paper_prod_seq = :lot_no AND ROWNUM = 1
                        )
                    ) P
                    -- 2. 오더 테이블과 조인 (지폭 추출: "710x4" → 710)
                    LEFT JOIN (
                        SELECT DISTINCT width 
                        FROM h3t_production_order
                        WHERE paper_prod_seq = :lot_no
                        AND rs_gubun = 'S'
                    ) O ON TO_NUMBER(SUBSTR(P.item_name, 1, INSTR(P.item_name, 'x') - 1)) = O.width
                    
                    -- 3. 유효 아이템만 있는 패턴 필터링
                    WHERE P.item_name IS NOT NULL
                    GROUP BY rn
                    HAVING 
                        -- 모든 유효 아이템이 오더 목록에 있는지 확인
                        COUNT(P.item_name) = COUNT(O.width)
                )
            """
            cursor.execute(query, lot_no=lot_no)
            rows = cursor.fetchall()
            
            db_patterns = []
            for row in rows:
                # None이나 빈 문자열이 아닌 유효한 아이템명만 필터링합니다.
                pattern_items = [item for item in row if item]
                if pattern_items:
                    db_patterns.append(pattern_items)
            
            print(f"Successfully fetched {len(db_patterns)} sheet patterns from th_pattern_tot_sheet for lot {lot_no}")
            return db_patterns
        except oracledb.Error as error:
            print(f"Error while fetching sheet patterns from DB: {error}")
            return []
        finally:
            if connection:
                self.pool.release(connection)

    def get_sheet_ca_patterns_from_db(self, lot_no):
        """th_pattern_tot_sheet 테이블에서 현재 오더의 지종/평량에 해당하는 모든 패턴을 가져옵니다.
        
        CA 공장은 복합폭("710x1+850x1") 등 복잡한 아이템을 사용하므로,
        SQL 단계에서는 지종/평량 일치 여부만 확인하고
        상세한 유효성 검증(현재 오더에 포함된 지폭인지 등)은 Python 로직에서 수행합니다.
        """
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            
            # CA는 복합폭 처리가 복잡하므로, 일단 해당 지종/평량의 모든 사용자 패턴을 가져옵니다.
            query = """
                SELECT DISTINCT
                    width1, width2, width3, width4, 
                    width5, width6, width7, width8
                FROM th_pattern_tot_sheet
                WHERE (paper_type, b_wgt) IN (
                    SELECT paper_type, b_wgt
                    FROM h3t_production_order
                    WHERE paper_prod_seq = :lot_no AND ROWNUM = 1
                )
            """
            cursor.execute(query, lot_no=lot_no)
            rows = cursor.fetchall()
            
            db_patterns = []
            for row in rows:
                # None이나 빈 문자열이 아닌 유효한 아이템명만 필터링합니다.
                pattern_items = [item for item in row if item]
                if pattern_items:
                    db_patterns.append(pattern_items)
            
            print(f"Successfully fetched {len(db_patterns)} sheet patterns from CA DB for lot {lot_no}")
            return db_patterns
        except oracledb.Error as error:
            print(f"Error while fetching CA sheet patterns from DB: {error}")
            return []
        finally:
            if connection:
                self.pool.release(connection)

import oracledb

class PatternGetters:
    def get_patterns_from_db(self, lot_no, version):
        """지정된 lot_no와 version에 대해 th_pattern_sequence에서 기존 패턴을 가져옵니다."""
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # rollwidth 대신 groupno 필드를 사용하여 패턴을 가져옵니다.
            # groupno가 복합폭 아이템의 고유 이름이므로 더 정확합니다.
            query = """
                SELECT 
                    groupno1, groupno2, groupno3, groupno4, 
                    groupno5, groupno6, groupno7, groupno8
                FROM th_pattern_sequence 
                WHERE lot_no = :lot_no AND version = :version
            """
            cursor.execute(query, lot_no=lot_no, version=version)
            rows = cursor.fetchall()
            
            db_patterns = []
            for row in rows:
                # None이나 빈 문자열이 아닌 유효한 groupno만 필터링합니다.
                pattern_items = [item for item in row if item]
                if pattern_items:
                    db_patterns.append(pattern_items)
            
            print(f"Successfully fetched {len(db_patterns)} existing patterns from DB for lot {lot_no} version {version}")
            return db_patterns
        except oracledb.Error as error:
            print(f"Error while fetching existing patterns from DB: {error}")
            return []
        finally:
            if connection:
                self.pool.release(connection)

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
                        where ( paper_type, b_wgt ) in (select paper_type, b_wgt from h3t_production_order where paper_prod_seq = :lot_no and rownum = 1 )
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

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

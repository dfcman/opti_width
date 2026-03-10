import oracledb

class DataInserters:
    """공통 데이터 처리 클래스 - 3개 공장에서 공통으로 사용하는 함수만 포함"""
    
    def update_lot_status(self, lot_no, version, status):
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            # 데몬용 쿼리 복원
            query = "update th_versions_manager set calc_successful = :status where lot_no = :lot_no and version = :version"
            cursor.execute(query, status=status, lot_no=lot_no, version=version)

            query = "update th_calculation_messages set message_seq = :status where lot_no = :lot_no and version = :version"
            cursor.execute(query, status=status, lot_no=lot_no, version=version)

            connection.commit()
            print(f"Successfully updated lot {lot_no} version {version} to status {status}")
            return True
        except oracledb.Error as error:
            print(f"Error while updating lot status: {error}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.pool.release(connection)

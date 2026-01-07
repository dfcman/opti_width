import oracledb
import sys
from .db_get_data_csv import CsvGetters
from .db_get_data_roll import RollGetters
from .db_get_data_sheet import SheetGetters
from .db_get_data_version import VersionGetters
from .db_insert_data import DataInserters
from .db_get_data_pattern import PatternGetters

class Database(CsvGetters, RollGetters, SheetGetters, VersionGetters, DataInserters, PatternGetters):
    def __init__(self, user, password, dsn, min_pool=1, max_pool=5, increment=1):
        self.user = user
        self.password = password
        self.dsn = dsn
        self.pool = None
        try:
            self.pool = oracledb.create_pool(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=min_pool,
                max=max_pool,
                increment=increment
            )
            print("Successfully created Oracle connection pool.")
        except oracledb.Error as error:
            print(f"Error while creating connection pool: {error}")
            raise

    def delete_optimization_results(self, lot_no, version):
        """지정된 lot_no와 version에 대한 모든 최적화 결과 데이터를 삭제합니다."""
        connection = None
        try:
            connection = self.pool.acquire()
            cursor = connection.cursor()
            tables_to_delete = [
                "th_pattern_sequence",
                "th_roll_sequence",
                "th_cut_sequence",
                "th_sheet_sequence",
                "th_order_group",
                "th_group_master"
            ]
            for table in tables_to_delete:
                query = f"DELETE FROM {table} WHERE lot_no = :lot_no AND version = :version"
                cursor.execute(query, lot_no=lot_no, version=version)
                print(f"Deleted data from {table} for lot {lot_no}, version {version}")
            
            connection.commit()
            print(f"Successfully deleted all optimization data for lot {lot_no}, version {version}")
            return True
        except oracledb.Error as error:
            print(f"Error while deleting optimization data: {error}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.pool.release(connection)

    def close_pool(self):
        if self.pool:
            self.pool.close()
            print("Oracle connection pool closed.")

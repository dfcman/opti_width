import oracledb
import sys
from .db_get_data_csv import CsvGetters
from .db_get_data_roll import RollGetters
from .db_get_data_sheet import SheetGetters
from .db_get_data_version import VersionGetters
from .db_insert_data import DataInserters
from .db_get_data_pattern import PatternGetters

class Database(CsvGetters, RollGetters, SheetGetters, VersionGetters, DataInserters, PatternGetters):
    def __init__(self, user, password, dsn, min_pool=1, max_pool=1, increment=1):
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

    def close_pool(self):
        if self.pool:
            self.pool.close()
            print("Oracle connection pool closed.")

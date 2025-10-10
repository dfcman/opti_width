import csv
import os
from pathlib import Path

# 이 파일(db_get_data_csv.py)이 있는 디렉토리(db)의 부모 디렉토리(프로젝트 루트)를 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_LOT_CSV_PATH = PROJECT_ROOT / 'csv' / 'target_lot.csv'
DEFAULT_ROLL_ORDERS_CSV_PATH = PROJECT_ROOT / 'csv' / 'roll_orders.csv'

class CsvGetters:
    def get_target_lot_csv(self, file_path=DEFAULT_TARGET_LOT_CSV_PATH):
        try:
            with open(file_path, mode='r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                header = next(reader)
                data = next(reader, None)
                if data:
                    # Convert numeric strings to appropriate types
                    converted_data = []
                    for item in data:
                        try:
                            converted_data.append(int(item))
                        except ValueError:
                            try:
                                converted_data.append(float(item))
                            except ValueError:
                                converted_data.append(item)
                    return tuple(converted_data)
            return (None,) * 16
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return (None,) * 16
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return (None,) * 16

    def get_roll_orders_from_db_csv(self, file_path=DEFAULT_ROLL_ORDERS_CSV_PATH):
        raw_orders = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    export_type = '수출' if row['export_yn'] == 'Y' else '내수'
                    raw_orders.append({
                        '오더번호': row['order_no'],
                        '지폭': int(row['width']),
                        '가로': int(row['length']),
                        '주문수량': int(row['order_roll_cnt']),
                        '주문톤': float(row['order_ton_cnt']),
                        '롤길이': int(row['roll_length']),
                        '등급': row['quality_grade'],
                        '수출내수': export_type
                    })
            print(f"Successfully fetched {len(raw_orders)} roll orders from {file_path}")
            return raw_orders
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return None

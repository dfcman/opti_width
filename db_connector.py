import oracledb
import sys

def get_orders_from_db(user, password, dsn):
    """
    Oracle DB에 연결하여 생산 오더를 가져와 raw_orders 리스트를 생성합니다.

    :param user: DB 사용자 이름
    :param password: DB 비밀번호
    :param dsn: DB 연결을 위한 DSN (e.g., 'localhost:1521/orcl')
    :return: raw_orders 리스트. 에러 발생 시 None 반환
    """
    try:
        # DB 연결
        connection = oracledb.connect(user=user, password=password, dsn=dsn)
        print("Successfully connected to Oracle Database")

        cursor = connection.cursor()

        # 데이터 조회를 위한 SQL 쿼리
        sql_query = """
            SELECT
                width,
                roll_length,
                quality_grade,
                order_roll_cnt,
                export_yn,
                order_no 
            FROM
                h3t_production_order
                --where paper_prod_seq = '3250900072'
                where paper_prod_seq = '3250900073'
                and rs_gubun = 'R'
                and order_no not in (
                '01003262960000200001',
                '00302720180004700001',
                '00302720170004700001',
                '00302720180005200001'
                )
                order by roll_length, width, dia, core
        """

        cursor.execute(sql_query)

        # 모든 데이터를 한 번에 가져오기
        rows = cursor.fetchall()

        # raw_orders 형식으로 데이터 변환
        raw_orders = []
        for row in rows:
            # 수정: 쿼리 결과와 변수 개수를 6개로 일치시킴
            width, roll_length, quality_grade, order_roll_cnt, export_yn, order_no = row
            
            # 'export_yn' 값을 '수출' 또는 '내수'로 변환
            export_type = '수출' if export_yn == 'Y' else '내수'

            raw_orders.append({
                '오더번호': order_no, # 실제 order_no를 사용하도록 변경
                '지폭': int(width),
                '주문수량': int(order_roll_cnt),
                '롤길이': int(roll_length),
                '등급': quality_grade,
                '수출내수': export_type
            })

        return raw_orders

    except oracledb.Error as error:
        print(f"Error while connecting to Oracle: {error}")
        return None
    finally:
        # 리소스 정리
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()
            print("Oracle connection is closed")

if __name__ == '__main__':
    # 이 파일을 직접 실행할 때 테스트를 위한 예시입니다.
    # config.ini 파일에서 DB 접속 정보를 읽어옵니다.
    import configparser

    config = configparser.ConfigParser()
    try:
        if not config.read('config.ini'):
            raise FileNotFoundError("config.ini 파일을 찾을 수 없습니다.")
        db_config = config['database']
        db_user = db_config['user']
        db_password = db_config['password']
        db_dsn = db_config['dsn']
    except (FileNotFoundError, KeyError) as e:
        print(f"설정 파일 오류: {e}")
        print("="*60)
        print(" [경고] db_connector.py를 직접 실행하려면 config.ini 파일이 필요합니다. ".center(60))
        print("="*60)
        sys.exit()

    # DB에서 오더 데이터 가져오기
    raw_orders_from_db = get_orders_from_db(user=db_user, password=db_password, dsn=db_dsn)

    if raw_orders_from_db:
        print("\n===== Fetched Orders from DB =====\n")
        # pandas가 설치되어 있다면 데이터프레임으로 예쁘게 출력
        try:
            import pandas as pd
            df = pd.DataFrame(raw_orders_from_db)
            print(df.to_string())
        except ImportError:
            # pandas가 없다면 그냥 리스트 출력
            for order in raw_orders_from_db:
                print(order)

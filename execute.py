import pandas as pd
import sys
import time
import configparser  # 설정 파일 처리를 위해 추가
from optimize import Optimize
from db_connector import get_orders_from_db  # db_connector.py 에서 함수 임포트

if __name__ == "__main__":
    t1 = time.time()

    # ===== 설정 파일에서 DB 접속 정보 읽기 ===== #
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
        print("프로그램을 종료합니다.")
        sys.exit()

    # ===== DB에서 오더 데이터 가져오기 ===== #
    raw_orders = get_orders_from_db(user=db_user, password=db_password, dsn=db_dsn)

    # 데이터 로드 실패 시 프로그램 종료
    if not raw_orders:
        print("DB에서 오더를 가져오지 못했습니다. 프로그램을 종료합니다.")
        sys.exit()
        
    print("===== 원본 주문 정보 (from DB) =====")
    print(pd.DataFrame(raw_orders).to_string())
    print("\n")

    # ===== 그룹별 최적화 파라미터 설정 (선택 사항) ===== #
    optimization_params = {
        # (롤길이, 등급, 수출내수) 를 키로 사용
        # 예: (2300, 'A', '내수'): {'max_pieces': 10, 'min_width': 4600},
    }
    default_params = {'max_pieces': 8, 'min_width': 4500}

    # ===== 데이터 그룹화 (롤길이, 등급, 수출내수 기준) ===== #
    df_orders = pd.DataFrame(raw_orders)
    grouped_orders = df_orders.groupby(['롤길이', '등급'])

    # ===== 그룹별 최적화 진행 ===== #
    for group_key, group in grouped_orders:
        # 현재 그룹의 파라미터 가져오기 (없으면 기본값 사용)
        params = optimization_params.get(group_key, default_params)
        
        roll_length, grade = group_key
        print(f"# ===== 그룹: (롤길이:{roll_length}, 등급:{grade}) 최적화 시작 ===== #")
        print(f"(적용 파라미터: max_pieces={params['max_pieces']}, min_width={params['min_width']})\n")

        df_spec_pre = group.copy()
        df_spec_pre['지종'] = 'DEFAULT'
        df_spec_pre['평량'] = 100
        
        optimizer = Optimize(
            df_spec_pre=df_spec_pre,
            max_width=4880,
            min_width=params['min_width'],
            max_pieces=params['max_pieces']
        )
        
        results = optimizer.run_optimize()

        # ===== 최종 결과 도출 ===== #
        if "error" in results:
            print(f"최적화 에러: {results['error']}\n")
        else:
            print("최적화 결과 (패턴별 생산량):")
            print(results["pattern_result"].to_string())
            print("\n\n# ================= 주문 충족 현황 ================== #\n")
            print(results["fulfillment_summary"].to_string())
            print("\n")

    print(f"\n총 연산 시간: {time.time() - t1:.2f}sec")
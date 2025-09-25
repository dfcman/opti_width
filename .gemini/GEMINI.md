## Gemini Added Memories
- 지폭 최적화 소스 개선 방안: 현재 코드는 모든 조합을 무작위로 탐색하고('Brute-force') 주문 수량을 고려하지 않는 한계가 있습니다. 이를 개선하기 위해, 실제 주문량을 만족시키면서 폐기물을 최소화하는 '절단 최적 문제(Cutting Stock Problem)'로 접근해야 합니다. Google의 'OR-Tools' 라이브러리를 사용하면 이 문제를 효율적으로 해결할 수 있습니다. 개선 방법은 1) OR-Tools 설치, 2) 가능한 커팅 패턴 생성, 3) 정수 계획법 모델을 사용해 '총 사용 롤 최소화'를 목표로 설정, 4) '각 지폭의 주문 수량 만족'을 제약조건으로 설정하여 최적해를 찾는 것입니다.
- The user has finalized the `update_lot_status` and `get_target_lot` functions in `db_connector.py` to match their production database. Do not modify these two functions in any future changes.
- execute.py : 최적해를 시작하는 메인 파일
- db_connector.py : db 연결 및 db 처리 함수
- roll_optimize.py : 롤지 오더에 대해서 최적해를 찾는 파일
- sheet_optimize.py : 쉬트지 오더에 대해서 정해진 표준길이로 최적해를 찾는 파일. 쉬트지오더를 롤수로 계산해서 패턴을 생성해서 최적해 찾음.
- sheet_optimize_var.py :  쉬트지 오더에 대해서 표준길이이 min, max 로 해서 최적해를 찾는 파일. 쉬트지오더를 장수로 계산해서 패턴을 생성해서 최적해 찾음.
- sheet_optimize_ca.py : 쉬트지 오더에 대해서 복합폭에 1개 이상의 규격이 포함되도록해서 최적해를 찾는 파일. 쉬트지오더를 장수로 계산해서 패턴을 생성해서 최적해 찾음.


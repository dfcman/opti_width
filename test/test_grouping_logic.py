import pandas as pd
import unittest

def apply_sheet_grouping(df_orders):
    """
    쉬트지 오더 그룹핑 로직을 적용합니다.
    
    그룹핑 기준:
    1. 기본: 가로, 등급
    2. 내수 (export_yn == 'N'):
       - 오더번호(order_no) 별로 그룹핑
    3. 수출 (export_yn == 'Y'):
       - order_gubun == 'A': 오더번호(order_no) 별로 그룹핑
       - 그 외: 세로(length) 범위로 그룹핑
         - 450 ~ 549
         - 550 ~ 699
         - 700 이상
         - 그 외 (범위 밖 - 사실상 없겠지만 예외처리)
    """
    
    def get_group_key(row):
        # 1. 내수
        if row['수출내수'] == '내수':
            return f"DOM_{row['order_no']}"
        
        # 2. 수출
        else: # 수출내수 == '수출'
            if row.get('order_gubun') == 'A':
                return f"EXP_A_{row['order_no']}"
            else:
                length = row['세로']
                if 450 <= length <= 549:
                    return "EXP_L_450_549"
                elif 550 <= length <= 699:
                    return "EXP_L_550_699"
                elif length >= 700:
                    return "EXP_L_700_PLUS"
                else:
                    return f"EXP_L_OTHER_{length}"

    df_orders['_group_key'] = df_orders.apply(get_group_key, axis=1)
    
    # 그룹핑 컬럼: 가로, 등급, _group_key
    # (세로는 범위로 묶이므로 그룹핑 키에서 제외하고, _group_key가 세로 범위를 대변함)
    # 하지만 결과적으로 그룹 내의 오더들은 '가로', '등급'은 같아야 함.
    
    return df_orders

class TestSheetGrouping(unittest.TestCase):
    def test_grouping_logic(self):
        data = [
            # Case 1: Domestic (내수) - Different Order No -> Different Groups
            {'order_no': 'ORD001', '가로': 1000, '세로': 500, '등급': 'A', '수출내수': '내수', 'order_gubun': 'B'},
            {'order_no': 'ORD002', '가로': 1000, '세로': 500, '등급': 'A', '수출내수': '내수', 'order_gubun': 'B'},
            
            # Case 2: Export (수출) Type A - Different Order No -> Different Groups
            {'order_no': 'ORD003', '가로': 1000, '세로': 500, '등급': 'A', '수출내수': '수출', 'order_gubun': 'A'},
            {'order_no': 'ORD004', '가로': 1000, '세로': 500, '등급': 'A', '수출내수': '수출', 'order_gubun': 'A'},
            
            # Case 3: Export (수출) Type B - Same Length Range (450-549) -> Same Group
            {'order_no': 'ORD005', '가로': 1000, '세로': 450, '등급': 'A', '수출내수': '수출', 'order_gubun': 'B'},
            {'order_no': 'ORD006', '가로': 1000, '세로': 549, '등급': 'A', '수출내수': '수출', 'order_gubun': 'B'},
            
            # Case 4: Export (수출) Type B - Different Length Range -> Different Groups
            {'order_no': 'ORD007', '가로': 1000, '세로': 550, '등급': 'A', '수출내수': '수출', 'order_gubun': 'B'}, # Range 550-699
            {'order_no': 'ORD008', '가로': 1000, '세로': 700, '등급': 'A', '수출내수': '수출', 'order_gubun': 'B'}, # Range 700+
            
            # Case 5: Different Width/Grade -> Different Groups (Implicit)
            {'order_no': 'ORD009', '가로': 1100, '세로': 500, '등급': 'A', '수출내수': '수출', 'order_gubun': 'B'},
            {'order_no': 'ORD010', '가로': 1000, '세로': 500, '등급': 'B', '수출내수': '수출', 'order_gubun': 'B'},
        ]
        
        df = pd.DataFrame(data)
        df = apply_sheet_grouping(df)
        
        # Verify Group Keys
        self.assertNotEqual(df.loc[0, '_group_key'], df.loc[1, '_group_key']) # Domestic different orders
        self.assertNotEqual(df.loc[2, '_group_key'], df.loc[3, '_group_key']) # Export A different orders
        self.assertEqual(df.loc[4, '_group_key'], df.loc[5, '_group_key'])    # Export B same range (450, 549)
        self.assertNotEqual(df.loc[4, '_group_key'], df.loc[6, '_group_key']) # Export B diff range (450 vs 550)
        self.assertNotEqual(df.loc[6, '_group_key'], df.loc[7, '_group_key']) # Export B diff range (550 vs 700)
        
        print("\nTest Data with Group Keys:")
        print(df[['order_no', '가로', '세로', '수출내수', 'order_gubun', '_group_key']])

if __name__ == '__main__':
    unittest.main()

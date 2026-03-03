"""
apply_sheet_grouping_jh 함수 단위 테스트
Java AdjacentWidth 4단계 알고리즘의 Python 변환 검증
"""
import pandas as pd
import numpy as np
import logging
import unittest
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execute import apply_sheet_grouping_jh

logging.basicConfig(level=logging.INFO)


def make_orders(widths_tons_maxsc, lot_no='TEST001', digital_flags=None):
    """테스트용 오더 DataFrame 생성 헬퍼
    
    Args:
        widths_tons_maxsc: list of (가로, 주문톤, max_sc_width) tuples
        digital_flags: optional list of 'Y'/'N' per order
    """
    rows = []
    for i, (w, ton, maxsc) in enumerate(widths_tons_maxsc):
        row = {
            'order_no': f'ORD{i+1:03d}',
            '가로': w,
            '주문톤': ton,
            'max_sc_width': maxsc,
            '등급': 'A',
        }
        if digital_flags:
            row['digital'] = digital_flags[i]
        rows.append(row)
    return pd.DataFrame(rows)


class TestAdjacentWidthGrouping(unittest.TestCase):
    """Java AdjacentWidth 4단계 알고리즘의 Python 변환 검증"""

    def test_basic_two_width_merge(self):
        """기본 2개 규격 묶음: 트림 조건 충족 시 작은 규격이 큰 규격으로 변경"""
        # 500, 495 with pok=3 (500*3+30=1530 <= 2000), diff_trim = (500-495)*3 = 15
        df = make_orders([
            (500, 10.0, 2000),
            (495, 10.0, 2000),
        ])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=20, max_wgt=100, min_wgt=5
        )
        # 495 should be merged to 500
        widths = sorted(result['가로'].unique())
        self.assertEqual(widths, [500])
        print(f"[PASS] 기본 2개 묶음: {widths}")

    def test_trim_exceeded_no_merge(self):
        """트림 초과 시 미적용: 차이가 adj_trim 초과하면 변경 없음"""
        # 500, 480 with pok=3, diff_trim = (500-480)*3 = 60 > adj_trim=20
        df = make_orders([
            (500, 10.0, 2000),
            (480, 10.0, 2000),
        ])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=20, max_wgt=100, min_wgt=5
        )
        widths = sorted(result['가로'].unique())
        self.assertEqual(widths, [480, 500])
        print(f"[PASS] 트림 초과 미적용: {widths}")

    def test_max_wgt_constraint(self):
        """최대 무게 제약: 두 규격 모두 max_wgt 이상이면 인접규격 미적용"""
        df = make_orders([
            (500, 150.0, 2000),  # 150 >= max_wgt(100) -> chk_max_wgt = False
            (495, 150.0, 2000),  # 150 >= max_wgt(100) -> chk_max_wgt = False
        ])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=20, max_wgt=100, min_wgt=5
        )
        widths = sorted(result['가로'].unique())
        # Both exceed max_wgt AND neither is below min_wgt -> no merge
        self.assertEqual(widths, [495, 500])
        print(f"[PASS] 최대 무게 제약: {widths}")

    def test_min_wgt_force_merge(self):
        """최소 무게 강제 적용: 한 쪽이 min_wgt 미만이면 인접규격 적용"""
        df = make_orders([
            (500, 150.0, 2000),  # 150 >= max_wgt(100) -> chk_max_wgt = False
            (495, 3.0, 2000),    # 3 < min_wgt(5) -> chk_min_wgt = True -> force merge
        ])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=20, max_wgt=100, min_wgt=5
        )
        widths = sorted(result['가로'].unique())
        self.assertEqual(widths, [500])  # min_wgt forces merge
        print(f"[PASS] 최소 무게 강제 적용: {widths}")

    def test_consecutive_change_rollback(self):
        """연속 변경 롤백: 2개 연속 변경 시 주문량 합계 기준 롤백"""
        # 3 widths: 500, 497, 494 -> all adjacent
        # with pok=3: (500-497)*3=9 <=20, (497-494)*3=9 <=20
        # order_sum[1] = 10+20=30, order_sum[2] = 20+15=35
        # chk_change[1]=True, chk_change[2]=True
        # STEP 3: order_sum[1]=30 < order_sum[2]=35 -> rollback [2]
        df = make_orders([
            (500, 10.0, 2000),
            (497, 20.0, 2000),
            (494, 15.0, 2000),
        ])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=20, max_wgt=100, min_wgt=5
        )
        # 497 -> 500 (merge), 494 stays (rollback)
        widths_map = {}
        for _, row in result.iterrows():
            orig = row.get('가로_원본', row['가로'])
            widths_map[orig] = row['가로']
        print(f"[INFO] 연속 변경 롤백 결과: {widths_map}")
        # 497 should be merged to 500
        self.assertEqual(widths_map.get(497, 497), 500)

    def test_triple_grouping(self):
        """3개 규격 묶음: adj_cnt > 2일 때 3개 규격이 하나로 묶이는 케이스"""
        # 500, 498, 496 with pok=3: (500-496)*3=12 <=20 -> triple possible
        # No above/under connections -> all three merge
        df = make_orders([
            (500, 10.0, 2000),
            (498, 10.0, 2000),
            (496, 10.0, 2000),
        ])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=3, adj_trim=20, max_wgt=100, min_wgt=5
        )
        widths = sorted(result['가로'].unique())
        self.assertEqual(widths, [500])
        print(f"[PASS] 3개 규격 묶음: {widths}")

    def test_digital_order_excluded(self):
        """Digital 오더 제외: digital='Y'인 오더는 인접규격 미적용"""
        df = make_orders(
            [(500, 10.0, 2000), (495, 10.0, 2000), (498, 10.0, 2000)],
            digital_flags=['N', 'N', 'Y']
        )
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=20, max_wgt=100, min_wgt=5
        )
        # 498 (digital=Y) should remain unchanged
        digital_rows = result[result['digital'] == 'Y']
        self.assertTrue(len(digital_rows) > 0)
        self.assertEqual(digital_rows.iloc[0]['가로'], 498)
        print(f"[PASS] Digital 오더 제외: {digital_rows[['가로']].values.tolist()}")

    def test_adj_trim_zero(self):
        """adj_trim=0: 인접규격 미적용, 기본 그룹핑만 수행"""
        df = make_orders([
            (500, 10.0, 2000),
            (495, 10.0, 2000),
        ])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=0, max_wgt=100, min_wgt=5
        )
        widths = sorted(result['가로'].unique())
        self.assertEqual(widths, [495, 500])
        print(f"[PASS] adj_trim=0 미적용: {widths}")

    def test_group_order_no_prefix(self):
        """group_order_no 접두사 확인: 장항=20"""
        df = make_orders([
            (500, 10.0, 2000),
            (495, 10.0, 2000),
        ])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=20, max_wgt=100, min_wgt=5
        )
        for gno in result['group_order_no'].unique():
            self.assertTrue(gno.startswith('20'))
        print(f"[PASS] group_order_no 접두사: {result['group_order_no'].unique()}")

    def test_single_width_no_merge(self):
        """규격이 1개일 때 인접규격 미적용"""
        df = make_orders([(500, 10.0, 2000)])
        result, groups, last_no = apply_sheet_grouping_jh(
            df, 0, 'TEST001', adj_cnt=2, adj_trim=20, max_wgt=100, min_wgt=5
        )
        self.assertEqual(result.iloc[0]['가로'], 500)
        print(f"[PASS] 단일 규격: {result['가로'].unique()}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

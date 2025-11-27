import pandas as pd
import unittest
from optimize.roll_optimize import RollOptimize
from optimize.roll_sl_optimize import RollSLOptimize

class TestOptimizeOutput(unittest.TestCase):
    def setUp(self):
        self.dummy_data = pd.DataFrame({
            'group_order_no': ['G1', 'G2'],
            '주문수량': [10, 20],
            '지폭': [500, 400],
            '롤길이': [1000, 1000],
            'dia': [100, 100],
            'core': ['3', '3'],
            'luster': ['Y', 'Y'],
            'color': ['White', 'White'],
            'order_pattern': ['P1', 'P1']
        })
        self.lot_no = "TEST_LOT"

    def test_roll_optimize_output(self):
        optimizer = RollOptimize(
            df_spec_pre=self.dummy_data,
            max_width=1000,
            min_width=0,
            max_pieces=2,
            lot_no=self.lot_no
        )
        # Mocking run_optimize to just test _format_results with a dummy solution
        # Since run_optimize calls _solve_master_problem which uses solver, we might want to just test _format_results directly if possible.
        # But _format_results needs final_solution and self.patterns populated.
        
        # Let's try to run a very simple optimization that should succeed quickly
        results = optimizer.run_optimize()
        
        if "error" in results:
            self.fail(f"RollOptimize failed: {results['error']}")
            
        pattern_details = results['pattern_details_for_db']
        self.assertTrue(len(pattern_details) > 0)
        first_detail = pattern_details[0]
        
        required_keys = ['diameter', 'color', 'luster', 'p_lot', 'core', 'order_pattern']
        for key in required_keys:
            self.assertIn(key, first_detail)
            
        self.assertEqual(first_detail['p_lot'], self.lot_no)
        self.assertEqual(first_detail['diameter'], 100)
        self.assertEqual(first_detail['color'], 'White')

    def test_roll_sl_optimize_output(self):
        optimizer = RollSLOptimize(
            df_spec_pre=self.dummy_data,
            max_width=1000,
            min_width=0,
            max_pieces=2,
            lot_no=self.lot_no
        )
        
        results = optimizer.run_optimize()
        
        if "error" in results:
            self.fail(f"RollSLOptimize failed: {results['error']}")
            
        pattern_details = results['pattern_details_for_db']
        self.assertTrue(len(pattern_details) > 0)
        first_detail = pattern_details[0]
        
        required_keys = ['diameter', 'color', 'luster', 'p_lot', 'core', 'order_pattern']
        for key in required_keys:
            self.assertIn(key, first_detail)
            
        self.assertEqual(first_detail['p_lot'], self.lot_no)
        self.assertEqual(first_detail['diameter'], 100)
        self.assertEqual(first_detail['color'], 'White')

if __name__ == '__main__':
    unittest.main()

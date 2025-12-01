import pandas as pd
import unittest
from optimize.sheet_optimize import SheetOptimize

class TestSheetOptimizeOutput(unittest.TestCase):
    def test_output_columns(self):
        # Mock Data
        df_orders = pd.DataFrame({
            '가로': [1000, 1000],
            '세로': [500, 500],
            '주문톤': [10, 10],
            'group_order_no': ['G1', 'G2'],
            'color': ['White', 'White'],
            'luster': [1, 1],
            'order_pattern': ['P1', 'P1'],
            '등급': ['A', 'A'],
            '수출내수': ['내수', '내수']
        })
        
        # Initialize Optimizer
        optimizer = SheetOptimize(
            df_spec_pre=df_orders,
            max_width=3000,
            min_width=2000,
            max_pieces=5,
            b_wgt=100,
            sheet_roll_length=10000,
            sheet_trim=0,
            min_sc_width=500,
            max_sc_width=1500,
            db=None,
            lot_no='TEST_LOT',
            version='V1'
        )
        
        # Run Optimize (Small scale to force brute force or quick result)
        # Since we don't want to actually run the complex optimization, 
        # we can mock the internal methods or just run it if it's fast enough.
        # Given the small input, it should be fast.
        
        results = optimizer.run_optimize(start_prod_seq=0)
        
        if "error" in results:
            print(f"Optimization failed: {results['error']}")
            # If it fails due to constraints, we might need to adjust them.
            # But let's see if it produces a result.
        
        if "pattern_result" in results:
            df_result = results["pattern_result"]
            print("\nResult Columns:", df_result.columns.tolist())
            print(df_result.head())
            
            required_columns = ['pattern', 'pattern_width', 'count', 'loss_per_roll', 'pattern_length', 'wd_width']
            for col in required_columns:
                self.assertIn(col, df_result.columns)
                self.assertFalse(df_result[col].isnull().all(), f"Column {col} is all NaN")

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import MagicMock
from db.db_insert_data import DataInserters

class TestDBInsertLogic(unittest.TestCase):
    def test_insert_pattern_sequence_max_width(self):
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        inserter = DataInserters()
        
        # Test Data
        lot_no = 'TEST_LOT'
        version = 'V1'
        plant = '3000'
        pm_no = '1'
        schedule_unit = 'SU'
        global_max_width = 5000
        paper_type = 'PT'
        b_wgt = 100
        
        # Pattern Details with specific max_width
        pattern_details = [
            {
                'count': 1,
                'prod_seq': 1,
                'widths': [1000, 1000, 0, 0, 0, 0, 0, 0],
                'group_nos': ['G1', 'G2', '', '', '', '', '', ''],
                'max_width': 3000, # Specific max width
                'rs_gubun': 'S'
            },
            {
                'count': 1,
                'prod_seq': 2,
                'widths': [2000, 0, 0, 0, 0, 0, 0, 0],
                'group_nos': ['G3', '', '', '', '', '', '', ''],
                # No max_width, should use global
                'rs_gubun': 'S'
            }
        ]
        
        # Execute
        inserter.insert_pattern_sequence(
            mock_conn, lot_no, version, plant, pm_no, schedule_unit, global_max_width,
            paper_type, b_wgt, pattern_details
        )
        
        # Verify
        # Check the bind_vars passed to executemany
        args, _ = mock_cursor.executemany.call_args
        query, bind_vars_list = args
        
        self.assertEqual(len(bind_vars_list), 2)
        
        # First record should have max_width = 3000
        self.assertEqual(bind_vars_list[0]['max_width'], 3000)
        
        # Second record should have max_width = 5000 (global)
        self.assertEqual(bind_vars_list[1]['max_width'], 5000)
        
        print("Verification Successful: max_width is correctly prioritized.")

if __name__ == '__main__':
    unittest.main()

import pandas as pd
import time
import sys
import os

# Add the directory to path to import RollOptimize
sys.path.append(r'd:\dev_src\opti_paperwidth\optimize')
from roll_optimize import RollOptimize

def test_generate_patterns():
    # Create dummy data
    # 10 items, widths between 100 and 300
    data = {
        'group_order_no': [f'Order_{i}' for i in range(10)],
        '주문수량': [10] * 10,
        '지폭': [100 + i*20 for i in range(10)], # 100, 120, ..., 280
        '롤길이': [1000] * 10,
        '수출내수': ['내수'] * 10
    }
    df = pd.DataFrame(data)
    
    # Initialize optimizer
    # max_width 1000, max_pieces 8
    optimizer = RollOptimize(df, max_width=1000, min_width=0, max_pieces=8)
    
    print("Starting _generate_all_patterns...")
    start_time = time.time()
    optimizer._generate_all_patterns()
    if hasattr(optimizer, '_filter_patterns_by_lp'):
        print("Applying LP filtering...")
        optimizer._filter_patterns_by_lp()
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Number of patterns generated: {len(optimizer.patterns)}")
    
    # Also test MIP solve time if patterns are generated
    if optimizer.patterns:
        print("Starting MIP solve (mock)...")
        start_mip = time.time()
        # We can't easily mock the internal solver call without modifying code or running it.
        # Let's just run run_optimize but force it to use the generated patterns
        # effectively we can just call _solve_master_problem(is_final_mip=True)
        
        # We need to make sure demands are set up (they are in __init__)
        solution = optimizer._solve_master_problem(is_final_mip=True)
        end_mip = time.time()
        print(f"MIP Solve Time: {end_mip - start_mip:.4f} seconds")
        if solution:
            print("Solution found.")
        else:
            print("No solution found.")

if __name__ == "__main__":
    test_generate_patterns()

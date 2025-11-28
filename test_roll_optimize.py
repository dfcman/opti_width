import pandas as pd
from optimize.roll_optimize import RollOptimize

def test_roll_optimization():
    # Mock data representing a typical roll order scenario
    data = {
        'group_order_no': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', '10'],
        '지폭': [606, 606, 667, 706, 706, 727, 788, 909, 970, 1091],
        '주문수량': [6, 8, 7, 7, 14, 7, 8, 5, 10, 4],
        '롤길이': [2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050]
    }
    df_spec_pre = pd.DataFrame(data)
    
    print("--- Test Case 1: Standard Scenario ---")
    optimizer = RollOptimize(
        df_spec_pre=df_spec_pre,
        max_width=4880,
        min_width=4500,
        max_pieces=8
    )
    
    results = optimizer.run_optimize()
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("Optimization Successful!")
        print(results['pattern_result'])
        print(f"Total Patterns Used: {len(results['pattern_result'])}")
        print(results['fulfillment_summary'])

    # Test Case 2: Scenario where pattern reduction is critical
    # Many small orders that could be combined in many ways, but we want few patterns
    print("\n--- Test Case 2: Pattern Reduction Scenario ---")
    data2 = {
        'group_order_no': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8'],
        '지폭': [606, 667, 706, 727, 788, 909, 970, 1091],
        '주문수량': [14, 7, 21, 7, 8, 5, 10, 4],
        '롤길이': [2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050]
    }
    df_spec_pre2 = pd.DataFrame(data2)
    
    optimizer2 = RollOptimize(
        df_spec_pre=df_spec_pre2,
        max_width=4880, # Changed from 1000 to 4880
        min_width=4500,
        max_pieces=10
    )
    
    results2 = optimizer2.run_optimize()
    
    if "error" in results2:
        print(f"Error: {results2['error']}")
    else:
        print("Optimization Successful!")
        print(results2['pattern_result'])
        print(f"Total Patterns Used: {len(results2['pattern_result'])}")
        print(results2['fulfillment_summary'])

    

if __name__ == "__main__":
    test_roll_optimization()

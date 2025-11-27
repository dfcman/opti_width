import pandas as pd
from optimize.roll_optimize import RollOptimize
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)

def run_optimization(data, label):
    print(f"\n{'='*20} {label} {'='*20}")
    df_orders = pd.DataFrame(data)
    print(f"Items count: {len(df_orders)}")
    
    start_time = time.time()
    optimizer = RollOptimize(
        df_spec_pre=df_orders,
        max_width=4900,
        min_width=4500,
        max_pieces=8
    )
    results = optimizer.run_optimize()
    end_time = time.time()
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Execution Time: {end_time - start_time:.2f}s")
        print(f"Total Patterns: {len(results['pattern_result'])}")
        print(f"Total Over-production: {results['fulfillment_summary']['과부족(롤)'].sum()}")
        print("Patterns:")
        print(results['pattern_result'][['pattern', 'count']])
        return len(results['pattern_result'])

# Case 1: Merged Orders
data_merged = [
    {'group_order_no': 'O1', '지폭': 606, '주문수량': 14, '롤길이': 2050},
    {'group_order_no': 'O2', '지폭': 667, '주문수량': 7, '롤길이': 2050},
    {'group_order_no': 'O3', '지폭': 706, '주문수량': 21, '롤길이': 2050},
    {'group_order_no': 'O4', '지폭': 727, '주문수량': 7, '롤길이': 2050},
    {'group_order_no': 'O5', '지폭': 788, '주문수량': 8, '롤길이': 2050},
    {'group_order_no': 'O6', '지폭': 909, '주문수량': 5, '롤길이': 2050},
    {'group_order_no': 'O7', '지폭': 970, '주문수량': 10, '롤길이': 2050},
    {'group_order_no': 'O8', '지폭': 1091, '주문수량': 4, '롤길이': 2050},
]

# Case 2: Split Orders
data_split = [
    {'group_order_no': 'O1_1', '지폭': 606, '주문수량': 6, '롤길이': 2050},
    {'group_order_no': 'O1_2', '지폭': 606, '주문수량': 8, '롤길이': 2050},
    {'group_order_no': 'O2', '지폭': 667, '주문수량': 7, '롤길이': 2050},
    {'group_order_no': 'O3_1', '지폭': 706, '주문수량': 7, '롤길이': 2050},
    {'group_order_no': 'O3_2', '지폭': 706, '주문수량': 14, '롤길이': 2050},
    {'group_order_no': 'O4', '지폭': 727, '주문수량': 7, '롤길이': 2050},
    {'group_order_no': 'O5', '지폭': 788, '주문수량': 8, '롤길이': 2050},
    {'group_order_no': 'O6', '지폭': 909, '주문수량': 5, '롤길이': 2050},
    {'group_order_no': 'O7', '지폭': 970, '주문수량': 10, '롤길이': 2050},
    {'group_order_no': 'O8', '지폭': 1091, '주문수량': 4, '롤길이': 2050},
]

count_merged = run_optimization(data_merged, "Merged Orders")
count_split = run_optimization(data_split, "Split Orders")

print(f"\nSummary: Merged Patterns={count_merged}, Split Patterns={count_split}")

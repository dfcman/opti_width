import pandas as pd
from optimize.roll_optimize import RollOptimize
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define the demands based on the user's log/description
# 606: 14, 667: 7, 706: 21, 727: 7, 788: 8, 909: 5, 970: 10, 1091: 4
data = [
    {'group_order_no': 'O1', '지폭': 606, '주문수량': 14, '롤길이': 2050},
    {'group_order_no': 'O2', '지폭': 667, '주문수량': 7, '롤길이': 2050},
    {'group_order_no': 'O3', '지폭': 706, '주문수량': 21, '롤길이': 2050},
    {'group_order_no': 'O4', '지폭': 727, '주문수량': 7, '롤길이': 2050},
    {'group_order_no': 'O5', '지폭': 788, '주문수량': 8, '롤길이': 2050},
    {'group_order_no': 'O6', '지폭': 909, '주문수량': 5, '롤길이': 2050},
    {'group_order_no': 'O7', '지폭': 970, '주문수량': 10, '롤길이': 2050},
    {'group_order_no': 'O8', '지폭': 1091, '주문수량': 4, '롤길이': 2050},
]

df_orders = pd.DataFrame(data)

print("--- Demands ---")
print(df_orders)

optimizer = RollOptimize(
    df_spec_pre=df_orders,
    max_width=4900,
    min_width=4500, # Assuming some min width, user didn't specify but usually it's around 2000-3000 or 0. Log said "min_width=2050" maybe? No, roll length is 2050.
    # Let's check log for min_width. "min_width=..." is in the log but value is not clear.
    # User's patterns sums: 4584, 4867, 4867, 4727, 4867.
    # Max width 4900.
    # Let's assume min_width is small enough.
    max_pieces=8
)

results = optimizer.run_optimize()

if "error" in results:
    print(f"Error: {results['error']}")
else:
    print("\n--- Optimization Results ---")
    print(results['pattern_result'])
    print(f"\nTotal Patterns: {len(results['pattern_result'])}")
    
    # Check fulfillment
    print("\n--- Fulfillment ---")
    print(results['fulfillment_summary'])

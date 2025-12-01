
import pandas as pd
import numpy as np
from optimize.sheet_optimize import SheetOptimize

# Mock data based on user sample
# 70 303250900073071 NaN NaN A NaN 2 0 550.0 910.0 내수 2.803 2.803 0.0 2.0
# Width: 550, Length: 910, Grade: A, Type: 내수, OrderTons: 2.803
# Needed Rolls: 2

data = {
    'plant': ['3000'],
    'pm_no': ['1'],
    'schedule_unit': ['1'],
    'order_no': ['ORD001'],
    '가로': [550],
    '세로': [910],
    '주문톤': [2.803],
    '등급': ['A'],
    '수출내수': ['내수'],
    'color': [''],
    'luster': [0],
    'order_pattern': ['']
}
df_orders = pd.DataFrame(data)
df_orders['group_order_no'] = '303250900073071'

# Parameters
# Assuming min_width is large enough to potentially cause issues, or small enough to work.
# User sample: Needed=2, Prod=0 (maybe).
# Let's try to reproduce Prod=0 first.
re_max_width = 5000
re_min_width = 2800 # If 550*4 = 2200 < 2800, it might fail if max_pieces=4
re_max_pieces = 4
b_wgt = 100.0
sheet_length_re = 10000
sheet_trim_size = 0
min_sc_width = 500
max_sc_width = 2000

print("--- Running SheetOptimize with mock data ---")
optimizer = SheetOptimize(
    df_spec_pre=df_orders.copy(),
    max_width=re_max_width,
    min_width=re_min_width,
    max_pieces=re_max_pieces,
    b_wgt=b_wgt,
    sheet_roll_length=sheet_length_re,
    sheet_trim=sheet_trim_size,
    min_sc_width=min_sc_width,
    max_sc_width=max_sc_width,
    lot_no='TEST',
    version='1'
)

results = optimizer.run_optimize()

if "error" in results:
    print(f"Optimization failed: {results['error']}")
else:
    print("Optimization success")
    summary = results['fulfillment_summary']
    print("\nFulfillment Summary Columns:", summary.columns.tolist())
    print("\nFulfillment Summary Content:")
    print(summary.to_string())
    
    # Check for NaNs
    if summary.isnull().values.any():
        print("\nWARNING: NaNs found in summary!")
        print(summary[summary.isnull().any(axis=1)])
    else:
        print("\nNo NaNs in summary.")

    # Check specific order
    row = summary[summary['group_order_no'] == '303250900073071']
    if not row.empty:
        print("\nTarget Order Row:")
        for col in summary.columns:
            print(f"{col}: {row[col].values[0]}")

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:22:18 2024

@author: Hansol
"""

import pandas as pd

df1 = pd.DataFrame({
    'key':['A', 'B', 'C', 'D'],
    'value' : range(4)
    })

df2 = pd.DataFrame({
    'key':['B', 'D', 'D', 'E'],
    'value' : range(4, 8)
    })

print(df1)
print(df2)

merged = pd.merge(df1, df2, on='key', how='left')
print(merged)
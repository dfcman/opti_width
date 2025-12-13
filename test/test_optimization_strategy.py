import pandas as pd
import time
import sys
from copy import deepcopy
import unittest.mock

sys.path.append(r'd:\dev_src\opti_paperwidth\optimize')
from roll_optimize import RollOptimize

def test_strategies():
    # Create dummy data
    data = {
        'group_order_no': [f'Order_{i}' for i in range(10)],
        '주문수량': [10] * 10,
        '지폭': [100 + i*20 for i in range(10)], # 100, 120, ..., 280
        '롤길이': [1000] * 10,
        '수출내수': ['내수'] * 10
    }
    df = pd.DataFrame(data)
    
    # Baseline: All Patterns
    print("--- Baseline: All Patterns ---")
    opt_base = RollOptimize(df, max_width=1000, min_width=0, max_pieces=8)
    
    start = time.time()
    opt_base._generate_all_patterns()
    sol_base = opt_base._solve_master_problem(is_final_mip=True)
    print(f"Objective: {sol_base['objective'] if sol_base else 'None'}")
    
    # Strategy 6: Generous RC Filtering
    print("\n--- Strategy 6: Generous RC Filtering ---")
    opt_strat6 = RollOptimize(df, max_width=1000, min_width=0, max_pieces=8)
    
    start = time.time()
    opt_strat6._generate_all_patterns()
    
    # 1. Solve LP
    lp_solution = opt_strat6._solve_master_problem(is_final_mip=False)
    duals = lp_solution['duals']
    
    kept_indices = set()
    
    # 2. Keep Basis
    for j, count in lp_solution['pattern_counts'].items():
        if count > 1e-6:
            kept_indices.add(j)
            
    print(f"Basis size: {len(kept_indices)}")
    
    # 3. Filter by RC < Threshold
    # Threshold = PATTERN_SETUP_COST (50000)
    # Actually, since we minimize, and RC is usually non-negative for optimal LP.
    # Wait, RC = c - z.
    # If we want to minimize, we look for negative RC to enter basis.
    # At optimality, all non-basic vars have RC >= 0.
    # We want to keep vars that are "close" to entering basis.
    # So RC < Threshold.
    
    THRESHOLD = 50000.0
    
    for j, pattern in enumerate(opt_strat6.patterns):
        if j in kept_indices:
            continue
            
        # RC = 0 - sum(dual * count) = -val
        # But wait, the duals from GLOP for maximization?
        # GLOP minimizes.
        # Constraints: Ax = b.
        # Duals y.
        # Reduced cost = c - yA.
        # Here c = 0 (for pattern vars).
        # So RC = - yA.
        # yA = sum(dual * count).
        # So RC = - sum(dual * count).
        # If duals are positive (shadow prices), then RC is negative?
        # Wait, if RC is negative, we should add it to basis to reduce objective.
        # At optimality, RC >= 0.
        # So -sum(dual * count) >= 0 => sum(dual * count) <= 0.
        # But duals for demand >= constraint are usually positive (marginal cost of demand).
        # So sum(dual * count) is positive.
        # So RC is negative.
        # This means my RC calculation is wrong or sign convention is different.
        
        # Let's check OR-Tools documentation or infer.
        # Solver minimizes.
        # Constraint: sum(pattern * x) >= demand.
        # Duals should be positive.
        # Reduced cost = c_j - sum(a_ij * y_i).
        # c_j = 0.
        # So RC = - sum(count * dual).
        # If dual > 0, RC < 0.
        # If RC < 0, we can improve solution by increasing x.
        # But we are at optimality.
        # So RC must be >= 0.
        # This implies duals must be <= 0?
        # Or constraints are <= ?
        # Constraints are `==` (with over_prod_vars).
        # production_expr == demand + over_prod.
        # production - over_prod == demand.
        # If we increase demand, cost increases. So dual should be positive?
        
        # Let's just use the value `sum(dual * count)`.
        # This is the "implied value" of the pattern.
        # We want patterns with HIGH implied value.
        # Because they satisfy demand efficiently.
        # In LP optimality, `sum(dual * count) <= c_j` (if maximizing) or `>=`?
        
        # Let's ignore signs and just say:
        # We want patterns that are "good".
        # Good patterns have high `sum(dual * count)`.
        # Because they cover high-valued demands.
        # So we sort by `sum(dual * count)` descending.
        # And keep top N.
        # Or keep all with `sum(dual * count) > some_value`.
        
        # But simpler:
        # Just keep top 1000 patterns by `sum(dual * count)`.
        # 1000 is 1/4 of 4079.
        # Should be safe.
        pass

    # Let's implement Top 1000 by Dual Value
    patterns_data = []
    for j, pattern in enumerate(opt_strat6.patterns):
        val = sum(duals.get(item, 0) * count for item, count in pattern.items())
        patterns_data.append({'index': j, 'val': val})
        
    patterns_data.sort(key=lambda x: x['val'], reverse=True)
    
    for i in range(min(1000, len(patterns_data))):
        kept_indices.add(patterns_data[i]['index'])
        
    print(f"Total kept patterns: {len(kept_indices)}")
    
    filtered_patterns = [opt_strat6.patterns[j] for j in kept_indices]
    opt_strat6.patterns = filtered_patterns
    opt_strat6._rebuild_pattern_cache()
    
    gen_time = time.time() - start
    print(f"Filtering Time: {gen_time:.4f}s")
    
    start = time.time()
    sol_strat6 = opt_strat6._solve_master_problem(is_final_mip=True)
    solve_time = time.time() - start
    print(f"Solve Time: {solve_time:.4f}s")
    print(f"Objective: {sol_strat6['objective'] if sol_strat6 else 'None'}")

    if sol_base and sol_strat6:
        diff = abs(sol_base['objective'] - sol_strat6['objective'])
        print(f"\nObjective Difference: {diff}")
        if diff < 1e-5:
            print("SUCCESS: Objectives match.")
        else:
            print("FAILURE: Objectives do not match.")

if __name__ == "__main__":
    test_strategies()

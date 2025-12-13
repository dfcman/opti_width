
try:
    from optimize.roll_optimize import RollOptimize
    from optimize.sheet_optimize import SheetOptimize
    print("Imports successful")
    import optimize.roll_optimize as ro
    print(f"RollOptimize NUM_THREADS: {ro.NUM_THREADS}")
except Exception as e:
    print(f"Import failed: {e}")

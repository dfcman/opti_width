# Copilot Instructions for opti_paperwidth Codebase
- The user prefers all explanations to be in Korean.
## Overview
This repository focuses on optimization algorithms for paper width and sheet scheduling, likely for manufacturing or logistics. The codebase is organized as a set of Python scripts, each handling a distinct part of the workflow.

## Architecture & Major Components
- **sheet_optimize.py & variants**: Core logic for sheet optimization. Variants (with dates) are experimental or historical versions.
- **roll_optimize.py**: Handles roll-based optimization, likely interacting with sheet logic.
- **band.py, processing.py**: Utility modules for band calculations and general data processing.
- **db_connector.py**: Database connectivity, likely for fetching or storing order and result data.
- **execute.py**: Orchestrates execution of optimization routines.
- **results/**: Stores output CSVs from optimization runs.
- **conf/config.ini, settings.json**: Configuration files for parameters and environment settings.

## Data Flow
- Input data is read from CSV files (e.g., `my_orders.csv`, `roll_orders.csv`, `target_lot.csv`).
- Optimization scripts process these inputs and write results to the `results/` directory as CSVs.
- Configuration is loaded from `conf/config.ini` and/or `settings.json`.

## Developer Workflows
- **Run optimization**: Execute scripts directly, e.g., `python sheet_optimize.py` or `python roll_optimize.py`.
- **Debugging**: Use `test.py`, `test1.py`, etc. for isolated logic tests. No formal test framework detected.
- **Configuration**: Adjust parameters in `conf/config.ini` or `settings.json` before running scripts.
- **Results**: Check output in `results/` for generated CSVs.

## Project-Specific Patterns
- Multiple versions of optimization scripts are kept for experimentation; newer files often have dates in their names.
- Data is exchanged via CSVs and simple config files, not via APIs or complex serialization.
- Utility modules are reused across scripts (e.g., `band.py`, `processing.py`).
- No package structure; scripts are top-level and imported as needed.

## Integration Points
- Database access via `db_connector.py` (details in code).
- External data via CSVs in the root directory.
- Results written to `results/` for downstream analysis.

## Example Usage
```powershell
python sheet_optimize.py
python roll_optimize.py
```

## Key Files
- `sheet_optimize.py`, `roll_optimize.py`, `band.py`, `processing.py`, `db_connector.py`, `conf/config.ini`, `settings.json`, `results/`

---
**Update this file if workflows or architecture change.**

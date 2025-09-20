#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_sheet_optimize.py
두 엔진(sheet_optimize.py vs sheet_optimize_250903.py)을 동일 입력으로 실행하고
패턴/롤수/과부족 요약을 비교하는 하네스.

로컬에서 실행 전, OR-Tools가 설치되어 있어야 합니다.
    pip install ortools pandas

사용법:
    python compare_sheet_optimize.py --data my_orders.csv --max_width 3200 --min_width 850 --max_pieces 4 \
        --b_wgt 0.0008 --sheet_roll_length 1000 --sheet_trim 20 --min_sc_width 850 --max_sc_width 2600

CSV 컬럼 예시(필요에 맞게 조정):
    group_order_no,가로,주문톤,길이
    G1,1200,1.2,1000
    G2,1400,1.0,1000
    G3,900,0.8,1000
"""
import argparse
import json
import sys
import traceback
from importlib.machinery import SourceFileLoader
from pathlib import Path
import types
import pandas as pd

def load_module(path: Path, name: str) -> types.ModuleType:
    loader = SourceFileLoader(name, str(path))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod

def run_engine(mod: types.ModuleType, df_spec_pre: pd.DataFrame, params: dict):
    result = {"ok": False, "error": None, "pattern_result": None, "fulfillment_summary": None, "raw": None}
    try:
        init_params_candidates = [
            ["df_spec_pre", "max_width", "min_width", "max_pieces", "b_wgt", "sheet_roll_length", "sheet_trim"],
            ["df_spec_pre", "max_width", "min_width", "max_pieces", "b_wgt", "sheet_roll_length", "sheet_trim", "min_sc_width", "max_sc_width"],
            ["df_spec_pre", "max_width", "min_width", "max_pieces"],
        ]
        last_err = None
        engine = None
        for cand in init_params_candidates:
            try:
                kwargs = {k: params[k] for k in cand}
                engine = mod.SheetOptimize(**kwargs)
                break
            except Exception as e:
                last_err = e
                engine = None
        if engine is None:
            raise RuntimeError(f"Failed to construct engine with any known signature: {last_err}")

        solve_name = next((n for n in ["solve","optimize","run","execute","run_optimization"] if hasattr(engine,n)), None)
        if solve_name is None:
            raise AttributeError("No solve/optimize method found on SheetOptimize")
        raw = getattr(engine, solve_name)()
        result["raw"] = raw

        pr = None
        fs = None
        if isinstance(raw, dict):
            pr = raw.get("pattern_result")
            fs = raw.get("fulfillment_summary")
        if pr is None and hasattr(engine, "pattern_result"):
            pr = getattr(engine, "pattern_result")
        if fs is None and hasattr(engine, "fulfillment_summary"):
            fs = getattr(engine, "fulfillment_summary")

        if isinstance(pr, list):
            pr = pd.DataFrame(pr)
        if isinstance(fs, list):
            fs = pd.DataFrame(fs)

        result["pattern_result"] = pr
        result["fulfillment_summary"] = fs
        result["ok"] = True
        return result
    except Exception as e:
        result["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
        return result

def compare_results(res_a: dict, res_b: dict) -> dict:
    def safe_len(df):
        try: return len(df) if df is not None else 0
        except: return 0
    def tot_rolls(df):
        if df is None: return None
        for c in ["Count","count","롤수","num_rolls","rolls"]:
            if c in getattr(df, "columns", []):
                try: return float(df[c].sum())
                except: pass
        return None
    def over_under_from_fs(fs):
        res = {"over_rolls": None, "under_rolls": None, "over_tons": None, "under_tons": None}
        if fs is None: return res
        cols = fs.columns if hasattr(fs, "columns") else []
        roll_over_cols = [c for c in cols if "과부족(롤)" in str(c)]
        ton_over_cols  = [c for c in cols if "과부족(톤)" in str(c)]
        try:
            if roll_over_cols:
                v = fs[roll_over_cols[0]]
                res["over_rolls"]  = float(v[v>0].sum())
                res["under_rolls"] = float(v[v<0].sum())
            if ton_over_cols:
                v = fs[ton_over_cols[0]]
                res["over_tons"]   = float(v[v>0].sum())
                res["under_tons"]  = float(v[v<0].sum())
        except: pass
        return res

    return {
        "patterns_count_a": safe_len(res_a.get("pattern_result")),
        "patterns_count_b": safe_len(res_b.get("pattern_result")),
        "total_rolls_a": tot_rolls(res_a.get("pattern_result")),
        "total_rolls_b": tot_rolls(res_b.get("pattern_result")),
        "fulfillment_a": over_under_from_fs(res_a.get("fulfillment_summary")),
        "fulfillment_b": over_under_from_fs(res_b.get("fulfillment_summary")),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file_a", default="sheet_optimize.py", help="엔진 A 파일 경로")
    p.add_argument("--file_b", default="sheet_optimize_250903.py", help="엔진 B 파일 경로")
    p.add_argument("--data", required=False, help="주문 CSV 경로 (미지정 시 내장 샘플 사용)")
    p.add_argument("--outdir", default=".", help="결과 저장 폴더")
    # 공통 파라미터
    p.add_argument("--max_width", type=int, default=3200)
    p.add_argument("--min_width", type=int, default=850)
    p.add_argument("--max_pieces", type=int, default=4)
    p.add_argument("--b_wgt", type=float, default=0.0008)
    p.add_argument("--sheet_roll_length", type=int, default=1000)
    p.add_argument("--sheet_trim", type=int, default=20)
    p.add_argument("--min_sc_width", type=int, default=850)
    p.add_argument("--max_sc_width", type=int, default=2600)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    file_a = Path(args.file_a)
    file_b = Path(args.file_b)

    # Load data
    if args.data:
        df = pd.read_csv(args.data, encoding="utf-8-sig")
    else:
        df = pd.DataFrame([
            {"group_order_no":"G1","가로":1200,"주문톤":1.2,"길이":1000},
            {"group_order_no":"G2","가로":1400,"주문톤":1.0,"길이":1000},
            {"group_order_no":"G3","가로":900, "주문톤":0.8,"길이":1000},
        ])

    params = dict(
        df_spec_pre=df,
        max_width=args.max_width,
        min_width=args.min_width,
        max_pieces=args.max_pieces,
        b_wgt=args.b_wgt,
        sheet_roll_length=args.sheet_roll_length,
        sheet_trim=args.sheet_trim,
        min_sc_width=args.min_sc_width,
        max_sc_width=args.max_sc_width,
    )

    # Load modules
    try:
        mod_a = load_module(file_a, "engine_a")
        mod_b = load_module(file_b, "engine_b")
    except Exception as e:
        print("[모듈 로드 실패]", e, file=sys.stderr)
        sys.exit(2)

    # Run both
    res_a = run_engine(mod_a, df, params)
    res_b = run_engine(mod_b, df, params)

    # Export raw tables if exist
    if isinstance(res_a.get("pattern_result"), pd.DataFrame):
        res_a["pattern_result"].to_csv(outdir / "engine_a_pattern_result.csv", index=False, encoding="utf-8-sig")
    if isinstance(res_b.get("pattern_result"), pd.DataFrame):
        res_b["pattern_result"].to_csv(outdir / "engine_b_pattern_result.csv", index=False, encoding="utf-8-sig")
    if isinstance(res_a.get("fulfillment_summary"), pd.DataFrame):
        res_a["fulfillment_summary"].to_csv(outdir / "engine_a_fulfillment_summary.csv", index=False, encoding="utf-8-sig")
    if isinstance(res_b.get("fulfillment_summary"), pd.DataFrame):
        res_b["fulfillment_summary"].to_csv(outdir / "engine_b_fulfillment_summary.csv", index=False, encoding="utf-8-sig")

    comparison = compare_results(res_a, res_b)
    report = {
        "engine_a": {"file": str(file_a), "ok": res_a["ok"], "error": res_a["error"],
                     "patterns_rows": None if res_a["pattern_result"] is None else len(res_a["pattern_result"]) },
        "engine_b": {"file": str(file_b), "ok": res_b["ok"], "error": res_b["error"],
                     "patterns_rows": None if res_b["pattern_result"] is None else len(res_b["pattern_result"]) },
        "comparison": comparison,
    }
    with open(outdir / "compare_sheet_optimize_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

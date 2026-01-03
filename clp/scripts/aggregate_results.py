
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple
from openpyxl import Workbook

# ---- CONFIG ----
# RESULTS_ROOT = Path(r"C:\Users\elahi\Desktop\clp\clp\results\BR-Original")
WD_DIR = [
    "BR-Original-baseline",
    "BR-Original",
    "BR-Original-two_phase",
    "BR-Modified-NSGA2_bi",
    "BR-Modified-NSGA2_tri",
]


BASE_ROOT = Path(r"C:\Users\elahi\Desktop\clp\clp\results")
RESULTS_ROOT = BASE_ROOT / WD_DIR[3]
OUT_FILE_NAME = WD_DIR[3]
OUT_DIR = RESULTS_ROOT   / "_summary" #/ WD_DIR[3]
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["six_way", "C1_respect"]
BR_CLASSES = [f"BR{i}" for i in range(16)]
CASES = range(1, 101)


def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_case(path: Path) -> Optional[Dict[str, Any]]:
    """
    Read one result json and extract the metrics we care about.
    Returns None if file is invalid or missing expected structure.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sol = data["pareto_front"][0]  # baseline single solution
        obj = sol["objectives"]
        cg = sol["cg"]
        diag = sol["diagnostics"]

        return {
            "z1": float(obj["Z1"]),
            "z2": float(obj.get("Z2", 0.0)),  # optional
            "z3": float(obj["Z3"]),
            "rdx": float(cg["rd_pct"]["x"]),
            "rdy": float(cg["rd_pct"]["y"]),
            "rdz": float(cg["rd_pct"]["z"]),
            "cgx": float(cg["loaded"]["x"]),
            "cgy": float(cg["loaded"]["y"]),
            "cgz": float(cg["loaded"]["z"]),
            "devx": float(cg["dev"]["x"]),
            "devy": float(cg["dev"]["y"]),
            "devz": float(cg["dev"]["z"]),
            "placed": int(diag["placed_count"]),
            "unplaced": int(diag["unplaced_count"]),
            "elapsed": float(diag["elapsed_sec"]),
            "timestamp_utc": _safe_get(data, ["timestamp_utc"], ""),
            "run_id": _safe_get(data, ["run_id"], ""),
        }
    except json.JSONDecodeError as e:
        print(f"[read_case] Invalid JSON in {path}: {e}")
    except KeyError as e:
        print(f"[read_case] Missing key {e} in {path}")
    except IndexError as e:
        print(f"[read_case] Empty pareto_front in {path}")
    except Exception as e:
        print(f"[read_case] Unexpected error in {path}: {e}")
        traceback.print_exc()
    return None



def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")



def write_xlsx(path, rows, sheet_name="data"):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    if not rows:
        wb.save(path)
        return

    headers = list(rows[0].keys())
    ws.append(headers)

    for r in rows:
        ws.append([r.get(h, "") for h in headers])

    wb.save(path)

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def std(vals):
    return stdev(vals) if len(vals) > 1 else 0.0

def col(rows, name):
    return [float(r[name]) for r in rows]


def aggregate_results(
    results_root: Path = RESULTS_ROOT,
    out_dir: Path = OUT_DIR,
    write_index: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Produces:
      - flat table: one row per (BR class, mode, case)
      - summary table: one row per (BR class, mode) with mean/median stats
      - optional run_index.json (one entry per case file)
    Returns (flat_rows, summary_rows).
    """
    flat_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []

    for br in BR_CLASSES:
        for mode in MODES:
            rows: List[Dict[str, Any]] = []
            folder = results_root / br / mode

            for case_id in CASES:
                path = folder / f"{case_id}.json"
                if not path.exists():
                    continue

                r = read_case(path)
                if r is None:
                    continue

                r.update({"br_class": br, "mode": mode, "case_id": case_id})
                rows.append(r)
                flat_rows.append(r)

                if write_index:
                    index_rows.append({
                        "dataset_family": "BR-Original",
                        "br_class": br,
                        "mode": mode,
                        "case_id": case_id,
                        "path": str(path),
                        "run_id": r.get("run_id", ""),
                        "timestamp_utc": r.get("timestamp_utc", ""),
                        "Z1": r["z1"],
                        "Z3": r["z3"],
                        "RDx": r["rdx"],
                        "RDy": r["rdy"],
                        "RDz": r["rdz"],
                    })

            if not rows:
                continue


            summary_rows.append({
                "br_class": br,
                "mode": mode,
                "n": len(rows),

                # ---- Z1 ----
                "mean_Z1": round(mean(col(rows, "z1")), 6),
                "median_Z1": round(median(col(rows, "z1")), 6),
                "std_Z1": round(std(col(rows, "z1")), 6),
                "min_Z1": round(min(col(rows, "z1")), 6),
                "max_Z1": round(max(col(rows, "z1")), 6),

                # ---- Z2 ----
                "mean_Z2": round(mean(col(rows, "z2")), 6),
                "median_Z2": round(median(col(rows, "z2")), 6),
                "std_Z2": round(std(col(rows, "z2")), 6),
                "min_Z2": round(min(col(rows, "z2")), 6),
                "max_Z2": round(max(col(rows, "z2")), 6),

                # ---- Z3 ----
                "mean_Z3": round(mean(col(rows, "z3")), 6),
                "median_Z3": round(median(col(rows, "z3")), 6),
                "std_Z3": round(std(col(rows, "z3")), 6),
                "min_Z3": round(min(col(rows, "z3")), 6),
                "max_Z3": round(max(col(rows, "z3")), 6),

                # ---- RDx ----
                "mean_RDx": round(mean(col(rows, "rdx")), 4),
                "median_RDx": round(median(col(rows, "rdx")), 4),
                "std_RDx": round(std(col(rows, "rdx")), 4),
                "min_RDx": round(min(col(rows, "rdx")), 4),
                "max_RDx": round(max(col(rows, "rdx")), 4),

                # ---- RDy ----
                "mean_RDy": round(mean(col(rows, "rdy")), 4),
                "median_RDy": round(median(col(rows, "rdy")), 4),
                "std_RDy": round(std(col(rows, "rdy")), 4),
                "min_RDy": round(min(col(rows, "rdy")), 4),
                "max_RDy": round(max(col(rows, "rdy")), 4),

                # ---- RDz ----
                "mean_RDz": round(mean(col(rows, "rdz")), 4),
                "median_RDz": round(median(col(rows, "rdz")), 4),
                "std_RDz": round(std(col(rows, "rdz")), 4),
                "min_RDz": round(min(col(rows, "rdz")), 4),
                "max_RDz": round(max(col(rows, "rdz")), 4),

                # ---- unplaced ----
                "mean_unplaced": round(mean(col(rows, "unplaced")), 4),
                "median_unplaced": round(median(col(rows, "unplaced")), 4),
                "std_unplaced": round(std(col(rows, "unplaced")), 4),
                "min_unplaced": round(min(col(rows, "unplaced")), 4),
                "max_unplaced": round(max(col(rows, "unplaced")), 4),

                # ---- runtime ----
                "mean_time_sec": round(mean(col(rows, "elapsed")), 6),
                "median_time_sec": round(median(col(rows, "elapsed")), 6),
                "std_time_sec": round(std(col(rows, "elapsed")), 6),
                "min_time_sec": round(min(col(rows, "elapsed")), 6),
                "max_time_sec": round(max(col(rows, "elapsed")), 6),
            })

            

    # ---- WRITE OUTPUTS ----
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flat (JSON + CSV)
    flat_json = out_dir / f"{OUT_FILE_NAME}_flat.json"
    flat_csv = out_dir / f"{OUT_FILE_NAME}_flat.csv"
    flat_xlsx = out_dir / f"{OUT_FILE_NAME}_flat.xlsx"


    # flat_json = out_dir / "br_original_flat.json"
    # flat_csv = out_dir /"br_original_flat.csv"
    # flat_xlsx = out_dir /"br_original_flat.xlsx"

    write_json(flat_json, flat_rows)

    flat_fields = [
        "br_class", "mode", "case_id",
        "z1", "z2", "z3",
        "rdx", "rdy", "rdz",
        "cgx", "cgy", "cgz",
        "devx", "devy", "devz",
        "placed", "unplaced", "elapsed",
        "timestamp_utc", "run_id",
    ]
    write_csv(flat_csv, flat_rows, flat_fields)
    write_xlsx(flat_xlsx, flat_rows, sheet_name="flat")

    # Summary (JSON + CSV)
    summary_json = out_dir / f"{OUT_FILE_NAME}_summary.json"
    summary_csv = out_dir / f"{OUT_FILE_NAME}_summary.csv"
    summary_xlsx = out_dir / f"{OUT_FILE_NAME}_summary.xlsx"


    # summary_json = out_dir /"br_original_summary.json"
    # summary_csv = out_dir /"br_original_summary.csv"
    # summary_xlsx = out_dir /"br_original_summary.xlsx"
    write_json(summary_json, summary_rows)

    summary_fields = [
        "br_class", "mode", "n",

        "mean_Z1", "median_Z1", "std_Z1", "min_Z1", "max_Z1",
        "mean_Z2", "median_Z2", "std_Z2", "min_Z2", "max_Z2",
        "mean_Z3", "median_Z3", "std_Z3", "min_Z3", "max_Z3",

        "mean_RDx", "median_RDx", "std_RDx", "min_RDx", "max_RDx",
        "mean_RDy", "median_RDy", "std_RDy", "min_RDy", "max_RDy",
        "mean_RDz", "median_RDz", "std_RDz", "min_RDz", "max_RDz",

        "mean_unplaced", "median_unplaced", "std_unplaced", "min_unplaced", "max_unplaced",

        "mean_time_sec", "median_time_sec", "std_time_sec", "min_time_sec", "max_time_sec",
    ]

    write_csv(summary_csv, summary_rows, summary_fields)
    write_xlsx(summary_xlsx, summary_rows, sheet_name="summary")
    # Optional index
    if write_index:
        index_path = out_dir / "run_index.json"
        write_json(index_path, index_rows)

    print("Wrote:")
    print(" -", flat_json)
    print(" -", flat_csv)
    print(" -", summary_json)
    print(" -", summary_csv)
    if write_index:
        print(" -", out_dir / "run_index.json")

    return flat_rows, summary_rows






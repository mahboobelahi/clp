# clp/scripts/write_dummy_run.py
from __future__ import annotations

from pathlib import Path
import random

from clp.results.schema import run_skeleton
from clp.results.writer import write_run
from clp.results.index import append_run

ROOT = Path(__file__).resolve().parents[1]  # .../clp
RESULTS_DIR = ROOT / "results"
INDEX_PATH = RESULTS_DIR / "run_index.json"


def main() -> None:
    # Dummy settings
    dataset_family = "BR-Modified"
    instance_id = "BR7_003"
    variant = "dblf_balance"
    seed = 7

    algo_params = {
        "pop_size": 50,
        "generations": 10,
        "crossover_p": 0.8,
        "pm1": 0.6,
        "pm2": 0.3,
    }
    container = {"L": 12.0, "W": 2.35, "H": 2.39}

    run = run_skeleton(
        dataset_family=dataset_family,
        instance_id=instance_id,
        variant=variant,
        seed=seed,
        algo_params=algo_params,
        container=container,
    )

    # Dummy progress
    rng = random.Random(seed)
    for g in range(algo_params["generations"] + 1):
        z1_max = min(0.50 + 0.03 * g + rng.random() * 0.01, 0.90)
        z1_avg = max(0.40, z1_max - 0.05 - rng.random() * 0.01)
        z1_min = max(0.20, z1_avg - 0.08 - rng.random() * 0.01)

        z2_min = max(0, 30 - 2 * g - rng.randint(0, 2))
        z2_avg = z2_min + rng.randint(5, 15)
        z2_max = z2_avg + rng.randint(5, 15)

        z3_min = max(0.0, 0.08 - 0.004 * g - rng.random() * 0.002)
        z3_avg = z3_min + 0.01 + rng.random() * 0.01
        z3_max = z3_avg + 0.01 + rng.random() * 0.01

        run["progress"]["per_generation"].append({
            "gen": g,
            "pop_size": algo_params["pop_size"],
            "Z1": {"min": round(z1_min, 6), "avg": round(z1_avg, 6), "max": round(z1_max, 6)},
            "Z2": {"min": int(z2_min), "avg": float(z2_avg), "max": int(z2_max)},
            "Z3": {"min": round(z3_min, 6), "avg": round(z3_avg, 6), "max": round(z3_max, 6)},
        })

    # Dummy pareto solutions (2-4 solutions)
    for s in range(3):
        run["pareto_front"].append({
            "solution_id": f"p{s:04d}",
            "objectives": {
                "Z1": round(0.70 + 0.02 * s + rng.random() * 0.01, 6),
                "Z2": int(10 - s),
                "Z3": round(0.02 + rng.random() * 0.01, 6),
            },
            "cg": {
                "target": {"x": 0.5, "y": 0.5, "z": 0.5},
                "deviation_pct_halfspan": {
                    "x": round(1.5 + rng.random() * 0.5, 3),
                    "y": round(2.0 + rng.random() * 0.5, 3),
                    "z": round(0.8 + rng.random() * 0.3, 3),
                }
            },
            "layout": {
                "container": container,
                "boxes": [
                    {"id": "i12", "x": 0.0, "y": 0.0, "z": 0.0, "l": 1.2, "w": 0.8, "h": 0.6,
                     "customer": 2, "fragile": 0, "weight": 40.0}
                ]
            },
            "diagnostics": {
                "ulo_count": int(10 - s),
                "ulo_pairs_sample": [["i12", "i44"]],
                "feasible": True,
                "notes": "dummy"
            }
        })

    # Output path
    out_dir = RESULTS_DIR / dataset_family / instance_id
    out_path = out_dir / f"run_{run['run_id']}.json"
    write_run(run, out_path)

    # Index entry
    append_run(INDEX_PATH, {
        "run_id": run["run_id"],
        "path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "dataset_family": dataset_family,
        "instance_id": instance_id,
        "variant": variant,
        "seed": seed,
        "timestamp_utc": run["timestamp_utc"],
    })

    print(f"✅ Wrote dummy run: {out_path}")
    print(f"✅ Updated index: {INDEX_PATH}")


if __name__ == "__main__":
    main()

# clp/clp/results/schema.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_run_id(timestamp_utc: str, seed: int, variant: str) -> str:
    # timestamp_utc like "2025-12-23T17:04:55Z"
    ts = timestamp_utc.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    return f"{ts}_seed{seed}_{variant}"


def run_skeleton(
    *,
    dataset_family: str,
    instance_id: str,
    variant: str,
    seed: int,
    algo_params: Dict[str, Any],
    container: Dict[str, float],
) -> Dict[str, Any]:
    """Return an empty-but-valid run dict you will populate later."""
    ts = utc_now_iso()
    run_id = make_run_id(ts, seed, variant)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": ts,
        "dataset": {
            "family": dataset_family,     # "BR-Original" or "BR-Modified"
            "instance_id": instance_id,   # e.g. "BR7_003"
            "source": "Bischoff-Ratcliff",
            "modification_protocol": {
                "enabled": dataset_family != "BR-Original",
                "seed": seed,
                "n_customers": None,
                "customer_assignment": None,
                "fragile_ratio": None,
                "fragile_assignment": None,
                "weights": None,
                "weight_limit": {"enabled": None, "Qmax": None, "note": ""},
                "orientation_rules": None,
                "notes": "",
            },
        },
        "algorithm": {
            "name": "NSGA-II",
            "variant": variant,    # e.g. "dblf_balance"
            "decoder": variant,    # keep same for now
            "seed": seed,
            "params": algo_params,
        },
        "objectives": {
            "Z1": {"name": "volume_utilization", "sense": "max"},
            "Z2": {"name": "unloading_obstacles", "sense": "min"},
            "Z3": {"name": "cg_deviation_avg_axes", "sense": "min"},
        },
        "progress": {"per_generation": []},   # list[ {gen, pop_size, Z1{min,avg,max}, ...} ]
        "pareto_front": [],                   # list[solutions]
        "meta": {
            "container": container,           # {L,W,H}
            "notes": "",
        },
    }

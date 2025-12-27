from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import random

# -----------------------------
# Config defaults
# -----------------------------
BR_CLASSES = [f"BR{i}" for i in range(16)]
CASES = range(1, 101)

CONTAINER_L = 587  # Length (X)
CONTAINER_W = 233  # Depth  (Y)
CONTAINER_H = 220  # Height (Z)

P_MAX_KG = 28200.0
TARGET_UTIL = 0.8

BETA_SETTINGS = [(2, 2), (5, 2), (2, 5)]  # paper2


def cm3_to_m3(v_cm3: float) -> float:
    return v_cm3 / 1_000_000.0


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def compute_container_density_params(alpha: int, beta: int) -> Dict[str, float]:
    Vc_cm3 = CONTAINER_L * CONTAINER_W * CONTAINER_H
    Vc_m3 = cm3_to_m3(Vc_cm3)

    Tc = TARGET_UTIL * Vc_m3
    Dc = P_MAX_KG / Tc  # kg/m3

    Dcmin = 0.5 * Dc
    Ex = beta_mean(alpha, beta)
    # Solve for Dcmax so E[density] == Dc
    Dcmax = Dcmin + (Dc - Dcmin) / Ex

    return {
        "Vc_m3": Vc_m3,
        "Tc_m3": Tc,
        "Dc": Dc,
        "Dcmin": Dcmin,
        "Dcmax": Dcmax,
    }


def sample_density(rng: random.Random, alpha: int, beta: int, Dcmin: float, Dcmax: float) -> float:
    x = rng.betavariate(alpha, beta)  # in [0,1]
    return Dcmin + x * (Dcmax - Dcmin)


def item_volume_m3(item: Dict[str, Any]) -> float:
    L = item["Length"]
    W = item["Depth"]   # Depth == width (Y)
    H = item["Height"]
    return cm3_to_m3(L * W * H)


def assign_customers_and_priorities(items: List[Dict[str, Any]], n_customers: int) -> None:
    # Heavy -> later unload -> higher priority number (Nc highest)
    # Sort by volume descending
    order = sorted(range(len(items)), key=lambda i: item_volume_m3(items[i]), reverse=True)

    # Map types to customer ids 1..n_customers (or <= types)
    # If more types than customers, bucket by rank.
    for rank, idx in enumerate(order):
        # bucket index 0..n_customers-1
        bucket = min(rank * n_customers // max(1, len(order)), n_customers - 1)
        customer_id = bucket + 1
        priority = customer_id  # same scale: 1..Nc (Nc highest priority)

        items[idx]["CustomerId"] = int(customer_id)
        items[idx]["Priority"] = int(priority)


def generate_modified_case(
    src_path: Path,
    dst_path: Path,
    seed: int,
    alpha: int,
    beta: int,
    stackable_prob_type: float = 0.01,
) -> None:
    data = json.loads(src_path.read_text(encoding="utf-8"))
    if "Items" not in data or "Objects" not in data:
        raise ValueError(f"Invalid BR JSON structure: {src_path}")

    items: List[Dict[str, Any]] = data["Items"]
    n_types = len(items)
    n_customers = max(1, n_types)  # your rule: customer per type for BR-original classes

    # Container density params
    dens_params = compute_container_density_params(alpha, beta)

    # RNG: combine base seed + case identity so it's reproducible and stable
    # (You can adjust this scheme if you want.)
    rng = random.Random(seed + hash((str(src_path), alpha, beta)) % 10_000_000)

    # Assign customers/priorities by type heaviness
    assign_customers_and_priorities(items, n_customers)

    # Inject per-item weights based on sampled density
    for it in items:
        vol_m3 = item_volume_m3(it)
        density = sample_density(rng, alpha, beta, dens_params["Dcmin"], dens_params["Dcmax"])
        weight = density * vol_m3

        it["Alpha"] = int(alpha)
        it["Beta"] = int(beta)
        it["Density"] = float(round(density, 6))
        it["Weight"] = float(round(weight, 6))

        # Stackable: mostly true; type-level probabilistic
        it["Stackable"] = 0 if rng.random() < stackable_prob_type else 1

    # Root metadata (optional, but strongly recommended)
    data["Modification"] = {
        "family": "BR-Modified",
        "seed": seed,
        "weight_model": "beta_density",
        "alpha": alpha,
        "beta": beta,
        "Pmax_kg": P_MAX_KG,
        "target_util": TARGET_UTIL,
        "container_dims": {"L": CONTAINER_L, "W": CONTAINER_W, "H": CONTAINER_H},
        "Dc_kg_per_m3": round(dens_params["Dc"], 6),
        "Dcmin": round(dens_params["Dcmin"], 6),
        "Dcmax": round(dens_params["Dcmax"], 6),
        "customer_rule": "per-type by volume desc; heavy -> higher customer_id/priority",
        "stackable_rule": f"type-level Bernoulli(p_nonstackable={stackable_prob_type})",
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    # Adjust these to your repo
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "datasets" / "br_original"

    base_seed = 123

    for alpha, beta in BETA_SETTINGS:
        dst_root = repo_root / "datasets" / f"br_modified_beta_{alpha}_{beta}"

        for br in BR_CLASSES:
            for case_id in CASES:
                src = src_root / br / f"{case_id}.json"
                dst = dst_root / br / f"{case_id}.json"
                if not src.exists():
                    continue

                generate_modified_case(
                    src_path=src,
                    dst_path=dst,
                    seed=base_seed,
                    alpha=alpha,
                    beta=beta,
                    stackable_prob_type=0.01,
                )

        print(f"✅ Generated: {dst_root}")

    print("Done.")


if __name__ == "__main__":
    main()

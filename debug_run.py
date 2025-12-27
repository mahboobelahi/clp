
from pathlib import Path
from pprint import pprint
from clp.clp.data.br_original import load_br_instance
from clp.clp.models.geometry import ( AABB, Dims,is_supported)
from clp.clp.decoders.dblf import decode_dblf
from clp.clp.decoders.two_phase import decode_two_phase
from clp.clp.eval.cg import compute_cg_metrics
from clp.clp.models.items import br_row_to_item_type, expand_demands
from clp.clp.polices.rotation import RotationMode  
# from clp.clp.models.geometry import AABB
# from clp.clp.models.items import Dims
from clp.clp.viz.debug_viz import plot_container_debug
from clp.clp.results.schema import run_skeleton
from clp.clp.results.writer import write_run
from typing import Any, Dict, List
from time import perf_counter
from random import seed
from clp.scripts.aggregate_results import aggregate_results

seed
container = Dims(589, 233, 220)
min_ratio: float = 0.8
WD_DIR = [
    "BR-Original-baseline",
    "BR-Original",
    "BR-Original-two_phase",
    "BR-Modified-NSGA2",
]
BASE_RESULTS = Path(r"C:\Users\elahi\Desktop\clp\clp\results")

def volume_utilization(placed: List[AABB], container: Dims) -> float:
    vol_loaded = sum(b.dims.L * b.dims.W * b.dims.H for b in placed)
    vol_container = container.L * container.W * container.H
    return (vol_loaded / vol_container) if vol_container > 0 else 0.0


def aabb_to_dict(b: AABB) -> Dict[str, Any]:
    return {
        "x": b.origin.x, "y": b.origin.y, "z": b.origin.z,
        "l": b.dims.L, "w": b.dims.W, "h": b.dims.H,
    }


def run_one_instance(
    *,
    dataset_root: Path,
    br_class: str,
    case_id: int,
    container: Dims,
    rotation_mode: RotationMode,
    seed: int,
) -> Dict[str, Any]:
    inst = load_br_instance(dataset_root, br_class, case_id)

    item_types = [br_row_to_item_type(r, i) for i, r in enumerate(inst.items)]
    instances = expand_demands(item_types)

    t0 = perf_counter()
    
    #! base-line DBLF
    # placed, unplaced = decode_dblf(
    #     container=container,
    #     item_types=item_types,
    #     instances=instances,
    #     rotation_mode=rotation_mode,
    #      min_ratio = min_ratio
    # )

  #! modified base-line Two-phase DBLF+Balance aware
    placed, unplaced = decode_two_phase(
                container=container,
                item_types=item_types,
                instances=instances,
                rotation_mode=RotationMode.SIX_WAY,
                box_order_policy="volume_then_maxface",  # ✅ ranking
                split_ratio=0.7,
                support_required=True,
                support_min_ratio=.80,
                is_supported_fn=is_supported,
            )



    elapsed = perf_counter() - t0

    cg = compute_cg_metrics(placed, container)
    z1 = volume_utilization(placed, container)
    z3 = cg.z3

    # Build schema run
    instance_id = f"{br_class}_{case_id:03d}"
    variant = "six_way" if rotation_mode == RotationMode.SIX_WAY else "C1_respect"

    run = run_skeleton(
        dataset_family=WD_DIR[1],
        instance_id=instance_id,
        variant=variant,
        seed=seed,
        algo_params={
            "decoder": "DBLF",
            "rotation_mode": variant,
            "support_required": True,
            "support_min_ratio": min_ratio,
        },
        container={"L": float(container.L), "W": float(container.W), "H": float(container.H)},
    )

    run["algorithm"]["name"] = "DBLF"
    run["algorithm"]["decoder"] = "DBLF"

    run["pareto_front"] = [{
        "solution_id": "baseline",
        "objectives": {
            "Z1": round(z1, 6),
            "Z2": 0,  # placeholder (ULO not computed in baseline DBLF)
            "Z3": round(z3, 6),
        },
        "cg": {
            "loaded": {"x": round(cg.cg_x, 4), "y": round(cg.cg_y, 4), "z": round(cg.cg_z, 4)},
            "container": {"x": round(cg.container_cg_x, 4), "y": round(cg.container_cg_y, 4), "z": round(cg.container_cg_z, 4)},
            "dev": {"x": round(cg.dev_x, 4), "y": round(cg.dev_y, 4), "z": round(cg.dev_z, 4)},
            "rd_pct": {"x": round(cg.rd_x_pct, 4), "y": round(cg.rd_y_pct, 4), "z": round(cg.rd_z_pct, 4)},
            "z3": round(cg.z3, 4),
            "total_mass": round(cg.total_mass, 4),
        },
        "layout": {
            "container": {"L": float(container.L), "W": float(container.W), "H": float(container.H)},
            "boxes": [aabb_to_dict(b) for b in placed],
        },
        "diagnostics": {
            "elapsed_sec": round(elapsed, 6),
            "placed_count": int(len(placed)),
            "unplaced_count": int(len(unplaced)),
            "notes": "",
        }
    }]

    run["meta"]["timing"] = {"elapsed_sec": round(elapsed, 6)}
    run["meta"]["counts"] = {"placed": int(len(placed)), "unplaced": int(len(unplaced))}
    run["meta"]["cg_summary"] = {
        "cg": [round(cg.cg_x, 4), round(cg.cg_y, 4), round(cg.cg_z, 4)],
        "dev": [round(cg.dev_x, 4), round(cg.dev_y, 4), round(cg.dev_z, 4)],
        "rd_pct": [round(cg.rd_x_pct, 4), round(cg.rd_y_pct, 4), round(cg.rd_z_pct, 4)],
        "z3": round(cg.z3, 4),
        "z1_util": round(z1, 6),
    }
    run["meta"]["ulo"] = {"implemented": False, "count": None, "note": "Z2 placeholder (not computed in baseline DBLF)"}


    return run


def main() -> None:
    # ----- Config -----
    container = Dims(589, 233, 220)  # BR original container
    dataset_root = Path("clp/datasets/br_original")  # your relative dataset root

    # Absolute results root (your requested Windows path)
    # results_root = Path(r"C:\Users\elahi\Desktop\clp\clp\results\BR-Original-two_phase-decoder")
    results_root = BASE_RESULTS / WD_DIR[2]

    seed = 0  # keep deterministic unless you shuffle instances

    br_classes = [f"BR{i}" for i in range(0, 16)]
    modes = [
        ("six_way", RotationMode.SIX_WAY),
        ("C1_respect", RotationMode.RESPECT_C1),
    ]

    # ----- Run batch -----
    overall_t0 = perf_counter()

    for br_class in br_classes:
        class_t0 = perf_counter()

        for mode_name, mode in modes:
            for case_id in range(1, 101):
                run = run_one_instance(
                    dataset_root=dataset_root,
                    br_class=br_class,
                    case_id=case_id,
                    container=container,
                    rotation_mode=mode,
                    seed=seed,
                )

                # Write exactly: BR0/C1_respect/1.json
                out_dir = results_root / br_class / mode_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{case_id}.json"
                write_run(run, out_path)

                print(f"✅ {br_class} {mode_name} case {case_id:03d} -> {out_path}")

         
        class_elapsed = perf_counter() - class_t0

        # write per-class timing summary
        summary_path = results_root / br_class / "class_summary.json"
        summary = {
            "br_class": br_class,
            "container": {"L": container.L, "W": container.W, "H": container.H},
            "elapsed_sec_total": round(class_elapsed, 6),
            "timestamp_note": "wall time for both modes and 100 cases",
        }
        write_run(summary, summary_path)
        print(f"🕒 {br_class} total time: {class_elapsed:.2f}s (summary: {summary_path})")

    overall_elapsed = perf_counter() - overall_t0
    overall_summary_path = results_root / "_overall_summary.json"
    overall_summary = {
        "dataset_family": "BR-Original",
        "elapsed_sec_total": round(overall_elapsed, 6),
        "br_classes": br_classes,
        "modes": [m[0] for m in modes],
        "cases_per_class": 100,
    }
    write_run(overall_summary, overall_summary_path)
    print(f"🏁 All done. Total time: {overall_elapsed:.2f}s (summary: {overall_summary_path})")


if __name__ == "__main__":
    # main()
    aggregate_results()
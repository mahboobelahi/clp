# debug_run.py
from pathlib import Path
from typing import Any, Dict, List
from time import perf_counter
import random

from clp.clp.data.br_original import load_br_instance
from clp.clp.models.geometry import AABB, Dims, is_supported
from clp.clp.decoders.dblf import decode_dblf
from clp.clp.decoders.two_phase import decode_two_phase
from clp.clp.eval.cg import compute_cg_metrics
from clp.clp.models.items import br_row_to_item_type, expand_demands
from clp.clp.polices.rotation import RotationMode

from clp.clp.results.schema import run_skeleton
from clp.clp.results.writer import write_run

from clp.clp.ga.population import init_population_groups, build_groups, expand_chromosome
from clp.clp.ga.nsga2 import run_nsga2_type1
from clp.configurations import(is_ULO, RESULTS_DIR_NAME, DECODER_KIND, BOX_ORDER_POLICY,
                                SPLIT_RATIO, SUPPORT_REQUIRED, SUPPORT_MIN_RATIO,
                                GA_TEST, GA_EVOLVE, POP_SIZE, GENERATIONS,
                                ENABLE_TEST_CLASS, ENABLE_TEST_CASE,SOFT_ROTATION,
                                ROTATION_MODE_SETTING)

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

BASE_RESULTS = Path(r"C:\Users\elahi\Desktop\clp\clp\results")


# =============================================================================
# Helpers
# =============================================================================
def volume_utilization(placed: List[AABB], container: Dims) -> float:
    vol_loaded = sum(b.dims.L * b.dims.W * b.dims.H for b in placed)
    vol_container = container.L * container.W * container.H
    return round((vol_loaded / vol_container), 4) if vol_container > 0 else 0.0


def aabb_to_dict(b: AABB) -> Dict[str, Any]:
    return {
        "x": b.origin.x, "y": b.origin.y, "z": b.origin.z,
        "l": b.dims.L, "w": b.dims.W, "h": b.dims.H,
        "type_id": getattr(b, "type_id", None),
    }


def pick_best_solution(solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
    # lexicographic: Z1 desc, Z3 asc
    best = None
    for s in solutions:
        z1 = float(s["objectives"]["Z1"])
        z3 = float(s["objectives"]["Z3"])
        if best is None:
            best = s
        else:
            bz1 = float(best["objectives"]["Z1"])
            bz3 = float(best["objectives"]["Z3"])
            if (z1 > bz1) or (z1 == bz1 and z3 < bz3):
                best = s
    return best if best is not None else solutions[0]


# =============================================================================
# Core: run one instance
# =============================================================================
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

    variant = "six_way" if rotation_mode == RotationMode.SIX_WAY else "C1_respect"

    t0 = perf_counter()

    solutions: List[Dict[str, Any]] = []
    progress: List[Dict[str, Any]] = []

    # ---------- BASELINE ----------
    if DECODER_KIND == "baseline":
        placed, unplaced = decode_dblf(
            container=container,
            item_types=item_types,
            instances=instances,
            rotation_mode=rotation_mode,
            support_required=SUPPORT_REQUIRED,
            support_min_ratio=float(SUPPORT_MIN_RATIO),
            is_supported_fn=is_supported,
        )
        cg = compute_cg_metrics(placed, container)
        z1 = volume_utilization(placed, container)
        z3 = float(cg.z3)

        elapsed = perf_counter() - t0

        solutions = [{
            "solution_id": "single_run",
            "objectives": {"Z1": round(float(z1), 6), "Z2": 0, "Z3": round(float(z3), 6)},
            "cg": {
                "loaded": {"x": round(cg.cg_x, 4), "y": round(cg.cg_y, 4), "z": round(cg.cg_z, 4)},
                "container": {"x": round(cg.container_cg_x, 4), "y": round(cg.container_cg_y, 4), "z": round(cg.container_cg_z, 4)},
                "dev": {"x": round(cg.dev_x, 4), "y": round(cg.dev_y, 4), "z": round(cg.dev_z, 4)},
                "rd_pct": {"x": round(cg.rd_x_pct, 4), "y": round(cg.rd_y_pct, 4), "z": round(cg.rd_z_pct, 4)},
                "z3": round(cg.z3, 4),
                "total_mass": round(cg.total_mass, 4),
            },
            "layout": {"container": {"L": float(container.L), "W": float(container.W), "H": float(container.H)},
                       "boxes": [aabb_to_dict(b) for b in placed]},
            "diagnostics": {"elapsed_sec": round(elapsed, 6),
                            "placed_count": int(len(placed)),
                            "unplaced_count": int(len(unplaced)),
                            "notes": ""},
        }]

        algo_name = "DBLF"
        algo_variant = variant
        algo_params = {"decoder": "dblf", "rotation_mode": variant}

        best_sol = solutions[0]

    # ---------- TWO PHASE ----------
    elif DECODER_KIND == "two_phase":
        # -------- Gen0-only test (no evolution) --------
        if GA_TEST and not GA_EVOLVE:
            pop = init_population_groups(
                instances=instances,
                item_types=item_types,
                rotation_mode=rotation_mode,
                pop_size=POP_SIZE,
                rng_seed=seed,
            )
            groups = build_groups(instances)
            rng = random.Random(seed)

            for idx, chrom in enumerate(pop):
                order = expand_chromosome(chrom, groups, rng=rng, shuffle_within_group=True)
                inst_ordered = [instances[i] for i in order]

                t_eval = perf_counter()
                placed_i, unplaced_i = decode_two_phase(
                    container=container,
                    item_types=item_types,
                    instances=inst_ordered,
                    rotation_mode=rotation_mode,
                    box_order_policy=None,  # GA provides order
                    box_order_seed=None,
                    split_ratio=SPLIT_RATIO,
                    support_required=SUPPORT_REQUIRED,
                    support_min_ratio=float(SUPPORT_MIN_RATIO),
                    is_supported_fn=is_supported,
                    rot_by_group=chrom.rot_map,
                    soft_rotation=SOFT_ROTATION,
                )
                elapsed_i = perf_counter() - t_eval

                cg_i = compute_cg_metrics(placed_i, container)
                z1_i = volume_utilization(placed_i, container)
                z3_i = float(cg_i.z3)

                solutions.append({
                    "solution_id": f"gen0_{idx:03d}",
                    "objectives": {"Z1": round(float(z1_i), 6), "Z2": 0, "Z3": round(float(z3_i), 6)},
                    "cg": {
                        "loaded": {"x": round(cg_i.cg_x, 4), "y": round(cg_i.cg_y, 4), "z": round(cg_i.cg_z, 4)},
                        "container": {"x": round(cg_i.container_cg_x, 4), "y": round(cg_i.container_cg_y, 4), "z": round(cg_i.container_cg_z, 4)},
                        "dev": {"x": round(cg_i.dev_x, 4), "y": round(cg_i.dev_y, 4), "z": round(cg_i.dev_z, 4)},
                        "rd_pct": {"x": round(cg_i.rd_x_pct, 4), "y": round(cg_i.rd_y_pct, 4), "z": round(cg_i.rd_z_pct, 4)},
                        "z3": round(cg_i.z3, 4),
                        "total_mass": round(cg_i.total_mass, 4),
                    },
                    "layout": {"container": {"L": float(container.L), "W": float(container.W), "H": float(container.H)},
                               "boxes": [aabb_to_dict(b) for b in placed_i]},
                    "diagnostics": {"elapsed_sec": round(elapsed_i, 6),
                                    "placed_count": int(len(placed_i)),
                                    "unplaced_count": int(len(unplaced_i)),
                                    "notes": ""},
                })

            elapsed = perf_counter() - t0
            best_sol = pick_best_solution(solutions)

            algo_name = "GA_GEN0 + TwoPhase"
            algo_variant = f"{variant}_gen0"
            algo_params = {
                "decoder": "two_phase_blocks",
                "rotation_mode": variant,
                "ga_test": True,
                "ga_evolve": False,
                "generations": 1,
                "pop_size": int(POP_SIZE),
                "soft_rotation": bool(SOFT_ROTATION),
            }

        # -------- FULL NSGA-II EVOLUTION --------
        elif GA_EVOLVE:
            res = run_nsga2_type1(
                instances=instances,
                item_types=item_types,
                container=container,
                rotation_mode=rotation_mode,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                seed=seed,
                split_ratio=SPLIT_RATIO,
                support_required=SUPPORT_REQUIRED,
                support_min_ratio=float(SUPPORT_MIN_RATIO),
                is_supported_fn=is_supported,   # IMPORTANT
                soft_rotation=SOFT_ROTATION,
            )
            front0 = res["front0"]          # List[EvalRec]
            progress = res["progress"]      # per-gen summary dicts

            # Convert front0 to your schema solution dicts
            for idx, r in enumerate(front0):
                cg_i = r.cg
                solutions.append({
                    "solution_id": f"front0_{idx:03d}",
                    "objectives": {"Z1": round(float(r.z1), 6), "Z2": round(float(r.z2), 4), "Z3": round(float(r.z3), 6)},
                    "cg": {
                        "loaded": {"x": round(cg_i.cg_x, 4), "y": round(cg_i.cg_y, 4), "z": round(cg_i.cg_z, 4)},
                        "container": {"x": round(cg_i.container_cg_x, 4), "y": round(cg_i.container_cg_y, 4), "z": round(cg_i.container_cg_z, 4)},
                        "dev": {"x": round(cg_i.dev_x, 4), "y": round(cg_i.dev_y, 4), "z": round(cg_i.dev_z, 4)},
                        "rd_pct": {"x": round(cg_i.rd_x_pct, 4), "y": round(cg_i.rd_y_pct, 4), "z": round(cg_i.rd_z_pct, 4)},
                        "z3": round(cg_i.z3, 4),
                        "total_mass": round(cg_i.total_mass, 4),
                    },
                    "layout": {"container": {"L": float(container.L), "W": float(container.W), "H": float(container.H)},
                               "boxes": [aabb_to_dict(b) for b in r.placed]},
                    "diagnostics": {"elapsed_sec": round(float(r.elapsed_sec), 6),
                                    "placed_count": int(len(r.placed)),
                                    "unplaced_count": int(len(r.unplaced)),
                                    "notes": ""},
                })

            elapsed = perf_counter() - t0
            best_sol = pick_best_solution(solutions)

            algo_name = "NSGA-II Type1 (Z1 max, Z3 min)"
            algo_variant = f"{variant}_nsga2"
            algo_params = {
                "decoder": "two_phase_blocks",
                "rotation_mode": variant,
                "ga_test": False,
                "ga_evolve": True,
                "generations": int(GENERATIONS),
                "pop_size": int(POP_SIZE),
                "soft_rotation": bool(SOFT_ROTATION),
                "split_ratio": float(SPLIT_RATIO),
                "support_required": bool(SUPPORT_REQUIRED),
                "support_min_ratio": float(SUPPORT_MIN_RATIO),
            }

        # -------- Plain two-phase heuristic (no GA) --------
        else:
            placed, unplaced = decode_two_phase(
                container=container,
                item_types=item_types,
                instances=instances,
                rotation_mode=rotation_mode,
                box_order_policy=BOX_ORDER_POLICY,
                box_order_seed=seed,
                split_ratio=SPLIT_RATIO,
                support_required=SUPPORT_REQUIRED,
                support_min_ratio=float(SUPPORT_MIN_RATIO),
                is_supported_fn=is_supported,
            )
            cg = compute_cg_metrics(placed, container)
            z1 = volume_utilization(placed, container)
            z3 = float(cg.z3)

            elapsed = perf_counter() - t0

            solutions = [{
                "solution_id": "single_run",
                "objectives": {"Z1": round(float(z1), 6), "Z2": 0, "Z3": round(float(z3), 6)},
                "cg": {
                    "loaded": {"x": round(cg.cg_x, 4), "y": round(cg.cg_y, 4), "z": round(cg.cg_z, 4)},
                    "container": {"x": round(cg.container_cg_x, 4), "y": round(cg.container_cg_y, 4), "z": round(cg.container_cg_z, 4)},
                    "dev": {"x": round(cg.dev_x, 4), "y": round(cg.dev_y, 4), "z": round(cg.dev_z, 4)},
                    "rd_pct": {"x": round(cg.rd_x_pct, 4), "y": round(cg.rd_y_pct, 4), "z": round(cg.rd_z_pct, 4)},
                    "z3": round(cg.z3, 4),
                    "total_mass": round(cg.total_mass, 4),
                },
                "layout": {"container": {"L": float(container.L), "W": float(container.W), "H": float(container.H)},
                           "boxes": [aabb_to_dict(b) for b in placed]},
                "diagnostics": {"elapsed_sec": round(elapsed, 6),
                                "placed_count": int(len(placed)),
                                "unplaced_count": int(len(unplaced)),
                                "notes": ""},
            }]
            best_sol = solutions[0]

            algo_name = "Two-Phase-DBLF"
            algo_variant = f"{variant}_two_phase"
            algo_params = {"decoder": "two_phase", "rotation_mode": variant}

    else:
        raise ValueError(f"Unknown DECODER_KIND={DECODER_KIND}")

    # =============================================================================
    # Build run schema (always)
    # =============================================================================
    instance_id = f"{br_class}_{case_id:03d}"
    run = run_skeleton(
        dataset_family=RESULTS_DIR_NAME,
        instance_id=instance_id,
        variant=variant,
        seed=seed,
        algo_params=algo_params,
        container={"L": float(container.L), "W": float(container.W), "H": float(container.H)},
    )

    run["algorithm"]["name"] = algo_name
    run["algorithm"]["variant"] = algo_variant
    run["algorithm"]["decoder"] = algo_params.get("decoder", algo_name)

    run["pareto_front"] = solutions

    # Progress logging for NSGA-II
    run["progress"]["per_generation"] = progress

    # Meta summary uses BEST solution (even if pareto_front has many)
    run["meta"]["timing"] = {"elapsed_sec": round(float(elapsed), 6)}
    run["meta"]["counts"] = {
        "placed": int(best_sol["diagnostics"]["placed_count"]),
        "unplaced": int(best_sol["diagnostics"]["unplaced_count"]),
    }
    run["meta"]["cg_summary"] = {
    "cg": [
        best_sol["cg"]["loaded"]["x"],
        best_sol["cg"]["loaded"]["y"],
        best_sol["cg"]["loaded"]["z"],
    ],
    "dev": [
        best_sol["cg"]["dev"]["x"],
        best_sol["cg"]["dev"]["y"],
        best_sol["cg"]["dev"]["z"],
    ],
    "rd_pct": [
        best_sol["cg"]["rd_pct"]["x"],
        best_sol["cg"]["rd_pct"]["y"],
        best_sol["cg"]["rd_pct"]["z"],
    ],
    "z3": best_sol["cg"]["z3"],                 # CG penalty (your Z3)
  }

    if is_ULO:
        run["meta"]["ulo"] = {
            "implemented": True,
            "count": best_sol["objectives"].get("Z2"),
            "note": None,
        }
    else:
        run["meta"]["ulo"] = {
            "implemented": False,
            "count": 0.0,
            "note": "Z2 disabled for this run",
        }

    return run


# =============================================================================
# Batch driver
# =============================================================================
def main() -> None:
    container = Dims(589, 233, 220)

    BR_DATA = ["br_original", "br_modified_beta_2_2", "br_modified_beta_2_5",
                "br_modified_beta_5_2","br_modified_cust_beta_2_2", "br_modified_cust_beta_2_5",
                "br_modified_cust_beta_5_2"]
    dataset_root = Path(f"clp/datasets/{BR_DATA[1]}")  # Type1 uses original BR

    results_root = BASE_RESULTS / RESULTS_DIR_NAME
    results_root.mkdir(parents=True, exist_ok=True)

    seed = 0
    random.seed(seed)

    br_classes = [f"BR{i}" for i in range(16)]
    case_ids = list(range(1, 101))

    if ROTATION_MODE_SETTING == "six":
        modes = [
            ("six_way", RotationMode.SIX_WAY),
        ]

    elif ROTATION_MODE_SETTING == "c1":
        modes = [
            ("C1_respect", RotationMode.RESPECT_C1),
        ]

    elif ROTATION_MODE_SETTING == "both":
        modes = [
            ("six_way", RotationMode.SIX_WAY),
            ("C1_respect", RotationMode.RESPECT_C1),
        ]

    else:
        raise ValueError(f"Unknown ROTATION_MODE_SETTING={ROTATION_MODE_SETTING!r}")

    overall_t0 = perf_counter()

    for br_class in br_classes:
        if br_class in ["BR0"]:continue
        class_t0 = perf_counter()

        for mode_name, mode in modes:
            for case_id in case_ids:
                # if case_id in [1,2]:continue#break
                # if case_id <=65:continue
                run = run_one_instance(
                    dataset_root=dataset_root,
                    br_class=br_class,
                    case_id=case_id,
                    container=container,
                    rotation_mode=mode,
                    seed=seed,
                )

                out_dir = results_root / br_class / mode_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{case_id}.json"
                write_run(run, out_path)
                print(f"✅ {br_class} {mode_name} case {case_id:03d} -> {out_path}")
                
                #! break case_id loop for quick test
                if ENABLE_TEST_CASE: break
        class_elapsed = perf_counter() - class_t0
        summary_path = results_root / br_class / "class_summary.json"
        write_run({
            "br_class": br_class,
            "container": {"L": container.L, "W": container.W, "H": container.H},
            "elapsed_sec_total": round(class_elapsed, 6),
        }, summary_path)

      #! break br_class loop for quick test
        if ENABLE_TEST_CLASS: break

    overall_elapsed = perf_counter() - overall_t0
    overall_summary_path = results_root / "_overall_summary.json"
    write_run({
        "dataset_family": RESULTS_DIR_NAME,
        "elapsed_sec_total": round(overall_elapsed, 6),
        "br_classes": br_classes,
        "modes": [m[0] for m in modes],
        "cases_executed": case_ids,
    }, overall_summary_path)

    print(f"🏁 Done. Total time: {overall_elapsed:.2f}s")


if __name__ == "__main__":
  # main()
  from clp.scripts.aggregate_results import aggregate_results
  aggregate_results()
# clp/clp/ga/evaluate.py
from typing import Dict, Any, List

from clp.clp.models.geometry import AABB, Dims, is_supported
from clp.clp.decoders.two_phase import decode_two_phase
from clp.clp.eval.cg import compute_cg_metrics


def volume_utilization(container: Dims, placed: List[AABB]) -> float:
    vol_loaded = sum(b.dims.L * b.dims.W * b.dims.H for b in placed)
    vol_container = container.L * container.W * container.H
    return round((vol_loaded / vol_container), 4) if vol_container > 0 else 0.0


def rot_by_type_from_individual(individual, instances) -> Dict[int, int]:
    """
    Moon-style: same type has same rotation in a chromosome.
    """
    rot_by_type: Dict[int, int] = {}
    for inst_idx, ridx in zip(individual.order, individual.rot_idx):
        tid = instances[inst_idx].type_id
        if tid not in rot_by_type:
            rot_by_type[tid] = int(ridx)
    return rot_by_type


def evaluate_individual(
    *,
    individual,
    instances,
    item_types,
    container: Dims,
    rotation_mode,
    split_ratio: float,
    support_required: bool,
    support_min_ratio: float,
    soft_rotation: bool = True,
    gap_tol: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate ONE GA individual using two-phase + block decoder.

    Returns dict with Z1, Z3 and placements.
    """

    # 1) chromosome → decoder inputs
    inst_ordered = [instances[i] for i in individual.order]
    rot_by_type = rot_by_type_from_individual(individual, instances)

    # 2) decode (decoder handles blocks + leftovers)
    placed, unplaced = decode_two_phase(
        container=container,
        item_types=item_types,
        instances=inst_ordered,
        rotation_mode=rotation_mode,
        box_order_policy=None,     # GA provides order
        split_ratio=split_ratio,
        support_required=support_required,
        support_min_ratio=support_min_ratio,
        is_supported_fn=is_supported if support_required else None,
        gap_tol=gap_tol,
        rot_by_type=rot_by_type,
        soft_rotation=soft_rotation,
    )

    # 3) objectives
    z1 = volume_utilization(container, placed)
    cg = compute_cg_metrics(placed, container)
    z3 = cg.z3

    return {
        "Z1": float(z1),
        "Z3": float(z3),
        "cg": cg,
        "placed": placed,
        "unplaced": unplaced,
    }

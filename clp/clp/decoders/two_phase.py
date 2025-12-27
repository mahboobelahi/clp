
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from clp.clp.models.geometry import Point, AABB, Dims, inside_container, overlaps
from clp.clp.polices.rotation import RotationMode, allowed_rotations
from clp.clp.decoders.extreme_points import update_eps, dblf_key  
from clp.clp.eval.cg import compute_cg_metrics 
from clp.clp.polices.box_order import apply_box_order


def rev_dblf_key(p: Point) -> Tuple[int, int, int]:
    # reverse DBLF: x desc, z desc, y desc
    return (-p.x, -p.z, -p.y)


def gap_score(
    cand: AABB,
    placed: List[AABB],
    container: Dims,
    eps: List[Point],
    *,
    tol: int = 0,
) -> float:
    """
    Higher is better.

    Very cheap 'alignment' proxy:
      +1 if cand.min face aligns with some existing max face (x or y or z)
      +1 if cand.max face aligns with some existing min face (x or y or z)
      + small reward for using smaller coordinates (encourage compactness)
    """
    x0, y0, z0 = cand.origin.x, cand.origin.y, cand.origin.z
    x1 = x0 + cand.dims.L
    y1 = y0 + cand.dims.W
    z1 = z0 + cand.dims.H

    # Collect planes from placed boxes
    x_mins = set()
    x_maxs = set()
    y_mins = set()
    y_maxs = set()
    z_mins = set()
    z_maxs = set()

    for b in placed:
        bx0, by0, bz0 = b.origin.x, b.origin.y, b.origin.z
        bx1 = bx0 + b.dims.L
        by1 = by0 + b.dims.W
        bz1 = bz0 + b.dims.H
        x_mins.add(bx0); x_maxs.add(bx1)
        y_mins.add(by0); y_maxs.add(by1)
        z_mins.add(bz0); z_maxs.add(bz1)

    # Container walls count as planes too (helps compact packing)
    x_mins.add(0); x_maxs.add(container.L)
    y_mins.add(0); y_maxs.add(container.W)
    z_mins.add(0); z_maxs.add(container.H)

    score = 0.0

    def aligned(val: int, plane_set: set[int]) -> bool:
        if tol == 0:
            return val in plane_set
        return any(abs(val - p) <= tol for p in plane_set)

    # Reward touching/aligning planes (reduces fragmentation)
    if aligned(x0, x_maxs): score += 1.0
    if aligned(y0, y_maxs): score += 1.0
    if aligned(z0, z_maxs): score += 1.0

    if aligned(x1, x_mins): score += 0.5
    if aligned(y1, y_mins): score += 0.5
    if aligned(z1, z_mins): score += 0.5

    # Small compactness reward (prefer smaller coordinates)
    score += 0.0001 * (-(x0 + y0 + z0))

    return score


def balance_score(
    cand: AABB,
    placed: List[AABB],
    container: Dims,
) -> float:
    """
    Higher is better. We maximize improvement (i.e., negative Z3).
    Computes CG metrics on (placed + cand) and returns -z3.
    """
    tmp = placed + [cand]
    cg = compute_cg_metrics(container=container, placed=tmp)  # must return .z3
    return -float(cg.z3)


def _first_phase_split(instances: List, split_ratio: float) -> Tuple[List, List]:
    n = len(instances)
    k = int(round(n * split_ratio))
    k = max(0, min(n, k))
    return instances[:k], instances[k:]


def decode_two_phase(
    *,
    container: Dims,
    item_types: List,
    instances: List,
    rotation_mode: RotationMode,
    split_ratio: float = 0.7,
    support_required: bool = True,
    support_min_ratio: float = 1.0,
    is_supported_fn=None,
    gap_tol: int = 0,
    # NEW:
    box_order_policy: Optional[str] = None,   # e.g. "volume_then_maxface"
    box_order_seed: Optional[int] = None,     # used by "random_tiebreak"
) -> Tuple[List[AABB], List]:
    """
    Two-phase greedy decoder with optional box-ranking:

    If box_order_policy is provided, instances are sorted once BEFORE decoding.
    If box_order_policy is None, instances order is used as-is (GA will provide it).
    """
    types_by_id: Dict[int, any] = {t.type_id: t for t in item_types}
    rots_by_type: Dict[int, List[Dims]] = {
        t.type_id: allowed_rotations(t, rotation_mode)
        for t in item_types
    }

    # ---- NEW: apply ranking once (if requested) ----
    if box_order_policy is not None:
        instances = apply_box_order(
            instances=instances,
            types_by_id=types_by_id,
            rotation_mode=rotation_mode,
            policy=box_order_policy,
            seed=box_order_seed,
        )

    placed: List[AABB] = []
    unplaced: List = []
    eps: List[Point] = [Point(0, 0, 0)]

    phaseA, phaseB = _first_phase_split(instances, split_ratio)

    def place_one(inst, ep_sort_key, scorer_fn) -> bool:
        nonlocal eps, placed

        rots = rots_by_type.get(inst.type_id, [])
        if not rots:
            return False

        eps.sort(key=ep_sort_key)

        best_cand: Optional[AABB] = None
        best_ep: Optional[Point] = None
        best_score: float = float("-inf")

        for ep in eps:
            for dims in rots:
                cand = AABB(ep, dims)

                if not inside_container(cand, container):
                    continue
                if any(overlaps(cand, b) for b in placed):
                    continue

                if support_required and cand.origin.z > 0:
                    if is_supported_fn is None:
                        raise RuntimeError("support_required=True but is_supported_fn not provided")
                    if not is_supported_fn(cand, placed, support_min_ratio):
                        continue

                s = scorer_fn(cand)
                if s > best_score:
                    best_score = s
                    best_cand = cand
                    best_ep = ep

        if best_cand is None or best_ep is None:
            return False

        placed.append(best_cand)
        eps.remove(best_ep)
        eps = update_eps(eps, placed, best_cand, container)
        return True

    # Phase A: gap/alignment
    for inst in phaseA:
        ok = place_one(
            inst,
            ep_sort_key=dblf_key,
            scorer_fn=lambda cand: gap_score(cand, placed, container, eps, tol=gap_tol),
        )
        if not ok:
            unplaced.append(inst)

    # Phase B: balance
    for inst in phaseB:
        ok = place_one(
            inst,
            ep_sort_key=rev_dblf_key,
            scorer_fn=lambda cand: balance_score(cand, placed, container),
        )
        if not ok:
            unplaced.append(inst)

    return placed, unplaced


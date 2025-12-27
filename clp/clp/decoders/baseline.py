
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from clp.clp.models.geometry import Point, AABB, Dims, inside_container, overlaps
from clp.clp.polices.rotation import allowed_rotations, RotationMode 
from clp.clp.decoders.extreme_points import update_eps , dblf_key  # if dblf_key is in your ep module


def decode_baseline_dblf(
    *,
    container: Dims,
    item_types: List,
    instances: List,
    rotation_mode: RotationMode,
    support_required: bool = True,
    support_min_ratio: float = 1.0,
    is_supported_fn=None,  # pass your is_supported(cand, placed, min_ratio) if you have it
) -> Tuple[List[AABB], List]:
    """
    Baseline DBLF:
      - EPs in DBLF order
      - rotations in allowed order
      - first feasible placement
    """
    types_by_id: Dict[int, any] = {t.type_id: t for t in item_types}

    rots_by_type: Dict[int, List[Dims]] = {
        t.type_id: allowed_rotations(t, rotation_mode)
        for t in item_types
    }

    placed: List[AABB] = []
    unplaced: List = []
    eps: List[Point] = [Point(0, 0, 0)]

    for inst in instances:
        placed_this = False
        rots = rots_by_type.get(inst.type_id, [])
        if not rots:
            unplaced.append(inst)
            continue

        # EPs already maintained sorted by update_eps; sort again defensively
        # eps.sort(key=dblf_key)

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

                # place
                placed.append(cand)
                eps.remove(ep)
                eps = update_eps(eps, placed, cand, container)
                placed_this = True
                break

            if placed_this:
                break

        if not placed_this:
            unplaced.append(inst)

    return placed, unplaced

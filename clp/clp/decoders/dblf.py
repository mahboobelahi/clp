

from typing import List, Tuple, Optional
from clp.clp.models.geometry import (Point, AABB, Dims, overlaps, inside_container,
                                     is_supported)
from clp.clp.models.items import ItemInstance, ItemType
from clp.clp.decoders.extreme_points import update_eps  
from clp.clp.polices.rotation import allowed_rotations, RotationMode 



def is_collision_free(candidate: AABB, placed: List[AABB]) -> bool:
    return not any(overlaps(candidate, b) for b in placed)


def decode_dblf(
    container: Dims,
    item_types: List[ItemType],
    instances: List[ItemInstance],
    rotation_mode: RotationMode,
    min_ratio: float = 0.8,
) -> Tuple[List[AABB], List[ItemInstance]]:
    """
    Baseline DBLF decoder:
    - EPs in DBLF order
    - try rotations
    - place first feasible
    """
    placed: List[AABB] = []
    unplaced: List[ItemInstance] = []

    eps: List[Point] = [Point(0, 0, 0)]

    # Precompute rotations per type_id
    rots_by_type = {
        it.type_id: allowed_rotations(it, rotation_mode)
        for it in item_types
    }

    for inst in instances:
        placed_this = False

        # Try each EP
        for ep_idx, ep in enumerate(list(eps)):  # snapshot because eps mutates
            # Try each allowed rotation
            for dims in rots_by_type[inst.type_id]:
                cand = AABB(ep, dims)

                if not inside_container(cand, container):
                    continue
                if not is_collision_free(cand, placed):
                    continue
                
                if cand.origin.z > 0 and not is_supported(cand, placed, min_ratio=min_ratio):
                    continue


                # Place it
                placed.append(cand)

                # remove used EP and update EPs
                eps.pop(ep_idx)
                eps = update_eps(eps, placed, cand, container)

                placed_this = True
                break

            if placed_this:
                break

        if not placed_this:
            unplaced.append(inst)

    return placed, unplaced

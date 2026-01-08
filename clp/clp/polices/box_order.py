from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import random

from clp.clp.models.items import ItemInstance, ItemType  # adjust if needed
from clp.clp.models.geometry import Dims
from clp.clp.polices.rotation import RotationMode, allowed_rotations


# ---------- Helpers (rotation-aware features) ----------

def _sorted_edges(d: Dims) -> Tuple[int, int, int]:
    # canonical edges d1>=d2>=d3 (rotation-invariant)
    a, b, c = d.L, d.W, d.H
    x, y, z = sorted((a, b, c), reverse=True)
    return x, y, z


def _max_face_area_from_edges(d1: int, d2: int, d3: int) -> int:
    # with sorted edges, max face area is d1*d2
    return d1 * d2


def _min_possible_height(item_type: ItemType, rotation_mode: RotationMode) -> int:
    rots = allowed_rotations(item_type, rotation_mode)
    if not rots:
        # no rotations allowed -> treat as "very tall" so it goes last
        return 10**9
    return min(r.H for r in rots)


def _max_possible_base_area(item_type: ItemType, rotation_mode: RotationMode) -> int:
    rots = allowed_rotations(item_type, rotation_mode)
    if not rots:
        return 0
    return max(r.L * r.W for r in rots)


# ---------- Policy API ----------

BoxOrderFn = Callable[
    [List[ItemInstance], Dict[int, ItemType], RotationMode, Optional[int], Optional[Dict[int, List[Dims]]]],
    List[ItemInstance],
]


def order_input(
    instances: List[ItemInstance],
    types_by_id: Dict[int, ItemType],
    rotation_mode: RotationMode,
    seed: Optional[int] = None,
    rots_by_type: Optional[Dict[int, List[Dims]]] = None,
) -> List[ItemInstance]:
    return list(instances)


def order_volume_then_maxface(
    instances: List[ItemInstance],
    types_by_id: Dict[int, ItemType],
    rotation_mode: RotationMode,
    seed: Optional[int] = None,
    rots_by_type: Optional[Dict[int, List[Dims]]] = None,
) -> List[ItemInstance]:
    """
    Default for BR-Original:
      - volume desc
      - max face area desc (rotation-invariant proxy via base dims)
      - edges desc (d1,d2,d3) as tie-breakers
    """
    def dims_for_instance(inst: ItemInstance) -> Dims:
        base = types_by_id[inst.type_id].base_dims
        if rots_by_type is None:
            return base
        rots = rots_by_type.get(inst.type_id)
        if not rots:
            return base
        pref = inst.rotation_pref
        if pref is None:
            return base
        idx = int(pref) % len(rots)
        return rots[idx]

    def key(inst: ItemInstance):
        dims = dims_for_instance(inst)
        d1, d2, d3 = _sorted_edges(dims)
        vol = dims.volume()
        maxface = _max_face_area_from_edges(d1, d2, d3)
        return (-vol, -maxface, -d1, -d2, -d3, inst.type_id, inst.instance_id)

    return sorted(instances, key=key)


def order_customer_then_volume_maxface(
    instances: List[ItemInstance],
    types_by_id: Dict[int, ItemType],
    rotation_mode: RotationMode,
    seed: Optional[int] = None,
    rots_by_type: Optional[Dict[int, List[Dims]]] = None,
) -> List[ItemInstance]:
    """
    Primary: customer_id asc, then use volume_then_maxface tie-breakers.
    """
    def dims_for_instance(inst: ItemInstance) -> Dims:
        base = types_by_id[inst.type_id].base_dims
        if rots_by_type is None:
            return base
        rots = rots_by_type.get(inst.type_id)
        if not rots:
            return base
        pref = inst.rotation_pref
        if pref is None:
            return base
        idx = int(pref) % len(rots)
        return rots[idx]

    def key(inst: ItemInstance):
        dims = dims_for_instance(inst)
        d1, d2, d3 = _sorted_edges(dims)
        vol = dims.volume()
        maxface = _max_face_area_from_edges(d1, d2, d3)
        customer = inst.customer_id if inst.customer_id is not None else 0
        return (customer, -vol, -maxface, -d1, -d2, -d3, inst.type_id, inst.instance_id)

    return sorted(instances, key=key)


def order_maxface_then_volume(
    instances: List[ItemInstance],
    types_by_id: Dict[int, ItemType],
    rotation_mode: RotationMode,
    seed: Optional[int] = None,
    rots_by_type: Optional[Dict[int, List[Dims]]] = None,
) -> List[ItemInstance]:
    """
    Sometimes better when many items have similar volumes but different shapes:
      - max face area desc
      - volume desc
      - edges desc
    """
    def key(inst: ItemInstance):
        base = types_by_id[inst.type_id].base_dims
        d1, d2, d3 = _sorted_edges(base)
        vol = base.volume()
        maxface = _max_face_area_from_edges(d1, d2, d3)
        return (-maxface, -vol, -d1, -d2, -d3, inst.type_id, inst.instance_id)

    return sorted(instances, key=key)


def order_min_height_then_volume(
    instances: List[ItemInstance],
    types_by_id: Dict[int, ItemType],
    rotation_mode: RotationMode,
    seed: Optional[int] = None,
    rots_by_type: Optional[Dict[int, List[Dims]]] = None,
) -> List[ItemInstance]:
    """
    Rotation-aware:
      - minimum achievable height asc (try to build stable low layers first)
      - volume desc
      - max achievable base area desc
    This uses allowed_rotations(...) so it respects C1 mode.
    """
    def key(inst: ItemInstance):
        t = types_by_id[inst.type_id]
        min_h = _min_possible_height(t, rotation_mode)
        vol = t.base_dims.volume()
        max_base = _max_possible_base_area(t, rotation_mode)
        # small min_h first -> ascending, so use +min_h
        return (min_h, -vol, -max_base, inst.type_id, inst.instance_id)

    return sorted(instances, key=key)


def order_random_tiebreak(
    instances: List[ItemInstance],
    types_by_id: Dict[int, ItemType],
    rotation_mode: RotationMode,
    seed: Optional[int] = None,
    rots_by_type: Optional[Dict[int, List[Dims]]] = None,
) -> List[ItemInstance]:
    """
    Deterministic randomness as *last* tie-breaker, while keeping meaningful geometry first.
    Good for GA diversity without making results unstable.
    """
    rng = random.Random(seed)
    noise = { (it.type_id, it.instance_id): rng.random() for it in instances }

    def key(inst: ItemInstance):
        base = types_by_id[inst.type_id].base_dims
        d1, d2, d3 = _sorted_edges(base)
        vol = base.volume()
        maxface = _max_face_area_from_edges(d1, d2, d3)
        return (-vol, -maxface, -d1, -d2, -d3, noise[(inst.type_id, inst.instance_id)])

    return sorted(instances, key=key)


POLICIES: Dict[str, BoxOrderFn] = {
    "input": order_input,
    "volume_then_maxface": order_volume_then_maxface,
    "customer_then_volume_maxface": order_customer_then_volume_maxface,
    "maxface_then_volume": order_maxface_then_volume,
    "min_height_then_volume": order_min_height_then_volume,
    "random_tiebreak": order_random_tiebreak,
}


def apply_box_order(
    instances: List[ItemInstance],
    types_by_id: Dict[int, ItemType],
    rotation_mode: RotationMode,
    policy: str = "volume_then_maxface",
    rots_by_type: Optional[Dict[int, List[Dims]]] = None,
) -> List[ItemInstance]:
    """
    Sort ItemInstances according to a named policy.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown box_order policy: {policy}. Available: {list(POLICIES.keys())}")
    return POLICIES[policy](instances, types_by_id, rotation_mode, None, rots_by_type)

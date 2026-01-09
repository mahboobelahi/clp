
from dataclasses import dataclass
from typing import Dict, List

# Keep this consistent with your MILP/overlap convention
TOUCH_IS_OVERLAP = False

@dataclass(frozen=True)
class ULOBox:
    idx: int
    x: int
    y: int
    z: int
    l: int
    w: int
    h: int
    pi: int

def _overlap_1d(a0: int, a1: int, b0: int, b1: int, *, touch_is_overlap: bool) -> bool:
    if touch_is_overlap:
        return not (a1 < b0 or b1 < a0)
    return (a0 < b1) and (b0 < a1)

def _overlap_xy(i: ULOBox, k: ULOBox) -> bool:
    return _overlap_1d(i.x, i.x + i.l, k.x, k.x + k.l, touch_is_overlap=TOUCH_IS_OVERLAP) and \
           _overlap_1d(i.y, i.y + i.w, k.y, k.y + k.w, touch_is_overlap=TOUCH_IS_OVERLAP)

def _overlap_yz(i: ULOBox, k: ULOBox) -> bool:
    return _overlap_1d(i.y, i.y + i.w, k.y, k.y + k.w, touch_is_overlap=TOUCH_IS_OVERLAP) and \
           _overlap_1d(i.z, i.z + i.h, k.z, k.z + k.h, touch_is_overlap=TOUCH_IS_OVERLAP)

def _in_front_from_door_xL(i: ULOBox, k: ULOBox) -> bool:
    # Door at rear face x = L => larger x is closer to door
    return k.x >= i.x + i.l

def _above(i: ULOBox, k: ULOBox) -> bool:
    return k.z >= i.z + i.h

def _is_ulo(i: ULOBox, k: ULOBox) -> bool:
    # Only count unloading obstacles: i must be earlier than k
    if not (i.pi < k.pi):
        return False

    front_block = _in_front_from_door_xL(i, k) and _overlap_yz(i, k)
    above_block = _above(i, k) and _overlap_xy(i, k)
    return front_block or above_block

def compute_ulo_count_from_placed(
    placed: List["AABB"],
    *,
    type_to_pi: Dict[int, int],
) -> int:
    """
    Count ULO pairs from decoder output AABBs.
    Requires a mapping: type_id -> delivery rank pi (smaller pi unload earlier).
    """
    boxes: List[ULOBox] = []
    for idx, a in enumerate(placed):
        if a.type_id not in type_to_pi:
            raise KeyError(f"type_id={a.type_id} missing from type_to_pi")
        boxes.append(ULOBox(
            idx=idx,
            x=int(a.origin.x), y=int(a.origin.y), z=int(a.origin.z),
            l=int(a.dims.L), w=int(a.dims.W), h=int(a.dims.H),
            pi=int(type_to_pi[a.type_id]),
        ))

    count = 0
    for i in boxes:
        for k in boxes:
            if i.idx != k.idx and _is_ulo(i, k):
                count += 1
    return count

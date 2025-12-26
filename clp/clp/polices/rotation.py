
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Set

from clp.clp.models.items import Dims, ItemType  # adjust import to your actual path


class RotationMode(str, Enum):
    RESPECT_C1 = "respect_c1"
    SIX_WAY = "six_way"

def _unique_permutations(d: Dims) -> List[Dims]:
    perms = [
        Dims(d.L, d.W, d.H),
        Dims(d.L, d.H, d.W),
        Dims(d.W, d.L, d.H),
        Dims(d.W, d.H, d.L),
        Dims(d.H, d.L, d.W),
        Dims(d.H, d.W, d.L),
    ]
    seen: Set[Tuple[int, int, int]] = set()
    out: List[Dims] = []
    for p in perms:
        key = (p.L, p.W, p.H)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def allowed_rotations(item, mode: RotationMode) -> List[Dims]:
    """
    Returns allowed orientation Dims(L,W,H) for this item type.

    SIX_WAY: all unique permutations.
    RESPECT_C1 (BR meaning): flag after each dimension indicates whether that
    dimension is allowed to be vertical (aligned with Z). If allowed, the other
    two can swap in X/Y.
    """
    base = item.base_dims

    if mode == RotationMode.SIX_WAY:
        return _unique_permutations(base)

    if mode != RotationMode.RESPECT_C1:
        raise ValueError(f"Unknown rotation mode: {mode}")

    L, W, H = base.L, base.W, base.H
    rots: List[Dims] = []

    # BR: 1 means that dimension may be vertical (z)
    if item.c1_height == 1:
        rots.append(Dims(L, W, H))
        rots.append(Dims(W, L, H))

    if item.c1_length == 1:
        rots.append(Dims(W, H, L))
        rots.append(Dims(H, W, L))

    if item.c1_width == 1:
        rots.append(Dims(L, H, W))
        rots.append(Dims(H, L, W))

    # deduplicate (in case dims repeat)
    seen: Set[Tuple[int, int, int]] = set()
    out: List[Dims] = []
    for r in rots:
        key = (r.L, r.W, r.H)
        if key not in seen:
            seen.add(key)
            out.append(r)

    return out
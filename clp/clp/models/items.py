

# put this in your chosen file, e.g. clp/clp/model.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class Dims:
    L: int
    W: int
    H: int

    def volume(self) -> int:
        return self.L * self.W * self.H


@dataclass(frozen=True)
class ItemType:
    type_id: int
    base_dims: Dims
    demand: int
    value: int  # volume proxy / objective value
    c1_length: int
    c1_width: int
    c1_height: int


@dataclass(frozen=True)
class ItemInstance:
    """One physical box (expanded from demand)."""
    type_id: int
    instance_id: int
    # Filled later by GA / customer logic:
    customer_id: Optional[int] = None
    rotation_pref: Optional[int] = None  # e.g., index into allowed rotations


def br_row_to_item_type(row, type_id: int) -> ItemType:
    # row is your BRItemRow from br_original.py
    dims = Dims(L=row.length, W=row.width, H=row.height)
    # value already provided; if you want, sanity-check against dims.volume()
    return ItemType(
        type_id=type_id,
        base_dims=dims,
        demand=row.demand,
        value=row.value,
        c1_length=row.c1_length,
        c1_width=row.c1_width,
        c1_height=row.c1_height,
    )


def expand_demands(item_types: List[ItemType]) -> List[ItemInstance]:
    instances: List[ItemInstance] = []
    k = 0
    for it in item_types:
        for _ in range(it.demand):
            instances.append(ItemInstance(type_id=it.type_id, instance_id=k))
            k += 1
    return instances

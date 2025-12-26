
from dataclasses import dataclass
from clp.clp.models.items import Dims

from typing import List, Tuple


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    z: int


@dataclass(frozen=True)
class AABB:
    origin: Point   # lower-back-left corner
    dims: Dims      # Dims(L, W, H)

    def __repr__(self) -> str:
        x0, y0, z0 = self.origin.x, self.origin.y, self.origin.z
        x1 = x0 + self.dims.L
        y1 = y0 + self.dims.W
        z1 = z0 + self.dims.H

        return (
            f"AABB("
            f"min=({x0},{y0},{z0}), "
            f"max=({x1},{y1},{z1}), "
            f"dims=({self.dims.L},{self.dims.W},{self.dims.H})"
            f")"
        )


#* Geometry utility functions

def overlaps(a: AABB, b: AABB) -> bool:
    """
    True only if there is positive-volume intersection.
    Touching faces/edges/corners is NOT overlap.
    """
    return not (
        a.origin.x + a.dims.L <= b.origin.x or
        b.origin.x + b.dims.L <= a.origin.x or
        a.origin.y + a.dims.W <= b.origin.y or
        b.origin.y + b.dims.W <= a.origin.y or
        a.origin.z + a.dims.H <= b.origin.z or
        b.origin.z + b.dims.H <= a.origin.z
    )


def inside_container(a: AABB, container_dims: Dims) -> bool:
    """
    Assumes origin is non-negative.
    """
    return (
        a.origin.x >= 0 and a.origin.y >= 0 and a.origin.z >= 0 and
        a.origin.x + a.dims.L <= container_dims.L and
        a.origin.y + a.dims.W <= container_dims.W and
        a.origin.z + a.dims.H <= container_dims.H
    )

def is_feasible(candidate: AABB, placed: list[AABB], container: Dims) -> bool:
    return inside_container(candidate, container) and not any(overlaps(candidate, p) for p in placed)


def _rect_intersection(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> Tuple[int,int,int,int] | None:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _union_area(rects: List[Tuple[int,int,int,int]]) -> int:
    """
    Compute union area of axis-aligned rectangles in 2D.
    rect = (x0,y0,x1,y1), half-open.
    Sweep-line on x with active y-interval union.
    Deterministic and exact for integers.
    """
    if not rects:
        return 0

    # events: x, type(+1 start / -1 end), y0, y1
    events = []
    for x0, y0, x1, y1 in rects:
        events.append((x0, 1, y0, y1))
        events.append((x1, -1, y0, y1))
    events.sort(key=lambda e: e[0])

    def y_union_length(intervals: List[Tuple[int,int]]) -> int:
        if not intervals:
            return 0
        intervals = sorted(intervals)
        total = 0
        cur_y0, cur_y1 = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 > cur_y1:
                total += cur_y1 - cur_y0
                cur_y0, cur_y1 = y0, y1
            else:
                cur_y1 = max(cur_y1, y1)
        total += cur_y1 - cur_y0
        return total

    area = 0
    active: List[Tuple[int,int]] = []
    prev_x = events[0][0]

    for x, typ, y0, y1 in events:
        dx = x - prev_x
        if dx > 0:
            area += dx * y_union_length(active)
            prev_x = x

        if typ == 1:
            active.append((y0, y1))
        else:
            # remove one matching interval
            for i, (ay0, ay1) in enumerate(active):
                if ay0 == y0 and ay1 == y1:
                    active.pop(i)
                    break

    return area


def is_supported(cand: "AABB", placed: List["AABB"], min_ratio: float = .8) -> bool:
    """
    True if cand is supported by floor (z==0) or by boxes directly below it.

    Support is measured as union area of overlaps between cand's base footprint
    and the top faces of boxes with top_z == cand.origin.z.

    min_ratio: required supported area / base area.
      1.0 = fully supported.
      0.8, 0.5 etc are permissive.
    """
    if cand.origin.z == 0:
        return True

    z_support = cand.origin.z

    cx0 = cand.origin.x
    cy0 = cand.origin.y
    cx1 = cx0 + cand.dims.L
    cy1 = cy0 + cand.dims.W

    base_area = (cx1 - cx0) * (cy1 - cy0)
    if base_area <= 0:
        return False

    overlaps_2d: List[Tuple[int,int,int,int]] = []

    for b in placed:
        top_z = b.origin.z + b.dims.H
        if top_z != z_support:
            continue

        bx0 = b.origin.x
        by0 = b.origin.y
        bx1 = bx0 + b.dims.L
        by1 = by0 + b.dims.W

        inter = _rect_intersection((cx0, cy0, cx1, cy1), (bx0, by0, bx1, by1))
        if inter is not None:
            overlaps_2d.append(inter)

    supported_area = _union_area(overlaps_2d)
    return supported_area >= min_ratio * base_area

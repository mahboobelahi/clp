from __future__ import annotations
from typing import List, Set, Tuple

from clp.clp.models.geometry import Point, AABB, Dims  # adjust if needed


def dblf_key(p: Point) -> Tuple[int, int, int]:
    # DBLF: x asc, then z asc, then y asc
    return (p.x, p.z, p.y)


def point_is_free(p: Point, placed: List[AABB]) -> bool:
    """
    EP is invalid if it's strictly inside an already placed AABB.
    Touching faces/edges is OK.
    """
    for b in placed:
        if (
            b.origin.x <= p.x < b.origin.x + b.dims.L and
            b.origin.y <= p.y < b.origin.y + b.dims.W and
            b.origin.z <= p.z < b.origin.z + b.dims.H
        ):
            return False
    return True

def lift_point_out_of_boxes(p: Point, placed: List[AABB]) -> Point:
    """
    If point p is inside the volume of any placed box (by x/y footprint and z),
    lift it vertically to the top of the highest such box covering (x,y).
    Repeat until it is no longer inside any box.
    """
    x, y, z = p.x, p.y, p.z

    while True:
        lifted = False
        best_top = None  # top z to lift to

        for b in placed:
            x0, y0, z0 = b.origin.x, b.origin.y, b.origin.z
            x1 = x0 + b.dims.L
            y1 = y0 + b.dims.W
            top = z0 + b.dims.H

            # Inside footprint (half-open like your overlap logic)
            in_xy = (x0 <= x < x1) and (y0 <= y < y1)

            # Inside volume in z (including bottom face, excluding top face)
            in_z = (z0 <= z < top)

            if in_xy and in_z:
                if best_top is None or top > best_top:
                    best_top = top
                    lifted = True

        if not lifted:
            return Point(x, y, z)

        # lift to the highest covering top
        z = best_top


def generate_eps_from_placed_box(new_box: AABB, container: Dims, placed: List[AABB]) -> List[Point]:
    """
    Moon-style EP generation with MPH for elevated placements.

    - If new_box is on floor (z0==0): P1=(x0+L, y0, 0), P2=(x0, y0+W, 0)
    - If elevated (z0>0): compute MPH_x and MPH_y (<= z0) from placed objects
      and set P1=(x0+L, y0, MPH_x), P2=(x0, y0+W, MPH_y)
    - P3 always: (x0, y0, z0+H)
    """

    Lc, Wc, Hc = container.L, container.W, container.H

    x0, y0, z0 = new_box.origin.x, new_box.origin.y, new_box.origin.z
    lx, ly, lz = new_box.dims.L, new_box.dims.W, new_box.dims.H

    # clamp to container bounds (avoid generating EP exactly outside)
    def clamp_x(x: int) -> int: return min(x, Lc)
    def clamp_y(y: int) -> int: return min(y, Wc)
    def clamp_z(z: int) -> int: return min(z, Hc)

    out: List[Point] = []

    # ----- P1/P2 -----
    if z0 == 0:
        p1 = Point(clamp_x(x0 + lx), y0, 0)  # along +x
        p2 = Point(x0, clamp_y(y0 + ly), 0)  # along +y

        if p1.x < Lc:
            out.append(p1)
        if p2.y < Wc:
            out.append(p2)

    else:
        MPH_x = 0
        MPH_y = 0

        # Use placed list (which includes new_box) but skip new_box itself
        for other in reversed(placed):
            if other is new_box:
                continue

            xj, yj, zj = other.origin.x, other.origin.y, other.origin.z
            lj, wj, hj = other.dims.L, other.dims.W, other.dims.H
            top = zj + hj

            # only supports at/below the base of this elevated placement
            if top > z0:
                continue

            # For P1 at (x0+lx, y0, ?)
            # other crosses plane x = x0+lx and overlaps y-range [y0, y0+ly)
            if (
                xj < x0 + lx and xj + lj > x0 + lx and
                yj < y0 + ly and yj + wj > y0
            ):
                if zj + hj <= z0 and zj + hj > MPH_x:#if top > MPH_x:
                    MPH_x = top

            # For P2 at (x0, y0+ly, ?)
            # other crosses plane y = y0+ly and overlaps x-range [x0, x0+lx)
            if (
                xj < x0 + lx and xj + lj > x0 and
                yj < y0 + ly and yj + wj > y0 + ly
            ):
                 if zj + hj <= z0 and zj + hj > MPH_y:#if top > MPH_y:
                    MPH_y = top

        p1 = Point(clamp_x(x0 + lx), y0, MPH_x)
        p2 = Point(x0, clamp_y(y0 + ly), MPH_y)

        if p1.x < Lc:
            out.append(p1)
        if p2.y < Wc:
            out.append(p2)

    # ----- P3 -----
    p3 = Point(x0, y0, clamp_z(z0 + lz))
    if p3.z < Hc:
        out.append(p3)
    # print("Generated EPs from box at", new_box, ":", out)
    return out


def update_eps(eps: List[Point], placed: List[AABB], new_box: AABB, container: Dims) -> List[Point]:
    """
    Update EP list after placing new_box using Moon-style EP generation (with MPH).
    """
    candidates = eps + generate_eps_from_placed_box(new_box, container, placed)

    # Filter + dedup
    filtered: List[Point] = []
    seen: Set[Tuple[int, int, int]] = set()

    for p in candidates:
        # point bounds (EP itself)
        if p.x < 0 or p.y < 0 or p.z < 0:
            continue
        if p.x > container.L or p.y > container.W or p.z > container.H:
            continue

        p = lift_point_out_of_boxes(p, placed)

        if not point_is_free(p, placed):
            continue

        key = (p.x, p.y, p.z)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(p)

    filtered.sort(key=dblf_key)
    return filtered

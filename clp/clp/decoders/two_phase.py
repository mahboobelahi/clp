# clp/clp/decoders/two_phase.py
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Callable, Hashable
from clp.clp.viz.debug_viz import plot_container_debug
from clp.clp.models.geometry import Point, AABB, Dims, inside_container, overlaps
from clp.clp.polices.rotation import RotationMode, allowed_rotations
from clp.clp.decoders.extreme_points import update_eps, dblf_key
from clp.configurations import (PLOT_PARTIAL_LAYOUT, debug, is_ULO,
                                    MAX_EPS_KEEP ,
                                    MAX_EP_PROBES ,
                                    MAX_ROT_TRIES,
                                     SPLIT_RATIO )


# ============================================================
# Speed/robustness caps (tune later)
# ============================================================


# ============================================================
# Grouping (Path-A): generic key for (customer,type) or (type)
# ============================================================
GroupKey = Tuple[Hashable, ...]


def default_group_key(inst) -> GroupKey:
    """
    Generic grouping:
      - If instance has customer_id -> (customer_id, type_id)
      - Else -> (type_id,)
    Works for: normal BR (one "implicit customer") and modified BR (many).
    """
    cid = getattr(inst, "customer_id", None)
    tid = inst.type_id
    return (tid,) if cid is None else (cid, tid)


def _group_sequence(instances: List, group_key_fn: Callable) -> List[GroupKey]:
    """
    Returns unique groups in the order they appear in 'instances'.
    This preserves chromosome intent when GA provides the instance order.
    """
    seen: Set[GroupKey] = set()
    out: List[GroupKey] = []
    for inst in instances:
        gk = group_key_fn(inst)
        if gk not in seen:
            seen.add(gk)
            out.append(gk)
    return out


# ============================================================
# EP sorting keys
# ============================================================
def rev_dblf_key(p: Point) -> Tuple[int, int, int]:
    # reverse DBLF: x desc, z desc, y desc
    return (-p.x, -p.z, -p.y)


def _dedupe_eps(eps: List[Point]) -> List[Point]:
    seen: Set[Tuple[int, int, int]] = set()
    out: List[Point] = []
    for p in eps:
        k = (p.x, p.y, p.z)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


# ============================================================
# Balance proxy via moments (fast; avoids full CG recompute per candidate)
# ============================================================
@dataclass
class MomentState:
    mx: float
    my: float
    mz: float
    left: float = 0.0
    right: float = 0.0
    front: float = 0.0
    rear: float = 0.0

    def score_with_box(self, aabb: AABB, mass: float) -> float:
        """
        Smaller is better.
        """
        xc = aabb.origin.x + aabb.dims.L / 2.0
        yc = aabb.origin.y + aabb.dims.W / 2.0

        left = self.left
        right = self.right
        front = self.front
        rear = self.rear

        if xc < self.mx:
            left += mass * abs(self.mx - xc)
        else:
            right += mass * abs(xc - self.mx)

        if yc < self.my:
            front += mass * abs(self.my - yc)
        else:
            rear += mass * abs(yc - self.my)

        lr = abs(left - right) / (left + right + 1e-9)
        fr = abs(front - rear) / (front + rear + 1e-9)
        return 0.5 * (lr + fr)

    def commit_box(self, aabb: AABB, mass: float) -> None:
        xc = aabb.origin.x + aabb.dims.L / 2.0
        yc = aabb.origin.y + aabb.dims.W / 2.0

        if xc < self.mx:
            self.left += mass * abs(self.mx - xc)
        else:
            self.right += mass * abs(xc - self.mx)

        if yc < self.my:
            self.front += mass * abs(self.my - yc)
        else:
            self.rear += mass * abs(yc - self.my)

    def score_with_block(
        self,
        origin: Point,
        dims: Dims,
        nx: int,
        ny: int,
        nz: int,
        mass_per_box: float
    ) -> float:
        """
        Smaller is better.
        Computes the moment diff if we add a filled block of nx*ny*nz boxes.
        This is still O(nx*ny*nz) but blocks reduce placement iterations a lot,
        and EP probing is capped.
        """
        left = self.left
        right = self.right
        front = self.front
        rear = self.rear

        for iz in range(nz):
            for ix in range(nx):
                x0 = origin.x + ix * dims.L
                xc = x0 + dims.L / 2.0
                for iy in range(ny):
                    y0 = origin.y + iy * dims.W
                    yc = y0 + dims.W / 2.0

                    if xc < self.mx:
                        left += mass_per_box * abs(self.mx - xc)
                    else:
                        right += mass_per_box * abs(xc - self.mx)

                    if yc < self.my:
                        front += mass_per_box * abs(self.my - yc)
                    else:
                        rear += mass_per_box * abs(yc - self.my)

        lr = abs(left - right) / (left + right + 1e-9)
        fr = abs(front - rear) / (front + rear + 1e-9)
        return 0.5 * (lr + fr)

    def commit_block(self, block_boxes: List[AABB], mass_fn) -> None:
        for b in block_boxes:
            self.commit_box(b, mass_fn(b))


# ============================================================
# Phase-A gap/alignment score (cheap)
# ============================================================
# def gap_score(extent: AABB, placed: List[AABB], container: Dims, *, tol: int = 0) -> float:
#     """
#     Higher is better. Uses block/box bounding AABB.
#     """
#     x0, y0, z0 = extent.origin.x, extent.origin.y, extent.origin.z
#     x1 = x0 + extent.dims.L
#     y1 = y0 + extent.dims.W
#     z1 = z0 + extent.dims.H

#     x_mins = set(); x_maxs = set()
#     y_mins = set(); y_maxs = set()
#     z_mins = set(); z_maxs = set()

#     for b in placed:
#         bx0, by0, bz0 = b.origin.x, b.origin.y, b.origin.z
#         bx1 = bx0 + b.dims.L
#         by1 = by0 + b.dims.W
#         bz1 = bz0 + b.dims.H
#         x_mins.add(bx0); x_maxs.add(bx1)
#         y_mins.add(by0); y_maxs.add(by1)
#         z_mins.add(bz0); z_maxs.add(bz1)

#     # container walls
#     x_mins.add(0); x_maxs.add(container.L)
#     y_mins.add(0); y_maxs.add(container.W)
#     z_mins.add(0); z_maxs.add(container.H)

#     def aligned(val: int, planes: set[int]) -> bool:
#         if tol == 0:
#             return val in planes
#         return any(abs(val - p) <= tol for p in planes)

#     s = 0.0
#     if aligned(x0, x_maxs): s += 1.0
#     if aligned(y0, y_maxs): s += 1.0
#     if aligned(z0, z_maxs): s += 1.0

#     if aligned(x1, x_mins): s += 0.5
#     if aligned(y1, y_mins): s += 0.5
#     if aligned(z1, z_mins): s += 0.5

#     # slight preference for smaller coordinates (compactness)
#     s += 0.0001 * (-(x0 + y0 + z0))
#     return s
def gap_score(
    extent: AABB,
    placed: List[AABB],
    container: Dims,
    *,
    used_qty: int,
    unit_volume: int,
    tol: int = 0,
    prefer_footprint: bool = True,
    prefer_height: bool = False,
) -> float:
    """
    Higher is better.

    Priorities:
      1) Place more boxes
      2) Use more volume
      3) Use floor footprint efficiently (DBLF)
      4) Optional: prefer filling height
      5) Very small contact bonus (tie-breaker only)
    """
    x0, y0, z0 = extent.origin.x, extent.origin.y, extent.origin.z
    ex, ey, ez = extent.dims.L, extent.dims.W, extent.dims.H

    sx = container.L - ex
    sy = container.W - ey
    sz = container.H - ez
    if sx < 0 or sy < 0 or sz < 0:
        return -1e18

    used_vol = used_qty * unit_volume

    s = 0.0

    # (1) boxes dominate everything
    s += 1e9 * used_qty

    # (2) volume
    s += 1e3 * used_vol

    # (3) footprint efficiency (DBLF preference)
    if prefer_footprint:
        s -= 10.0 * (sx + sy)

    # (4) optional height preference
    if prefer_height:
        s -= 2.0 * sz
    else:
        # still penalize height slack lightly
        s -= 1.0 * sz

    # (5) tiny contact bonus (only after first placement)
    if placed:
        x_mins = {0}; x_maxs = {container.L}
        y_mins = {0}; y_maxs = {container.W}
        z_mins = {0}; z_maxs = {container.H}

        for b in placed:
            bx0, by0, bz0 = b.origin.x, b.origin.y, b.origin.z
            bx1 = bx0 + b.dims.L
            by1 = by0 + b.dims.W
            bz1 = bz0 + b.dims.H
            x_mins.add(bx0); x_maxs.add(bx1)
            y_mins.add(by0); y_maxs.add(by1)
            z_mins.add(bz0); z_maxs.add(bz1)

        def aligned(v, planes):
            if tol == 0:
                return v in planes
            return any(abs(v - p) <= tol for p in planes)

        if aligned(x0, x_mins): s += 0.3
        if aligned(y0, y_mins): s += 0.3
        if aligned(z0, z_mins): s += 0.15
        if aligned(x0 + ex, x_maxs): s += 0.15
        if aligned(y0 + ey, y_maxs): s += 0.15
        if aligned(z0 + ez, z_maxs): s += 0.05

    return s


# ============================================================
# Phase split helper
# ============================================================
def _first_phase_split(instances: List, split_ratio: float) -> Tuple[List, List]:
    n = len(instances)
    k = int(round(n * split_ratio))
    k = max(0, min(n, k))
    return instances[:k], instances[k:]


# ============================================================
# Rotation lookup for group
# ============================================================
def _rotation_dims_for_group(
    *,
    group_key: GroupKey,
    type_id: int,
    rots_by_type: Dict[int, List[Dims]],
    rot_by_group: Optional[Dict[GroupKey, int]],
    soft_rotation: bool,
) -> List[Dims]:
    rots = rots_by_type.get(type_id, [])
    if not rots:
        return []

    # if GA doesn't provide a rotation for this group, allow all
    if rot_by_group is None or group_key not in rot_by_group:
        return rots[:MAX_ROT_TRIES]

    idx = rot_by_group[group_key] % len(rots)
    if not soft_rotation:
        return [rots[idx]]

    # soft mode: try preferred first, then fall back to others
    head = [rots[idx]]
    tail = [r for j, r in enumerate(rots) if j != idx]
    return (head + tail)[:MAX_ROT_TRIES]


# ============================================================
# Strict Moon block builder (no incomplete columns/layers)
# ============================================================
def _try_build_strict_block(
    *,
    ep: Point,
    type_id: int,
    dims: Dims,
    remaining_qty: int,
    placed: List[AABB],
    container: Dims,
    support_required: bool,
    support_min_ratio: float,
    is_supported_fn,
) -> Optional[Tuple[List[AABB], Dims, int, Tuple[int, int, int]]]:
    """
    Build a strict cuboid block at EP with identical boxes of the same type & rotation.

    Rules:
      - block has integer nx*ny*nz boxes fully filled
      - no incomplete columns/layers (cuboid only)
      - limited by container bounds, overlaps, support, and remaining_qty

    Returns:
      (block_boxes, extent_dims, used_qty, (nx,ny,nz)) or None
    """
    if remaining_qty <= 0:
        return None

    # max counts within container bounds from this EP
    max_nx = (container.L - ep.x) // dims.L
    max_ny = (container.W - ep.y) // dims.W
    max_nz = (container.H - ep.z) // dims.H
    if max_nx <= 0 or max_ny <= 0 or max_nz <= 0:
        return None

    # Start from maximum footprint. If demand can't fill even one layer, shrink footprint.
    nx = int(max_nx)
    ny = int(max_ny)

    # Shrink footprint until nx*ny <= remaining_qty (so at least one full layer is possible)
    while nx >= 1 and ny >= 1 and (nx * ny) > remaining_qty:
        if ny > 1:
            ny -= 1
        elif nx > 1:
            nx -= 1
        else:
            break

    if nx <= 0 or ny <= 0:
        return None

    nz = min(int(max_nz), remaining_qty // (nx * ny))

    if nz <= 0:
        return None

    # Try feasibility. If fails, shrink height first, then footprint.
    while nx >= 1 and ny >= 1 and nz >= 1:
        used = nx * ny * nz
        if used <= 0:
            return None

        block_boxes: List[AABB] = []
        feasible = True

        for iz in range(nz):
            for ix in range(nx):
                for iy in range(ny):
                    o = Point(ep.x + ix * dims.L, ep.y + iy * dims.W, ep.z + iz * dims.H)
                    cand = AABB(o, dims, type_id=type_id)

                    if not inside_container(cand, container):
                        feasible = False
                        break
                    if any(overlaps(cand, b) for b in placed):
                        feasible = False
                        break

                    if support_required and cand.origin.z > 0:
                        if is_supported_fn is None:
                            raise RuntimeError("support_required=True but is_supported_fn not provided")
                        
                        if not is_supported_fn(cand, placed + block_boxes, support_min_ratio):
                            feasible = False
                            break

                    block_boxes.append(cand)
            #!
            #     if not feasible:
            #         break
            # if not feasible:
            #     break

        if feasible:
            extent = Dims(nx * dims.L, ny * dims.W, nz * dims.H)
            return block_boxes, extent, used, (nx, ny, nz)

        #! shrink strategy
        if nz > 1:
            nz -= 1
        elif ny > 1:
            ny -= 1
            nz = min(int(max_nz), remaining_qty // (nx * ny)) if (nx * ny) > 0 else 0
        elif nx > 1:
            nx -= 1
            nz = min(int(max_nz), remaining_qty // (nx * ny)) if (nx * ny) > 0 else 0
        else:
            break

    return None


# ============================================================
# Main decoder
# ============================================================
def decode_two_phase(
    *,
    container: Dims,
    item_types: List,
    instances: List,
    rotation_mode: RotationMode,
    split_ratio: float = SPLIT_RATIO,
    support_required: bool = True,
    support_min_ratio: float = 1.0,
    is_supported_fn=None,
    gap_tol: int = 0,
    box_order_policy: Optional[str] = None,
    box_order_seed: Optional[int] = None,

    # Backward compatible: old callers may pass rot_by_type
    rot_by_type: Optional[Dict[int, int]] = None,

    # Path-A: group-based rotation control
    rot_by_group: Optional[Dict[GroupKey, int]] = None,
    group_key_fn: Callable = default_group_key,

    soft_rotation: bool = False,#True,
) -> Tuple[List[AABB], List]:
    """
    Two-phase greedy with Moon-style strict blocks first, then leftovers as single boxes.

    - If box_order_policy is provided, decoder applies internal sorting (baseline).
      For GA, set box_order_policy=None to respect chromosome order.
    - Blocks are built per GROUP (default: (customer_id,type_id) else (type_id,)).
    - Rotation control is per GROUP via rot_by_group, with optional soft fallback.

    Returns: (placed_boxes, unplaced_instances)
    """

    # Allowed rotations per type under the chosen mode
    rots_by_type: Dict[int, List[Dims]] = {
        t.type_id: allowed_rotations(t, rotation_mode) for t in item_types
    }

    # Optional baseline sorting (do NOT use for GA)
    if box_order_policy is not None:
        from clp.clp.polices.box_order import apply_box_order
        types_by_id = {t.type_id: t for t in item_types}
        instances = apply_box_order(
            instances=instances,
            types_by_id=types_by_id,
            rotation_mode=rotation_mode,
            policy=box_order_policy,
            rots_by_type=rots_by_type,
        )
        if debug:
            print(f"[BOX_ORDER] policy={box_order_policy} total={len(instances)}")
            for i, inst in enumerate(instances[:10]):
                base = types_by_id[inst.type_id].base_dims
                dims = base
                pref = inst.rotation_pref
                if pref is not None:
                    rots = rots_by_type.get(inst.type_id, [])
                    if rots:
                        dims = rots[int(pref) % len(rots)]
                print(
                    f"  {i:03d} type={inst.type_id} inst={inst.instance_id} "
                    f"rot_pref={pref} dims=({dims.L},{dims.W},{dims.H})"
                )

    # Backward compatibility:
    # If rot_by_group not provided but rot_by_type is, convert for groups present.
    if rot_by_group is None and rot_by_type is not None:
        rot_by_group = {}
        for inst in instances:
            gk = group_key_fn(inst)
            tid = inst.type_id
            if gk not in rot_by_group and tid in rot_by_type:
                rot_by_group[gk] = rot_by_type[tid]

    # Mass function for moment proxy (use weight if defined, else volume)
    types_by_id = {t.type_id: t for t in item_types}

    def mass_fn(b: AABB) -> float:
        it = types_by_id[b.type_id]
        w = getattr(it, "weight", None)
        return float(w) if w is not None else float(b.dims.volume())

    # State
    placed: List[AABB] = []
    unplaced: List = []
    eps: List[Point] = [Point(0, 0, 0)]

    # Moment state for balance proxy
    mstate = MomentState(mx=container.L / 2.0, my=container.W / 2.0, mz=container.H / 2.0)

    # Phase split in the given instance order
    phaseA_insts, phaseB_insts = _first_phase_split(instances, SPLIT_RATIO)

    # Remaining quantities per group
    remA: Dict[GroupKey, int] = {}
    for inst in phaseA_insts:
        gk = group_key_fn(inst)
        remA[gk] = remA.get(gk, 0) + 1

    remB: Dict[GroupKey, int] = {}
    for inst in phaseB_insts:
        gk = group_key_fn(inst)
        remB[gk] = remB.get(gk, 0) + 1

    group_seq_A = _group_sequence(phaseA_insts, group_key_fn)
    group_seq_B = _group_sequence(phaseB_insts, group_key_fn)

    def _cap_eps(sort_key=dblf_key) -> None:
        nonlocal eps
        eps = _dedupe_eps(eps)
        if len(eps) > MAX_EPS_KEEP:
            eps.sort(key=sort_key)
            eps = eps[:MAX_EPS_KEEP]

    def place_best_block_for_group(
        *,
        group_key: GroupKey,
        type_id: int,
        remaining_qty: int,
        ep_sort_key,
        phase: str,  # "A" or "B"
        soft_rotation=soft_rotation,
        support_min_ratio=support_min_ratio) -> Optional[int]:
        """
        Try to place one strict block for this group (type fixed, rotation chosen).
        Returns how many boxes were consumed, else None.
        """
        nonlocal eps, placed, mstate

        if remaining_qty <= 0:
            return None

        rot_dims_list = _rotation_dims_for_group(
            group_key=group_key,
            type_id=type_id,
            rots_by_type=rots_by_type,
            rot_by_group=rot_by_group,
            soft_rotation=soft_rotation,
        )
        if not rot_dims_list:
            return None

        eps.sort(key=ep_sort_key)
        _cap_eps(sort_key=ep_sort_key)
        eps_to_probe = eps[:MAX_EP_PROBES]

        best = None  # (score, ep, block_boxes, extent_dims, used_qty, nxnyz, base_dims)

        for ep in eps_to_probe:
            for dims in rot_dims_list:
                built = _try_build_strict_block(
                    ep=ep,
                    type_id=type_id,
                    dims=dims,
                    remaining_qty=remaining_qty,
                    placed=placed,
                    container=container,
                    support_required=support_required,
                    support_min_ratio=support_min_ratio,
                    is_supported_fn=is_supported_fn,
                )
                if built is None:
                    continue

                block_boxes, extent_dims, used_qty, (nx, ny, nz) = built
                extent_aabb = AABB(ep, extent_dims, type_id=type_id)

                # if phase == "A" and soft_rotation:
                #     sc = gap_score(extent_aabb, placed, container, tol=gap_tol)
                # else:
                #     # moment diff smaller is better; invert
                #     tmp_box = AABB(Point(0, 0, 0), dims, type_id=type_id)
                #     mpb = mass_fn(tmp_box)
                #     md = mstate.score_with_block(ep, dims, nx, ny, nz, mpb)
                #     sc = -md

                # if (best is None) or (sc > best[0]):
                
                best = (0, ep, block_boxes, extent_dims, used_qty)
                break

            if best != None:
                break

        # if best is None:
        #     return None
        if best is not None:
            _, best_ep, block_boxes, extent_dims, used_qty = best

            placed.extend(block_boxes)
            mstate.commit_block(block_boxes, mass_fn)

            eps.remove(best_ep)
            eps = update_eps(eps, placed, AABB(best_ep, extent_dims, type_id=type_id), container)
            _cap_eps(sort_key=ep_sort_key)
            
            if PLOT_PARTIAL_LAYOUT:
                plot_container_debug(
                    placed=placed,
                    eps= eps,
                    container_dims=(container.L, container.W, container.H),
                    title="NSGA-II Type1 Decoded Layout",
                    show_box_labels=True,
                )
            return int(used_qty)

    # ============================================================
    # Block phase A (gap/alignment)
    # ============================================================
    for gk in group_seq_A:
        tid = int(gk[-1])  # type_id always last in our default key
        while remA.get(gk, 0) > 0:
            used = place_best_block_for_group(
                group_key=gk,
                type_id=tid,
                remaining_qty=remA[gk],
                ep_sort_key=dblf_key,
                phase="A",
            )
            if used is None:
                break
            remA[gk] -= used

    # ============================================================
    # Block phase B (balance proxy)
    # ============================================================
    for gk in group_seq_B:
        tid = int(gk[-1])
        while remB.get(gk, 0) > 0:
            used = place_best_block_for_group(
                group_key=gk,
                type_id=tid,
                remaining_qty=remB[gk],
                ep_sort_key=rev_dblf_key,
                phase="B",
                support_min_ratio=0.75,
                soft_rotation=False,#True
            )
            if used is None:
                break
            remB[gk] -= used

    # ============================================================
    # Leftovers ONLY after block phase ends (your requirement)
    # ============================================================
    leftovers: List = []

    # Build leftovers list from remaining counts in remA/remB
    for inst in phaseA_insts:
        gk = group_key_fn(inst)
        if remA.get(gk, 0) > 0:
            leftovers.append(inst)
            remA[gk] -= 1

    for inst in phaseB_insts:
        gk = group_key_fn(inst)
        if remB.get(gk, 0) > 0:
            leftovers.append(inst)
            remB[gk] -= 1

    # Remaining qty per leftover group
    remL: Dict[GroupKey, int] = {}
    for inst in leftovers:
        gk = group_key_fn(inst)
        remL[gk] = remL.get(gk, 0) + 1

    # Stable group order for leftovers
    group_seq_L = _group_sequence(leftovers, group_key_fn)

    # ============================================================
    # Try placing leftovers in block-mode
    # ============================================================
    for gk in group_seq_L:
        remaining = remL.get(gk, 0)
        if remaining <= 0:
            continue

        tid = int(gk[-1])

        while remL.get(gk, 0) > 0:
            used = place_best_block_for_group(
                group_key=gk,
                type_id=tid,
                remaining_qty=remL[gk],
                ep_sort_key=dblf_key,
                phase="A",
                soft_rotation=False,#True,
                support_min_ratio=0.7,
            )
            if used is None or used <= 0:
                break

            remL[gk] -= used

    # ============================================================
    # Reconstruct unplaced instances from what remains in remL
    # ============================================================
    group_to_insts: Dict[GroupKey, List] = defaultdict(list)
    for inst in leftovers:
        group_to_insts[group_key_fn(inst)].append(inst)

    unplaced_L: List = []
    for gk, insts in group_to_insts.items():
        rem = remL.get(gk, 0)
        if rem > 0:
            # take any rem instances (deterministic: keep original order)
            unplaced_L.extend(insts[-rem:])


    return placed, unplaced_L

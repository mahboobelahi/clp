
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable
import math

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
    x0, y0, z0 = cand.origin.x, cand.origin.y, cand.origin.z
    x1 = x0 + cand.dims.L
    y1 = y0 + cand.dims.W
    z1 = z0 + cand.dims.H

    x_mins = set(); x_maxs = set()
    y_mins = set(); y_maxs = set()
    z_mins = set(); z_maxs = set()

    for b in placed:
        bx0, by0, bz0 = b.origin.x, b.origin.y, b.origin.z
        bx1 = bx0 + b.dims.L
        by1 = by0 + b.dims.W
        bz1 = bz0 + b.dims.H
        x_mins.add(bx0); x_maxs.add(bx1)
        y_mins.add(by0); y_maxs.add(by1)
        z_mins.add(bz0); z_maxs.add(bz1)

    x_mins.add(0); x_maxs.add(container.L)
    y_mins.add(0); y_maxs.add(container.W)
    z_mins.add(0); z_maxs.add(container.H)

    score = 0.0

    def aligned(val: int, plane_set: set[int]) -> bool:
        if tol == 0:
            return val in plane_set
        return any(abs(val - p) <= tol for p in plane_set)

    if aligned(x0, x_maxs): score += 1.0
    if aligned(y0, y_maxs): score += 1.0
    if aligned(z0, z_maxs): score += 1.0

    if aligned(x1, x_mins): score += 0.5
    if aligned(y1, y_mins): score += 0.5
    if aligned(z1, z_mins): score += 0.5

    score += 0.0001 * (-(x0 + y0 + z0))
    return score


def make_mass_fn(item_types):
    types = {t.type_id: t for t in item_types}

    def mass_fn(b: AABB) -> float:
        it = types[b.type_id]
        if getattr(it, "weight", None) is not None:
            return it.weight
        return b.dims.volume()  # BR-Original fallback

    return mass_fn


def balance_score(
    cand: AABB,
    placed: List[AABB],
    container: Dims,
    *,
    mass_fn,
) -> float:
    tmp = placed + [cand]
    cg = compute_cg_metrics(tmp, container, mass_fn=mass_fn)
    return -float(cg.z3)


def _first_phase_split(instances: List, split_ratio: float) -> Tuple[List, List]:
    n = len(instances)
    k = int(round(n * split_ratio))
    k = max(0, min(n, k))
    return instances[:k], instances[k:]


def _type_sequence_from_instances(instances: List) -> List[int]:
    """Preserve first-occurrence order of types."""
    seen = set()
    out = []
    for inst in instances:
        tid = inst.type_id
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def _rotation_dims_for_type(
    *,
    type_id: int,
    rots_by_type: Dict[int, List[Dims]],
    rot_by_type: Optional[Dict[int, int]],
    soft_rotation: bool,
) -> List[Dims]:
    """Return rotation dims list in priority order (GA choice first if provided)."""
    rots = rots_by_type.get(type_id, [])
    if not rots:
        return []
    if rot_by_type is None or type_id not in rot_by_type:
        return rots  # baseline behavior: try all rotations (as your old code)
    idx = rot_by_type[type_id] % len(rots)
    if not soft_rotation:
        return [rots[idx]]
    # soft: try GA-picked first, then others
    return [rots[idx]] + [r for j, r in enumerate(rots) if j != idx]


def _try_build_block(
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
) -> Optional[Tuple[List[AABB], Dims, int]]:
    """
    Moon-style strict cuboid block (nx * ny * nz identical boxes).
    No incomplete columns/layers: block is full grid.

    Returns (block_boxes, block_extent_dims, used_qty) or None.
    """
    if remaining_qty <= 0:
        return None

    # Max by container bounds (width->length->height conceptually)
    max_nx = (container.L - ep.x) // dims.L
    max_ny = (container.W - ep.y) // dims.W
    max_nz = (container.H - ep.z) // dims.H
    if max_nx <= 0 or max_ny <= 0 or max_nz <= 0:
        return None

    # Start from widest/longest/tallest, then shrink until feasible.
    nx, ny, nz = int(max_nx), int(max_ny), int(max_nz)

    # Enforce availability by shrinking height first, then length, then width
    def shrink_for_qty(nx, ny, nz, q):
        while nx * ny * nz > q and nz > 1:
            nz -= 1
        while nx * ny * nz > q and nx > 1:
            nx -= 1
        while nx * ny * nz > q and ny > 1:
            ny -= 1
        # If still too big, force to 1×1×1 if at least 1 available
        if nx * ny * nz > q:
            nx = ny = nz = 1
        return nx, ny, nz

    nx, ny, nz = shrink_for_qty(nx, ny, nz, remaining_qty)

    # Validate/shrink by feasibility (overlap/support). Shrink order: nz -> nx -> ny
    while nx >= 1 and ny >= 1 and nz >= 1:
        used = nx * ny * nz
        if used <= 0:
            return None

        # Build all AABBs for the candidate cuboid
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
                        if not is_supported_fn(cand, placed, support_min_ratio):
                            feasible = False
                            break

                    block_boxes.append(cand)
                if not feasible:
                    break
            if not feasible:
                break

        if feasible:
            extent = Dims(nx * dims.L, ny * dims.W, nz * dims.H)
            return block_boxes, extent, used

        # shrink order: drop height, then length, then width
        if nz > 1:
            nz -= 1
        elif nx > 1:
            nx -= 1
        elif ny > 1:
            ny -= 1
        else:
            break

    return None


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
    box_order_policy: Optional[str] = None,
    box_order_seed: Optional[int] = None,
    # NEW for GA:
    rot_by_type: Optional[Dict[int, int]] = None,   # GA rotation choice per type_id
    soft_rotation: bool = True,
) -> Tuple[List[AABB], List]:
    """
    Two-phase decoder with Moon-style block placement FIRST,
    then leftover singleton placement AFTER block phases finish.

    - If box_order_policy is None: uses instances as-is (GA provides type sequence).
    - rot_by_type: if provided, tries GA rotation first per type; if soft_rotation,
      can fall back to other allowed rotations.

    Returns:
      placed: list of AABBs (boxes, blocks expanded into boxes)
      unplaced: list of instances that were not packed (leftovers after singleton pass)
    """
    types_by_id: Dict[int, any] = {t.type_id: t for t in item_types}
    rots_by_type: Dict[int, List[Dims]] = {
        t.type_id: allowed_rotations(t, rotation_mode)
        for t in item_types
    }
    mass_fn = make_mass_fn(item_types)

    if box_order_policy is not None:
        instances = apply_box_order(
            instances=instances,
            types_by_id=types_by_id,
            rotation_mode=rotation_mode,
            policy=box_order_policy,
            seed=box_order_seed,
        )

    placed: List[AABB] = []
    eps: List[Point] = [Point(0, 0, 0)]

    phaseA_insts, phaseB_insts = _first_phase_split(instances, split_ratio)

    # Remaining inventory per phase, per type
    remA: Dict[int, int] = {}
    for inst in phaseA_insts:
        remA[inst.type_id] = remA.get(inst.type_id, 0) + 1

    remB: Dict[int, int] = {}
    for inst in phaseB_insts:
        remB[inst.type_id] = remB.get(inst.type_id, 0) + 1

    type_seq_A = _type_sequence_from_instances(phaseA_insts)
    type_seq_B = _type_sequence_from_instances(phaseB_insts)

    def place_best_block_for_type(
        *,
        type_id: int,
        remaining_qty: int,
        ep_sort_key,
        score_block_fn,  # function(extent_aabb)->score
    ) -> Optional[int]:
        nonlocal eps, placed

        if remaining_qty <= 0:
            return None

        rot_dims_list = _rotation_dims_for_type(
            type_id=type_id,
            rots_by_type=rots_by_type,
            rot_by_type=rot_by_type,
            soft_rotation=soft_rotation,
        )
        if not rot_dims_list:
            return None

        eps.sort(key=ep_sort_key)

        best_boxes = None
        best_extent = None
        best_ep = None
        best_used = None
        best_score = float("-inf")

        for ep in eps:
            for dims in rot_dims_list:
                built = _try_build_block(
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
                block_boxes, extent_dims, used_qty = built

                # Score block using its bounding cuboid AABB (cheap and consistent)
                extent_aabb = AABB(ep, extent_dims, type_id=type_id)
                s = score_block_fn(extent_aabb)

                if s > best_score:
                    best_score = s
                    best_boxes = block_boxes
                    best_extent = extent_dims
                    best_ep = ep
                    best_used = used_qty

        if best_boxes is None or best_ep is None or best_used is None:
            return None

        # Commit placement
        placed.extend(best_boxes)
        eps.remove(best_ep)
        # Update eps using the block's extent (one update is usually fine)
        # If you find missed opportunities, you can update per each box later.
        eps = update_eps(eps, placed, AABB(best_ep, best_extent, type_id=type_id), container)
        return best_used

    # -------- Block placement phase A (gap/alignment) --------
    for tid in type_seq_A:
        while remA.get(tid, 0) > 0:
            used = place_best_block_for_type(
                type_id=tid,
                remaining_qty=remA[tid],
                ep_sort_key=dblf_key,
                score_block_fn=lambda extent_aabb: gap_score(extent_aabb, placed, container, eps, tol=gap_tol),
            )
            if used is None:
                break
            remA[tid] -= used

    # -------- Block placement phase B (balance) --------
    for tid in type_seq_B:
        while remB.get(tid, 0) > 0:
            used = place_best_block_for_type(
                type_id=tid,
                remaining_qty=remB[tid],
                ep_sort_key=rev_dblf_key,
                score_block_fn=lambda extent_aabb: balance_score(extent_aabb, placed, container, mass_fn=mass_fn),
            )
            if used is None:
                break
            remB[tid] -= used

    # -------- Leftover singleton placement AFTER block phases finish --------
    leftovers: List = []
    # rebuild leftover instance list (we only tracked counts)
    # keep original relative order: phaseA leftovers then phaseB leftovers
    if remA:
        for inst in phaseA_insts:
            if remA.get(inst.type_id, 0) > 0:
                leftovers.append(inst)
                remA[inst.type_id] -= 1
    if remB:
        for inst in phaseB_insts:
            if remB.get(inst.type_id, 0) > 0:
                leftovers.append(inst)
                remB[inst.type_id] -= 1

    unplaced: List = []

    # Original single-box placer, but rotation comes from GA first (soft fallback allowed)
    def place_one_single(inst, ep_sort_key, scorer_fn) -> bool:
        nonlocal eps, placed

        rot_dims_list = _rotation_dims_for_type(
            type_id=inst.type_id,
            rots_by_type=rots_by_type,
            rot_by_type=rot_by_type,
            soft_rotation=soft_rotation,
        )
        if not rot_dims_list:
            return False

        eps.sort(key=ep_sort_key)

        best_cand: Optional[AABB] = None
        best_ep: Optional[Point] = None
        best_score: float = float("-inf")

        for ep in eps:
            for dims in rot_dims_list:
                cand = AABB(ep, dims, type_id=inst.type_id)

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

    # Apply the same 2-phase split logic to leftovers (still after blocks)
    leftA, leftB = _first_phase_split(leftovers, split_ratio)

    for inst in leftA:
        ok = place_one_single(
            inst,
            ep_sort_key=dblf_key,
            scorer_fn=lambda cand: gap_score(cand, placed, container, eps, tol=gap_tol),
        )
        if not ok:
            unplaced.append(inst)

    for inst in leftB:
        ok = place_one_single(
            inst,
            ep_sort_key=rev_dblf_key,
            scorer_fn=lambda cand: balance_score(cand, placed, container, mass_fn=mass_fn),
        )
        if not ok:
            unplaced.append(inst)

    return placed, unplaced



# from __future__ import annotations

# from typing import Dict, List, Tuple, Optional

# from clp.clp.models.geometry import Point, AABB, Dims, inside_container, overlaps
# from clp.clp.polices.rotation import RotationMode, allowed_rotations
# from clp.clp.decoders.extreme_points import update_eps, dblf_key  
# from clp.clp.eval.cg import compute_cg_metrics 
# from clp.clp.polices.box_order import apply_box_order


# def rev_dblf_key(p: Point) -> Tuple[int, int, int]:
#     # reverse DBLF: x desc, z desc, y desc
#     return (-p.x, -p.z, -p.y)


# def gap_score(
#     cand: AABB,
#     placed: List[AABB],
#     container: Dims,
#     eps: List[Point],
#     *,
#     tol: int = 0,
# ) -> float:
#     """
#     Higher is better.

#     Very cheap 'alignment' proxy:
#       +1 if cand.min face aligns with some existing max face (x or y or z)
#       +1 if cand.max face aligns with some existing min face (x or y or z)
#       + small reward for using smaller coordinates (encourage compactness)
#     """
#     x0, y0, z0 = cand.origin.x, cand.origin.y, cand.origin.z
#     x1 = x0 + cand.dims.L
#     y1 = y0 + cand.dims.W
#     z1 = z0 + cand.dims.H

#     # Collect planes from placed boxes
#     x_mins = set()
#     x_maxs = set()
#     y_mins = set()
#     y_maxs = set()
#     z_mins = set()
#     z_maxs = set()

#     for b in placed:
#         bx0, by0, bz0 = b.origin.x, b.origin.y, b.origin.z
#         bx1 = bx0 + b.dims.L
#         by1 = by0 + b.dims.W
#         bz1 = bz0 + b.dims.H
#         x_mins.add(bx0); x_maxs.add(bx1)
#         y_mins.add(by0); y_maxs.add(by1)
#         z_mins.add(bz0); z_maxs.add(bz1)

#     # Container walls count as planes too (helps compact packing)
#     x_mins.add(0); x_maxs.add(container.L)
#     y_mins.add(0); y_maxs.add(container.W)
#     z_mins.add(0); z_maxs.add(container.H)

#     score = 0.0

#     def aligned(val: int, plane_set: set[int]) -> bool:
#         if tol == 0:
#             return val in plane_set
#         return any(abs(val - p) <= tol for p in plane_set)

#     # Reward touching/aligning planes (reduces fragmentation)
#     if aligned(x0, x_maxs): score += 1.0
#     if aligned(y0, y_maxs): score += 1.0
#     if aligned(z0, z_maxs): score += 1.0

#     if aligned(x1, x_mins): score += 0.5
#     if aligned(y1, y_mins): score += 0.5
#     if aligned(z1, z_mins): score += 0.5

#     # Small compactness reward (prefer smaller coordinates)
#     score += 0.0001 * (-(x0 + y0 + z0))

#     return score

# def make_mass_fn(item_types):
#     types = {t.type_id: t for t in item_types}

#     def mass_fn(b: AABB) -> float:
#         it = types[b.type_id]
#         if it.weight is not None:
#             return it.weight
#         return b.dims.volume()  # BR-Original fallback

#     return mass_fn

# def balance_score(
#     cand: AABB,
#     placed: List[AABB],
#     container: Dims,
#     *,
#     mass_fn,
# ) -> float:
#     """
#     Higher is better. We maximize improvement (i.e., negative Z3).
#     Computes CG metrics on (placed + cand) and returns -z3.
#     """
#     tmp = placed + [cand]
#     cg = compute_cg_metrics(tmp, container, mass_fn=mass_fn)
#     return -float(cg.z3)



# def _first_phase_split(instances: List, split_ratio: float) -> Tuple[List, List]:
#     n = len(instances)
#     k = int(round(n * split_ratio))
#     k = max(0, min(n, k))
#     return instances[:k], instances[k:]





# def decode_two_phase(
#     *,
#     container: Dims,
#     item_types: List,
#     instances: List,
#     rotation_mode: RotationMode,
#     split_ratio: float = 0.7,
#     support_required: bool = True,
#     support_min_ratio: float = 1.0,
#     is_supported_fn=None,
#     gap_tol: int = 0,
#     # NEW:
#     box_order_policy: Optional[str] = None,   # e.g. "volume_then_maxface"
#     box_order_seed: Optional[int] = None,     # used by "random_tiebreak"
# ) -> Tuple[List[AABB], List]:
#     """
#     Two-phase greedy decoder with optional box-ranking:

#     If box_order_policy is provided, instances are sorted once BEFORE decoding.
#     If box_order_policy is None, instances order is used as-is (GA will provide it).
#     """
#     types_by_id: Dict[int, any] = {t.type_id: t for t in item_types}
#     rots_by_type: Dict[int, List[Dims]] = {
#         t.type_id: allowed_rotations(t, rotation_mode)
#         for t in item_types
#     }
    
#     mass_fn = make_mass_fn(item_types)


#     # ---- NEW: apply ranking once (if requested) ----
#     if box_order_policy is not None:
#         instances = apply_box_order(
#             instances=instances,
#             types_by_id=types_by_id,
#             rotation_mode=rotation_mode,
#             policy=box_order_policy,
#             seed=box_order_seed,
#         )

#     placed: List[AABB] = []
#     unplaced: List = []
#     eps: List[Point] = [Point(0, 0, 0)]

#     phaseA, phaseB = _first_phase_split(instances, split_ratio)

#     def place_one(inst, ep_sort_key, scorer_fn) -> bool:
#         nonlocal eps, placed

#         rots = rots_by_type.get(inst.type_id, [])
#         if not rots:
#             return False

#         eps.sort(key=ep_sort_key)

#         best_cand: Optional[AABB] = None
#         best_ep: Optional[Point] = None
#         best_score: float = float("-inf")

#         for ep in eps:
#             for dims in rots:
#                 # cand = AABB(ep, dims)
#                 cand = AABB(ep, dims, type_id=inst.type_id)


#                 if not inside_container(cand, container):
#                     continue
#                 if any(overlaps(cand, b) for b in placed):
#                     continue

#                 if support_required and cand.origin.z > 0:
#                     if is_supported_fn is None:
#                         raise RuntimeError("support_required=True but is_supported_fn not provided")
#                     if not is_supported_fn(cand, placed, support_min_ratio):
#                         continue

#                 s = scorer_fn(cand)
#                 if s > best_score:
#                     best_score = s
#                     best_cand = cand
#                     best_ep = ep

#         if best_cand is None or best_ep is None:
#             return False

#         placed.append(best_cand)
#         eps.remove(best_ep)
#         eps = update_eps(eps, placed, best_cand, container)
#         return True

#     # Phase A: gap/alignment
#     for inst in phaseA:
#         ok = place_one(
#             inst,
#             ep_sort_key=dblf_key,
#             scorer_fn=lambda cand: gap_score(cand, placed, container, eps, tol=gap_tol),
#         )
#         if not ok:
#             unplaced.append(inst)

#     # Phase B: balance
#     for inst in phaseB:
#         ok = place_one(
#             inst,
#             ep_sort_key=rev_dblf_key,
#             scorer_fn=lambda cand: balance_score(cand, placed, container, mass_fn=mass_fn),
#         )
#         if not ok:
#             unplaced.append(inst)

#     return placed, unplaced


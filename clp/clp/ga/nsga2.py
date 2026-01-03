# clp/clp/ga/nsga2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import random
from time import perf_counter
from clp.clp.viz.debug_viz import plot_container_debug
from clp.clp.eval.cg import compute_cg_metrics
from clp.clp.eval.ULO_count import compute_ulo_count_from_placed
from clp.clp.models.geometry import Dims
from clp.clp.decoders.two_phase import decode_two_phase
from clp.clp.polices.rotation import RotationMode

from clp.clp.ga.population import (
    init_population_groups,
    build_groups,
    expand_chromosome,
    GroupChromosome,
)
from clp.configurations import PLOT_POP_EVALUATION, debug, is_ULO

# ============================================================
# Evaluated individual record (keeps EVERYTHING needed for JSON)
# ============================================================
@dataclass
class EvalRec:
    chrom: GroupChromosome
    order: List[int]          # expanded instance indices

    # Objectives
    z1: float                 # maximize (volume utilization)
    z2: float                 # minimize (ULO count)
    z3: float                 # minimize (CG penalty / z3)

    placed: list              # List[AABB]
    unplaced: list            # List[ItemInstance] (or whatever your type is)
    cg: Any                   # CGMetrics
    elapsed_sec: float

    # NSGA-II fields
    rank: int = 10**9
    crowding: float = 0.0

    def __repr__(self) -> str:
        cg_str = "None"
        if self.cg is not None:
            cg_str = (
                f"(dx={self.cg.rd_x_pct:.4f}%, "
                f"dy={self.cg.rd_y_pct:.4f}%, "
                f"dz={self.cg.rd_z_pct:.4f}%)"
            )

        return (
            "EvalRec("
            f"Z1={self.z1:.4f}, "
            f"Z2={self.z2:.0f}, "
            f"Z3={self.z3:.4f}, "
            f"CG={cg_str}, "
            f"placed={len(self.placed)}, "
            f"unplaced={len(self.unplaced)}, "
            f"rank={self.rank}, "
            f"crowding={self.crowding:.3f}"
            ")"
        )

# ============================================================
# Objective helpers
# ============================================================
def volume_utilization(placed, container: Dims) -> float:
    vol_loaded = sum(b.dims.L * b.dims.W * b.dims.H for b in placed)
    vol_container = container.L * container.W * container.H
    return (vol_loaded / vol_container) if vol_container > 0 else 0.0

# ============================================================
EPS_Z1 = 1e-4
EPS_Z2 = 1.0   # ULO is a count; keep 1.0 unless you want smoothing
EPS_Z3 = 1e-4

def dominates(a, b, *, is_ULO: bool) -> bool:
    # quantize to reduce noise / ties
    a_z1 = round(a.z1 / EPS_Z1) * EPS_Z1
    b_z1 = round(b.z1 / EPS_Z1) * EPS_Z1

    a_z3 = round(a.z3 / EPS_Z3) * EPS_Z3
    b_z3 = round(b.z3 / EPS_Z3) * EPS_Z3

    if is_ULO:
        a_z2 = round(a.z2 / EPS_Z2) * EPS_Z2
        b_z2 = round(b.z2 / EPS_Z2) * EPS_Z2

        # Z1 maximize, Z2 minimize, Z3 minimize
        better_or_equal = (a_z1 >= b_z1) and (a_z2 <= b_z2) and (a_z3 <= b_z3)
        strictly_better = (a_z1 > b_z1) or (a_z2 < b_z2) or (a_z3 < b_z3)
        return better_or_equal and strictly_better

    # 2 objectives: Z1 maximize, Z3 minimize
    better_or_equal = (a_z1 >= b_z1) and (a_z3 <= b_z3)
    strictly_better = (a_z1 > b_z1) or (a_z3 < b_z3)
    return better_or_equal and strictly_better


# ============================================================
# NSGA-II: fast non-dominated sorting
# ============================================================
def fast_nondominated_sort(pop: List[EvalRec], *, is_ULO: bool) -> List[List[EvalRec]]:
    S: Dict[int, List[int]] = {}
    n: Dict[int, int] = {}
    fronts: List[List[int]] = []

    for p_idx, p in enumerate(pop):
        S[p_idx] = []
        n[p_idx] = 0
        for q_idx, q in enumerate(pop):
            if p_idx == q_idx:
                continue
            if dominates(p, q, is_ULO=is_ULO):
                S[p_idx].append(q_idx)
            elif dominates(q, p, is_ULO=is_ULO):
                n[p_idx] += 1

        if n[p_idx] == 0:
            p.rank = 0

    front0 = [i for i in range(len(pop)) if n[i] == 0]
    fronts.append(front0)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    pop[q_idx].rank = i + 1
                    next_front.append(q_idx)
        i += 1
        if next_front:
            fronts.append(next_front)

    return [[pop[idx] for idx in f] for f in fronts if f]


# ============================================================
def crowding_distance(front: List[EvalRec], *, is_ULO: bool) -> None:
    if not front:
        return

    for p in front:
        p.crowding = 0.0

    def add_obj_dist(key_fn):
        order = sorted(front, key=key_fn)
        order[0].crowding = float("inf")
        order[-1].crowding = float("inf")

        zmin = key_fn(order[0])
        zmax = key_fn(order[-1])
        denom = (zmax - zmin) if (zmax - zmin) != 0 else 1.0

        for i in range(1, len(order) - 1):
            if order[i].crowding != float("inf"):
                order[i].crowding += (key_fn(order[i + 1]) - key_fn(order[i - 1])) / denom

    # Z1 (max) -> distance calculation same
    add_obj_dist(lambda r: r.z1)

    # Z3 (min)
    add_obj_dist(lambda r: r.z3)

    # Z2 (ULO min) only if enabled
    if is_ULO:
        add_obj_dist(lambda r: r.z2)



# ============================================================
# Selection: tournament by (rank asc, crowding desc)
# ============================================================
def tournament_select(pop: List[EvalRec], rng: random.Random, k: int = 2) -> EvalRec:
    cand = rng.sample(pop, k)
    cand.sort(key=lambda r: (r.rank, -r.crowding))
    return cand[0]


# ============================================================
# Operators on GroupChromosome (Path-A)
# ============================================================
def ox_crossover(
    seq1: List[Tuple],
    seq2: List[Tuple],
    rng: random.Random,
) -> Tuple[List[Tuple], List[Tuple]]:
    n = len(seq1)
    if n < 2:
        return seq1[:], seq2[:]

    a, b = sorted(rng.sample(range(n), 2))

    def ox(p1, p2):
        child = [None] * n
        child[a:b + 1] = p1[a:b + 1]
        fill = [g for g in p2 if g not in child]
        j = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[j]
                j += 1
        return child

    return ox(seq1, seq2), ox(seq2, seq1)



def mutate_rot_map_two_step(
    chrom: GroupChromosome,
    rng: random.Random,
    *,
    p_swap: float = 0.6,
    p_reset: float = 0.3,
) -> GroupChromosome:
    """
    Two-step mutation:
      1) With prob p_swap: swap rotation indices between two random groups.
      2) For each group with prob p_reset: set to a random allowed rotation for its type.
    """
    out: Dict[GroupKey, int] = dict(chrom.rot_map)
    keys = list(out.keys())
    if not keys:
        return chrom
    
    if debug:
        print("\n[MUTATION] before:", out)

    # =========================
    # Step 1: SWAP mutation
    # =========================
    if len(keys) >= 2 and random.random() < p_swap:
        a, b = random.sample(keys, 2)

        if debug:
            print(
                f"[SWAP] groups {a} <-> {b} | "
                f"rot {out[a]} <-> {out[b]}"
            )

        out[a], out[b] = out[b], out[a]

        # clamp after swap
        for gk in (a, b):
            k = _allowed_rot_count(chrom, gk)
            before = out[gk]
            out[gk] = int(out[gk]) % k
            if debug and before != out[gk]:
                print(
                    f"[SWAP-CLAMP] group {gk} | "
                    f"{before} -> {out[gk]} (k={k})"
                )

    # =========================
    # Step 2: RESET mutation
    # =========================
    for gk in keys:
        if random.random() < p_reset:
            k = _allowed_rot_count(chrom, gk)
            old = out[gk]
            new = random.randrange(0, k)
            out[gk] = new

            if debug:
                print(
                    f"[RESET] group {gk} | "
                    f"{old} -> {new} (k={k})"
                )

    if debug:
        print("[MUTATION] after :", out)

    return GroupChromosome(
        group_seq=chrom.group_seq,
        rot_map=out,
        rots_by_type=chrom.rots_by_type,
    )


# ============================================================
# Evaluation (decoder-driven, returns full artifacts)
# ============================================================
BOX_ORDER_POLICY = "volume_then_maxface"
def evaluate_chromosome(
    *,
    chrom: GroupChromosome,
    groups: Dict,
    instances: List,
    item_types: List,
    container: Dims,
    rotation_mode: RotationMode,
    rng: random.Random,
    shuffle_within_group: bool,
    split_ratio: float,
    support_required: bool,
    support_min_ratio: float,
    is_supported_fn,
    soft_rotation: bool,
) -> EvalRec:
    order = expand_chromosome(
        chrom,
        groups,
        rng=rng,
        shuffle_within_group=shuffle_within_group,
    )
    inst_ordered = [instances[i] for i in order]

    # Build mapping once (still better to do outside eval, but this works)
    type_to_pi: Dict[int, int] = {}
    for inst in inst_ordered:
        cid = getattr(inst, "customer_id", None)
        if cid is None:
            continue
        type_to_pi.setdefault(int(inst.type_id), int(cid))

    t0 = perf_counter()
    placed, unplaced = decode_two_phase(
        container=container,
        item_types=item_types,
        instances=inst_ordered,
        rotation_mode=rotation_mode,
        box_order_policy=BOX_ORDER_POLICY,
        box_order_seed=None,
        split_ratio=split_ratio,
        support_required=support_required,
        support_min_ratio=float(support_min_ratio),
        is_supported_fn=is_supported_fn,
        rot_by_group=chrom.rot_map,
        soft_rotation=soft_rotation,
    )
    elapsed = perf_counter() - t0

    if PLOT_POP_EVALUATION:
        plot_container_debug(
            placed=placed,
            container_dims=(container.L, container.W, container.H),
            title="NSGA-II Type1 Decoded Layout",
            show_box_labels=True,
        )


    ulo = float(compute_ulo_count_from_placed(placed, type_to_pi=type_to_pi)) if is_ULO else 0.0
    cg = compute_cg_metrics(placed, container)
    z1 = float(volume_utilization(placed, container))
    z3 = float(cg.z3)

    res = EvalRec(
        chrom=chrom,
        order=order,
        z1=z1,
        z2=ulo,
        z3=z3,
        placed=placed,
        unplaced=unplaced,
        cg=cg,
        elapsed_sec=float(elapsed),
    )

    print(f" Evaluated: {res}")
    return res

# ============================================================
# Main loop: bi-objective NSGA-II (Type-1: Z1 max, Z3 min)
# ============================================================
def _allowed_rot_count(chrom: GroupChromosome, gk: GroupKey) -> int:
    _, type_id = gk
    rots = chrom.rots_by_type[type_id]
    if not rots:
        raise ValueError(f"No allowed rotations for type_id={type_id}")
    return len(rots)


def run_nsga2_type1(
    *,
    instances: List,
    item_types: List,
    container: Dims,
    rotation_mode: RotationMode,
    pop_size: int,
    generations: int,
    seed: int = 0,
    split_ratio: float = 0.9,
    support_required: bool = True,
    support_min_ratio: float = 0.8,
    is_supported_fn=None,
    soft_rotation: bool = True,
    p_cx: float = 0.8,
    p_mut_seq: float = 0.6,
    p_mut_rot: float = 0.3,
    shuffle_within_group: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      population: final evaluated population (EvalRec list)
      front0: final non-dominated front (EvalRec list)
      progress: per-generation summary list
    """
    if is_supported_fn is None:
        raise RuntimeError("run_nsga2_type1 requires is_supported_fn")

    rng = random.Random(seed)

    # groups = build_groups(instances)

    pop_chrom,groups = init_population_groups(
        instances=instances,
        item_types=item_types,
        rotation_mode=rotation_mode,
        pop_size=pop_size,
        rng_seed=seed,
    )

    eval_pop = [
        evaluate_chromosome(
            chrom=c,
            groups=groups,
            instances=instances,
            item_types=item_types,
            container=container,
            rotation_mode=rotation_mode,
            rng=rng,
            shuffle_within_group=shuffle_within_group,
            split_ratio=split_ratio,
            support_required=support_required,
            support_min_ratio=support_min_ratio,
            is_supported_fn=is_supported_fn,
            soft_rotation=soft_rotation,
        )
        for c in pop_chrom
    ]

    progress: List[Dict[str, Any]] = []

    for gen in range(generations):
        print(f"🧬 NSGA-II Type1 Generation {gen}...")
        fronts = fast_nondominated_sort(eval_pop, is_ULO=is_ULO)
        for f in fronts:
            crowding_distance(f, is_ULO=is_ULO)


        best_z1 = max(r.z1 for r in eval_pop)
        best_z2 = min(r.z2 for r in eval_pop) if is_ULO else 0.0
        best_z3 = min(r.z3 for r in eval_pop)
        mean_t = sum(r.elapsed_sec for r in eval_pop) / max(1, len(eval_pop))
        progress.append({
            "gen": gen,
            "best_z1": best_z1,
            "best_z2": best_z2 if is_ULO else 0.0,
            "best_z3": best_z3,
            "mean_eval_sec": mean_t,
            "front0_size": len(fronts[0]) if fronts else 0,
        })

        
        
        
        offspring: List[GroupChromosome] = []
        while len(offspring) < pop_size:
            p1 = tournament_select(eval_pop, rng)
            p2 = tournament_select(eval_pop, rng)

            c1_seq = p1.chrom.group_seq[:]
            c2_seq = p2.chrom.group_seq[:]

            c1_rot = dict(p1.chrom.rot_map)
            c2_rot = dict(p2.chrom.rot_map)

            if rng.random() < p_cx:
                c1_seq, c2_seq = ox_crossover(p1.chrom.group_seq, p2.chrom.group_seq, rng)

            # build children with rots_by_type attached
            child1 = GroupChromosome(
                group_seq=c1_seq,
                rot_map=c1_rot,
                rots_by_type=p1.chrom.rots_by_type,
            )
            child2 = GroupChromosome(
                group_seq=c2_seq,
                rot_map=c2_rot,
                rots_by_type=p2.chrom.rots_by_type,
            )

            # mutate (now mutation has access to allowed rotation counts)
            child1 = mutate_rot_map_two_step(child1, rng, p_swap=p_mut_seq, p_reset=p_mut_rot)
            child2 = mutate_rot_map_two_step(child2, rng, p_swap=p_mut_seq, p_reset=p_mut_rot)

            offspring.append(child1)
            if len(offspring) < pop_size:
                offspring.append(child2)

        
        eval_off = [
            evaluate_chromosome(
                chrom=c,
                groups=groups,
                instances=instances,
                item_types=item_types,
                container=container,
                rotation_mode=rotation_mode,
                rng=rng,
                shuffle_within_group=shuffle_within_group,
                split_ratio=split_ratio,
                support_required=support_required,
                support_min_ratio=support_min_ratio,
                is_supported_fn=is_supported_fn,
                soft_rotation=soft_rotation,
            )
            for c in offspring
        ]

        # Environmental selection
        combined = eval_pop + eval_off
        fronts = fast_nondominated_sort(combined, is_ULO=is_ULO)

        next_pop: List[EvalRec] = []
        for f in fronts:
            crowding_distance(f, is_ULO=is_ULO)
            if len(next_pop) + len(f) <= pop_size:
                next_pop.extend(f)
            else:
                f.sort(key=lambda r: (-r.crowding))
                next_pop.extend(f[: pop_size - len(next_pop)])
                break

        eval_pop = next_pop

    # final front
    fronts = fast_nondominated_sort(eval_pop, is_ULO=is_ULO)
    for f in fronts:
        crowding_distance(f, is_ULO=is_ULO)

    def get_front(fronts, k):
        return fronts[k] if k < len(fronts) else []

    return {
        "population": eval_pop,
        "front0": get_front(fronts, 0),
        "front1": get_front(fronts, 1),
        "progress": progress,
    }
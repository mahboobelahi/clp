# 
# clp/clp/ga/population.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Sequence, Tuple, Optional, Hashable
import random
from types import MappingProxyType
from clp.clp.polices.rotation import RotationMode, allowed_rotations


# ----------------------------
# Group key (generic)
# ----------------------------
GroupKey = Tuple[Hashable, ...]  # e.g. (type_id,) or (customer_id, type_id)


def default_group_key(inst) -> GroupKey:
    """
    Generic grouping:
      - If instances have customer_id -> group by (customer_id, type_id)
      - Else -> group by (type_id,)
    Works for normal BR (customer_id None) and modified BR.
    """
    cid = getattr(inst, "customer_id", None)
    tid = inst.type_id
    if cid is None:
        return (tid,)
    return (cid, tid)


@dataclass(frozen=True)
class GroupChromosome:
    """
    Path-A genome:
      - group_seq: permutation of groups
      - rot_map: one rotation index per group (index into allowed_rotations(type))
    """
    group_seq: List[GroupKey]
    rot_map: Dict[GroupKey, int]
    rots_by_type: Mapping[int, List[Dims]]

    def __post_init__(self) -> None:
            # Freeze rot_map so crossover/mutation can't accidentally mutate parents.
            object.__setattr__(self, "rot_map", MappingProxyType(dict(self.rot_map)))
            # Freeze rots_by_type mapping too (contents lists still mutable, but mapping not)
            object.__setattr__(self, "rots_by_type", MappingProxyType(dict(self.rots_by_type)))


def build_rots_by_type(item_types, rotation_mode: RotationMode) -> Dict[int, List]:
    return {t.type_id: allowed_rotations(t, rotation_mode) for t in item_types}


def build_groups(
    instances: Sequence,
    group_key_fn=default_group_key,
) -> Dict[GroupKey, List[int]]:
    """
    Returns mapping: group_key -> list of instance indices
    """
    groups: Dict[GroupKey, List[int]] = {}
    for idx, inst in enumerate(instances):
        gk = group_key_fn(inst)
        groups.setdefault(gk, []).append(idx)
    return groups


# def expand_chromosome(
#     chrom: GroupChromosome,
#     groups: Dict[GroupKey, List[int]],
#     instances: Optional[Sequence] = None,
#     rng: Optional[random.Random] = None,
#     shuffle_within_group: bool = True,
# ) -> List[int]:
#     """
#     Convert group chromosome to an item-level order (list of instance indices).
#     Keeps groups contiguous, optionally shuffles within each group for diversity.
#     If instances is provided, updates each instance.rotation_pref with the group's rot_map value.
#     """
#     order: List[int] = []
#     for gk in chrom.group_seq:
#         block = list(groups[gk])
#         if shuffle_within_group and rng is not None and len(block) > 1:
#             rng.shuffle(block)
#         if instances is not None:
#             rot_idx = chrom.rot_map[gk]
#             for idx in block:
#                 instances[idx].rotation_pref = rot_idx
#         order.extend(block)
#     return order

def expand_chromosome(
    chrom: GroupChromosome,
    groups: Dict[GroupKey, List[int]],
    instances: Sequence,
    rng: Optional[random.Random] = None,
    shuffle_within_group: bool = True,
) -> List:
    """
    Returns a NEW ordered list of ItemInstance with rotation_pref set.
    Does not mutate original frozen dataclass instances.
    """
    inst_ordered: List = []

    for gk in chrom.group_seq:
        block = list(groups[gk])
        if shuffle_within_group and rng is not None and len(block) > 1:
            rng.shuffle(block)

        rot_idx = int(chrom.rot_map[gk])
        for idx in block:
            inst_ordered.append(replace(instances[idx], rotation_pref=rot_idx))

    return inst_ordered


def init_population_groups(
    *,
    instances: Sequence,
    item_types: Sequence,
    rotation_mode: RotationMode,
    pop_size: int,
    rng_seed: int = 0,
    group_key_fn=default_group_key,
) -> Tuple[List[GroupChromosome], Dict[GroupKey, List[int]]]:
    """
    generalized initialization:
      - all items in the same group are adjacent
      - all items in the same group share the same rotation index
    Returns:
      (population, groups)
    """
    rng = random.Random(rng_seed)

    rots_by_type = build_rots_by_type(item_types, rotation_mode)
    for tid, rots in rots_by_type.items():
        if not rots:
            raise ValueError(f"No allowed rotations for type_id={tid} under mode={rotation_mode}")

    groups: Dict[GroupKey, List[int]] = build_groups(instances, group_key_fn=group_key_fn)
    group_keys: List[GroupKey] = list(groups.keys())

    population: List[GroupChromosome] = []

    for _ in range(pop_size):
        seq = list(group_keys)
        rng.shuffle(seq)

        rot_map: Dict[GroupKey, int] = {}
        for gk in seq:
            # Assumption: GroupKey ends with type_id, e.g., (customer_id, type_id)
            tid = gk[-1]
            if tid not in rots_by_type:
                raise KeyError(f"type_id={tid} from group key {gk} not found in rots_by_type")
            rot_map[gk] = rng.randrange(0, len(rots_by_type[tid]))

        population.append(
            GroupChromosome(
                group_seq=seq,
                rot_map=rot_map,
                rots_by_type=rots_by_type,
            )
        )

    return population, groups

# def init_population_groups(
#     *,
#     instances: Sequence,
#     item_types: Sequence,
#     rotation_mode: RotationMode,
#     pop_size: int,
#     rng_seed: int = 0,
#     group_key_fn=default_group_key,
# ) -> List[GroupChromosome]:
#     """
#     generalized initialization:
#       - all items in the same group are adjacent
#       - all items in the same group share the same rotation index
#     """
#     rng = random.Random(rng_seed)

#     rots_by_type = build_rots_by_type(item_types, rotation_mode)
#     for tid, rots in rots_by_type.items():
#         if not rots:
#             raise ValueError(f"No allowed rotations for type_id={tid} under mode={rotation_mode}")

#     groups = build_groups(instances, group_key_fn=group_key_fn)
#     group_keys = list(groups.keys())

#     population: List[GroupChromosome] = []

#     for _ in range(pop_size):
#         seq = list(group_keys)
#         rng.shuffle(seq)

#         rot_map: Dict[GroupKey, int] = {}
#         for gk in seq:
#             # type_id is always the last element in our default key
#             tid = int(gk[-1])
#             rot_map[gk] = rng.randrange(len(rots_by_type[tid]))

#         population.append(GroupChromosome(group_seq=seq, rot_map=rot_map, 
#                                           rots_by_type=rots_by_type))

#     return (population,groups)



def test_rot_constraints(
    pop: List,
    *,
    instances,
    item_types,
    rotation_mode: RotationMode,
    limit: int = 10,          # print only first N chromosomes
    do_assert: bool = False,  # set True for C1 hard check
):
    types_by_id = {t.type_id: t for t in item_types}

    def group_key(inst):
        cid = getattr(inst, "customer_id", None)
        tid = inst.type_id
        return (tid,) if cid is None else (cid, tid)

    def any_instance_for_group(g):
        for idx, inst in enumerate(instances):
            if group_key(inst) == g:
                return idx
        raise RuntimeError(f"No instance found for group {g}")

    def assert_c1(inst_type, dims):
        # C1 means: which original dimension is allowed to be vertical (== dims.H)
        base = inst_type.base_dims
        if dims.H == base.H:
            assert inst_type.c1_height == 1
        elif dims.H == base.L:
            assert inst_type.c1_length == 1
        elif dims.H == base.W:
            assert inst_type.c1_width == 1
        else:
            raise AssertionError("dims.H does not match any base dimension (should never happen)")

    print(f"\n=== Rotation debug: mode={rotation_mode}, pop={len(pop)} (printing {min(limit,len(pop))}) ===")

    for i, chrom in enumerate(pop[:limit]):
        print(f"\nChromosome {i:02d}")
        print("  Group sequence:", chrom.group_seq)

        print("  Rotations per group:")
        for g in chrom.group_seq:  # use group_seq order for readability
            ridx = chrom.rot_map[g]
            inst_idx = any_instance_for_group(g)
            inst = instances[inst_idx]
            it = types_by_id[inst.type_id]

            rots = allowed_rotations(it, rotation_mode)
            if not rots:
                raise RuntimeError(f"type_id={inst.type_id} has 0 allowed rotations under mode={rotation_mode}")

            ridx = ridx % len(rots)
            dims = rots[ridx]

            if do_assert and rotation_mode == RotationMode.RESPECT_C1:
                assert_c1(it, dims)

            print(f"    group {g}: rot_idx={ridx} -> dims=({dims.L},{dims.W},{dims.H})")

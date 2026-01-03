#clp\clp\ga\population.py
from clp.clp.polices.rotation import RotationMode, allowed_rotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import random

@dataclass(frozen=True)
class Individual:
    order: List[int]    # indices into `instances`
    rot_idx: List[int]  # per-gene rotation index into rots_by_type[inst.type_id]

def build_rots_by_type(item_types, rotation_mode):
    return {t.type_id: allowed_rotations(t, rotation_mode) for t in item_types}


def sample_rot_indices_for_order(order, instances, rots_by_type, rng: random.Random) -> List[int]:
    """
    For each instance in `order`, sample a valid rotation index for its type.
    """
    out = []
    for inst_idx in order:
        inst = instances[inst_idx]
        rots = rots_by_type[inst.type_id]
        out.append(rng.randrange(len(rots)))
    return out



def volume_key(inst, types_by_id):
    d = types_by_id[inst.type_id].base_dims
    return d.L * d.W * d.H

def max_face_key(inst, types_by_id):
    d = types_by_id[inst.type_id].base_dims
    faces = [d.L*d.W, d.L*d.H, d.W*d.H]
    return max(faces)

def make_type1_orderings(instances, item_types):
    types_by_id = {t.type_id: t for t in item_types}

    policies = {
        "volume_desc": lambda i: (-volume_key(instances[i], types_by_id),),
        "maxface_desc": lambda i: (-max_face_key(instances[i], types_by_id),),
        "volume_then_maxface": lambda i: (-volume_key(instances[i], types_by_id),
                                          -max_face_key(instances[i], types_by_id)),
        "random": None,
    }
    return policies, types_by_id


def init_population(
    *,
    instances: Sequence,
    item_types: Sequence,
    rotation_mode: RotationMode,
    pop_size: int,
    rng_seed: int = 0,
) -> List[Individual]:
    rng = random.Random(rng_seed)

    # Allowed rotations per TYPE
    rots_by_type: Dict[int, List] = {
        t.type_id: allowed_rotations(t, rotation_mode) for t in item_types
    }
    for tid, rots in rots_by_type.items():
        if not rots:
            raise ValueError(f"No allowed rotations for type_id={tid} under mode={rotation_mode}")

    # Group instance indices by type_id
    by_type: Dict[int, List[int]] = {}
    for idx, inst in enumerate(instances):
        by_type.setdefault(inst.type_id, []).append(idx)

    type_ids = list(by_type.keys())
    population: List[Individual] = []

    for _ in range(pop_size):
        # 1) random type sequence
        type_seq = type_ids[:]
        rng.shuffle(type_seq)

        # 2) concatenated type blocks (adjacent)
        order: List[int] = []
        for tid in type_seq:
            block = by_type[tid][:]
            rng.shuffle(block)  # diversity within block, still adjacent
            order.extend(block)

        # 3) one rotation per type
        rot_per_type = {tid: rng.randrange(len(rots_by_type[tid])) for tid in type_seq}
        rot_idx = [rot_per_type[instances[i].type_id] for i in order]

        population.append(Individual(order=order, rot_idx=rot_idx))

    return population


# def init_population_type1(
#     *,
#     instances: Sequence,
#     item_types: Sequence,
#     rotation_mode: RotationMode,
#     pop_size: int,
#     seed: int = 0,
#     seed_policies: Optional[List[str]] = None,   # e.g. ["volume_then_maxface", "maxface_desc", ...]
# ) -> List[Individual]:
#     rng = random.Random(seed)

#     rots_by_type = build_rots_by_type(item_types, rotation_mode)
#     policies, _types_by_id = make_type1_orderings(instances, item_types)

#     if seed_policies is None:
#         # sensible defaults (geometry-only)
#         seed_policies = ["volume_then_maxface", "volume_desc", "maxface_desc"]

#     population: List[Individual] = []
#     n = len(instances)
#     base_indices = list(range(n))

#     # 1) Strategy seeds
#     for p in seed_policies:
#         if len(population) >= pop_size:
#             break
#         key_fn = policies.get(p)
#         if key_fn is None:
#             continue
#         order = sorted(base_indices, key=key_fn)
#         rot_idx = sample_rot_indices_for_order(order, instances, rots_by_type, rng)
#         population.append(Individual(order=order, rot_idx=rot_idx))

#     # 2) Random fill
#     while len(population) < pop_size:
#         order = base_indices[:]
#         rng.shuffle(order)
#         rot_idx = sample_rot_indices_for_order(order, instances, rots_by_type, rng)
#         population.append(Individual(order=order, rot_idx=rot_idx))

#     return population




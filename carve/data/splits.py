"""Source-unit-disjoint split assignment with duplicate grouping.

A split unit is a source video for TAD and SurveillanceCrash and an accident
event for CADP, so a unit (and both paired CADP clips it yields) never
crosses splits. Reposted or near-duplicate uploads are grouped by perceptual
hash before splitting and always land in the same split. Split ratios are
supplied per dataset by the caller.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np

HashValue = Union[str, int]


def _as_int(value: HashValue) -> int:
    return int(value, 16) if isinstance(value, str) else int(value)


def hamming_distance(a: HashValue, b: HashValue) -> int:
    """Bit-level Hamming distance between two perceptual hashes."""
    return bin(_as_int(a) ^ _as_int(b)).count("1")


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def group_by_hash(
    hashes: Mapping[str, HashValue], max_hamming: int = 0
) -> list[list[str]]:
    """Group unit ids whose precomputed perceptual hashes (near-)collide.

    Hashes are supplied by the caller (hex strings or ints). With
    ``max_hamming=0`` grouping is exact-bucket; otherwise a pairwise pass over
    unique hash values links near-duplicates, which is quadratic in the number
    of unique hashes. Output is deterministic: members sorted within groups,
    groups sorted by first member.
    """
    ids = sorted(hashes)
    if max_hamming <= 0:
        buckets: dict[int, list[str]] = {}
        for unit_id in ids:
            buckets.setdefault(_as_int(hashes[unit_id]), []).append(unit_id)
        groups = list(buckets.values())
    else:
        values = [_as_int(hashes[unit_id]) for unit_id in ids]
        uf = _UnionFind(len(ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if bin(values[i] ^ values[j]).count("1") <= max_hamming:
                    uf.union(i, j)
        merged: dict[int, list[str]] = {}
        for i, unit_id in enumerate(ids):
            merged.setdefault(uf.find(i), []).append(unit_id)
        groups = list(merged.values())
    return sorted((sorted(g) for g in groups), key=lambda g: g[0])


def assign_splits(
    units: Sequence[str],
    ratios: Mapping[str, float],
    rng: np.random.Generator,
    duplicate_groups: Optional[Sequence[Sequence[str]]] = None,
    weights: Optional[Mapping[str, int]] = None,
) -> dict[str, str]:
    """Assign source units to splits, keeping duplicate groups together.

    ``ratios`` maps split names to fractions summing to one; ``weights``
    optionally gives the clip count each unit contributes (default 1).
    Groups are shuffled with ``rng`` and placed largest-first into the split
    with the largest remaining deficit, so realized fractions track the
    requested ratios at group granularity. Deterministic given ``rng``.
    """
    total_ratio = float(sum(ratios.values()))
    if not ratios or abs(total_ratio - 1.0) > 1e-6 or min(ratios.values()) < 0:
        raise ValueError("split ratios must be non-negative and sum to one")
    unit_set = set(units)
    if len(unit_set) != len(units):
        raise ValueError("duplicate unit ids in split input")

    grouped: list[list[str]] = []
    seen: set[str] = set()
    for group in duplicate_groups or []:
        members = [u for u in group if u in unit_set]
        if not members:
            continue
        if seen & set(members):
            raise ValueError("a unit appears in more than one duplicate group")
        seen.update(members)
        grouped.append(sorted(members))
    grouped.extend([u] for u in units if u not in seen)

    def weight(group: Sequence[str]) -> int:
        if weights is None:
            return len(group)
        return sum(int(weights.get(u, 1)) for u in group)

    order = rng.permutation(len(grouped))
    shuffled = [grouped[int(i)] for i in order]
    shuffled.sort(key=weight, reverse=True)  # stable: ties keep shuffled order

    split_names = list(ratios)
    total_weight = sum(weight(g) for g in grouped)
    targets = {name: ratios[name] * total_weight for name in split_names}
    assigned = {name: 0.0 for name in split_names}

    result: dict[str, str] = {}
    for group in shuffled:
        deficits = [(targets[name] - assigned[name], name) for name in split_names]
        best = max(deficits, key=lambda item: item[0])[1]
        for unit_id in group:
            result[unit_id] = best
        assigned[best] += weight(group)
    return result


__all__ = ["assign_splits", "group_by_hash", "hamming_distance"]

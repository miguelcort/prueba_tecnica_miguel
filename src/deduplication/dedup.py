"""Exact and near-duplicate detection.

Near duplicates are dangerous in technical docs: two pages can be 95% equal and differ in the
one endpoint that matters. So similarity alone never drops a document. A pair is dropped only
if the technical tokens agree; otherwise it is flagged as a version difference and reported.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

from ..common import technical_tokens

MAX_HASH = (1 << 61) - 1


def shingles(text: str, size: int) -> set:
    words = text.lower().split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _coefficients(num_perm: int, seed: int) -> List[Tuple[int, int]]:
    generator = random.Random(seed)
    return [(generator.randrange(1, MAX_HASH), generator.randrange(MAX_HASH)) for _ in range(num_perm)]


def minhash(text: str, shingle_words: int, num_perm: int, seed: int) -> List[int]:
    """Fixed-size signature: documents never need to be compared in full."""
    values = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=7).digest(), "big") for s in shingles(text, shingle_words)]
    if not values:
        return [MAX_HASH] * num_perm
    return [min((a * value + b) % MAX_HASH for value in values) for a, b in _coefficients(num_perm, seed)]


def similarity(left: List[int], right: List[int]) -> float:
    return round(sum(a == b for a, b in zip(left, right)) / len(left), 4)


def lsh_candidates(signatures: Dict[str, List[int]], bands: int) -> List[Tuple[str, str]]:
    """Banding avoids comparing all pairs, which is what lets this grow past a sample."""
    pairs = set()
    for band in range(bands):
        buckets: Dict[tuple, List[str]] = defaultdict(list)
        for key in sorted(signatures):
            signature = signatures[key]
            rows = len(signature) // bands
            buckets[tuple(signature[band * rows : (band + 1) * rows])].append(key)
        for members in buckets.values():
            pairs.update(combinations(sorted(members), 2))
    return sorted(pairs)


def technical_difference(left: str, right: str) -> Dict[str, List[str]]:
    a, b = set(technical_tokens(left)), set(technical_tokens(right))
    return {"only_in_first": sorted(a - b)[:25], "only_in_second": sorted(b - a)[:25]}


def group_exact(rows: Iterable[dict]) -> Dict[tuple, List[dict]]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        groups[row["content_hash"]].append(row)
    return groups


def threshold_sweep(signatures: Dict[str, List[int]], projects: Dict[str, str], thresholds: List[float]) -> List[dict]:
    """Pairs per threshold on the sample. It justifies the configured value; it is O(n^2)."""
    scores = [similarity(signatures[a], signatures[b]) for a, b in combinations(sorted(signatures), 2) if projects[a] == projects[b]]
    return [{"threshold": t, "pairs_at_or_above": sum(score >= t for score in scores)} for t in thresholds]

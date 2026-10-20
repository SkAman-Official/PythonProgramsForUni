"""Check Bloom filter guarantees, sizing and empirical false-positive rates."""

import hashlib
import math
import random

import pytest

from bloom_filter import BloomFilter


@pytest.mark.parametrize("target", [0.01, 0.001, 0.0001])
def test_no_false_negatives_and_measured_rate(target: float) -> None:
    """All inserted strings must match; use a fixed sample for the FP bound."""
    rng = random.Random(2026)
    members = [f"member:{i}:{rng.getrandbits(128):032x}" for i in range(10_000)]
    bloom = BloomFilter(len(members), target)
    for item in members:
        bloom.add(item)
    assert all(item in bloom for item in members)
    false_positives = sum(
        f"query:{i}:{rng.getrandbits(128):032x}" in bloom for i in range(100_000)
    )
    assert false_positives / 100_000 <= 1.5 * target


@pytest.mark.parametrize("n, target", [(1, 0.9), (25, 0.5), (10000, 0.001)])
def test_sizing_formulas(n: int, target: float) -> None:
    """Round the bit count first, then compute and round the hash count."""
    bloom = BloomFilter(n, target)
    expected_m = math.ceil(-n * math.log(target) / math.log(2) ** 2)
    assert bloom.m == expected_m
    assert bloom.k == math.ceil((expected_m / n) * math.log(2))
    assert len(bloom._bits) == math.ceil(expected_m / 8)


@pytest.mark.parametrize("n", [0, -1, 1.5, True, "10", None])
def test_invalid_capacity(n: object) -> None:
    """Reject nonpositive capacities and values that are not integers."""
    with pytest.raises(ValueError, match="expected_items"):
        BloomFilter(n, 0.01)


@pytest.mark.parametrize("target", [
    0, 1, -0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.01", None,
])
def test_invalid_rate(target: object) -> None:
    """Only finite numeric rates strictly between zero and one are valid."""
    with pytest.raises(ValueError, match="false_positive_rate"):
        BloomFilter(10, target)


def test_empty_filter() -> None:
    """No bit is set before insertion, including for empty or Unicode strings."""
    bloom = BloomFilter(100, 0.01)
    assert all(item not in bloom for item in ["", "hello", "你好", "🐍", "a" * 10000])
    assert bloom.count == 0
    assert bloom.estimated_false_positive_rate() == 0


def test_hash_positions_and_packed_bits() -> None:
    """Check digest chunk order and bit offsets against an independent fixture."""
    bloom = BloomFilter(20, 0.01)
    item = "café 🐍"
    digest = hashlib.sha256(item.encode("utf-8")).digest()
    h1, h2 = int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:16], "big")
    positions = [(h1 + i * h2) % bloom.m for i in range(bloom.k)]
    assert list(bloom._positions(item)) == positions
    expected = bytearray(math.ceil(bloom.m / 8))
    for position in positions:
        expected[position // 8] |= 1 << (position % 8)
    bloom.add(item)
    assert bloom._bits == expected
    assert item in bloom


def test_duplicates_and_rate_estimate() -> None:
    """Repeating an item preserves bits but still increments the add count."""
    bloom = BloomFilter(100, 0.01)
    bloom.add("")
    before = bytes(bloom._bits)
    bloom.add("")
    assert bloom._bits == before
    assert "" in bloom
    assert bloom.count == 2
    expected = (1 - math.exp(-bloom.k * 2 / bloom.m)) ** bloom.k
    assert bloom.estimated_false_positive_rate() == pytest.approx(expected)

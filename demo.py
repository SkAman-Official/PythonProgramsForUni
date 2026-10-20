"""Measure Bloom filter false positives on reproducible random strings."""

import random

from bloom_filter import BloomFilter


def main() -> None:
    """Insert 10,000 strings and test 100,000 guaranteed non-members per rate."""
    rng = random.Random(2026)
    # Prefixes keep the two groups disjoint; the index also prevents duplicates.
    members = [f"member:{i}:{rng.getrandbits(128):032x}" for i in range(10_000)]
    non_members = [
        f"query:{i}:{rng.getrandbits(128):032x}" for i in range(100_000)
    ]
    print("10,000 insertions; 100,000 non-member queries; seed=2026")
    print(f"{'Target rate':>12} {'m (bits)':>10} {'k':>4} {'Measured FP rate':>18}")
    for target in (0.01, 0.001, 0.0001):
        bloom = BloomFilter(len(members), target)
        for item in members:
            bloom.add(item)
        false_positives = sum(item in bloom for item in non_members)
        measured = false_positives / len(non_members)
        print(f"{target:12.4%} {bloom.m:10d} {bloom.k:4d} {measured:18.4%}")


if __name__ == "__main__":
    main()

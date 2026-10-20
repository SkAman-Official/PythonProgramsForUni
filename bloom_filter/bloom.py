"""Compact membership checks with possible false positives and no deletions."""

import hashlib
import math
from collections.abc import Iterator


class BloomFilter:
    """Store strings in a bit array sized for an expected number of insertions.

    A negative lookup is definite; a positive lookup means possibly present.
    ``count`` tracks add calls, including duplicates, rather than distinct items.
    """

    def __init__(self, expected_items: int, false_positive_rate: float) -> None:
        """Validate the capacity and target rate, then allocate the bit array."""
        if (
            isinstance(expected_items, bool)
            or not isinstance(expected_items, int)
            or expected_items <= 0
        ):
            raise ValueError("expected_items must be a positive integer")
        if (
            isinstance(false_positive_rate, bool)
            or not isinstance(false_positive_rate, (int, float))
            or not 0 < false_positive_rate < 1
        ):
            raise ValueError("false_positive_rate must be between 0 and 1")

        self.m = math.ceil(
            -expected_items * math.log(false_positive_rate) / math.log(2) ** 2
        )
        self.k = math.ceil((self.m / expected_items) * math.log(2))
        self.count = 0
        self._bits = bytearray((self.m + 7) // 8)

    def _positions(self, item: str) -> Iterator[int]:
        """Double-hash using the first two big-endian 64-bit digest chunks."""
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big")
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item: str) -> None:
        """Set the item's bits and count one insertion, even for a duplicate."""
        for position in self._positions(item):
            self._bits[position // 8] |= 1 << (position % 8)
        self.count += 1

    def __contains__(self, item: str) -> bool:
        """Return False if definitely absent, or True if possibly present."""
        return all(
            self._bits[position // 8] & (1 << (position % 8))
            for position in self._positions(item)
        )

    def estimated_false_positive_rate(self) -> float:
        """Estimate the rate using the insertion count and standard formula."""
        # expm1 keeps small estimates from losing precision near zero.
        return (-math.expm1(-self.k * self.count / self.m)) ** self.k

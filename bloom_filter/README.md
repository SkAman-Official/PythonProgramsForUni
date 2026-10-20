# Bloom filter

A Bloom filter answers “have we seen this string?” using a compact bit array.
A negative answer is definite. A positive answer may be a false positive, so
applications should confirm it against their actual data when correctness matters.

This implementation uses Python 3.11+ and only the standard library.
From the repository root:

```bash
python demo.py
python -m pytest -q tests/test_bloom.py
```

```python
from bloom_filter import BloomFilter

seen = BloomFilter(expected_items=10_000, false_positive_rate=0.01)
seen.add("example")
assert "example" in seen
print(seen.estimated_false_positive_rate())
```

## How it works

For capacity `n` and target rate `p`, the filter allocates
`m = ceil(-n * ln(p) / ln(2)**2)` bits. It then chooses
`k = ceil((m / n) * ln(2))` hash positions per string, using the rounded `m`.
Bits are packed into `ceil(m / 8)` bytes of a `bytearray`.

SHA-256 hashes the UTF-8 string. The first eight digest bytes form `h1` and
the next eight form `h2`, both unsigned and big-endian. The remaining bytes are
unused. Positions are `(h1 + i*h2) % m`, for `i` from zero through `k-1`.
Adding sets those bits; lookup returns False as soon as any of them is unset.
Empty strings and Unicode are supported. Strings are not normalized.

The estimated rate is `(1 - exp(-k * count / m))**k`. The implementation uses
`expm1` for numerical accuracy near zero. `count` is the number of calls to
`add`, including duplicates. Repeating a string leaves its bits unchanged,
so duplicate-heavy input makes this count-based estimate less representative.

## Reproducible demonstration

`demo.py` inserts 10,000 unique strings and checks 100,000 non-members for each
rate, using seed 2026. Random suffixes provide varied inputs; separate prefixes
ensure no query is accidentally an inserted item. All three filters use the
same strings for comparison.

| Target rate | m (bits) | k | Measured false-positive rate |
|---|---:|---:|---:|
| 1% | 95,851 | 7 | 1.0200% |
| 0.1% | 143,776 | 10 | 0.1030% |
| 0.01% | 191,702 | 14 | 0.0130% |

These are measurements on a fixed sample, not guaranteed bounds. The estimate
assumes approximately uniform hashing; double hashing and finite samples can
produce differences. Adding more items than planned increases false positives.
The filter cannot list its items, delete them safely, or report a distinct count.
It is not designed for concurrent mutation.

Operations take O(k) bit accesses plus the cost of hashing the input string.
Storage is O(m) bits. Tests cover membership, the requested 1.5× empirical bound,
sizing, exact digest/bit conventions, invalid parameters, empty filters,
Unicode and duplicate insertions.

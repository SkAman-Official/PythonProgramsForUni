# Validation record

## Bloom filter addition

The full suite passed locally under Python 3.11.17: 95 tests, including 25
Bloom filter cases. The new tests also passed under Python 3.12.14. Ruff passed
repository-wide. CI now discovers root-level tests and runs `python demo.py`
in addition to the existing project demonstrations.

The seeded demo inserted 10,000 unique strings and checked 100,000 guaranteed
non-members per target. Measured false-positive rates were 1.0200%, 0.1030% and
0.0130% for targets of 1%, 0.1% and 0.01%. All met the requested 1.5× empirical
bound on this fixed sample. The Bloom filter README records the bit counts,
hash counts, hash-byte convention and limitations.

## Hidden Markov Model addition

The expanded eleven-project repository passed Ruff and all 70 tests locally
under Python 3.11. The new project contributes 12 tests, including exhaustive
path enumeration for all length-three sensor sequences, impossible sequences,
deterministic ties, empty and unknown inputs, invalid probability tables, and
a 10,000-observation stability check. The default HMM CLI also ran successfully.

GitHub Actions for pull request #2 passed on Python 3.11 and 3.12, checking lint,
the full test suite and all eleven default demonstrations. The workflow discovers
projects automatically, so no CI configuration change was needed.

## Initial ten-project validation

The complete repository was checked locally with CPython 3.11.17 and 3.12.14.
Both environments installed `requirements-dev.txt` successfully and passed
`python -m pip check`. CI repeats lint, tests and all ten demonstrations on
Python 3.11 and 3.12; the original local results are recorded below.

## Commands and outcomes

```bash
ruff check .
python -m pytest -q
for program in projects/*/main.py; do python "$program"; done
```

- Ruff: clean with the configured E, F, I, UP and B rule groups.
- Pytest: 58 passed on each Python version; no skips or expected failures.
- Every default CLI: successful on both versions with a 60-second timeout.
- Offline check: tests and demonstrations also passed with Python Internet
  socket connections rejected by an injected startup guard. The guard itself
  was checked before running the demonstrations.
- Alternate CLI checks: JSON job input, seeded 1,000-customer simulation,
  on-disk inventory, local phrase search, UTF-8 file compression/decompression,
  and a different polynomial/root interval all passed offline.
- Both PNG files were regenerated with Agg and visually checked.
- Each of the original ten project READMEs has 100–150 whitespace-separated words, including its
  command and output blocks. Each original source module has 150–219 lines.

## Default demonstration timings

These are wall-clock observations from this machine, not performance guarantees.
Startup and plotting are included.

| Project | Python 3.11 seconds | Python 3.12 seconds |
|---|---:|---:|
| 01_dependency_scheduler | 0.054 | 0.049 |
| 02_queue_simulator | 0.039 | 0.041 |
| 03_file_inventory | 0.046 | 0.051 |
| 04_text_search | 0.039 | 0.051 |
| 05_huffman_codec | 0.040 | 0.051 |
| 06_sparse_polynomials | 0.035 | 0.039 |
| 07_logistic_regression | 0.930 | 0.927 |
| 08_backprop_network | 1.008 | 1.006 |
| 09_kmeans_pca | 1.165 | 1.213 |
| 10_q_learning | 0.628 | 0.687 |

## Learning results

| Experiment | Seed | Measured result | Required threshold |
|---|---:|---|---|
| Logistic regression | 13 | Loss 0.6931 → 0.0289; test accuracy 0.9778; baseline 0.9778 | Accuracy ≥ 0.90; baseline gap ≤ 0.05 |
| Backpropagation network | 19 | Loss 0.6449 → 0.0934; test accuracy 0.9333 | Accuracy ≥ 0.85 |
| K-means | 23 | Inertia 1581.877 → 752.096; adjusted Rand 1.0000 | Adjusted Rand ≥ 0.90 |
| Q-learning | 29 | Greedy cost 100 → 4; all-start success 1.000; initial-state route 8 steps | Success ≥ 0.95 |

Q-learning's mean absolute TD error also decreases from about 0.8913 over the
first 100 episodes to numerical noise over the final 100. Individual errors
are not required to decrease monotonically. The held-out classification splits
are separate from training, and scaling is fitted only on training observations.
Cluster labels use synthetic ground truth solely for evaluation, not fitting.

## Scope

All runtime datasets are generated in code. Tests and examples require no
credentials, network services or GPU after dependencies are installed. The
first six projects and project 11 use only the standard library. Timings and quality scores
apply to these small, fixed-seed examples; each project documents its limitations.
Installation needs registry access. No repository settings were changed.

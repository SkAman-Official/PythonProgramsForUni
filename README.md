# Python programs for AI study

Ten small, offline Python projects exploring algorithms, software design and
machine learning. Each directory contains a runnable demonstration, focused
tests and notes about its results and limitations. The selection and acceptance
criteria are in [the plan](docs/PLAN.md).

| Project | Description |
|---|---|
| [Dependency scheduler](projects/01_dependency_scheduler/README.md) | Order prerequisites and calculate parallel completion times. |
| [Queue simulator](projects/02_queue_simulator/README.md) | Measure waiting times in a discrete-event service queue. |
| [File inventory](projects/03_file_inventory/README.md) | Hash directory contents and compare JSON snapshots. |
| [Text search](projects/04_text_search/README.md) | Rank local documents with a positional TF-IDF index. |
| [Huffman codec](projects/05_huffman_codec/README.md) | Compress Unicode text using a portable prefix-code archive. |
| [Sparse polynomials](projects/06_sparse_polynomials/README.md) | Combine, differentiate and solve sparse polynomials. |
| [Logistic regression](projects/07_logistic_regression/README.md) | Compare NumPy gradient descent with scikit-learn. |
| [Backpropagation network](projects/08_backprop_network/README.md) | Train a small nonlinear classifier with explicit gradients. |
| [K-means and PCA](projects/09_kmeans_pca/README.md) | Recover synthetic clusters and plot a two-dimensional projection. |
| [Q-learning](projects/10_q_learning/README.md) | Learn a route through a small blocked grid. |

## Setup

Use Python 3.11 or 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

Installation needs package-registry access; all examples and tests run offline
afterward. The first six projects need only the standard library. The last four
list their runtime dependencies in their own requirements files.

Run one example with `python projects/01_dependency_scheduler/main.py`, or all:

```bash
for program in projects/*/main.py; do python "$program"; done
```

Plots use the noninteractive Agg backend and are saved beside their projects.
Seeds are fixed; small floating-point differences across platforms are expected.
CI checks Python 3.11 and 3.12. MIT licensed; see [LICENSE](LICENSE).

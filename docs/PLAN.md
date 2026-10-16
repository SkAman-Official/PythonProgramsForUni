# Portfolio plan

I use six core computer-science projects and four AI projects. Each program
runs offline on a CPU, uses Python 3.11+, and targets about 150–400 source lines.
The datasets are synthetic or bundled with scikit-learn. Each CLI has fixed
inputs or a seed so its results can be checked again.

## Fifteen candidates

| # | Candidate | Decision |
|---|---|---|
| 1 | Dependency graph scheduler | Select: graph algorithms and cycle detection |
| 2 | Discrete-event queue simulator | Select: heaps and object-oriented modeling |
| 3 | Transactional file inventory | Select: file I/O, hashing and safe comparison |
| 4 | Positional text search | Select: indexing, ranking and text processing |
| 5 | Prefix-code compressor | Select: trees, binary formats and round trips |
| 6 | Sparse polynomial algebra | Select: numerical computing and data modeling |
| 7 | Logistic regression from scratch | Select: optimization and baseline comparison |
| 8 | Neural network with backpropagation | Select: gradients and nonlinear learning |
| 9 | K-means with PCA visualization | Select: unsupervised learning and plots |
| 10 | Q-learning grid navigator | Select: sequential decisions and evaluation |
| 11 | AVL tree visualizer | Drop: overlaps graph/tree work without adding enough range |
| 12 | CSV schema inference | Drop: file inventory and search already cover parsing |
| 13 | Cellular automaton | Drop: weaker measurable outcome than the simulator |
| 14 | Naive Bayes classifier | Drop: overlaps supervised classification |
| 15 | Sudoku constraint solver | Drop: duplicates search coverage from the navigator |

## Selected projects and acceptance criteria

| Directory / title | Goal | Concepts | Libraries | Measurement |
|---|---|---|---|---|
| 01_dependency_scheduler | Schedule prerequisite-constrained jobs | DAGs, topological order, critical paths | Standard library | Valid order; known makespan; cycles rejected |
| 02_queue_simulator | Measure a single-server queue | Dataclasses, event ordering, simulation | Standard library | Exact hand-calculated waits; stable seeded runs |
| 03_file_inventory | Compare directory snapshots | pathlib, SHA-256, JSON, safe validation | Standard library | Detect added/removed/modified files; round-trip snapshots |
| 04_text_search | Rank documents for multiword queries | Unicode normalization, inverted index, TF-IDF | Standard library | Known retrieval order; empty and unknown queries |
| 05_huffman_codec | Encode text with a portable prefix code | Heaps, trees, bytes, format validation | Standard library | Unicode round trips; malformed archives rejected |
| 06_sparse_polynomials | Evaluate and combine sparse polynomials | Immutable objects, algebra, differentiation | Standard library | Algebraic identities; analytic derivative; root residual |
| 07_logistic_regression | Compare a handwritten classifier to a baseline | NumPy, gradient descent, train/test split | NumPy, scikit-learn | Falling loss; held-out accuracy ≥ 0.90; baseline gap ≤ 0.05 |
| 08_backprop_network | Learn nonlinear decision boundaries | Chain rule, minibatch-free optimization | NumPy, scikit-learn | Gradient finite differences; falling loss; accuracy ≥ 0.85 |
| 09_kmeans_pca | Recover synthetic groups and visualize them | Lloyd iterations, SVD/PCA, cluster evaluation | NumPy, scikit-learn, matplotlib | Nonincreasing inertia; adjusted Rand score ≥ 0.90; saved plot |
| 10_q_learning | Learn a route around blocked cells | Bellman updates, exploration, policies | NumPy, matplotlib | TD-error shapes; falling evaluation cost; goal success ≥ 0.95 |

## Working assumptions

These are independent educational command-line programs, not deployed services.
No runtime network access, secrets, GPU, downloaded data, or deep-learning
frameworks are needed. Tests include at least four cases per project; learning
projects also check shapes, optimization progress and seeded quality thresholds.
For Q-learning, optimization progress means improved greedy-policy episode cost;
raw temporal-difference errors need not decrease monotonically.
For k-means, inertia is the loss and adjusted Rand score measures recovery rather
than supervised accuracy. Plots use Agg and are committed under results/.

A single branch contains a scaffold commit followed by one commit per project.
Repository settings are left unchanged. Remote publication requires verified
private visibility; unknown visibility is insufficient to push. Design choices,
validation evidence and access limitations will be recorded in the PR description.

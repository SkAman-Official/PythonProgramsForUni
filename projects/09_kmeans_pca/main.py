"""Cluster generated observations and save a PCA projection."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "portfolio-mpl")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

LOGGER = logging.getLogger(__name__)
Array = NDArray[np.float64]


@dataclass(frozen=True)
class Clusters:
    """Collect centers, nearest-center labels, and the inertia trace."""

    centers: Array
    labels: NDArray[np.int64]
    losses: Array


def validate_features(features: Array) -> Array:
    """Require a finite, nonempty feature matrix."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or 0 in features.shape or not np.isfinite(features).all():
        raise ValueError("Expected a finite nonempty matrix")
    return features


def distances(features: Array, centers: Array) -> Array:
    """Return squared Euclidean distances for every row-center pair."""
    return np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def initialize_centers(features: Array, count: int, seed: int) -> Array:
    """Choose k-means++ centers, handling duplicate observations explicitly."""
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(features)))]
    while len(chosen) < count:
        nearest = distances(features, features[chosen]).min(axis=1)
        if nearest.sum() == 0:
            remaining = sorted(set(range(len(features))) - set(chosen))
            chosen.append(remaining[0])
        else:
            chosen.append(int(rng.choice(len(features), p=nearest / nearest.sum())))
    return features[chosen].copy()


def kmeans(
    features: Array,
    count: int = 3,
    seed: int = 23,
    max_iterations: int = 100,
) -> Clusters:
    """Run Lloyd updates and retain empty centers to preserve nonincreasing loss."""
    features = validate_features(features)
    if not 1 <= count <= len(features) or max_iterations < 1:
        raise ValueError("Invalid cluster count or iteration limit")
    centers = initialize_centers(features, count, seed)
    losses = []
    for _ in range(max_iterations):
        squared = distances(features, centers)
        labels = np.argmin(squared, axis=1)
        losses.append(float(squared[np.arange(len(features)), labels].sum()))
        updated = centers.copy()
        for cluster in range(count):
            members = features[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers, atol=1e-10, rtol=0):
            centers = updated
            break
        centers = updated
    squared = distances(features, centers)
    labels = np.argmin(squared, axis=1)
    losses.append(float(squared[np.arange(len(features)), labels].sum()))
    return Clusters(centers, labels, np.asarray(losses))


def project_pca(features: Array, components: int = 2) -> tuple[Array, Array]:
    """Center observations and project onto right singular vectors."""
    features = validate_features(features)
    if not 1 <= components <= min(features.shape):
        raise ValueError("Invalid PCA component count")
    centered = features - features.mean(axis=0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular**2))
    explained = singular[:components] ** 2 / total if total else np.zeros(components)
    return centered @ vectors[:components].T, explained


def dataset(seed: int = 23) -> tuple[Array, NDArray[np.int64]]:
    """Generate three well-separated groups in five dimensions."""
    return make_blobs(
        n_samples=360, n_features=5, centers=3, cluster_std=0.65, random_state=seed
    )


def save_plot(features: Array, result: Clusters, output: Path) -> None:
    """Save an Agg scatter plot with PCA variance labels."""
    projected, explained = project_pca(features)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(
        projected[:, 0],
        projected[:, 1],
        c=result.labels,
        cmap="viridis",
        s=16,
        alpha=0.8,
    )
    axis.set(
        xlabel=f"PC1 ({explained[0]:.1%})",
        ylabel=f"PC2 ({explained[1]:.1%})",
        title="K-means on synthetic five-dimensional observations",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=120, metadata={"Software": "matplotlib"})
    plt.close(figure)


def main() -> None:
    """Fit seeded clusters, report recovery, and write the projection image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "clusters.png",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    features, truth = dataset(args.seed)
    result = kmeans(features, seed=args.seed)
    save_plot(features, result, args.output)
    LOGGER.info("Inertia: %.3f -> %.3f", result.losses[0], result.losses[-1])
    LOGGER.info("Adjusted Rand score: %.4f", adjusted_rand_score(truth, result.labels))
    LOGGER.info("Saved %s", args.output.name)


if __name__ == "__main__":
    main()

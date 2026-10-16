"""Compare handwritten binary logistic regression with scikit-learn."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
Array = NDArray[np.float64]


def sigmoid(values: Array) -> Array:
    """Compute logistic probabilities without overflowing exponentials."""
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def validate_features(features: Array) -> Array:
    """Require a finite, nonempty two-dimensional feature matrix."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or 0 in features.shape or not np.isfinite(features).all():
        raise ValueError("Features must be a finite nonempty matrix")
    return features


def objective_gradient(
    features: Array, targets: Array, weights: Array, bias: float
) -> tuple[float, Array, float]:
    """Return mean binary cross-entropy and its exact parameter gradients."""
    logits = features @ weights + bias
    loss = np.mean(np.logaddexp(0, logits) - targets * logits)
    residual = sigmoid(logits) - targets
    return float(loss), features.T @ residual / len(features), float(residual.mean())


@dataclass
class LogisticModel:
    """Fit a linear binary classifier with full-batch gradient descent."""

    learning_rate: float = 0.2
    iterations: int = 1000
    weights: Array | None = None
    bias: float = 0.0
    losses: list[float] | None = None

    def fit(self, features: Array, targets: Array) -> LogisticModel:
        """Reset parameters and minimize unregularized cross-entropy."""
        features = validate_features(features)
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (len(features),) or not np.isin(targets, [0, 1]).all():
            raise ValueError("Targets must be one binary label per row")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if self.iterations < 1:
            raise ValueError("Iterations must be positive")
        self.weights = np.zeros(features.shape[1])
        self.bias = 0.0
        self.losses = []
        for _ in range(self.iterations):
            loss, gradient, bias_gradient = objective_gradient(
                features, targets, self.weights, self.bias
            )
            self.losses.append(loss)
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * bias_gradient
        final_loss, _, _ = objective_gradient(
            features, targets, self.weights, self.bias
        )
        self.losses.append(final_loss)
        return self

    def predict_proba(self, features: Array) -> Array:
        """Return positive-class probabilities for a fitted model."""
        if self.weights is None:
            raise ValueError("Fit the model before prediction")
        features = validate_features(features)
        if features.shape[1] != len(self.weights):
            raise ValueError("Feature count differs from training")
        return sigmoid(features @ self.weights + self.bias)

    def predict(self, features: Array) -> NDArray[np.int64]:
        """Classify rows at the fixed probability threshold of one half."""
        return (self.predict_proba(features) >= 0.5).astype(np.int64)

    def score(self, features: Array, targets: Array) -> float:
        """Return mean binary accuracy after checking target shape and values."""
        predictions = self.predict(features)
        targets = np.asarray(targets)
        if targets.shape != predictions.shape or not np.isin(targets, [0, 1]).all():
            raise ValueError("Expected one binary label per prediction")
        return float(np.mean(predictions == targets))


def dataset(seed: int = 13) -> tuple[Array, Array, Array, Array]:
    """Create a stratified synthetic split and fit scaling on training rows only."""
    features, targets = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        flip_y=0.01,
        random_state=seed,
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features, targets, test_size=0.3, random_state=seed, stratify=targets
    )
    scaler = StandardScaler().fit(train_x)
    return scaler.transform(train_x), scaler.transform(test_x), train_y, test_y


def experiment(seed: int = 13, iterations: int = 1000) -> dict[str, float]:
    """Train both unregularized models and report held-out accuracy."""
    train_x, test_x, train_y, test_y = dataset(seed)
    model = LogisticModel(iterations=iterations).fit(train_x, train_y)
    baseline = LogisticRegression(penalty=None, max_iter=1000).fit(train_x, train_y)
    return {
        "initial_loss": model.losses[0],
        "final_loss": model.losses[-1],
        "numpy_accuracy": model.score(test_x, test_y),
        "baseline_accuracy": float(baseline.score(test_x, test_y)),
    }


def main() -> None:
    """Run the seeded offline comparison and log measured scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name, value in experiment(args.seed, args.iterations).items():
        LOGGER.info("%s: %.4f", name, value)


if __name__ == "__main__":
    main()

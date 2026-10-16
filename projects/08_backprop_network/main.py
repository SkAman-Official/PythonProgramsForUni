"""Train a one-hidden-layer network using explicit NumPy backpropagation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
Array = NDArray[np.float64]


def sigmoid(values: Array) -> Array:
    """Return stable logistic probabilities for arbitrary finite logits."""
    return np.exp(-np.logaddexp(0, -values))


def loss_and_gradients(
    features: Array, targets: Array, parameters: dict[str, Array]
) -> tuple[float, dict[str, Array]]:
    """Compute mean cross-entropy and gradients through a tanh hidden layer."""
    hidden = np.tanh(features @ parameters["w1"] + parameters["b1"])
    logits = hidden @ parameters["w2"] + parameters["b2"]
    labels = targets.reshape(-1, 1)
    loss = np.mean(np.logaddexp(0, logits) - labels * logits)
    output_delta = (sigmoid(logits) - labels) / len(features)
    hidden_delta = (output_delta @ parameters["w2"].T) * (1 - hidden**2)
    return float(loss), {
        "w1": features.T @ hidden_delta,
        "b1": hidden_delta.sum(axis=0),
        "w2": hidden.T @ output_delta,
        "b2": output_delta.sum(axis=0),
    }


@dataclass
class Network:
    """Fit a binary classifier with a small dense hidden layer."""

    hidden_units: int = 16
    learning_rate: float = 0.1
    iterations: int = 2500
    seed: int = 19
    parameters: dict[str, Array] | None = None
    losses: list[float] | None = None

    def initialize(self, features: int) -> dict[str, Array]:
        """Initialize weights with variance scaled to their fan-in."""
        if features < 1 or self.hidden_units < 1:
            raise ValueError("Layer widths must be positive")
        rng = np.random.default_rng(self.seed)
        self.parameters = {
            "w1": rng.normal(size=(features, self.hidden_units)) / np.sqrt(features),
            "b1": np.zeros(self.hidden_units),
            "w2": rng.normal(size=(self.hidden_units, 1)) / np.sqrt(self.hidden_units),
            "b2": np.zeros(1),
        }
        return self.parameters

    def fit(self, features: Array, targets: Array) -> Network:
        """Reset and train the network with deterministic full-batch updates."""
        features = validate_features(features)
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (len(features),) or not np.isin(targets, [0, 1]).all():
            raise ValueError("Expected one binary label per row")
        if self.iterations < 1 or not np.isfinite(self.learning_rate):
            raise ValueError("Invalid optimization controls")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        parameters = self.initialize(features.shape[1])
        self.losses = []
        for _ in range(self.iterations):
            loss, gradients = loss_and_gradients(features, targets, parameters)
            self.losses.append(loss)
            for name in parameters:
                parameters[name] -= self.learning_rate * gradients[name]
        self.losses.append(loss_and_gradients(features, targets, parameters)[0])
        return self

    def predict_proba(self, features: Array) -> Array:
        """Return positive-class probabilities from the fitted parameters."""
        if self.parameters is None:
            raise ValueError("Initialize or fit before prediction")
        features = validate_features(features)
        parameters = self.parameters
        if features.shape[1] != parameters["w1"].shape[0]:
            raise ValueError("Feature width differs from the fitted network")
        hidden = np.tanh(features @ parameters["w1"] + parameters["b1"])
        return sigmoid(hidden @ parameters["w2"] + parameters["b2"]).ravel()

    def predict(self, features: Array) -> NDArray[np.int64]:
        """Classify rows using a fixed half-probability threshold."""
        return (self.predict_proba(features) >= 0.5).astype(np.int64)

    def score(self, features: Array, targets: Array) -> float:
        """Return mean binary accuracy after checking target shape and values."""
        predictions = self.predict(features)
        targets = np.asarray(targets)
        if targets.shape != predictions.shape or not np.isin(targets, [0, 1]).all():
            raise ValueError("Expected one binary label per prediction")
        return float(np.mean(predictions == targets))


def validate_features(features: Array) -> Array:
    """Require finite rows with at least one feature."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or 0 in features.shape or not np.isfinite(features).all():
        raise ValueError("Expected a finite nonempty matrix")
    return features


def dataset(seed: int = 19) -> tuple[Array, Array, Array, Array]:
    """Generate noisy moons and scale without leaking held-out observations."""
    features, targets = make_moons(n_samples=500, noise=0.16, random_state=seed)
    train_x, test_x, train_y, test_y = train_test_split(
        features, targets, test_size=0.3, stratify=targets, random_state=seed
    )
    scaler = StandardScaler().fit(train_x)
    return scaler.transform(train_x), scaler.transform(test_x), train_y, test_y


def experiment(seed: int = 19, iterations: int = 2500) -> dict[str, float]:
    """Measure training loss and held-out accuracy on generated moons."""
    train_x, test_x, train_y, test_y = dataset(seed)
    model = Network(seed=seed, iterations=iterations).fit(train_x, train_y)
    return {
        "initial_loss": model.losses[0],
        "final_loss": model.losses[-1],
        "test_accuracy": model.score(test_x, test_y),
    }


def main() -> None:
    """Run a reproducible nonlinear classification experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--iterations", type=int, default=2500)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name, value in experiment(args.seed, args.iterations).items():
        LOGGER.info("%s: %.4f", name, value)


if __name__ == "__main__":
    main()

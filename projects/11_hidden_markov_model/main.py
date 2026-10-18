"""Discrete Hidden Markov Model inference, using only the standard library."""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass


def logsumexp(values: Sequence[float]) -> float:
    """Sum probabilities in log space without underflow."""
    largest = max(values)
    if largest == -math.inf:
        return -math.inf
    return largest + math.log(sum(math.exp(x - largest) for x in values))


def log_probability(value: float) -> float:
    return math.log(value) if value else -math.inf


@dataclass(frozen=True)
class HiddenMarkovModel:
    states: tuple[str, ...]
    symbols: tuple[str, ...]
    initial: tuple[float, ...]
    transition: tuple[tuple[float, ...], ...]
    emission: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.states)
        if not n or not self.symbols:
            raise ValueError("States and symbols must be nonempty")
        if len(set(self.states)) != n or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("State and symbol labels must be unique")
        if len(self.transition) != n or len(self.emission) != n:
            raise ValueError("One transition and emission row is needed per state")
        rows = [(self.initial, n)]
        rows += [(row, n) for row in self.transition]
        rows += [(row, len(self.symbols)) for row in self.emission]
        for row, width in rows:
            if len(row) != width:
                raise ValueError("Probability row has the wrong length")
            if any(not math.isfinite(p) or p < 0 or p > 1 for p in row):
                raise ValueError("Probabilities must be finite and between 0 and 1")
            if not math.isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Probability rows must sum to 1")

    def _encode(self, observations: Sequence[str]) -> list[int]:
        lookup = {symbol: i for i, symbol in enumerate(self.symbols)}
        try:
            return [lookup[symbol] for symbol in observations]
        except KeyError as error:
            raise ValueError(f"Unknown observation: {error.args[0]}") from error

    def forward(self, observations: Sequence[str]) -> float:
        """Return log P(observations), summing over all hidden paths."""
        encoded = self._encode(observations)
        if not encoded:
            return 0.0
        n = len(self.states)
        scores = [log_probability(self.initial[s])
                  + log_probability(self.emission[s][encoded[0]]) for s in range(n)]
        for symbol in encoded[1:]:
            scores = [
                logsumexp([scores[previous]
                           + log_probability(self.transition[previous][current])
                           for previous in range(n)])
                + log_probability(self.emission[current][symbol])
                for current in range(n)
            ]
        return logsumexp(scores)

    def viterbi(self, observations: Sequence[str]) -> tuple[list[str], float]:
        """Return the most likely path and its joint log probability.

        Ties use the first state in model order. Impossible sequences have no path.
        """
        encoded = self._encode(observations)
        if not encoded:
            return [], 0.0
        n = len(self.states)
        scores = [log_probability(self.initial[s])
                  + log_probability(self.emission[s][encoded[0]]) for s in range(n)]
        backpointers = []
        for symbol in encoded[1:]:
            next_scores, parents = [], []
            for current in range(n):
                candidates = [scores[previous]
                              + log_probability(self.transition[previous][current])
                              for previous in range(n)]
                parent = max(range(n), key=lambda s: candidates[s])
                parents.append(parent)
                next_scores.append(candidates[parent]
                                   + log_probability(self.emission[current][symbol]))
            backpointers.append(parents)
            scores = next_scores
        last = max(range(n), key=lambda s: scores[s])
        best_score = scores[last]
        if best_score == -math.inf:
            return [], best_score
        path = [last]
        for parents in reversed(backpointers):
            path.append(parents[path[-1]])
        return [self.states[s] for s in reversed(path)], best_score


def machine_model() -> HiddenMarkovModel:
    """Illustrative probabilities, not estimates from real machinery."""
    return HiddenMarkovModel(
        states=("healthy", "worn"),
        symbols=("quiet", "vibration", "alarm"),
        initial=(0.9, 0.1),
        transition=((0.95, 0.05), (0.10, 0.90)),
        emission=((0.80, 0.18, 0.02), (0.10, 0.40, 0.50)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", nargs="*", default=[
        "quiet", "quiet", "vibration", "alarm", "alarm"])
    args = parser.parse_args()
    model = machine_model()
    try:
        likelihood = model.forward(args.observations)
        path, joint = model.viterbi(args.observations)
    except ValueError as error:
        parser.error(str(error))
    print("Observations:", " -> ".join(args.observations))
    print("Best path:   ", " -> ".join(path) if path else "No possible path")
    print(f"Sequence log likelihood: {likelihood:.6f}")
    print(f"Best path joint log probability: {joint:.6f}")
    if likelihood != -math.inf:
        print(f"Best path posterior probability: {math.exp(joint - likelihood):.6f}")


if __name__ == "__main__":
    main()

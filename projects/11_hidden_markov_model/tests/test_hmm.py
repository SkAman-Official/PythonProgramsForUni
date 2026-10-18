"""Compare dynamic programming against exhaustive enumeration on short inputs."""

import itertools
import math
from dataclasses import replace

import pytest


def test_against_exhaustive_paths(project):
    model = project.machine_model()
    for observations in itertools.product(model.symbols, repeat=3):
        paths = list(itertools.product(range(2), repeat=3))
        weights = []
        for path in paths:
            weight = model.initial[path[0]]
            for t, state in enumerate(path):
                weight *= model.emission[state][model.symbols.index(observations[t])]
                if t:
                    weight *= model.transition[path[t - 1]][state]
            weights.append(weight)
        decoded, score = model.viterbi(observations)
        expected = paths[max(range(len(paths)), key=lambda i: weights[i])]
        assert decoded == [model.states[s] for s in expected]
        assert math.exp(score) == pytest.approx(max(weights))
        assert math.exp(model.forward(observations)) == pytest.approx(sum(weights))


def test_long_sequence_stays_finite(project):
    model = project.machine_model()
    observations = ["alarm"] * 10000
    assert math.isfinite(model.forward(observations))
    path, score = model.viterbi(observations)
    assert len(path) == len(observations)
    assert math.isfinite(score)
    assert score <= model.forward(observations)


def test_empty_and_unknown(project):
    model = project.machine_model()
    assert model.forward([]) == 0
    assert model.viterbi([]) == ([], 0)
    for infer in (model.forward, model.viterbi):
        with pytest.raises(ValueError, match="Unknown"):
            infer(["missing"])


def test_impossible_and_tie(project):
    model = project.HiddenMarkovModel(
        ("a", "b"), ("yes", "no"), (0.5, 0.5),
        ((0.5, 0.5), (0.5, 0.5)), ((1.0, 0.0), (1.0, 0.0)))
    assert model.forward(["no"]) == -math.inf
    assert model.viterbi(["yes", "no"]) == ([], -math.inf)
    assert model.viterbi(["yes", "yes"])[0] == ["a", "a"]


@pytest.mark.parametrize("changes", [
    {"initial": (0.2, 0.2)}, {"initial": (math.nan, 0.0)},
    {"initial": (-0.1, 1.1)}, {"transition": ((1.0,), (0.1, 0.9))},
    {"emission": ()}, {"states": ("same", "same")},
    {"symbols": ()}, {"initial": (math.inf, 0.0)},
])
def test_invalid_model(project, changes):
    with pytest.raises(ValueError):
        replace(project.machine_model(), **changes)

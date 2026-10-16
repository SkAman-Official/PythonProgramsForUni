import numpy as np
import pytest


@pytest.fixture(scope="module")
def trained(project):
    grid = project.Grid()
    return grid, project.train(grid)


def test_shapes_and_learning_progress(trained):
    grid, result = trained
    assert result.values.shape == (grid.size**2, 4)
    assert result.td_errors.shape == (1500,)
    assert result.policy_costs.ndim == 1
    assert np.isfinite(result.values).all()
    assert result.td_errors[-100:].mean() < result.td_errors[:100].mean()
    assert result.policy_costs[-1] < result.policy_costs[0] * 0.2


def test_success_and_valid_route(project, trained):
    grid, result = trained
    success, _ = project.evaluate(grid, result.values)
    assert success >= 0.95
    path = project.rollout(grid, result.values)
    assert path[-1] == grid.goal
    assert len(path) - 1 == 8
    assert not set(path) & grid.blocked
    assert len(set(path)) == len(path)


def test_collisions_and_terminal_state(project):
    grid = project.Grid()
    assert grid.step(0, 0) == (0, -1, False)
    assert grid.step(1, 2) == (1, -1, False)
    assert grid.step(23, 1) == (24, 20, True)
    assert grid.step(24, 0) == (24, 0, True)


def test_invalid_inputs(project):
    with pytest.raises(ValueError):
        project.Grid(blocked=frozenset({24}))
    with pytest.raises(ValueError):
        project.train(project.Grid(), episodes=0)
    with pytest.raises(ValueError):
        project.Grid().step(6, 0)
    with pytest.raises(ValueError):
        project.rollout(project.Grid(), np.zeros((2, 4)))


def test_plot_and_seed_reproducibility(project, trained, tmp_path):
    grid, result = trained
    first = project.train(grid, episodes=30)
    second = project.train(grid, episodes=30)
    np.testing.assert_array_equal(first.values, second.values)
    output = tmp_path / "policy.png"
    project.save_plot(grid, result, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

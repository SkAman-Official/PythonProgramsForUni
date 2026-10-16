"""Learn a deterministic grid-navigation policy with tabular Q-learning."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from collections import deque
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

LOGGER = logging.getLogger(__name__)
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class Grid:
    """Represent a square world with an absorbing bottom-right goal."""

    size: int = 5
    blocked: frozenset[int] = frozenset({6, 7, 11, 16})

    def __post_init__(self) -> None:
        if self.size < 2 or any(
            cell < 0 or cell >= self.size**2 for cell in self.blocked
        ):
            raise ValueError("Invalid grid size or blocked cell")
        if self.goal in self.blocked:
            raise ValueError("Goal must be traversable")

    @property
    def goal(self) -> int:
        """Return the terminal state's flattened index."""
        return self.size**2 - 1

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        """Move one cell or stay on collision; return successor, reward, terminal."""
        if state < 0 or state > self.goal or state in self.blocked:
            raise ValueError("Invalid state")
        if action not in range(4):
            raise ValueError("Invalid action")
        if state == self.goal:
            return state, 0.0, True
        row, column = divmod(state, self.size)
        dr, dc = MOVES[action]
        new_row, new_column = row + dr, column + dc
        successor = state
        if 0 <= new_row < self.size and 0 <= new_column < self.size:
            candidate = new_row * self.size + new_column
            if candidate not in self.blocked:
                successor = candidate
        terminal = successor == self.goal
        return successor, 20.0 if terminal else -1.0, terminal

    def reachable_starts(self) -> list[int]:
        """Find all nonterminal cells connected to the goal."""
        seen = {self.goal}
        queue = deque([self.goal])
        while queue:
            state = queue.popleft()
            for action in range(4):
                # The goal is absorbing, so inspect its neighbors geometrically.
                row, column = divmod(state, self.size)
                dr, dc = MOVES[action]
                nr, nc = row + dr, column + dc
                candidate = nr * self.size + nc
                if (
                    0 <= nr < self.size
                    and 0 <= nc < self.size
                    and candidate not in self.blocked
                    and candidate not in seen
                ):
                    seen.add(candidate)
                    queue.append(candidate)
        return sorted(seen - {self.goal})


@dataclass(frozen=True)
class Training:
    """Store Q values, per-episode TD errors, and greedy evaluation costs."""

    values: NDArray[np.float64]
    td_errors: NDArray[np.float64]
    policy_costs: NDArray[np.float64]


def rollout(
    grid: Grid,
    values: NDArray[np.float64],
    start: int = 0,
    max_steps: int = 100,
) -> list[int]:
    """Follow the greedy policy until termination or the supplied step budget."""
    if values.shape != (grid.size**2, 4) or not np.isfinite(values).all():
        raise ValueError("Q table has an invalid shape or nonfinite values")
    if start not in grid.reachable_starts() and start != grid.goal:
        raise ValueError("Start must be connected to the goal")
    if max_steps < 1:
        raise ValueError("Step budget must be positive")
    path = [start]
    for _ in range(max_steps):
        if path[-1] == grid.goal:
            break
        successor, _, _ = grid.step(path[-1], int(np.argmax(values[path[-1]])))
        path.append(successor)
    return path


def evaluate(grid: Grid, values: NDArray[np.float64]) -> tuple[float, float]:
    """Return all-start success fraction and mean cost, penalizing failures by 100."""
    paths = [rollout(grid, values, start) for start in grid.reachable_starts()]
    successes = [path[-1] == grid.goal for path in paths]
    costs = [
        len(path) - 1 if success else 100
        for path, success in zip(paths, successes, strict=True)
    ]
    return float(np.mean(successes)), float(np.mean(costs))


def train(grid: Grid, episodes: int = 1500, seed: int = 29) -> Training:
    """Learn from seeded exploratory episodes starting across reachable cells."""
    if episodes < 1:
        raise ValueError("Episodes must be positive")
    starts = grid.reachable_starts()
    if not starts:
        raise ValueError("Grid needs a reachable nonterminal cell")
    rng = np.random.default_rng(seed)
    values = np.zeros((grid.size**2, 4))
    errors = []
    costs = [evaluate(grid, values)[1]]
    for episode in range(episodes):
        state = int(rng.choice(starts))
        epsilon = max(0.05, 1 - episode / (episodes * 0.8))
        episode_errors = []
        for _ in range(100):
            if rng.random() < epsilon:
                action = int(rng.integers(4))
            else:
                action = int(np.argmax(values[state]))
            successor, reward, terminal = grid.step(state, action)
            target = reward if terminal else reward + 0.95 * values[successor].max()
            error = target - values[state, action]
            values[state, action] += 0.3 * error
            episode_errors.append(abs(error))
            state = successor
            if terminal:
                break
        errors.append(float(np.mean(episode_errors)))
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            costs.append(evaluate(grid, values)[1])
    return Training(values, np.asarray(errors), np.asarray(costs))


def save_plot(grid: Grid, result: Training, output: Path) -> None:
    """Save the learned policy alongside its greedy evaluation cost trace."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, (world, curve) = plt.subplots(1, 2, figsize=(9, 4))
    for state in range(grid.size**2):
        row, column = divmod(state, grid.size)
        if state in grid.blocked:
            world.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, color="0.3"))
        elif state == grid.goal:
            world.text(column, row, "Goal", ha="center", va="center", fontsize=8)
        else:
            dr, dc = MOVES[int(np.argmax(result.values[state]))]
            world.arrow(column, row, dc * 0.3, dr * 0.3, head_width=0.12, color="teal")
    world.set(
        xlim=(-0.5, grid.size - 0.5),
        ylim=(grid.size - 0.5, -0.5),
        title="Greedy policy",
        xticks=range(grid.size),
        yticks=range(grid.size),
    )
    world.set_aspect("equal")
    curve.plot(result.policy_costs, marker="o", markersize=3)
    curve.set(
        xlabel="Evaluation checkpoint (every 100 episodes)",
        ylabel="Mean steps (failure = 100)",
        title="Policy evaluation cost",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=120, metadata={"Software": "matplotlib"})
    plt.close(figure)


def main() -> None:
    """Train and evaluate a seeded policy, then save a noninteractive image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results" / "policy.png"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    grid = Grid()
    result = train(grid, args.episodes, args.seed)
    success, cost = evaluate(grid, result.values)
    save_plot(grid, result, args.output)
    LOGGER.info("All-start success: %.3f; mean steps: %.2f", success, cost)
    LOGGER.info(
        "Policy cost: %.2f -> %.2f", result.policy_costs[0], result.policy_costs[-1]
    )
    LOGGER.info("Route from 0: %s", rollout(grid, result.values))


if __name__ == "__main__":
    main()

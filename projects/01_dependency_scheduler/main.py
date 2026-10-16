"""Schedule prerequisite-constrained jobs on unlimited parallel workers."""

from __future__ import annotations

import argparse
import heapq
import json
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    """Describe one named job and its immediate prerequisites."""

    name: str
    duration: float
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class Slot:
    """Record a job's earliest feasible start and finish times."""

    name: str
    start: float
    finish: float


def validate(jobs: list[Job]) -> dict[str, Job]:
    """Validate names, durations, and dependency references."""
    import math

    by_name = {job.name: job for job in jobs}
    if len(by_name) != len(jobs):
        raise ValueError("Job names must be unique")
    for job in jobs:
        if not job.name or not math.isfinite(job.duration) or job.duration < 0:
            raise ValueError("Names must be nonempty and durations finite/nonnegative")
        if len(set(job.dependencies)) != len(job.dependencies):
            raise ValueError("Repeated prerequisite")
        if any(name not in by_name for name in job.dependencies):
            raise ValueError("Unknown prerequisite")
    return by_name


def topological_order(jobs: list[Job]) -> list[str]:
    """Return a deterministic order or reject a cyclic graph."""
    by_name = validate(jobs)
    incoming = {name: len(job.dependencies) for name, job in by_name.items()}
    children: dict[str, list[str]] = {name: [] for name in by_name}
    for job in jobs:
        for dependency in job.dependencies:
            children[dependency].append(job.name)
    ready = [name for name, count in incoming.items() if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for child in children[name]:
            incoming[child] -= 1
            if incoming[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(jobs):
        raise ValueError("Dependency cycle detected")
    return order


def schedule(jobs: list[Job]) -> list[Slot]:
    """Calculate earliest execution slots under unlimited parallelism."""
    by_name = validate(jobs)
    finishes: dict[str, float] = {}
    slots = []
    for name in topological_order(jobs):
        job = by_name[name]
        start = max((finishes[dep] for dep in job.dependencies), default=0.0)
        finish = start + job.duration
        finishes[name] = finish
        slots.append(Slot(name, start, finish))
    return slots


def critical_path(jobs: list[Job]) -> list[str]:
    """Return one longest prerequisite path, breaking ties by name."""
    by_name = validate(jobs)
    totals: dict[str, float] = {}
    paths: dict[str, list[str]] = {}
    for name in topological_order(jobs):
        job = by_name[name]
        if job.dependencies:
            parent = min(job.dependencies, key=lambda dep: (-totals[dep], dep))
            totals[name] = totals[parent] + job.duration
            paths[name] = paths[parent] + [name]
        else:
            totals[name] = job.duration
            paths[name] = [name]
    if not totals:
        return []
    last = min(totals, key=lambda name: (-totals[name], name))
    return paths[last]


def load_jobs(path: Path) -> list[Job]:
    """Read a JSON list of job objects without changing the input."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list")
    jobs = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each job must be an object")
        name = item.get("name")
        duration = item.get("duration")
        dependencies = item.get("dependencies", [])
        if not isinstance(name, str) or isinstance(duration, bool):
            raise ValueError("Invalid job fields")
        if not isinstance(duration, (int, float)):
            raise ValueError("Duration must be numeric")
        if not isinstance(dependencies, list) or not all(
            isinstance(dep, str) for dep in dependencies
        ):
            raise ValueError("Dependencies must be a list of names")
        jobs.append(Job(name, float(duration), tuple(dependencies)))
    validate(jobs)
    return jobs


def demo_jobs() -> list[Job]:
    """Return a small pipeline with two parallel branches."""
    return [
        Job("prepare", 2),
        Job("analyse", 4, ("prepare",)),
        Job("review", 3, ("prepare",)),
        Job("report", 1, ("analyse", "review")),
    ]


def main() -> None:
    """Run a built-in example or schedule jobs from a JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    jobs = load_jobs(args.input) if args.input else demo_jobs()
    slots = schedule(jobs)
    for slot in slots:
        LOGGER.info("%s: %.1f -> %.1f", slot.name, slot.start, slot.finish)
    LOGGER.info("Makespan: %.1f", max((slot.finish for slot in slots), default=0))
    LOGGER.info("Critical path: %s", " -> ".join(critical_path(jobs)))


if __name__ == "__main__":
    main()

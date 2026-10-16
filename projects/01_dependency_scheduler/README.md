# Dependency scheduler

I wanted to model a pipeline where some jobs can begin together and others must wait. I represent each job with a frozen dataclass and use a heap-backed topological traversal for deterministic ordering. Earliest finish times also identify a critical path.

From the repository root, after the main setup:

```bash
python projects/01_dependency_scheduler/main.py
pytest projects/01_dependency_scheduler/tests -q
```

Example output:

```text
prepare: 0.0 -> 2.0
analyse: 2.0 -> 6.0
review: 2.0 -> 5.0
report: 6.0 -> 7.0
Makespan: 7.0
```

The example finishes in seven time units because analysis and review overlap. I test cycles, unknown prerequisites, duplicate names, zero durations and JSON loading. The model assumes unlimited workers and ignores resource contention. A critical path can have ties; I resolve those by name. To use another graph, I pass `--input jobs.json` with a JSON list containing `name`, `duration` and optional `dependencies` fields.

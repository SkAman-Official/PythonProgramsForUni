# Queue simulator

I wanted to see how waiting time emerges from arrivals and service durations. I use customer and event dataclasses, with a heap as the event calendar. Completions take precedence over arrivals at the same time, and equal-time arrivals keep their input order.

From the repository root, after the main setup:

```bash
python projects/02_queue_simulator/main.py
pytest projects/02_queue_simulator/tests -q
```

Example output:

```text
Customers: 3
Mean wait: 0.667; max wait: 2.000
Utilization: 1.000
```

The built-in case has waits of zero, two and zero time units. I test that calculation, empty queues, idle periods, simultaneous arrivals and deterministic generation. For a longer run I added `--customers 1000 --seed 7`. Rates default to 0.7 arrivals and 1.0 completions per time unit. This is one server with an unbounded waiting room; I do not model priorities, abandonment or confidence intervals. Finite-run utilization excludes time before the first arrival.

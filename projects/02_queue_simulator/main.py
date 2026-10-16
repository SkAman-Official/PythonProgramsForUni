"""Simulate a first-come, first-served queue with one server."""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import random
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Customer:
    """Store an arrival time and required service duration."""

    arrival: float
    service: float


@dataclass(order=True, frozen=True)
class Event:
    """Order service completions before simultaneous arrivals."""

    time: float
    priority: int
    sequence: int
    customer: int = field(compare=False)


@dataclass(frozen=True)
class Visit:
    """Record one customer's arrival, service start, and departure."""

    arrival: float
    start: float
    departure: float

    @property
    def wait(self) -> float:
        """Return the time spent waiting before service."""
        return self.start - self.arrival


@dataclass(frozen=True)
class Summary:
    """Collect average waiting time and observed server utilization."""

    count: int
    mean_wait: float
    max_wait: float
    utilization: float
    horizon: float


def simulate(customers: list[Customer]) -> list[Visit]:
    """Process arrivals and completions using a heap-based event calendar."""
    for customer in customers:
        if any(
            not math.isfinite(x) or x < 0 for x in (customer.arrival, customer.service)
        ):
            raise ValueError("Arrival and service times must be finite/nonnegative")
    calendar = [
        Event(customer.arrival, 1, i, i) for i, customer in enumerate(customers)
    ]
    heapq.heapify(calendar)
    waiting: list[int] = []
    waiting_head = 0
    busy = False
    visits: dict[int, Visit] = {}
    sequence = len(customers)

    def start_service(index: int, time: float) -> None:
        nonlocal sequence
        customer = customers[index]
        departure = time + customer.service
        visits[index] = Visit(customer.arrival, time, departure)
        heapq.heappush(calendar, Event(departure, 0, sequence, index))
        sequence += 1

    while calendar:
        event = heapq.heappop(calendar)
        if event.priority == 0:
            if waiting_head < len(waiting):
                index = waiting[waiting_head]
                waiting_head += 1
                start_service(index, event.time)
            else:
                busy = False
        elif busy:
            waiting.append(event.customer)
        else:
            busy = True
            start_service(event.customer, event.time)
    return [visits[i] for i in range(len(customers))]


def summarize(visits: list[Visit]) -> Summary:
    """Summarize a run over the interval from first arrival to last departure."""
    if not visits:
        return Summary(0, 0, 0, 0, 0)
    horizon = max(v.departure for v in visits) - min(v.arrival for v in visits)
    busy_time = sum(v.departure - v.start for v in visits)
    return Summary(
        len(visits),
        sum(v.wait for v in visits) / len(visits),
        max(v.wait for v in visits),
        busy_time / horizon if horizon else 0,
        horizon,
    )


def generate_customers(
    count: int, arrival_rate: float, service_rate: float, seed: int = 7
) -> list[Customer]:
    """Generate exponential interarrival and service times with a local RNG."""
    if count < 0 or not all(
        math.isfinite(rate) and rate > 0 for rate in (arrival_rate, service_rate)
    ):
        raise ValueError("Count must be nonnegative and rates finite/positive")
    rng = random.Random(seed)
    time = 0.0
    customers = []
    for _ in range(count):
        time += rng.expovariate(arrival_rate)
        customers.append(Customer(time, rng.expovariate(service_rate)))
    return customers


def main() -> None:
    """Run a deterministic hand-checkable example or a seeded simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=0.7)
    parser.add_argument("--service-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.customers:
        customers = generate_customers(
            args.customers, args.arrival_rate, args.service_rate, args.seed
        )
    else:
        customers = [Customer(0, 3), Customer(1, 2), Customer(5, 1)]
    summary = summarize(simulate(customers))
    LOGGER.info("Customers: %d", summary.count)
    LOGGER.info("Mean wait: %.3f; max wait: %.3f", summary.mean_wait, summary.max_wait)
    LOGGER.info("Utilization: %.3f", summary.utilization)


if __name__ == "__main__":
    main()

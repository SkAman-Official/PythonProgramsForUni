"""Manipulate sparse real polynomials and find bracketed roots."""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, init=False)
class Polynomial:
    """Store sorted nonzero (nonnegative exponent, coefficient) pairs."""

    terms: tuple[tuple[int, float], ...]

    def __init__(self, terms: Iterable[tuple[int, float]] = ()) -> None:
        combined: dict[int, float] = {}
        for exponent, coefficient in terms:
            if type(exponent) is not int or exponent < 0:
                raise ValueError("Exponents must be nonnegative integers")
            if not math.isfinite(coefficient):
                raise ValueError("Coefficients must be finite")
            combined[exponent] = combined.get(exponent, 0) + coefficient
            if not math.isfinite(combined[exponent]):
                raise ValueError("Coefficient sum overflowed")
        object.__setattr__(
            self,
            "terms",
            tuple(
                sorted(
                    (power, float(value))
                    for power, value in combined.items()
                    if value != 0
                )
            ),
        )

    @property
    def degree(self) -> int:
        """Return the largest exponent, or -1 for the zero polynomial."""
        return self.terms[-1][0] if self.terms else -1

    def __call__(self, x: float) -> float:
        """Evaluate with gap-aware Horner steps instead of filling zero terms."""
        if not math.isfinite(x):
            raise ValueError("Evaluation point must be finite")
        if not self.terms:
            return 0.0
        descending = list(reversed(self.terms))
        previous, value = descending[0]
        for exponent, coefficient in descending[1:]:
            value = value * x ** (previous - exponent) + coefficient
            previous = exponent
        return value * x**previous

    def __add__(self, other: Polynomial) -> Polynomial:
        return Polynomial(self.terms + other.terms)

    def __sub__(self, other: Polynomial) -> Polynomial:
        return self + other.scale(-1)

    def __mul__(self, other: Polynomial) -> Polynomial:
        return Polynomial((a + b, x * y) for a, x in self.terms for b, y in other.terms)

    def scale(self, factor: float) -> Polynomial:
        """Multiply every coefficient by a finite scalar."""
        if not math.isfinite(factor):
            raise ValueError("Scale must be finite")
        return Polynomial((power, value * factor) for power, value in self.terms)

    def derivative(self) -> Polynomial:
        """Return the analytic first derivative."""
        return Polynomial(
            (power - 1, power * value) for power, value in self.terms if power
        )

    def integral(self, constant: float = 0) -> Polynomial:
        """Return an antiderivative with the requested integration constant."""
        return Polynomial(
            [(0, constant)]
            + [(power + 1, value / (power + 1)) for power, value in self.terms]
        )

    def integrate(self, lower: float, upper: float) -> float:
        """Evaluate a definite integral using the analytic antiderivative."""
        primitive = self.integral()
        return primitive(upper) - primitive(lower)

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        return " + ".join(
            f"({value:g})x^{power}" for power, value in reversed(self.terms)
        )


def bisect_root(
    polynomial: Polynomial,
    lower: float,
    upper: float,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Find a sign-changing root using a bounded bisection search."""
    if not all(math.isfinite(x) for x in (lower, upper, tolerance)):
        raise ValueError("Bounds and tolerance must be finite")
    if lower >= upper or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid interval or convergence controls")
    left_value = polynomial(lower)
    right_value = polynomial(upper)
    if left_value == 0:
        return lower
    if right_value == 0:
        return upper
    if (left_value > 0) == (right_value > 0):
        raise ValueError("Interval must bracket a sign change")
    for _ in range(max_iterations):
        middle = lower / 2 + upper / 2
        value = polynomial(middle)
        if abs(value) <= tolerance or (upper - lower) / 2 <= tolerance:
            return middle
        if (value > 0) == (left_value > 0):
            lower, left_value = middle, value
        else:
            upper = middle
    raise RuntimeError("Bisection did not converge")


def parse_terms(text: str) -> Polynomial:
    """Parse comma-separated exponent:coefficient pairs."""
    if not text.strip():
        return Polynomial()
    terms = []
    for pair in text.split(","):
        power, value = pair.split(":")
        terms.append((int(power), float(value)))
    return Polynomial(terms)


def main() -> None:
    """Evaluate a polynomial and optionally solve a bracketed root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", default="2:1,0:-2")
    parser.add_argument("--x", type=float, default=2)
    parser.add_argument("--bracket", type=float, nargs=2, default=[0, 2])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    polynomial = parse_terms(args.terms)
    LOGGER.info("p(x) = %s", polynomial)
    LOGGER.info("p(%g) = %g", args.x, polynomial(args.x))
    LOGGER.info("Derivative: %s", polynomial.derivative())
    LOGGER.info("Root: %.8f", bisect_root(polynomial, *args.bracket))


if __name__ == "__main__":
    main()

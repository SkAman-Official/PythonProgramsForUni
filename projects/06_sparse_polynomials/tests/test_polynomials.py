import math

import pytest


def test_canonical_representation(project):
    polynomial = project.Polynomial([(5, 1), (5, -1), (0, 3)])
    assert polynomial.terms == ((0, 3.0),)
    assert project.Polynomial().degree == -1
    assert project.Polynomial()(10) == 0


def test_algebraic_identities(project):
    a = project.Polynomial([(3, 2), (0, -1)])
    b = project.Polynomial([(1, 3)])
    c = project.Polynomial([(0, 2)])
    assert a * (b + c) == a * b + a * c
    assert (a - a).terms == ()
    assert (a * b)(2) == pytest.approx(a(2) * b(2))


def test_calculus(project):
    p = project.Polynomial([(3, 2), (1, 4)])
    assert p.derivative()(2) == 28
    assert p.integral().derivative() == p
    assert p.integrate(0, 1) == pytest.approx(2.5)
    h = 1e-5
    assert p.derivative()(1.3) == pytest.approx((p(1.3 + h) - p(1.3 - h)) / (2 * h))


def test_bisection(project):
    p = project.parse_terms("2:1,0:-2")
    root = project.bisect_root(p, 0, 2)
    assert root == pytest.approx(math.sqrt(2), abs=1e-9)
    assert abs(p(root)) < 1e-8
    with pytest.raises(ValueError, match="bracket"):
        project.bisect_root(p, 2, 3)
    with pytest.raises(RuntimeError):
        project.bisect_root(p, 0, 2, max_iterations=1)


def test_invalid_coefficients_and_exponents(project):
    for terms in ([(-1, 2)], [(1, float("inf"))], [(True, 2)]):
        with pytest.raises(ValueError):
            project.Polynomial(terms)

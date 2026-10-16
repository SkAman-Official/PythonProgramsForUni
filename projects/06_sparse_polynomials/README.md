# Sparse polynomial algebra

I wanted a numerical example where the representation matters. I store only nonzero coefficients, merge repeated exponents and evaluate using gaps between powers. Frozen polynomial objects support addition, multiplication, differentiation and analytic integration. A bounded bisection routine finds roots when an interval changes sign.

From the repository root, after the main setup:

```bash
python projects/06_sparse_polynomials/main.py
pytest projects/06_sparse_polynomials/tests -q
```

Example output:

```text
p(x) = (1)x^2 + (-2)x^0
p(2) = 2
Derivative: (2)x^1
Root: 1.41421356
```

The default example approximates the square root of two. I test canonical zero handling, distributivity, derivative finite differences, integrals, invalid terms and convergence failures. Another example is `--terms "3:1,0:-1" --x 2 --bracket 0 2`. Coefficients use floating-point arithmetic, and exponents must be nonnegative integers. Bisection requires a sign change and can miss even-multiplicity roots. Very large powers can overflow; this is not a symbolic algebra system.

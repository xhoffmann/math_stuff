"""Performs vector and matrix operations.

2020, Xavier R. Hoffmann <xrhoffmann@gmail.com>
"""

import numpy as np

from mathstuff import root_finding

from typing import Callable, Tuple


def legendre_polynomial(x: float, n: int) -> float:
    """Evaluate n-order Legendre polynomial.

    Args:
        x: Abscissa to evaluate.
        n: Polynomial order.

    Returns:
        Value of polynomial.
    """
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        polynomials = [1, x]
        for k in range(1, n):
            new = ((2 * k + 1) * x * polynomials[-1] - k * polynomials[-2]) / (k + 1)
            polynomials.append(new)
        return polynomials[-1]


def find_bisection_bounds(
    *, n: int, polynomial: Callable
) -> Tuple[Tuple[float, float], ...]:
    """Find root bounds for n-order polynomial.

    Args:
        n: Polynomial order.
        polynomial: Function to evaluate polynomial.

    Returns:
        Bisection bounds for n roots.
    """
    k = 1
    while True:
        bounds = np.linspace(-1, 1, k * n + 1)
        values = np.array([polynomial(x, n) for x in bounds])
        mask = values[:-1] * values[1:] < 0
        if sum(mask) == n:
            return tuple(x for x in zip(bounds[:-1][mask], bounds[1:][mask]))
        else:
            k += 1


def integrate_legendre(func: Callable, n: int, a: float, b: float) -> float:
    """Integrate with Legendre polynomials of order n.

    Args:
        func: Function to integrate.
        n: Polynomial order.
        a: Left bound of integration interval.
        b: Right bound of integration interval.

    Returns:
        Value of integral.
    """
    # find bisection bounds of polynomial roots
    bounds = find_bisection_bounds(n=n, polynomial=legendre_polynomial)
    # find roots of polynomial
    roots = [
        root_finding.hybrid_secant_bisection(
            x_left=bound[0],
            x_right=bound[1],
            func=legendre_polynomial,
            func_args=(n,),
        )
        for bound in bounds
    ]
    # compute integration weights
    weights = np.array(
        [
            2.0 * (1.0 - root ** 2) / ((n * legendre_polynomial(root, n - 1)) ** 2)
            for root in roots
        ]
    )
    # compute function values
    func_values = np.array([func(0.5 * ((b - a) * root + a + b)) for root in roots])
    return 0.5 * (b - a) * sum(weights * func_values)

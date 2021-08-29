"""Performs vector and matrix operations.

2020, Xavier R. Hoffmann <xrhoffmann@gmail.com>
"""

import numpy as np

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
    n: int, polynomial: Callable
) -> Tuple[Tuple[float, float], ...]:
    """Find root bounds for n-order polynomial.

    Args:
        n: Polynomial order.
        polynomial: Function to evaluate polynomial.

    Returns:
        Value of polynomial.
    """
    k = 1
    while True:
        bounds = np.linspace(-1, 1, k * n + 1)
        values = np.array([polynomial(x, n) for x in bounds])
        left = values[:-1]
        right = values[1:]
        mask = left * right < 0
        if sum(mask) == n:
            return tuple(x for x in zip(left[mask], right[mask]))
        else:
            k += 1

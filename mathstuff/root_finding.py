"""Functions for finding roots.

2021, Xavier R. Hoffmann <xrhoffmann@gmail.com>
"""

from typing import Callable, Tuple, Dict

_EPS = 1e-10


def bisection(
    *,
    x_left: float,
    x_right: float,
    func: Callable,
    func_args: Tuple = (),
    func_kwargs: Dict = {},
    eps: float = _EPS,
) -> float:
    """Find root with bisection method.

    Args:
        x_left: Left boundary.
        x_right: Right boundary.
        func: Function to evaluate.
        func_args: Arguments for function.
        func_kwargs: Keyword arguments for function.
        eps: Required precision for root.

    Returns:
        Value of root.

    Raises:
        ValueError: If left and/or right bounds are roots.
        ValueError: If function has same sign at left and right bounds.
    """
    xl = x_left
    xr = x_right
    fl = func(xl, *func_args, **func_kwargs)
    fr = func(xr, *func_args, **func_kwargs)
    # check restrictions on bounds
    if fl == 0:
        err = "Left bound is a root."
        raise ValueError(err)
    elif fr == 0:
        err = "Right bound is a root."
        raise ValueError(err)
    if fl * fr > 0:
        err = f"Function has same sign at left ({fl}) and right ({fr}) bounds."
        raise ValueError(err)

    while abs(xr - xl) > eps:
        x2 = 0.5 * (xl + xr)
        f2 = func(x2, *func_args, **func_kwargs)
        if fl * f2 > 0:
            xl = x2
            fl = f2
        elif fr * f2 > 0:
            xr = x2
            fr = f2
        else:
            break
    return 0.5 * (xr + xl)


def hybrid_secant_bisection(
    *,
    x_left: float,
    x_right: float,
    func: Callable,
    func_args: Tuple = (),
    func_kwargs: Dict = {},
    eps: float = _EPS,
) -> float:
    """Find root with hybrid secant-bisection method.

    Requires left and right boundary where function has opposite sign.

    Args:
        x_left: Left boundary.
        x_right: Right boundary.
        func: Function to evaluate.
        func_args: Arguments for function.
        func_kwargs: Keyword arguments for function.
        eps: Required precision for root.

    Returns:
        Value of root.

    Raises:
        ValueError: If left and/or right bounds are roots.
        ValueError: If function has same sign at left and right bounds.
    """
    # prep bisection bounds
    xl = x_left
    xr = x_right
    fl = func(xl, *func_args, **func_kwargs)
    fr = func(xr, *func_args, **func_kwargs)
    # check restrictions on bounds
    if fl == 0:
        err = "Left bound is a root."
        raise ValueError(err)
    elif fr == 0:
        err = "Right bound is a root."
        raise ValueError(err)
    if fl * fr > 0:
        err = f"Function has same sign at left ({fl}) and right ({fr}) bounds."
        raise ValueError(err)

    # prep secant values
    x0 = xl
    x1 = xr
    f0 = fl
    f1 = fr
    while abs(x0 - x1) > eps:
        # compute new value
        try:
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if x2 < xl or x2 > xr:
                x2 = 0.5 * (xl + xr)
        except ZeroDivisionError:
            x2 = 0.5 * (xl + xr)
        finally:
            f2 = func(x2, *func_args, **func_kwargs)
        # update secant values
        x0, f0 = x1, f1
        x1, f1 = x2, f2
        # update bisection bounds
        if fl * f2 > 0:
            xl = x2
            fl = f2
        elif fr * f2 > 0:
            xr = x2
            fr = f2
        else:
            break
    return 0.5 * (x0 + x1)

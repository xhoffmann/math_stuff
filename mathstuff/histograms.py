"""Functions for computing histograms.

2019, Xavier R. Hoffmann <xrhoffmann@gmail.com>
"""

from typing import Sequence, Optional, Union, Tuple

import numpy as np


def binning_lin(
    *, x_min: float, x_max: float, bin_width: float, fuse_last_bin: bool = True
) -> np.ndarray:
    """Computes linear binning from range and bin width.

    All bins except last have size `bin_width`. If `fuse_last_bin` is
    False, the last bin will be smaller than or equal to `bin_width`. If
    `fuse_last_bin` is True, the remaining part will be fused to the
    previous bin.

    Args:
        x_min: Lower range of the bins.
        x_max: Upper range of the bins, must be larger
            than `x_min`.
        bin_width: Length of equal width bins, must be positive.
        fuse_last_bin: Join last and one-to-last bins.

    Returns:
        Monotonically increasing array of bin edges, including the
            rightmost edge.

    Raises:
        ValueError: If `x_max` is smaller or equal to `x_min`.
        ValueError: If `bin_width` is negative or zero.
    """
    # control requirements
    if x_max < x_min:
        err = f"x_max ({x_max}) must be larger than x_min ({x_min})."
        raise ValueError(err)
    if bin_width <= 0:
        err = f"bin_width ({bin_width}) must be positive."
        raise ValueError(err)

    num_bins = int((x_max - x_min) / bin_width)
    bins = np.array(range(num_bins + 1)) * bin_width + x_min
    if bins[-1] < x_max:
        bins = np.append(bins, (x_max,))
        if fuse_last_bin and len(bins) > 2:
            bins = np.append(bins[:-2], (bins[-1],))
    return bins


def binning_log(
    *,
    x_min: float,
    x_max: float,
    bin_width: float,
    bin_factor: float,
    fuse_last_bin: bool = True,
) -> np.ndarray:
    """Computes logarithmic binning from range, bin width, bin factor.

    All bins except last have size `bin_width * bin_factor ** k`. If
    `fuse_last_bin` is False, the last bin will be smaller than or equal
     to the corresponding sequence exponentially increasing sequence. If
    `fuse_last_bin` is True, the remaining part will be fused to the
    previous bin.


    Args:
        x_min: Lower range of the bins, must be positive.
        x_max: Upper range of the bins, must be larger
            than `x_min`.
        bin_width: Width of first bin, must be positive.
        bin_factor: Exponential increase of bin widths, must be
            larger than or equal to 1.
        fuse_last_bin: Join last and one-to-last bins.

    Returns:
        Monotonically increasing array of bin edges, including the
            rightmost edge.

    Raises:
        ValueError: If `x_min` is negative or zero.
        ValueError: If `x_max` is smaller or equal to `x_min`.
        ValueError: If `bin_width` is negative or zero.
        ValueError: If `bin_factor` is smaller than or equal to 1.
    """
    # control requirements
    if x_min <= 0:
        err = f"x_min ({x_min}) must be positive."
        raise ValueError(err)
    if x_max < x_min:
        err = f"x_max ({x_max}) must be larger than x_min ({x_min})."
        raise ValueError(err)
    if bin_width <= 0:
        err = f"bin_width ({bin_width}) must be positive."
        raise ValueError(err)
    if bin_factor <= 1:
        err = f"bin_factor ({bin_factor}) must be larger than 1."
        raise ValueError(err)

    # find upper bound for number of bins
    max_bound = int(1 + np.log((x_max - x_min) / bin_width) / np.log(bin_factor))
    bins_list = np.cumsum(bin_width * (bin_factor ** np.array(range(max_bound + 1))))
    # select bins with right edge < x_max
    mask = bins_list < (x_max - x_min)
    bins = np.append(np.append((x_min,), bins_list[mask] + x_min), (x_max,))
    if len(bins) > 2 and bins[-1] not in bins_list + x_min and fuse_last_bin:
        bins = np.append(bins[:-2], (bins[-1],))
    return bins


def binning_align(
    *, bin_edges: Sequence, x_align: Optional[str] = None, log: bool = False
) -> np.ndarray:
    """Computes the bin abscissas given the bin edges.

    Bin abscissas can be 'left', 'center' or 'right' aligned.
    Default align is 'center' for linear binning.
    Default align is 'left' for logarithmic binning.

    Args:
        bin_edges: Array of bin edges, length n.
        x_align: Abscissas align mode.
        log: If `True`, binning is logarithmic. If `False`, binning is
            linear.

    Returns:
        Array of bin abscissas, length n-1.

    Raises:
        ValueError: If `x_align` is not None, 'left', 'center'
            or 'right'.
    """
    # default align modes
    _x_align_default_lin = "center"
    _x_align_default_log = "left"

    bin_edges = np.array(bin_edges)
    if x_align is None:
        if log:
            x = binning_align(
                bin_edges=bin_edges, x_align=_x_align_default_log, log=log
            )
        else:
            x = binning_align(
                bin_edges=bin_edges, x_align=_x_align_default_lin, log=log
            )
    elif x_align == "left":
        x = bin_edges[:-1]
    elif x_align == "right":
        x = bin_edges[1:]
    elif x_align == "center":
        if log:
            x = bin_edges[:-1] * np.sqrt(bin_edges[1:] / bin_edges[:-1])
        else:
            x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    else:
        err = f"x_align mode ({x_align}) is not valid."
        raise ValueError(err)
    return x


def histogram_lin(
    *,
    x_data: Sequence,
    bins: Optional[Union[Sequence, int]] = None,
    bin_width: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    density: bool = True,
    weights: Optional[Sequence] = None,
    x_align: Optional[str] = None,
    fuse_last_bin: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a histogram with linear binning.

    Bin edges given or computed from number of equal-sized bins or
        computed from bin width.
    If `bin_width` specified, don't pass `bins`. In this case, the last
        bin will be smaller than or equal to `bin_width` (if not
        `fuse_last_bin`, else last and one-to-last are fused).
    Range may be specified through `x_min` and `x_max`, else defaults to
        min and max of input data.
    Bin abscissas can be 'left', 'center' or 'right' aligned.

    Args:
        x_data: Input data. The histogram is computed over the flattened
            array.
        bins: If `bins` is an int, it defines the number of equal-width
            bins in the given range. If `bins` is a sequence, it defines
            a monotonically increasing array of bin edges, including the
            rightmost edge, allowing for non-uniform bin widths (x_min
            and x_max are ignored). If `bins`is None, `bin_width` is
            required.
        bin_width: Length of equal width bins, must be positive.
        x_min: Lower range of the bins, smaller values are ignored.
        x_max: Upper range of the bins, larger values are ignored, must
            be larger than x_min.
        density: If ``False``, the result will contain the number of
            samples in each bin. If ``True``, the result is the value of
            the probability *density* function at the bin, normalized
            such that the *integral* over the range is 1.
        x_align: Abscissas align mode, can be 'left', 'center' or
            'right'. Defaults to 'center'.
        weights: An array of weights, of the same shape as `x_data`.
            Each value in `x_data` only contributes its associated
            weight towards the bin count (instead of 1).
        fuse_last_bin: Join last and one-to-last bins if ``True``.

    Returns:
        Abscissas, counts/pdf.

    Raises:
        ValueError: If `bin_width` is negative or zero.
        ValueError: If type of `bins` not supported.

    Notes:
        All bins except last are half-open intervals [·,·), last bin is
            closed interval [·,·].
    """
    # obtain x_min and x_max
    if x_min is None:
        x_min = min(x_data)
    if x_max is None or x_max < x_min:
        x_max = max(x_data)

    # construct bins_list
    if bins is None:
        # construct list from bin_width
        if bin_width is None or bin_width <= 0:
            err = f"bin_width ({bin_width}) must be positive."
            raise ValueError(err)
        bins_list = binning_lin(
            x_min=x_min, x_max=x_max, bin_width=bin_width, fuse_last_bin=fuse_last_bin
        )
    elif type(bins) is int:
        # construct list from number of bins
        bins_list = np.linspace(x_min, x_max, bins + 1)
    elif hasattr(bins, "__len__"):
        # cast list to numpy array
        bins_list = np.array(bins)
        bins_list.sort()
    else:
        # raise error
        err = f"Unsupported bins type ({type(bins)})."
        raise ValueError(err)

    y, bin_edges = np.histogram(
        x_data, bins=bins_list, density=density, weights=weights
    )
    x = binning_align(bin_edges=bin_edges, x_align=x_align, log=False)
    return x, y

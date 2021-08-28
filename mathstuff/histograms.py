"""Functions for computing histograms.

2019, Xavier R. Hoffmann <xrhoffmann@gmail.com>
"""

from typing import Sequence, Optional

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
    if len(bins) > 2 and bins[-1] not in bins_list+x_min and fuse_last_bin:
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
            x = binning_align(bin_edges=bin_edges, x_align=_x_align_default_log, log=log)
        else:
            x = binning_align(bin_edges=bin_edges, x_align=_x_align_default_lin, log=log)
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



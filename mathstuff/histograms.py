"""Functions for computing histograms.

2019, Xavier R. Hoffmann <xrhoffmann@gmail.com>
"""

import numpy as np


def binning_lin(
    *, x_min: float, x_max: float, bin_width: float, fuse_last_bin: bool = False
) -> np.ndarray:
    """
    Computes linear binning from range (start and end) and bin width.

    All bins except last have size `bin_width`. If `fuse_last_bin` is
    False, the last bin will be smaller than or equal to `bin_width`. If
    `fuse_last_bin` is True, the remaining part will be fused to the
    one-to-last bin.

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
        if fuse_last_bin:
            bins[-1] = x_max
        else:
            bins = np.append(bins, (x_max,))
    return bins



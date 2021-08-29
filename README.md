# math_stuff
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Various numerical mathematical stuff: root finding, histograms, triangular matrices, Gaussian quadratures, etc.  

2021, Xavier R. Hoffmann <xrhoffmann@gmail.com>


## `root_finding`

Module to find roots of univariate functions.  Two methods are implemented:
- bisection (`bisection`)
- hybrid secant-bisection (`hybrid_secant_bisection`)

See [docs](https://github.com/xhoffmann/math_stuff/blob/main/docs/root_finding.pdf) for details.

## `histograms`

Module to compute histograms and related operations.
- linear binning from a given range (`binning_lin`)
- logarithmic binning from a given range (`binning_log`)
- left, center or right align bin abscissas (linear or logarithmic) (`binning_align`)
- compute a histogram with linear bins (`histrogram_lin`)
- compute a histogram with logarithmic bins (`histogram_log`)
- transform a linear histogram into another linear histogram (`histogram_lin2lin`)
- transform a linear histogram into a logarithmic histogram (`histrogram_lin2log`)

## `triangle_matrix`

Module to compute sums and transformations with vectors and triangular matrices.
- reverse cumulative sum of a vector (`reverse_cumsum`)
- matrix-vector product of upper-triangular terms (`triangular_dot`)
- sum rows of upper-triangular matrix terms (`triangular_sum_rows`)
- sum columns of upper-triangular matrix terms (`triangular_sum_colums`)
- sum terms of upper-triangular sub-matrices (`triangular_sum_chunks`)

See docs for details.

## `guassian_quadratures`

Module to integrate univariate functions using Gaussian quadratures.

- CODE
- DOCS: adapt
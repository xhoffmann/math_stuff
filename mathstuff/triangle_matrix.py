"""Performs vector and matrix operations.

2020, Xavier Hoffmann <xrhoffmann@gmail.com>
"""

# external imports
import numpy as np


def shift_vector(vec: np.ndarray, shift: int) -> np.ndarray:
    """Shifts positions of a 1d array.

    If `shift` is negative, values move down.
    Replaces null values with 0.
    Output vector has same length as input vector.

    Args:
        vec: Input vector to shift, 1d array
        shift: Shift positions up (positive) or down (negative).

    Returns:
        Shifted vector.

    Note:
        With shift=k, w(i)=v(i+k).
    """
    if shift < 0:
        return np.append(np.zeros_like(vec[shift:]), vec[:shift])
    elif shift > 0:
        return np.append(vec[shift:], np.zeros_like(vec[:shift]))
    else:
        return vec


def shift_matrix_rows(mat: np.ndarray, shift: int) -> np.ndarray:
    """Shifts rows of a 2d array.

    If `shift` is negative, rows move down.
    Replaces null values with 0.
    Output array has same shape as input array.

    Args:
        mat: Input matrix to shift, 2d array
        shift: Shift rows up (positive) or down (negative).

    Returns:
        Shifted matrix.

    Note:
        With shift=k, B(i,j)=A(i+k,j).
    """
    if shift < 0:
        aux = np.roll(mat, -shift, axis=0)
        aux[:-shift, :] = 0.0
        return aux
    elif shift > 0:
        aux = np.roll(mat, -shift, axis=0)
        aux[-shift:, :] = 0.0
        return aux
    else:
        return mat


def reverse_cumsum(vec: np.ndarray, shift: int = 0) -> np.ndarray:
    """Anti-cumulative sum of (shifted) vector.

    Sums from position forward.
    Shifts positions of summed vector.
    Replaces null values with 0.
    Output vector has same length as input vector.

    Args:
        vec: Input vector to sum, 1d array.
        shift: Shift positions up (positive) or down (negative).

    Returns:
        Anti-cumulative sum of input vector, 1d array.

    Note:
        With shift=k, w(i)=sum_{j=i+k)^{end}v(j).
    """
    aux = np.flip(np.cumsum(np.flip(vec)))
    return shift_vector(aux, shift)


def triangular_dot(mat: np.ndarray, vec: np.ndarray, shift: int = 0) -> np.ndarray:
    """(Shifted) Matrix product of upper-triangular terms.

    Computes matrix product reduced to upper-triangular terms.
    Shifts positions of summed vector.
    Replaces null values with 0.
    Output array has same shape as input array.

    Args:
        mat: Arbitrary square matrix, 2d array.
        vec: Arbitrary vector, 1d array.
        shift: Shift rows up (positive) or down (negative).

    Returns:
        Shifted

    Note:
        With shift=k, w(i)=sum_{j=i+k}^{end}A(i+k,j)*v(j).
    """
    aux_mat = np.triu(mat)
    aux_vec = np.dot(aux_mat, vec)
    return shift_vector(aux_vec, shift)


def triangular_sum_rows(mat: np.ndarray, shift: int = 0) -> np.ndarray:
    """Sum matrix rows of upper-triangular terms.

    Sums matrix by rows, reduced to upper-triangular terms.
    Shifts positions of summed vector.
    Replaces null values with 0.

    Args:
        mat: Arbitrary square matrix, 2d array.
        shift: Shift positions up (positive) or down (negative).

    Returns:
        Summed vector, 1d array.

    Note:
        With shift=k, w(i)=sum_{j=i+k}^{end}A(i+k,j).
    """
    aux_mat = np.triu(mat)
    aux_vec = np.sum(aux_mat, axis=1)
    return shift_vector(aux_vec, shift)


def triangular_sum_columns(mat: np.ndarray, row_shift: int = 0) -> np.ndarray:
    """Sum matrix columns of triangular terms.

    Sums matrix by columns, reduced to triangular terms.
    Optionally, exclude above-diagonal or include below-diagonal terms.

    Args:
        mat: Arbitrary square matrix, 2d array.
        row_shift: Include below-diagonal terms (positive) or exclude
            above-diagonal terms (negative).

    Returns:
        Summed vector, 1d array.

    Note:
        With row_shift=k, w(i)=sum_{j=1}^{i+k}M(j,i).
    """
    aux = np.triu(mat, k=-row_shift)
    return np.sum(aux, axis=0)


def triangular_sum_chunks(
    mat: np.ndarray, row_shift: int = 0, col_shift: int = 0
) -> np.ndarray:
    """Sum matrix as upper-triangular chunks.

    Sum submatrix defined by diagonal element and upper-right corner.
    Optionally, exclude above-diagonal or include below-diagonal terms.
    Replaces null values with 0.
    Output array has same shape as input array.

    Args:
        mat: Arbitrary square matrix, 2d array.
        row_shift: Include below-diagonal terms (positive) or exclude
            above-diagonal terms (negative).
        col_shift: Include left-diagonal terms (positive) or exclude
            right-diagonal terms (negative).

    Returns:
        Summed matrix, 2d array.

    Note:
        With With row_shift=k_row and col_shift=k_col,
            w(i)=sum_{m=1}^{i+k_row}sum_{j=i+k_col}^{end}A(m,j)
    """
    aux_mat = np.apply_along_axis(reverse_cumsum, 1, mat)
    aux_shift = col_shift - row_shift
    aux_mat = np.triu(aux_mat, k=aux_shift)
    aux_vec = np.sum(aux_mat, axis=0)
    return shift_vector(aux_vec, shift=col_shift)

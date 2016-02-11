""":py:mod:`maxvolpy.aux` module contains some auxiliary routines, useful when working with different maxvol functions from :py:mod:`maxvolpy.maxvol` and when building approximations with help of functions from :py:mod:`maxvolpy.cross`."""
from __future__ import print_function, division, absolute_import

__all__ = ['svd_cut', 'reduce_approximation']

import numpy as np

def svd_cut(A, tol, norm = 2):
    """Computes singular values decomposition of matrix **A** and returns only largest singular values and vectors, corresponding to relative tolerance **tol**.

:Parameters:
    **A**: *numpy.ndarray*
        Real or complex matrix or matrix-like object.
    **tol**: *float*
        Tolerance of cutting singular values operation.
    **norm**: {2, 'fro'}, optional
        Defines norm, that is chosen when cutting singular values.
:Returns:
    **U**: *numpy.ndarray*
        Left singular vectors, corresponding to largest singular values.
    **S**: *numpy.ndarray*
        Largest singular values.
    **V**: *numpy.ndarray*
        Right singular vectors, corresponding to largest singular values."""
    U, S, V = np.linalg.svd(A, full_matrices = 0)
    if norm == 2:
        S1 = S[::-1]
        tmp_S = tol*S1[-1]
        rank = S1.shape[0] - S1.searchsorted(tmp_S, 'left')
    elif norm == 'fro':
        S1 = (S*S)[::-1]
        for i in range(1, S1.shape[0]):
            S1[i] += S1[i-1]
        tmp_S = S1[-1]*tol*tol
        rank = S1.shape[0] - S1.searchsorted(tmp_S, 'left')
    else:
        raise ValueError("Invalid parameter norm value")
    return U[:,:rank].copy(), S[:rank].copy(), V[:rank].copy()

def reduce_approximation(U, V, tol):
    """Performs :py:func:`svd_cut` procedure for matrix **U**.dot(**V**)

:Parameters:
    **U**, **V**: *numpy.ndarray*
        Two matrices, such that **U**.dot(**V**) makes sense.
    **tol**: *float*
        Tolerance of cutting singular values of a matrix **U**.dot(**V**).
:Returns:
     **U**: *numpy.ndarray*
        Left singular vectors, corresponding to largest singular values.
    **S**: *numpy.ndarray*
        Largest singular values.
    **V**: *numpy.ndarray*
        Right singular vectors, corresponding to largest singular values."""
    Q1, R1 = np.linalg.qr(U)
    U1, S1, V1 = svd_cut(R1.dot(V), tol)
    return Q1.dot(U1), S1, V1
    

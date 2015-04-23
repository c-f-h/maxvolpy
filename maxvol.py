# setting functions and variables for 'import *'
from __future__ import division, print_function
__all__ = ['py_rect_maxvol', 'py_maxvol', 'rect_maxvol', 'maxvol', 'rect_maxvol_svd', 'maxvol_svd', 'rect_maxvol_qr', 'maxvol_qr']
from .aux import svd_cut

def py_rect_maxvol(A, tol = 1., maxK = None, min_add_K = None, minK = None, start_maxvol_iters = 10, identity_submatrix = True, top_k_index = -1):
    """Python implementation of rectangular 2-volume maximization. For information see :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>` function. Can be used to learn rect_maxvol algorithm."""
    # tol2 - square of parameter tol
    tol2 = tol**2
    # N - number of rows, r - number of columns of matrix A
    N, r = A.shape
    # some work on parameters
    if N <= r:
        return np.arange(N, dtype = np.int32), np.eye(N, dtype = A.dtype)
    if maxK is None or maxK > N:
        maxK = N
    if maxK < r:
        maxK = r
    if minK is None or minK < r:
        minK = r
    if minK > N:
        minK = N
    if min_add_K is not None:
        minK = max(minK, r + min_add_K) 
    if minK > maxK:
        minK = maxK
        #raise ValueError('minK value cannot be greater than maxK value')
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r
    # choose initial submatrix and coefficients according to maxvol algorithm
    index = np.zeros(N, dtype = np.int32)
    chosen = np.ones(top_k_index)
    tmp_index, C = py_maxvol(A, 1.05, start_maxvol_iters, top_k_index)
    index[:r] = tmp_index
    chosen[tmp_index] = 0
    C = np.asfortranarray(C)
    # compute square 2-norms of each row in coefficients matrix C
    row_norm_sqr = np.array([chosen[i]*np.linalg.norm(C[i], 2)**2 for i in range(top_k_index)])
    # find maximum value in row_norm_sqr
    i = np.argmax(row_norm_sqr)
    K = r
    # set cgeru or zgeru for complex numbers and dger or sger for float numbers
    try:
        ger = get_blas_funcs('geru', [C])
    except:
        ger = get_blas_funcs('ger', [C])
    # augment maxvol submatrix with each iteration
    while (row_norm_sqr[i] > tol2 and K < maxK) or K < minK:
        # add i to index and recompute C and square norms of each row by SVM-formula
        index[K] = i
        chosen[i] = 0
        c = C[i].copy()
        v = C.dot(c.conj())
        l = 1.0/(1+v[i])
        ger(-l,v,c,a=C,overwrite_a=1)
        C = np.hstack([C, l*v.reshape(-1,1)])
        row_norm_sqr -= (l*v[:top_k_index]*v[:top_k_index].conj()).real
        row_norm_sqr *= chosen
        # find maximum value in row_norm_sqr
        i = row_norm_sqr.argmax()
        K += 1
    # parameter identity_submatrix is True, set submatrix, corresponding to maxvol rows, equal to identity matrix
    if identity_submatrix:
        C[index[:K]] = np.eye(K, dtype = C.dtype)
    return index[:K].copy(), C

def py_maxvol(A, tol = 1.05, max_iters = 100, top_k_index = -1):
    """Python implementation of 1-volume maximization. For information see :py:func:`maxvol()<maxvolpy.maxvol.maxvol>` function. Can be used to learn maxvol algorithm."""
    # some work on parameters
    if tol < 1:
        tol = 1.0
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype = np.int32), np.eye(N, dtype = A.dtype)
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r
    # set auxiliary matrices and get corresponding *GETRF function from lapack
    B = np.copy(A[:top_k_index], order = 'F')
    C = np.copy(A.T, order = 'F')
    H, ipiv, info = get_lapack_funcs('getrf', [B])(B, overwrite_a = 1)
    # compute pivots from ipiv (result of *GETRF)
    index = np.arange(N, dtype = np.int32)
    for i in range(r):
        tmp = index[i]
        index[i] = index[ipiv[i]]
        index[ipiv[i]] = tmp
    # solve A = CH, H is in LU format
    B = H[:r]
    # It will be much faster to use *TRSM instead of *TRTRS
    trtrs = get_lapack_funcs('trtrs', [B])
    trtrs(B, C, trans = 1, lower = 0, unitdiag = 0, overwrite_b = 1)
    trtrs(B, C, trans = 1, lower = 1, unitdiag = 1, overwrite_b = 1)
    # C has shape (r, N) -- it is stored transposed
    # find max value in C
    i, j = divmod(abs(C[:,:top_k_index]).argmax(), top_k_index)
    # set cgeru or zgeru for complex numbers and dger or sger for float numbers
    try:
        ger = get_blas_funcs('geru', [C])
    except:
        ger = get_blas_funcs('ger', [C])
    # set number of iters to 0
    iters = 0
    # check if need to swap rows
    while abs(C[i,j]) > tol and iters < max_iters:
        # add j to index and recompute C by SVM-formula
        index[i] = j
        tmp_row = C[i].copy()
        tmp_column = C[:,j].copy()
        tmp_column[i] -= 1.
        alpha = -1./C[i,j]
        ger(alpha, tmp_column, tmp_row, a = C, overwrite_a = 1)
        iters += 1
        i, j = divmod(abs(C[:,:top_k_index]).argmax(), top_k_index)
    return index[:r].copy(), C.T

def rect_maxvol(A, tol = 1., maxK = None, min_add_K = None, minK = None, start_maxvol_iters = 10, identity_submatrix = True, top_k_index = -1):
    """Computes rectangular submatrix :math:`H` of maximum 2-volume in given nonsingular matrix :math:`A`.
Returns :math:`H` as rows of matrix :math:`A`.
Uses greedy maximization of the 2-volume and Sherman-Woodbury-Morrison formula for fast rank-1 update of pseudo-inverse matrices.
Uses cython-compiled function :py:func:`c_rect_maxvol()<maxvolpy.maxvol.c_rect_maxvol>` if extension was succesfully compiled and :py:func:`py_rect_maxvol()<maxvolpy.maxvol.py_rect_maxvol>` otherwise.

:Parameters:
    **A**: :math:`(N, r)` *ndarray*
        Real or complex matrix, :math:`N \ge r`.
    **tol**: *float*
        Upper bound for euclidian norm of rows of resulting matrix :math:`C`.
    **maxK**: *integer*
        Maximum number of rows in submatrix :math:`H`.
    **minK**: *integer*
        Minimum number of rows in submatrix :math:`H`.
    **min_add_K**: *integer*
        Minimum number of rows to add to the square submatrix. :math:`H` will have minimum of :math:`r` +min_add_K rows.
    **start_maxvol_iters**: *integer*
        How many iterations of square maxvol (optimization of 1-volume) is required to be done before actual rectangular 2-volume maximization.
    **identity_submatrix**: *bool*
        If true, submatrix :math:`C[piv]` of returned matrix :math:`C` will be equal to identity.
    **top_k_index**: *integer*
        Rows, returned by :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`, will be in range from 0 to (**top_k_index**-1). This restriction is ignored, if **top_k_index** == -1
:Returns:
    **piv**: :math:`(K,)` *array of integers*
        Rows of matrix :math:`A`, maximizing 2-volume. :math:`H = A[piv]`.
    **C**: :math:`(N, K)` *ndarray*
        Matrix of coefficients, such that :math:`A = C.dot(H)`.

>>> import numpy as np
>>> from maxvolpy.maxvol import rect_maxvol
>>> np.random.seed(100)
>>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
>>> piv, C = rect_maxvol(a, 1.0)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 7.64870068114e-16
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 0.992612890258
>>> piv, C = rect_maxvol(a, 1.5)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 7.91336090209e-16
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 1.47552960395
>>> piv, C = rect_maxvol(a, 2.0)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 7.10992920009e-16
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 1.98552688328
"""
    return rect_maxvol_func(A, tol, maxK, min_add_K, minK, start_maxvol_iters, identity_submatrix, top_k_index)

def maxvol(A, tol = 1.05, max_iters = 100, top_k_index = -1):
    """Computes square submatrix :math:`H` of maximum 1-volume in given nonsingular matrix :math:`A`.
Returns :math:`H` as rows of matrix :math:`A`.
Uses greedy maximization of the 1-volume and Sherman-Woodbury-Morrison formula for fast rank-1 update of inverse matrices.
Uses cython-compiled function :py:func:`c_maxvol()<maxvolpy.maxvol.c_maxvol>` if extension was succesfully compiled and :py:func:`py_maxvol()<maxvolpy.maxvol.py_maxvol>` otherwise.

:Parameters:
    **A**: :math:`(N, r)` *ndarray*
        Real or complex matrix, :math:`N \ge r`.
    **tol**: *float*
        Upper bound for Chebyshev norm of resulting matrix :math:`C`. Minimum value is 1.
    **max_iters**: *integer*
        Maximum number of iterations. Each iteration swaps 2 rows.
    **top_k_index**: *integer*
        Rows, returned by :py:func:`maxvol()<maxvolpy.maxvol.maxvol>`, will be in range from 0 to (**top_k_index**-1). This restriction is ignored, if **top_k_index** == -1
:Returns:
    **piv**: :math:`(r,)` *array of integers*
        Rows of matrix :math:`A`, maximizing 1-volume. :math:`H = A[piv]`.
    **C**: :math:`(N, r)` *ndarray*
        Matrix of coefficients, such that :math:`A = C.dot(H)`.

>>> import numpy as np
>>> from maxvolpy.maxvol import maxvol
>>> np.random.seed(100)
>>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
>>> piv, C = maxvol(a, 1.0)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 1.1169921841e-15
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.0
>>> piv, C = maxvol(a, 1.05)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 1.20365952104e-15
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.0464068226
>>> piv, C = maxvol(a, 1.10)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 9.97068873972e-16
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.07853911436"""
    return maxvol_func(A, tol = tol, max_iters = max_iters, top_k_index = top_k_index)

def rect_maxvol_svd(A, svd_tol = 1e-3, tol = 1, maxK = None, min_add_K = None, minK = None, start_maxvol_iters = 10, identity_submatrix = True, job = 'F', top_k_index = -1):
    """Computes SVD for **top_k_index** rows and/or columns of given matrix **A**, cuts off singular values, lower than **svd_tol** (relatively, getting only **r** highest singular vectors), projects rows and/or columns, starting from **top_k_index**, to space of first **top_k_index** rows and/or columns and runs :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>` for left and/or right singular vectors.

:Parameters:
    **A**: :math:`(N, M)` *ndarray*
        Real or complex matrix.
    **svd_tol**: *float*
        Cut-off singular values parameter.
    **tol**: *float*
        Sets upper bound for euclidian norm of rows of resulting matrix or matrices :math:`C`.
    **maxK**: *integer*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. Maximum number of rows or columns in submatrix or submatrices :math:`H`.
    **minK**: *integer*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. Minimum number of rows or columns in submatrix or submatrices :math:`H`.
    **min_add_K**: *integer*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. Minimum number of rows or columns to add to the square submatrix or submatrices.
    **start_maxvol_iters**: *integer*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. How many iterations of square maxvol (optimization of 1-volume) is required to do before actual rectangular 2-volume maximization.
    **identity_submatrix**: *bool*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. If true, submatrix :math:`C[piv]` of returned matrix :math:`C` will be equal to identity.
    **job**: *character*
        'R' for right singular vectors, 'C' for left singular vectors, 'F' for both of them.
    **top_k_index**: *integer*
        Rows and/or columns, returned by :py:func:`rect_maxvol_svd()<maxvolpy.maxvol.rect_maxvol_svd>`, will be in range from 0 to (**top_k_index**-1). This restriction is ignored, if **top_k_index** == -1
:Returns:
    Depending on **job** parameter, returns rows **piv** and matrix **C** for left and/or right singular vectors.

    **piv**: :math:`(r,)` *array of integers*
        Rows or columns of matrix :math:`A`, maximizing 2-volume. :math:`H = A[piv]` or :math:`H = A[:, piv]`.
    **C**: :math:`(N, r)` *ndarray*
        Matrix of coefficients, such that :math:`A \\approx C.dot(H)` or :math:`A \\approx H.dot(C.T.conj())`.

>>> import numpy as np
>>> from maxvolpy.maxvol import rect_maxvol_svd
>>> np.random.seed(100)
>>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
>>> piv, C = rect_maxvol_svd(a, svd_tol = 1e-1, tol = 1.0, job = 'R')
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 0.145389409144
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 0.977165148845
>>> piv, C = rect_maxvol_svd(a, svd_tol = 1e-1, tol = 1.5, job = 'R')
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 0.188595873914
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 1.40581472836
>>> piv, C = rect_maxvol_svd(a, svd_tol = 1e-1, tol = 2.0, job = 'R')
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 0.226287540714
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 1.85668430673
"""
    if job == 'R':
        if top_k_index == -1:
            top_k_index = A.shape[0]
        # compute largest singular values and vectors for first top_k_index rows
        U, S, V = svd_cut(A[:top_k_index], svd_tol)
        # find projection coefficients of all other rows to subspace of largest singular vectors of first rows
        B = A[top_k_index:].dot(V.T.conj())*(1.0/S).reshape(1,-1)
        # apply rect_maxvol for projection coefficients
        return rect_maxvol(np.vstack([U, B]), tol, maxK, min_add_K, minK, start_maxvol_iters, identity_submatrix, top_k_index)
    elif job == 'C':
        if top_k_index == -1:
            top_k_index = A.shape[1]
        # compute largest singular values and vectors for first top_k_index columns
        U, S, V = svd_cut(A[:,:top_k_index], svd_tol)
        # find projection coefficients of all other columns to subspace of largest singular vectors of first columns
        B = (1.0/S).reshape(-1,1)*U.T.conj().dot(A[:,top_k_index:])
        # apply rect_maxvol for projection coefficients
        return rect_maxvol(np.vstack([V.T.conj(), B.T.conj()]), tol, maxK, min_add_K, minK, start_maxvol_iters, identity_submatrix, top_k_index)
    elif job == 'F':
        # procede with job = 'R' and job = 'C' simultaneously
        if top_k_index != -1:
            value0, value1 = rect_maxvol_svd(A, svd_tol, tol, maxK, min_add_K, minK, start_maxvol_iters, identity_submatrix, 'R', top_k_index)
            value2, value3 = rect_maxvol_svd(A, svd_tol, tol, maxK, min_add_K, minK, start_maxvol_iters, identity_submatrix, 'C', top_k_index)
            return value0, value1, value2, value3
        U, S, V = svd_cut(A, svd_tol)
        value0, value1 = rect_maxvol(U, tol, maxK, min_add_K, minK, start_maxvol_iters, identity_submatrix, top_k_index)
        value2, value3 = rect_maxvol(V.T.conj(), tol, maxK, min_add_K, minK, start_maxvol_iters, identity_submatrix, top_k_index)
        return value0, value1, value2, value3

def maxvol_svd(A, svd_tol = 1e-3, tol = 1.05, max_iters = 100, job = 'F', top_k_index = -1):
    """Computes SVD for **top_k_index** rows and/or columns of given matrix **A**, cuts off singular values, lower than **svd_tol** (getting only **r** highest singular vectors), projects rows and/or columns, starting from **top_k_index**, to space of first **top_k_index** rows and/or columns and runs :py:func:`maxvol()<maxvolpy.maxvol.maxvol>` for left and/or right singular vectors.

:Parameters:
    **A**: :math:`(N, M)` *ndarray*
        Real or complex matrix.
    **svd_tol**: *float*
        cut-off singular values parameter.
    **tol**: *float*
        Parameter for :py:func:`maxvol()<maxvolpy.maxvol.maxvol>`. Sets upper bound for Chebyshev norm of resulting matrix or matrices :math:`C`.
    **max_iters**: *integer*
        Parameter for :py:func:`maxvol()<maxvolpy.maxvol.maxvol>`. Maximum number of iterations.
    **job**: *character*
        'R' for right singular vectors, 'C' for left singular vectors, 'F' for both of them.
    **top_k_index**: *integer*
        Rows and/or columns, returned by :py:func:`maxvol_svd()<maxvolpy.maxvol.maxvol_svd>`, will be in range from 0 to (**top_k_index**-1). This restriction is ignored, if **top_k_index** == -1
:Returns:
    Depending on **job** parameter, returns rows **piv** and matrix **C** for left and/or right singular vectors.

    **piv**: :math:`(r,)` *array of integers*
        Rows or columns of matrix :math:`A`, maximizing 1-volume. :math:`H = A[piv]` or :math:`H = A[:, piv]`.
    **C**: :math:`(N, r)` *ndarray*
        Matrix of coefficients, such that :math:`A \\approx C.dot(H)` or :math:`A \\approx H.dot(C.T.conj())`.

>>> import numpy as np
>>> from maxvolpy.maxvol import maxvol_svd
>>> np.random.seed(100)
>>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
>>> piv, C = maxvol_svd(a, svd_tol = 1e-1, tol = 1.0, job = 'R')
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 0.246839403405
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.0
>>> piv, C = maxvol_svd(a, svd_tol = 1e-1, tol = 1.05, job = 'R')
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 0.246839403405
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.0
>>> piv, C = maxvol_svd(a, svd_tol = 1e-1, tol = 1.10, job = 'R')
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 0.25484511962
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.06824913975
"""
    if tol < 1:
        tol = 1.0
    if job == 'R':
        if top_k_index == -1:
            top_k_index = A.shape[0]
        # compute largest singular values and vectors for first top_k_index rows
        U, S, V = svd_cut(A[:top_k_index], svd_tol)
        # find projection coefficients of all other rows to subspace of largest singular vectors of first rows
        B = A[top_k_index:].dot(V.T.conj())*(1.0/S).reshape(1,-1)
        # apply maxvol for projection coefficients
        return maxvol(np.vstack([U, B]), tol, max_iters, top_k_index)
    elif job == 'C':
        if top_k_index == -1:
            top_k_index = A.shape[1]
        # compute largest singular values and vectors for first top_k_index columns
        U, S, V = svd_cut(A[:,:top_k_index], svd_tol)
        # find projection coefficients of all other columns to subspace of largest singular vectors of first columns
        B = (1.0/S).reshape(-1,1)*U.T.conj().dot(A[:,top_k_index:])
        # apply rect_maxvol for projection coefficients
        return maxvol(np.vstack([V.T.conj(), B.T.conj()]), tol, max_iters, top_k_index)
    elif job == 'F':
        # procede with job = 'R' and job = 'C' simultaneously
        if top_k_index != -1:
            value0, value1 = maxvol_svd(A, svd_tol, tol, max_iters, 'R', top_k_index)
            value2, value3 = maxvol_svd(A, svd_tol, tol, max_iters, 'C', top_k_index)
            return value0, value1, value2, value3
        U, S, V = svd_cut(A, svd_tol)
        value0, value1 = maxvol(U, tol, max_iters, top_k_index)
        value2, value3 = maxvol(V.T.conj(), tol, max_iters, top_k_index)
        return value0, value1, value2, value3

def rect_maxvol_qr(A, tol = 1, maxK = None, min_add_K = None, minK = None, start_maxvol_iters = 10, identity_submatrix = True, top_k_index = -1):
    """Computes QR for given matrix **A**  and runs :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>` for factor **Q**.
Easy way to make nonsingular matrix to call :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>` without errors.

:Parameters:
    **A**: :math:`(N, r)` *ndarray*
        Real or complex matrix, :math:`N \ge r`.
    **tol**: *float*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. Sets upper bound for euclidian norm of rows of resulting matrix or matrices :math:`C`.
    **maxK**: *integer*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. Maximum number of rows or columns in submatrix or submatrices :math:`H`.
    **minK**: *integer*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. Minimum number of rows or columns in submatrix or submatrices :math:`H`.
    **min_add_K**: *integer*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. Minimum number of rows or columns to add to the square submatrix or submatrices.
    **start_maxvol_iters**: *integer*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. How many iterations of square maxvol (optimization of 1-volume) is required to do before actual rectangular 2-volume maximization.
    **identity_submatrix**: *bool*
        Parameter for :py:func:`rect_maxvol()<maxvolpy.maxvol.rect_maxvol>`. If true, submatrix :math:`C[piv]` of returned matrix :math:`C` will be equal to identity.
    **top_k_index**: *integer*
        Rows, returned by :py:func:`rect_maxvol_qr()<maxvolpy.maxvol.rect_maxvol_qr>`, will be in range from 0 to (**top_k_index**-1). This restriction is ignored, if **top_k_index** == -1
:Returns:
    **piv**: :math:`(r,)` *array of integers*
        Rows of matrix :math:`A`, maximizing 2-volume. :math:`H = A[piv]`.
    **C**: :math:`(N, r)` *ndarray*
        Matrix of coefficients, such that :math:`A = C.dot(H)`.

>>> import numpy as np
>>> from maxvolpy.maxvol import rect_maxvol_qr
>>> np.random.seed(100)
>>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
>>> piv, C = rect_maxvol_qr(a, 1.0)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 1.08443862923e-15
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 0.992612890258
>>> piv, C = rect_maxvol_qr(a, 1.5)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 1.09090399035e-15
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 1.47552960395
>>> piv, C = rect_maxvol_qr(a, 2.0)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 1.05516303649e-15
>>> print('maximum euclidian norm of row in matrix C: {}'.format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
maximum euclidian norm of row in matrix C: 1.98552688328
"""
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype = np.int32), np.eye(N, dtype = A.dtype)
    Q = np.linalg.qr(A)[0]
    return rect_maxvol(Q, tol, maxK, min_add_K, minK, start_maxvol_iters, identity_submatrix, top_k_index)

def maxvol_qr(A, tol = 1.05, max_iters = 100, top_k_index = -1):
    """Computes QR for given matrix **A**  and runs :py:func:`maxvol()<maxvolpy.maxvol.maxvol>` for factor **Q**.
Easy way to make nonsingular matrix to call :py:func:`maxvol()<maxvolpy.maxvol.maxvol>` without errors.

:Parameters:
    **A**: :math:`(N, r)` *ndarray*
        Real or complex matrix, :math:`N \ge r`.
    **tol**: *float*
        Parameter for :py:func:`maxvol()<maxvolpy.maxvol.maxvol>`. Sets upper bound for Chebyshev norm of resulting matrix or matrices :math:`C`. Minimum value is 1.
    **max_iters**: *integer*
        Parameter for :py:func:`maxvol()<maxvolpy.maxvol.maxvol>`. Maximum number of iterations.
    **top_k_index**: *integer*
        Rows, returned by :py:func:`maxvol_qr()<maxvolpy.maxvol.maxvol_qr>`, will be in range from 0 to (**top_k_index**-1). This restriction is ignored, if **top_k_index** == -1
:Returns:
    **piv**: :math:`(r,)` *array of integers*
        Rows of matrix :math:`A`, maximizing 1-volume. :math:`H = A[piv]`.
    **C**: :math:`(N, r)` *ndarray*
        Matrix of coefficients, such that :math:`A = C.dot(H)`.

>>> import numpy as np
>>> from maxvolpy.maxvol import maxvol_qr
>>> np.random.seed(100)
>>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
>>> piv, C = maxvol_qr(a, 1.0)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 1.42187726922e-15
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.0
>>> piv, C = maxvol_qr(a, 1.05)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 1.45909734856e-15
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.0464068226
>>> piv, C = maxvol_qr(a, 1.10)
>>> print('relative maxvol approximation error: {}'.format(np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
relative maxvol approximation error: 1.10355731792e-15
>>> print('Chebyshev norm of matrix C: {}'.format(abs(C).max()))
Chebyshev norm of matrix C: 1.07853911436
"""
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype = np.int32), np.eye(N, dtype = A.dtype)
    Q = np.linalg.qr(A)[0]
    return maxvol(Q, tol, max_iters, top_k_index)

import numpy as np
from scipy.linalg import solve_triangular, get_lapack_funcs, get_blas_funcs

try:
    from ._maxvol import c_rect_maxvol, c_maxvol
    rect_maxvol_func = c_rect_maxvol
    maxvol_func = c_maxvol
    __all__.extend(['c_rect_maxvol', 'c_maxvol'])
except:
    print('warning: fast C maxvol functions are not compiled, continue with python maxvol functions')
    rect_maxvol_func = py_rect_maxvol
    maxvol_func = py_maxvol

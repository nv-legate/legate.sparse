# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are also subject to the following license:
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cunumeric
import numpy

from .base import pack_to_rect1_store
from .coo import coo_array
from .csc import csc_array
from .csr import csr_array
from .dia import dia_array
from .utils import get_store_from_cunumeric_array


def spdiags(data, diags, m, n, format=None):
    """
    Return a sparse matrix from diagonals.
    Parameters
    ----------
    data : array_like
        Matrix diagonals stored row-wise
    diags : sequence of int or an int
        Diagonals to set:
        * k = 0  the main diagonal
        * k > 0  the kth upper diagonal
        * k < 0  the kth lower diagonal
    m, n : int
        Shape of the result
    format : str, optional
        Format of the result. By default (format=None) an appropriate sparse
        matrix format is returned. This choice is subject to change.
    See Also
    --------
    diags : more convenient form of this function
    dia_matrix : the sparse DIAgonal format.
    Examples
    --------
    >>> from scipy.sparse import spdiags
    >>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    >>> diags = np.array([0, -1, 2])
    >>> spdiags(data, diags, 4, 4).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])
    """
    if format is not None:
        raise NotImplementedError
    return dia_array((data, diags), shape=(m, n), dtype=cunumeric.float64)


def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse matrix from diagonals.
    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the matrix diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse matrix format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the matrix.
    See Also
    --------
    spdiags : construct matrix from diagonals
    Notes
    -----
    This function differs from `spdiags` in the way it handles
    off-diagonals.
    The result from `diags` is the sparse equivalent of::
        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])
    Repeated diagonal offsets are disallowed.
    .. versionadded:: 0.11
    Examples
    --------
    >>> from scipy.sparse import diags
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags(diagonals, [0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])
    Broadcasting of scalars is supported (but shape needs to be
    specified):
    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])
    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:
    >>> diags([1, 2, 3], 1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    # if offsets is not a sequence, assume that there's only one diagonal
    if numpy.isscalar(offsets):
        # now check that there's actually only one diagonal
        if len(diagonals) == 0 or numpy.isscalar(diagonals[0]):
            diagonals = [cunumeric.atleast_1d(diagonals)]
        else:
            raise ValueError("Different number of diagonals and offsets.")
    else:
        diagonals = list(map(cunumeric.atleast_1d, diagonals))

    # Basic check
    if len(diagonals) != len(offsets):
        raise ValueError("Different number of diagonals and offsets.")

    # Determine shape, if omitted
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)

    # Determine data type, if omitted
    if dtype is None:
        raise NotImplementedError

    if format is not None and format not in ["csr", "dia"]:
        raise NotImplementedError

    # Construct data array
    m, n = shape

    M = max(
        [min(m + offset, n - offset) + max(0, offset) for offset in offsets]
    )
    M = max(0, M)
    data_arr = cunumeric.zeros((len(offsets), M), dtype=dtype)

    K = min(m, n)

    for j, diagonal in enumerate(diagonals):
        offset = int(offsets[j])
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError(
                "Offset %d (index %d) out of bounds" % (offset, j)
            )
        try:
            data_arr[j, k : k + length] = diagonal[..., :length]
        except ValueError as e:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError(
                    "Diagonal length (index %d: %d at offset %d) does not "
                    "agree with matrix size (%d, %d)."
                    % (j, len(diagonal), offset, m, n)
                ) from e
            raise

    # We importantly don't perform this conversion to cunumeric (involving
    # an attach operation) until we're done indexing into the list. This
    # avoid a cunumeric crash involving restrictions in attach in pde.py.
    offsets = cunumeric.atleast_1d(offsets)
    dia = dia_array((data_arr, offsets), shape=(m, n), dtype=dtype)
    if format == "csr":
        return dia.tocsr()
    return dia


def eye(m, n=None, k=0, dtype=numpy.float64, format="csr"):
    if format not in ["csr", "dia"]:
        raise NotImplementedError
    if n is None:
        n = m
    if format == "csr" and k == 0 and m == n:
        row_lo = cunumeric.arange(m, dtype=numpy.int64)
        row_hi = row_lo + 1
        # TODO (rohany): Make a version of this function that enables control
        # over whether or not rectanges are inclusive or exclusive. That way we
        # can avoid the copy here.
        pos = pack_to_rect1_store(
            get_store_from_cunumeric_array(row_lo),
            get_store_from_cunumeric_array(row_hi),
        )
        crd = get_store_from_cunumeric_array(row_lo)
        vals = get_store_from_cunumeric_array(
            cunumeric.ones(m, dtype=numpy.float64)
        )
        return csr_array((vals, crd, pos), dtype=dtype, shape=(m, n))

    diags = cunumeric.ones((1, max(0, min(m + k, n))), dtype=dtype)
    dia = spdiags(diags, k, m, n)
    if format == "csr":
        return dia.tocsr()
    return dia


def identity(n, dtype=cunumeric.float64, format=None):
    return eye(n, dtype=dtype, format=format)


def kron(A, B, format=None):
    """kronecker product of sparse matrices A and B
    Parameters
    ----------
    A : sparse or dense matrix
        first matrix of the product
    B : sparse or dense matrix
        second matrix of the product
    format : str, optional
        format of the result (e.g. "csr")
    Returns
    -------
    kronecker product in a sparse matrix format
    Examples
    --------
    >>> from scipy import sparse
    >>> A = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))
    >>> B = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))
    >>> sparse.kron(A, B).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])
    >>> sparse.kron(A, [[1, 2], [3, 4]]).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])
    """
    # This implementation was nearly directly lifted from scipy's
    # implementation of Kroneker products.

    A = A.tocoo()
    B = B.tocoo()
    output_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    if A.nnz == 0 or B.nnz == 0:
        # Kronecker product against zero matrix is the zero matrix.
        return coo_array(output_shape).asformat(format)

    # Expand entries of A into blocks.
    row = cunumeric.repeat(A.row, B.nnz)
    col = cunumeric.repeat(A.col, B.nnz)
    data = cunumeric.repeat(A.data, B.nnz)

    if (
        max(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
        > numpy.iinfo("int32").max
    ):
        # TODO (rohany): We have some problems if these shapes get too large in
        # our construction of matrices (I was lazy and didn't write sort
        # methods), so have some warning signs if we hit here.
        raise NotImplementedError
        # row = row.astype(np.int64)
        # col = col.astype(np.int64)

    row *= B.shape[0]
    col *= B.shape[1]

    # Increment block indices.
    row, col = row.reshape((-1, B.nnz)), col.reshape((-1, B.nnz))
    # Cunumeric hasn't yet implemented add.__call__, so don't use +=.
    row = row + B.row
    col = col + B.col
    row, col = row.reshape(-1), col.reshape(-1)

    # compute block entries
    data = data.reshape((-1, B.nnz)) * B.data
    data = data.reshape(-1)

    return coo_array((data, (row, col)), shape=output_shape).asformat(format)


# is_sparse_matrix returns whether or not an object is a legate
# sparse created sparse matrix.
def is_sparse_matrix(o):
    return any(
        (
            isinstance(o, csr_array),
            isinstance(o, csc_array),
            isinstance(o, coo_array),
            isinstance(o, dia_array),
        )
    )


issparse = is_sparse_matrix
isspmatrix = is_sparse_matrix


# Variants for each particular format type.
def isspmatrix_csc(o):
    return isinstance(o, csc_array)


def isspmatrix_csr(o):
    return isinstance(o, csr_array)


def isspmatrix_coo(o):
    return isinstance(o, coo_array)


def isspmatrix_dia(o):
    return isinstance(o, dia_array)


def random(
    m,
    n,
    density=0.01,
    format="coo",
    dtype=None,
    random_state=None,
    data_rvs=None,
):
    """Generate a sparse matrix of the given shape and density with randomly
    distributed values.
    This function is adapted from the implementation in scipy.sparse.
    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        This random state will be used
        for sampling the sparsity structure, but not necessarily for sampling
        the values of the structurally nonzero entries of the matrix.
    data_rvs : callable, optional
        Samples a requested number of random values.
        This function should take a single argument specifying the length
        of the ndarray that it will return. The structurally nonzero entries
        of the sparse random matrix will be taken from the array sampled
        by this function. By default, uniform [0, 1) random values will be
        sampled using the same random state as is used for sampling
        the sparsity structure.
    Returns
    -------
    res : sparse matrix
    Notes
    -----
    Only float types are supported for now.
    Examples
    --------
    >>> from scipy.sparse import random
    >>> from scipy import stats
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> rvs = stats.poisson(25, loc=10).rvs
    >>> S = random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
    >>> S.A
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])
    >>> from scipy.sparse import random
    >>> from scipy.stats import rv_continuous
    >>> class CustomDistribution(rv_continuous):
    ...     def _rvs(self,  size=None, random_state=None):
    ...         return random_state.standard_normal(size)
    >>> X = CustomDistribution(seed=rng)
    >>> Y = X()  # get a frozen version of the distribution
    >>> S = random(3, 4, density=0.25, random_state=rng, data_rvs=Y.rvs)
    >>> S.A
    array([[ 0.        ,  0.        ,  0.        ,  0.        ],   # random
           [ 0.13569738,  1.9467163 , -0.81205367,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ]])
    """
    import cunumeric as np

    if density < 0 or density > 1:
        raise ValueError("density expected to be 0 <= density <= 1")
    dtype = np.dtype(dtype)

    mn = m * n

    tp = np.intc
    if mn > np.iinfo(tp).max:
        tp = np.int64

    if mn > np.iinfo(tp).max:
        msg = """\
Trying to generate a random sparse matrix such as the product of dimensions is
greater than %d - this is not supported on this machine
"""
        raise ValueError(msg % np.iinfo(tp).max)

    # Number of non zero values
    k = int(round(density * m * n))

    # We don't support setting of random_state right now.
    assert random_state is None
    # We also don't support setting of data_rvs right now.
    assert data_rvs is None
    ind = np.random.choice(mn, size=k, replace=False)
    j = np.floor(ind * 1.0 / m).astype(tp, copy=False)
    i = (ind - j * m).astype(tp, copy=False)
    vals = np.random.rand(k).astype(dtype, copy=False)
    return coo_array((vals, (i, j)), shape=(m, n)).asformat(format, copy=False)


def rand(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    """Generate a sparse matrix of the given shape and density with uniformly
    distributed values.
    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    Returns
    -------
    res : sparse matrix
    Notes
    -----
    Only float types are supported for now.
    See Also
    --------
    scipy.sparse.random : Similar function that allows a user-specified random
        data source.
    Examples
    --------
    >>> from scipy.sparse import rand
    >>> matrix = rand(3, 4, density=0.25, format="csr", random_state=42)
    >>> matrix
    <3x4 sparse matrix of type '<class 'numpy.float64'>'
       with 3 stored elements in Compressed Sparse Row format>
    >>> matrix.toarray()
    array([[0.05641158, 0.        , 0.        , 0.65088847],
           [0.        , 0.        , 0.        , 0.14286682],
           [0.        , 0.        , 0.        , 0.        ]])
    """
    return random(m, n, density, format, dtype, random_state)

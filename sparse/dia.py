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
import scipy

import sparse

from .base import CompressedBase
from .coverage import clone_scipy_arr_kind
from .types import coord_ty
from .utils import (
    cast_arr,
    get_store_from_cunumeric_array,
    store_to_cunumeric_array,
)


@clone_scipy_arr_kind(scipy.sparse.dia_array)
class dia_array(CompressedBase):
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        if copy:
            raise NotImplementedError
        if shape is None:
            raise NotImplementedError
        assert isinstance(arg, tuple)
        data, offsets = arg
        if isinstance(offsets, int):
            offsets = cunumeric.full((1,), offsets)
        data, offsets = cast_arr(data), cast_arr(offsets)
        if dtype is not None:
            data = data.astype(dtype)
        dtype = data.dtype
        assert dtype is not None
        if not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)

        self.dtype = dtype
        self.shape = shape
        self._offsets = get_store_from_cunumeric_array(offsets)
        self._data = get_store_from_cunumeric_array(data)

    @property
    def data(self):
        return store_to_cunumeric_array(self._data)

    @property
    def offsets(self):
        return store_to_cunumeric_array(self._offsets)

    def copy(self):
        data = cunumeric.array(self.data)
        offsets = cunumeric.array(self.offsets)
        return dia_array((data, offsets), shape=self.shape, dtype=self.dtype)

    def astype(self, dtype, casting="unsafe", copy=True):
        data = self.data.copy() if copy else self.data
        offsets = self.offsets.astype(dtype, casting=casting, copy=copy)
        return dia_array((data, offsets), shape=self.shape, dtype=dtype)

    # This implementation of nnz on DIA matrices is lifted from scipy.sparse.
    @property
    def nnz(self):
        M, N = self.shape
        nnz = 0
        for k in self.offsets:
            if k > 0:
                nnz += min(M, N - k)
            else:
                nnz += min(M + k, N)
        return int(nnz)

    # This implementation of diagonal() on DIA matrices is lifted from
    # scipy.sparse.
    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cunumeric.empty(0, dtype=self.data.dtype)
        (idx,) = cunumeric.nonzero(self.offsets == k)
        first_col = max(0, k)
        last_col = min(rows + k, cols)
        result_size = last_col - first_col
        if idx.size == 0:
            return cunumeric.zeros(result_size, dtype=self.data.dtype)
        result = self.data[int(idx[0]), first_col:last_col]
        padding = result_size - len(result)
        if padding > 0:
            result = cunumeric.pad(result, (0, padding), mode="constant")
        return result

    # This implementation of tocoo() is lifted from scipy.sparse.
    def tocoo(self, copy=False):
        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = cunumeric.arange(offset_len)

        row = offset_inds - self.offsets[:, None]
        mask = row >= 0
        mask &= row < num_rows
        mask &= offset_inds < num_cols
        mask &= self.data != 0
        row = cunumeric.array(row[mask])
        col = cunumeric.array(
            cunumeric.tile(offset_inds, num_offsets)[mask.ravel()]
        )
        data = cunumeric.array(self.data[mask])

        return sparse.coo_array(
            (data, (row, col)), shape=self.shape, dtype=self.dtype
        )

    # For memory utilization purposes, we aim to avoid turning matrices
    # into COO matrices and sorting if we can. Since we can do transposes
    # on DIA matrices efficiently, and have an efficient DIA->CSC conversion
    # routine, we can utilize this to get a fast DIA->CSR converter. Note
    # that I attempted to get the DIA->CSC routine to work for DIA->CSR
    # conversion but couldn't quite make it.
    def tocsr(self):
        return self.T.tocsc().T

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(
                (
                    "Sparse matrices do not support "
                    "an 'axes' parameter because swapping "
                    "dimensions is the only logical permutation."
                )
            )
        if copy:
            raise AssertionError

        num_rows, num_cols = self.shape
        max_dim = max(self.shape)

        # flip diagonal offsets
        offsets = -self.offsets

        # re-align the data matrix
        r = cunumeric.arange(len(offsets), dtype=coord_ty)[:, None]
        c = (
            cunumeric.arange(num_rows, dtype=coord_ty)
            - (offsets % max_dim)[:, None]
        )
        pad_amount = max(0, max_dim - self.data.shape[1])
        data = cunumeric.hstack(
            (
                self.data,
                cunumeric.zeros(
                    (self.data.shape[0], pad_amount), dtype=self.data.dtype
                ),
            )
        )
        data = data[r, c]
        return dia_array(
            (data, offsets),
            shape=(num_cols, num_rows),
            copy=copy,
            dtype=self.dtype,
        )

    T = property(transpose)

    # This routine is lifted from scipy.sparse's converter.
    def tocsc(self, copy=False):
        if copy:
            raise AssertionError
        if self.nnz == 0:
            return sparse.csc_array.make_empty(self.shape, self.dtype)

        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = cunumeric.arange(offset_len)

        row = offset_inds - self.offsets[:, None]
        mask = row >= 0
        mask &= row < num_rows
        mask &= offset_inds < num_cols
        mask &= self.data != 0

        idx_dtype = coord_ty
        indptr = cunumeric.zeros(num_cols + 1, dtype=idx_dtype)
        indptr[1 : offset_len + 1] = cunumeric.cumsum(
            mask.sum(axis=0)[:num_cols]
        )
        if offset_len < num_cols:
            indptr[offset_len + 1 :] = indptr[offset_len]
        indices = row.T[mask.T].astype(idx_dtype, copy=False)
        data = self.data.T[mask.T]
        return sparse.csc_array(
            (data, indices, indptr), shape=self.shape, dtype=self.dtype
        )

    def todense(self):
        return self.tocoo().todense()


# Declare an alias for this type.
dia_matrix = dia_array

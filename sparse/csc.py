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
from legate.core import Store
from legate.core.shape import Shape
from legate.core.types import ReductionOp

import sparse

from .base import (
    CompressedBase,
    DenseSparseBase,
    pack_to_rect1_store,
    unpack_rect1_store,
)
from .config import SparseOpCode
from .coverage import clone_scipy_arr_kind
from .partition import CompressedImagePartition
from .runtime import ctx, runtime
from .types import coord_ty, float64, nnz_ty
from .utils import (
    cast_arr,
    cast_to_store,
    get_store_from_cunumeric_array,
    require_float64_dtypes,
    store_to_cunumeric_array,
)


@clone_scipy_arr_kind(scipy.sparse.csc_array)
class csc_array(CompressedBase, DenseSparseBase):
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        if copy:
            raise NotImplementedError
        super().__init__()

        if isinstance(arg, numpy.ndarray):
            arg = cunumeric.array(arg)
        if isinstance(arg, cunumeric.ndarray):
            assert arg.ndim == 2
            shape = arg.shape
            # Similarly to the CSR from dense case, we'll do a column based
            # distribution.
            arg_store = get_store_from_cunumeric_array(arg)
            q_nnz = ctx.create_store(nnz_ty, shape=(arg.shape[1]))
            promoted_q_nnz = q_nnz.promote(0, shape[0])
            task = ctx.create_task(SparseOpCode.DENSE_TO_CSC_NNZ)
            task.add_output(promoted_q_nnz)
            task.add_input(arg_store)
            task.add_broadcast(promoted_q_nnz, 0)
            task.add_alignment(promoted_q_nnz, arg_store)
            task.execute()
            # Assemble the output CSC array using the non-zeros per column.
            self.pos, nnz = self.nnz_to_pos(q_nnz)
            self.crd = ctx.create_store(coord_ty, shape=(nnz))
            self.vals = ctx.create_store(arg.dtype, shape=(nnz))
            promoted_pos = self.pos.promote(0, shape[0])
            task = ctx.create_task(SparseOpCode.DENSE_TO_CSC)
            task.add_output(promoted_pos)
            task.add_output(self.crd)
            task.add_output(self.vals)
            task.add_input(arg_store)
            task.add_input(promoted_pos)
            # Partition the columns.
            task.add_broadcast(promoted_pos, 0)
            task.add_alignment(promoted_pos, arg_store)
            task.add_image_constraint(
                promoted_pos,
                self.crd,
                range=True,
                functor=CompressedImagePartition,
            )
            task.add_image_constraint(
                promoted_pos,
                self.vals,
                range=True,
                functor=CompressedImagePartition,
            )
            task.execute()
        elif isinstance(arg, tuple):
            if shape is None:
                raise AssertionError("Unable to infer shape.")
            (data, indices, indptr) = arg
            # Handle when someone passes a CSC indptr array as input.
            if isinstance(indptr, cunumeric.ndarray):
                assert indptr.shape[0] == shape[1] + 1
                los = indptr[:-1]
                his = indptr[1:]
                self.pos = pack_to_rect1_store(
                    get_store_from_cunumeric_array(los),
                    get_store_from_cunumeric_array(his),
                )
            else:
                assert isinstance(indptr, Store)
                self.pos = indptr

            # TODO (rohany): Allow for different variation of coordinate
            #  type, i.e. choosing int32 or int64.
            self.crd = cast_to_store(cast_arr(indices, coord_ty))
            data = cast_arr(data)
            self.vals = cast_to_store(data)
        else:
            raise NotImplementedError

        assert shape is not None
        self.shape = shape

        # Use the user's dtype if requested, otherwise infer it from
        # the input data.
        if dtype is None:
            dtype = self.data.dtype
        else:
            self.data = self.data.astype(dtype)

        if not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)
        self.dtype = dtype

        # Manually adjust the key partition of the pos array to distribute the
        # sparse matrix by the rows across all processors. This makes the
        # solver understand that not everything should be replicated just
        # because the matrix construction is not parallelized.
        tile_size = (
            self.pos.shape[0] + runtime.num_procs - 1
        ) // runtime.num_procs
        pos_part = self.pos.partition_by_tiling(Shape(tile_size))
        self.pos.set_key_partition(pos_part.partition)
        # Override the volume calculation on pos regions, since repartitioning
        # the pos region will most definitely cause moving around the crd and
        # vals arrays. Importantly, we need to compute the volume here and then
        # return it in the closure. If we compute it inside the closure, we end
        # up creating a reference cycle between self.pos._storage and self,
        # leading us to never collect self, leaking futures stored here.
        volume = (
            self.crd.comm_volume()
            + self.vals.comm_volume()
            + self.pos.extents.volume()
        )

        def compute_volume():
            return volume

        self.pos._storage.volume = compute_volume

    # Enable direct operation on the values array.
    def get_data(self):
        return store_to_cunumeric_array(self.vals)

    def set_data(self, data):
        if isinstance(data, numpy.ndarray):
            data = cunumeric.array(data)
        assert isinstance(data, cunumeric.ndarray)
        self.vals = get_store_from_cunumeric_array(data)

    data = property(fget=get_data, fset=set_data)

    @classmethod
    def make_empty(cls, shape, dtype):
        M, N = shape
        # Make an empty pos array.
        q = cunumeric.zeros((N,), dtype=coord_ty)
        pos, _ = CompressedBase.nnz_to_pos_cls(
            get_store_from_cunumeric_array(q)
        )
        crd = ctx.create_store(coord_ty, (0,), optimize_scalar=False)
        vals = ctx.create_store(dtype, (0,), optimize_scalar=False)
        return cls((vals, crd, pos), shape=shape, dtype=dtype)

    def astype(self, dtype, casting="unsafe", copy=True):
        pos = self.copy_pos() if copy else self.pos
        crd = (
            cunumeric.array(store_to_cunumeric_array(self.crd))
            if copy
            else self.crd
        )
        vals = self.data.astype(dtype, casting=casting, copy=copy)
        return csc_array.make_with_same_nnz_structure(
            self, (vals, crd, pos), dtype=dtype
        )

    def copy(self):
        pos = self.copy_pos()
        crd = cunumeric.array(store_to_cunumeric_array(self.crd))
        vals = cunumeric.array(store_to_cunumeric_array(self.vals))
        return csc_array.make_with_same_nnz_structure(self, (vals, crd, pos))

    def tocsr(self, copy=False):
        if copy:
            raise NotImplementedError
        return self.tocoo().tocsr()

    def tocsc(self, copy=False):
        if copy:
            raise NotImplementedError
        return self

    def tocoo(self, copy=False):
        if copy:
            raise NotImplementedError
        # The conversion to COO is pretty straightforward. The crd and values
        # arrays are already set up for COO, we just need to expand the pos
        # array into coordinates.
        cols_expanded = ctx.create_store(coord_ty, shape=self.crd.shape)
        task = ctx.create_task(SparseOpCode.EXPAND_POS_TO_COORDINATES)
        task.add_input(self.pos)
        task.add_output(cols_expanded)
        task.add_image_constraint(
            self.pos,
            cols_expanded,
            range=True,
            functor=CompressedImagePartition,
        )
        task.execute()
        return sparse.coo_array(
            (self.vals, (self.crd, cols_expanded)),
            shape=self.shape,
            dtype=self.dtype,
        )

    def __mul__(self, other):
        if not isinstance(other, csc_array):
            raise NotImplementedError
        # We can actually re-use the CSR * CSR method to add two CSC matrices.
        this_csr = sparse.csr_array(
            (self.vals, self.crd, self.pos),
            shape=(self.shape[1], self.shape[0]),
        )
        other_csr = sparse.csr_array(
            (other.vals, other.crd, other.pos),
            shape=(other.shape[1], other.shape[0]),
        )
        result = this_csr * other_csr
        # Now unpack it into a CSC matrix.
        return csc_array(
            (result.vals, result.crd, result.pos), shape=self.shape
        )

    def __add__(self, other):
        if not isinstance(other, csc_array):
            raise NotImplementedError
        # We can actually re-use the CSR + CSR method to add two CSC matrices.
        this_csr = sparse.csr_array(
            (self.vals, self.crd, self.pos),
            shape=(self.shape[1], self.shape[0]),
        )
        other_csr = sparse.csr_array(
            (other.vals, other.crd, other.pos),
            shape=(other.shape[1], other.shape[0]),
        )
        result = this_csr + other_csr
        # Now unpack it into a CSC matrix.
        return csc_array(
            (result.vals, result.crd, result.pos), shape=self.shape
        )

    def diagonal(self, k=0):
        return sparse.csr_array(
            (self.vals, self.crd, self.pos),
            shape=(self.shape[1], self.shape[0]),
        ).diagonal(k=-k)

    def transpose(self, copy=False):
        if copy:
            raise NotImplementedError
        return sparse.csr_array.make_with_same_nnz_structure(
            self,
            (self.vals, self.crd, self.pos),
            shape=Shape((self.shape[1], self.shape[0])),
        )

    T = property(transpose)

    # other / mat is defined to be the element-wise division of a other by mat
    # over the non-zero coordinates of mat. For now, we restrict this operation
    # to be on scalars only.
    def __rtruediv__(self, other):
        if not cunumeric.isscalar(other):
            raise NotImplementedError
        vals_arr = store_to_cunumeric_array(self.vals)
        new_vals = other / vals_arr
        return csc_array.make_with_same_nnz_structure(
            self,
            (get_store_from_cunumeric_array(new_vals), self.crd, self.pos),
        )

    # TODO (rohany): This is a diversion from the API, but we'll set copy to be
    # false for now.
    def conj(self, copy=False):
        if copy:
            raise NotImplementedError
        if self.dtype != float64:
            raise NotImplementedError
        return self

    def dot(self, other, out=None):
        require_float64_dtypes(self, other, out)
        if out is not None:
            assert isinstance(out, cunumeric.ndarray)
        if len(other.shape) == 1 or (
            len(other.shape) == 2 and other.shape[1] == 1
        ):
            if not isinstance(other, cunumeric.ndarray):
                other = cunumeric.array(other)
            other_originally_2d = False
            if len(other.shape) == 2 and other.shape[1] == 1:
                other = other.squeeze(1)
                other_originally_2d = True
            assert self.shape[1] == other.shape[0]
            # Ensure that we write into the output when possible.
            if out is not None:
                if other_originally_2d:
                    assert out.shape == (self.shape[0], 1)
                    out = out.squeeze(1)
                else:
                    assert out.shape == (self.shape[0],)
                out.fill(0.0)
                output = out
                output_store = get_store_from_cunumeric_array(out)
            else:
                output = cunumeric.zeros((self.shape[0],), dtype=float64)
                output_store = get_store_from_cunumeric_array(output)
            other_store = get_store_from_cunumeric_array(other)
            task = ctx.create_task(SparseOpCode.CSC_SPMV_COL_SPLIT)
            task.add_reduction(output_store, ReductionOp.ADD)
            task.add_input(self.pos)
            task.add_input(self.crd)
            task.add_input(self.vals)
            task.add_input(other_store)
            task.add_broadcast(output_store)
            task.add_alignment(self.pos, other_store)
            task.add_image_constraint(
                self.pos,
                self.crd,
                range=True,
                functor=CompressedImagePartition,
            )
            task.add_alignment(self.crd, self.vals)
            task.execute()
            if other_originally_2d:
                output = output.reshape((-1, 1))
            return output
        else:
            return self.tocsr().dot(other, out=out)

    def __matmul__(self, other):
        return self.dot(other)

    def __rmatmul__(self, other):
        raise NotImplementedError

    def todense(self, order=None, out=None):
        require_float64_dtypes(self, out)
        if order is not None:
            raise NotImplementedError
        if out is not None:
            out = cunumeric.array(out)
            out = get_store_from_cunumeric_array(out)
        elif out is None:
            out = ctx.create_store(self.dtype, shape=self.shape)
        # TODO (rohany): We'll do a col-based distribution for now, but a
        # non-zero based distribution of this operation with overlapping output
        # regions being reduced into also seems possible.
        promoted_pos = self.pos.promote(0, self.shape[0])
        task = ctx.create_task(SparseOpCode.CSC_TO_DENSE)
        task.add_output(out)
        task.add_input(promoted_pos)
        task.add_input(self.crd)
        task.add_input(self.vals)
        # We aren't partitioning the rows.
        task.add_broadcast(out, 0)
        task.add_alignment(out, promoted_pos)
        task.add_image_constraint(
            promoted_pos,
            self.crd,
            range=True,
            functor=CompressedImagePartition,
        )
        task.add_alignment(self.crd, self.vals)
        task.execute()
        return store_to_cunumeric_array(out)

    # sddmm computes a sampled dense-dense matrix multiplication operation
    # by fusing the element-wise multiply by the sparse matrix into the
    # dense matrix multiplication of the C and D operands. This function
    # is _not_ part of the scipy.sparse package but is prudent to add as
    # a kernel in many emerging workloads.
    def sddmm(self, C, D):
        # We'll start out with a row-based distribution of the CSR matrix.  In
        # the future, we can look into doing a non-zero based distribution of
        # the computation, as there aren't really any downsides of doing it
        # versus a row-based distribution. The problem with both is that they
        # require replicating the D matrix onto all processors. Doing a
        # partitioning strategy that partitions up the j dimension of the
        # computation is harder.  This operation is also non-zero structure
        # preserving, so we'll just write into an output array of values and
        # share the pos and crd arrays.
        # TODO (rohany): An option is also partitioning up the `k` dimension of
        # the computation (allows for partitioning C twice and D once), but
        # requires reducing into the output.
        if isinstance(C, numpy.ndarray):
            C = cunumeric.array(C)
        if isinstance(D, numpy.ndarray):
            D = cunumeric.array(D)
        assert len(C.shape) == 2 and len(D.shape) == 2
        assert (
            self.shape[0] == C.shape[0]
            and self.shape[1] == D.shape[1]
            and C.shape[1] == D.shape[0]
        )
        require_float64_dtypes(self, C, D)
        C_store = get_store_from_cunumeric_array(C)
        D_store = get_store_from_cunumeric_array(D)
        result_vals = ctx.create_store(self.dtype, shape=self.vals.shape)

        promoted_pos = self.pos.promote(0, D_store.shape[0])
        task = ctx.create_task(SparseOpCode.CSC_SDDMM)
        task.add_output(result_vals)
        task.add_input(promoted_pos)
        task.add_input(self.crd)
        task.add_input(self.vals)
        task.add_input(C_store)
        task.add_input(D_store)
        # Partition the rows of the sparse matrix and C.
        task.add_broadcast(promoted_pos, 0)
        task.add_alignment(promoted_pos, D_store)
        task.add_image_constraint(
            promoted_pos,
            result_vals,
            range=True,
            functor=CompressedImagePartition,
        )
        task.add_image_constraint(
            promoted_pos,
            self.crd,
            range=True,
            functor=CompressedImagePartition,
        )
        task.add_image_constraint(
            promoted_pos,
            self.vals,
            range=True,
            functor=CompressedImagePartition,
        )
        task.add_broadcast(C_store)
        task.execute()
        return csc_array.make_with_same_nnz_structure(
            self, (result_vals, self.crd, self.pos)
        )

    # TODO (rohany): Deduplicate these methods against csr_matrix.
    def __str__(self):
        los, his = self._unpack_pos()
        crdarr = store_to_cunumeric_array(self.crd)
        valsarr = store_to_cunumeric_array(self.vals)
        return (
            f"{store_to_cunumeric_array(los)}, "
            f"{store_to_cunumeric_array(his)}, {crdarr}, {valsarr}"
        )

    def _unpack_pos(self):
        return unpack_rect1_store(self.pos)


csc_matrix = csc_array

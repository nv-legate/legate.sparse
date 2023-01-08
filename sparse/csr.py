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

import warnings
from typing import Any

import cunumeric
import numpy
import scipy
from legate.core import (
    Future,
    FutureMap,
    Point,
    Rect,
    Store,
    ffi,
    track_provenance,
    types,
)
from legate.core.partition import DomainPartition, ImagePartition, Tiling
from legate.core.shape import Shape
from legate.core.store import StorePartition
from legate.core.types import ReductionOp

import sparse

from .base import (
    CompressedBase,
    DenseSparseBase,
    pack_to_rect1_store,
    unpack_rect1_store,
)
from .config import SparseOpCode, SparseProjectionFunctor, rect1
from .coverage import clone_scipy_arr_kind
from .partition import (
    CompressedImagePartition,
    DensePreimage,
    MinMaxImagePartition,
)
from .runtime import ctx, runtime
from .types import coord_ty, nnz_ty
from .utils import (
    cast_arr,
    cast_to_common_type,
    cast_to_store,
    factor_int,
    find_last_user_stacklevel,
    get_store_from_cunumeric_array,
    store_to_cunumeric_array,
)


@clone_scipy_arr_kind(scipy.sparse.csr_array)
class csr_array(CompressedBase, DenseSparseBase):
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        self.ndim = 2
        super().__init__()

        if isinstance(arg, numpy.ndarray):
            arg = cunumeric.array(arg)
        if isinstance(arg, cunumeric.ndarray):
            assert arg.ndim == 2
            shape = arg.shape
            # Conversion from dense arrays is pretty easy. We'll do a row-wise
            # distribution and use a two-pass algorithm that first counts the
            # non-zeros per row and then fills them in.
            arg_store = get_store_from_cunumeric_array(arg)
            q_nnz = ctx.create_store(nnz_ty, shape=(arg.shape[0]))
            task = ctx.create_task(SparseOpCode.DENSE_TO_CSR_NNZ)
            promoted_q_nnz = q_nnz.promote(1, shape[1])
            task.add_output(promoted_q_nnz)
            task.add_input(arg_store)
            task.add_broadcast(promoted_q_nnz, 1)
            task.add_alignment(promoted_q_nnz, arg_store)
            task.execute()

            # Assemble the output CSR array using the non-zeros per row.
            self.pos, nnz = self.nnz_to_pos(q_nnz)
            self.crd = ctx.create_store(coord_ty, shape=(nnz))
            self.vals = ctx.create_store(arg.dtype, shape=(nnz))
            task = ctx.create_task(SparseOpCode.DENSE_TO_CSR)
            promoted_pos = self.pos.promote(1, shape[1])
            task.add_output(promoted_pos)
            task.add_output(self.crd)
            task.add_output(self.vals)
            task.add_input(arg_store)
            task.add_input(promoted_pos)
            # Partition the rows.
            task.add_broadcast(promoted_pos, 1)
            task.add_alignment(promoted_pos, arg_store)
            task.add_image_constraint(
                promoted_pos,
                self.crd,
                range=True,
                functor=CompressedImagePartition,
            )
            task.add_alignment(self.crd, self.vals)
            task.execute()
        elif isinstance(arg, scipy.sparse.csr_array) or isinstance(
            arg, scipy.sparse.csr_matrix
        ):
            shape = arg.shape
            self.vals = get_store_from_cunumeric_array(
                cunumeric.array(arg.data)
            )
            # TODO (rohany): Allow for non int64 typed coordinates.
            self.crd = get_store_from_cunumeric_array(
                cunumeric.array(arg.indices).astype(coord_ty)
            )
            # Cast the indptr array in the scipy.csr_matrix into our Rect<1>
            # based pos array.
            indptr = cunumeric.array(arg.indptr, dtype=cunumeric.int64)
            los = indptr[:-1]
            his = indptr[1:]
            self.pos = pack_to_rect1_store(
                get_store_from_cunumeric_array(los),
                get_store_from_cunumeric_array(his),
            )
        elif isinstance(arg, tuple):
            if copy:
                raise NotImplementedError
            if shape is None:
                raise AssertionError("Cannot infer shape in this case.")

            if len(arg) == 2:
                # If the tuple has two arguments, then it must be of the form
                # (data, (row, col)), so just pass it to the COO constructor
                # and transform it into a CSR matrix.
                data, (row, col) = arg
                result = sparse.coo_array(
                    (data, (row, col)), shape=shape
                ).tocsr()
                self.pos = result.pos
                self.crd = result.crd
                self.vals = result.vals
                shape = result.shape
            elif len(arg) == 3:
                (data, indices, indptr) = arg
                if isinstance(indptr, cunumeric.ndarray):
                    assert indptr.shape[0] == shape[0] + 1
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
                raise AssertionError
        else:
            raise NotImplementedError

        assert shape is not None
        # Ensure that we don't accidentally include ndarray
        # objects as the elements of our shapes, as that can
        # lead to reference cycles or issues when talking to
        # legate under the hood.
        self.shape = tuple(int(i) for i in shape)

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

    # Enable direct operation on the indices array.
    def get_indices(self):
        return store_to_cunumeric_array(self.crd)

    def set_indices(self, indices):
        if isinstance(indices, numpy.ndarray):
            indices = cunumeric.array(indices)
        assert isinstance(indices, cunumeric.ndarray)
        self.crd = get_store_from_cunumeric_array(indices)

    indices = property(fget=get_indices, fset=set_indices)

    @classmethod
    def make_empty(cls, shape, dtype):
        M, N = shape
        # Make an empty pos array.
        q = cunumeric.zeros((M,), dtype=coord_ty)
        pos, _ = CompressedBase.nnz_to_pos_cls(
            get_store_from_cunumeric_array(q)
        )
        crd = ctx.create_store(coord_ty, (0,), optimize_scalar=False)
        vals = ctx.create_store(dtype, (0,), optimize_scalar=False)
        return cls((vals, crd, pos), shape=shape, dtype=dtype)

    @property
    def nnz(self):
        return self.vals.shape[0]

    def astype(self, dtype, casting="unsafe", copy=True):
        if not copy and dtype == self.dtype:
            return self
        pos = self.copy_pos() if copy else self.pos
        crd = (
            cunumeric.array(store_to_cunumeric_array(self.crd))
            if copy
            else self.crd
        )
        vals = self.data.astype(dtype, casting=casting, copy=copy)
        return csr_array.make_with_same_nnz_structure(
            self, (vals, crd, pos), dtype=dtype
        )

    def copy(self):
        pos = self.copy_pos()
        crd = cunumeric.array(store_to_cunumeric_array(self.crd))
        vals = cunumeric.array(store_to_cunumeric_array(self.vals))
        return csr_array.make_with_same_nnz_structure(self, (vals, crd, pos))

    def conj(self, copy=True):
        if copy:
            return self.copy().conj(copy=False)
        return csr_array.make_with_same_nnz_structure(
            self,
            (
                get_store_from_cunumeric_array(self.data.conj()),
                self.crd,
                self.pos,
            ),
        )

    @track_provenance(runtime.legate_context, nested=True)
    def tropical_spmv(self, other, out=None):
        if not isinstance(other, cunumeric.ndarray):
            other = cunumeric.array(other)
        assert len(other.shape) == 2
        # TODO (rohany): Add checks around dtypes.
        assert self.shape[1] == other.shape[0]
        if out is None:
            output = ctx.create_store(
                coord_ty, shape=(self.shape[0], other.shape[1])
            )
        else:
            assert isinstance(out, cunumeric.ndarray)
            assert (
                out.shape[0] == self.shape[0]
                and out.shape[1] == other.shape[1]
            )
            output = get_store_from_cunumeric_array(out)
        other_store = get_store_from_cunumeric_array(other)

        # An auto-parallelized version of the kernel.
        promoted_pos = self.pos.promote(1, dim_size=output.shape[1])
        task = ctx.create_task(
            SparseOpCode.CSR_SPMV_ROW_SPLIT_TROPICAL_SEMIRING
        )
        task.add_output(output)
        task.add_input(promoted_pos)
        task.add_input(self.crd)
        task.add_input(other_store)
        # Add partitioning. We make sure the field dimensions aren't
        # partitioned.
        task.add_broadcast(output, 1)
        task.add_alignment(output, promoted_pos)
        task.add_image_constraint(
            promoted_pos,
            self.crd,
            range=True,
            functor=CompressedImagePartition,
        )
        task.add_alignment(self.crd, self.vals)

        # In order to do the image from the coordinates into the corresponding
        # rows of other, we have to apply an AffineProjection from the
        # coordinates to cast them up to reference rows of other, rather than
        # single points. The API for this is a bit restrictive, so we have to
        # pass a staged MinMaxImagePartition functor through to the image
        # constraint.
        def partFunc(*args, **kwargs):
            return MinMaxImagePartition(*args, proj_dims=[0], **kwargs)

        task.add_image_constraint(
            self.crd,
            other_store,
            range=False,
            disjoint=False,
            complete=False,
            functor=partFunc,
        )
        task.execute()
        return store_to_cunumeric_array(output)

    def to_scipy_sparse_csr(self):
        import scipy.sparse

        los, _ = unpack_rect1_store(self.pos)
        los = store_to_cunumeric_array(los)
        indptr = cunumeric.append(los, [self.crd.shape[0]])
        return scipy.sparse.csr_array(
            (
                store_to_cunumeric_array(self.vals),
                store_to_cunumeric_array(self.crd),
                indptr,
            ),
            shape=self.shape,
            dtype=self.dtype,
        )

    def dot(self, other, out=None, spmv_domain_part=False):
        if out is not None:
            assert isinstance(out, cunumeric.ndarray)
        if isinstance(other, numpy.ndarray):
            other = cunumeric.array(other)
        # We're doing a SpMV or an SpMSpV.
        if len(other.shape) == 1 or (
            len(other.shape) == 2 and other.shape[1] == 1
        ):
            # We don't have an SpMSpV implementation, so just convert the input
            # sparse matrix into a dense vector first.
            other_originally_sparse = False
            if sparse.is_sparse_matrix(other):
                other = other.todense()
                other_originally_sparse = True
            if not isinstance(other, cunumeric.ndarray):
                other = cunumeric.array(other)
            assert self.shape[1] == other.shape[0]
            other_originally_2d = False
            if len(other.shape) == 2 and other.shape[1] == 1:
                other = other.squeeze(1)
                other_originally_2d = True

            # The SpMV code uses an image from the coordinates into the
            # other vector to reduce the amount of communication needed,
            # see below for more details. When the other vector is
            # transformed, we can't do this optimization without an affine
            # transformation on the image. Worse, when some more complicated
            # transforms have been applied to the input vector, we may end
            # up sending a sliced instance to the cuSPARSE tasks that is not
            # a contiguous piece of memory. So, when the input vector is
            # transformed, make a temporary copy into a contiguous instance
            # to allow for both the image optimization and to sidestep any
            # problems with leaf tasks. It seems like part of the problem
            # would be alleviated to use an `exact` mapping in the legate
            # sparse mapper, but this still seems like a better solution
            # as we would avoid replication of the other vector when it is
            # transformed.
            other_store = get_store_from_cunumeric_array(other)
            if other_store.transformed:
                level = find_last_user_stacklevel()
                warnings.warn(
                    "CSR SpMV creating an implicit copy due to "
                    "transformed x vector.",
                    category=RuntimeWarning,
                    stacklevel=level,
                )
                other = cunumeric.array(other)

            # Coerce A and x into a common type. Use that coerced type
            # to find the type of the output.
            A, x = cast_to_common_type(self, other)
            if out is None:
                # Annoyingly, we don't seem to be able to use
                # cunumeric.empty here (as done below), as we get an
                # error from Legion about trying to create an accessor
                # onto an empty future under load. So, just use create_store
                # directly.
                # y = cunumeric.empty(self.shape[0], dtype=A.dtype)
                y = store_to_cunumeric_array(
                    ctx.create_store(A.dtype, shape=(self.shape[0],))
                )
            else:
                # We can't use the output if it not the correct type,
                # as then we can't guarantee that we would write into
                # it. So, error out if the output type doesn't match
                # the resolved type of A and x.
                if out.dtype != A.dtype:
                    raise ValueError(
                        f"Output type {out.dtype} is not consistent "
                        f"with resolved dtype {A.dtype}"
                    )
                if other_originally_2d:
                    assert out.shape == (self.shape[0], 1)
                    assert not spmv_domain_part
                    out = out.squeeze(1)
                else:
                    assert out.shape == (self.shape[0],)
                y = out

            # If we're going to end up reducing into the output, reset it
            # to zero before launching tasks.
            if spmv_domain_part:
                # Importantly, use a fill instead of y[:] = 0. This
                # allows Legion to optimize the reduction by not
                # contributing each write of 0 into the output, and
                # instead can lazily apply the contribution of the fill.
                y.fill(0)

            # Invoke the SpMV after the setup.
            spmv(A, x, y, domain_part=spmv_domain_part)

            output = y
            if other_originally_2d:
                output = output.reshape((-1, 1))
            if other_originally_sparse:
                output = csr_array(
                    output, shape=output.shape, dtype=output.dtype
                )
            return output
        elif isinstance(other, sparse.csc_array):
            if out is not None:
                raise ValueError("Cannot specify out for CSRxCSC matmul.")
            assert self.shape[1] == other.shape[0]
            return spgemm_csr_csr_csc(*cast_to_common_type(self, other))
        elif isinstance(other, csr_array):
            if out is not None:
                raise ValueError("Cannot provide out for CSRxCSC matmul.")
            assert self.shape[1] == other.shape[0]
            return spgemm_csr_csr_csr(*cast_to_common_type(self, other))
        elif isinstance(other, cunumeric.ndarray):
            # We can dispatch to SpMM here. There are different implementations
            # that one can go for, like the 2-D distribution, or the 1-D
            # non-zero balanced distribution.
            assert self.shape[1] == other.shape[0]
            A, B = cast_to_common_type(self, other)
            if out is None:
                C = store_to_cunumeric_array(
                    ctx.create_store(
                        A.dtype, shape=(self.shape[0], other.shape[1])
                    )
                )
            else:
                if out.dtype != A.dtype:
                    raise ValueError(
                        f"Output type {out.dtype} is not consistent "
                        f"with resolved dtype {A.dtype}"
                    )
                assert out.shape == (self.shape[0], other.shape[1])
                C = out
            spmm(A, B, C)
            return C
        else:
            raise NotImplementedError

    def matvec(self, other):
        return self @ other

    def tocsr(self, copy=False):
        if copy:
            return self.copy().tocsr(copy=False)
        return self

    def tocsc(self, copy=False):
        if copy:
            return self.copy().tocsc(copy=False)
        return self.tocoo().tocsc()

    def tocoo(self, copy=False):
        if copy:
            return self.copy().tocoo(copy=False)
        # The conversion to COO is pretty straightforward. The crd and values
        # arrays are already set up for COO, we just need to expand the pos
        # array into coordinates.
        rows_expanded = ctx.create_store(coord_ty, shape=self.crd.shape)
        task = ctx.create_task(SparseOpCode.EXPAND_POS_TO_COORDINATES)
        task.add_input(self.pos)
        task.add_output(rows_expanded)
        task.add_image_constraint(
            self.pos,
            rows_expanded,
            range=True,
            functor=CompressedImagePartition,
        )
        task.execute()
        return sparse.coo_array(
            (self.vals, (rows_expanded, self.crd)),
            shape=self.shape,
            dtype=self.dtype,
        )

    def transpose(self, copy=False):
        if copy:
            return self.copy().transpose(copy=False)
        return sparse.csc_array.make_with_same_nnz_structure(
            self,
            (self.vals, self.crd, self.pos),
            shape=Shape((self.shape[1], self.shape[0])),
        )

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cunumeric.empty(0, dtype=self.dtype)
        output = ctx.create_store(
            self.dtype, shape=(min(rows + min(k, 0), cols - max(k, 0)))
        )
        # TODO (rohany): Just to get things working for the AMG example, we'll
        # just support k == 0.
        if k != 0:
            raise NotImplementedError
        task = ctx.create_task(SparseOpCode.CSR_DIAGONAL)
        task.add_output(output)
        self._add_to_task(task)
        task.add_alignment(output, self.pos)
        task.add_image_constraint(
            self.pos, self.crd, range=True, functor=CompressedImagePartition
        )
        task.add_alignment(self.crd, self.vals)
        task.execute()
        return store_to_cunumeric_array(output)

    T = property(transpose)

    def todense(self, order=None, out=None):
        if order is not None:
            raise NotImplementedError
        if out is not None:
            out = cunumeric.array(out)
            if out.dtype != self.dtype:
                raise ValueError(
                    f"Output type {out.dtype} is not consistent "
                    f"with dtype {self.dtype}"
                )
            out = get_store_from_cunumeric_array(out)
        elif out is None:
            out = ctx.create_store(self.dtype, shape=self.shape)
        # TODO (rohany): We'll do a row-based distribution for now, but a
        # non-zero based distribution of this operation with overlapping output
        # regions being reduced into also seems possible.
        promoted_pos = self.pos.promote(1, self.shape[1])
        task = ctx.create_task(SparseOpCode.CSR_TO_DENSE)
        task.add_output(out)
        task.add_input(promoted_pos)
        task.add_input(self.crd)
        task.add_input(self.vals)
        # We aren't partitioning the columns.
        task.add_broadcast(out, 1)
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
    @track_provenance(runtime.legate_context, nested=True)
    def sddmm(self, C, D):
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
        return sddmm_impl(*cast_to_common_type(self, C, D))

    def multiply(self, other):
        return self * other

    # other / mat is defined to be the element-wise division of a other by mat
    # over the non-zero coordinates of mat. For now, we restrict this operation
    # to be on scalars only.
    def __rtruediv__(self, other):
        if not cunumeric.isscalar(other):
            raise NotImplementedError
        vals_arr = store_to_cunumeric_array(self.vals)
        new_vals = other / vals_arr
        return csr_array.make_with_same_nnz_structure(
            self,
            (get_store_from_cunumeric_array(new_vals), self.crd, self.pos),
        )

    # This is an element-wise operation now.
    def __mul__(self, other):
        if isinstance(other, csr_array):
            assert self.shape == other.shape
            B, C = cast_to_common_type(self, other)
            return mult(B, C)

        # If we got a sparse matrix type that we know about, try and convert it
        # to csr to complete the addition.
        if sparse.is_sparse_matrix(other):
            return self * other.tocsr()

        # At this point, we have objects that we might not understand. Case to
        # try and figure out what they are.
        if isinstance(other, numpy.ndarray):
            other = cunumeric.ndarray(other)
        if cunumeric.ndim(other) == 0:
            # If we have a scalar, then do an element-wise multiply on the
            # values array.
            new_vals = store_to_cunumeric_array(self.vals) * other
            return csr_array.make_with_same_nnz_structure(
                self,
                (get_store_from_cunumeric_array(new_vals), self.crd, self.pos),
            )
        elif isinstance(other, cunumeric.ndarray):
            assert self.shape == other.shape
            return mult_dense(*cast_to_common_type(self, other))
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        # A not-too-optimized implementation of subtract: multiply by -1 then
        # add.
        return self + (other * -1.0)

    def __add__(self, other):
        if isinstance(other, numpy.ndarray):
            other = cunumeric.array(other)
        # If we're being added against a dense matrix, there's no point of
        # doing anything smart. Convert ourselves to a dense matrix and do the
        # addition.
        if isinstance(other, cunumeric.ndarray):
            return self.todense() + other
        # If the other operand is sparse, then cast it into the format we know
        # how to deal with.
        if sparse.is_sparse_matrix(other):
            other = other.tocsr()
        else:
            raise NotImplementedError
        assert self.shape == other.shape
        return add(*cast_to_common_type(self, other))

    # rmatmul represents the operation other @ self.
    def __rmatmul__(self, other):
        if isinstance(other, numpy.ndarray):
            other = cunumeric.array(other)
        if len(other.shape) == 1:
            raise NotImplementedError
        elif len(other.shape) == 2:
            assert other.shape[1] == self.shape[0]
            A, B = cast_to_common_type(other, self)
            # Similiarly to other places in the code, we need to
            # ensure that any futures created by cunumeric are
            # promoted to regions for our tasks to work.
            C = store_to_cunumeric_array(
                get_store_from_cunumeric_array(
                    cunumeric.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
                )
            )
            rspmm(A, B, C)
            return C
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        return self.dot(other)

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

    def _add_to_task(self, task, input=True, vals=True):
        if input:
            task.add_input(self.pos)
            task.add_input(self.crd)
            if vals:
                task.add_input(self.vals)
        else:
            task.add_output(self.pos)
            task.add_output(self.crd)
            if vals:
                task.add_output(self.vals)


def scan_local_results_and_scale_pos(
    weights: FutureMap, pos: Store, pos_part: StorePartition, num_procs: int
):
    weights.wait()
    val = 0
    scanVals = []
    for i in range(0, num_procs):
        fut = weights.get_future(Point(i))
        count = int.from_bytes(fut.get_buffer(), "little")
        scanVals.append(val)
        val += count
    scanFutures = []
    for value in scanVals:
        buf = ffi.buffer(ffi.new("int64_t*", value))
        scanFutures.append(
            Future.from_buffer(runtime.legate_runtime.legion_runtime, buf)
        )
    scanFutureMap = FutureMap.from_list(
        runtime.legate_runtime.legion_context,
        runtime.legate_runtime.legion_runtime,
        scanFutures,
    )
    assert (
        pos_part.partition.color_shape.ndim == 1
        and pos_part.partition.color_shape[0] == num_procs
    )
    task = ctx.create_manual_task(
        SparseOpCode.SCALE_RECT_1, launch_domain=Rect(hi=(num_procs,))
    )
    task.add_output(pos_part)
    task.add_input(pos_part)
    task._scalar_future_maps.append(scanFutureMap)
    task.execute()


# spmv computes y = A @ x.
def spmv(
    A: csr_array, x: cunumeric.ndarray, y: cunumeric.ndarray, domain_part=False
):
    x_store = get_store_from_cunumeric_array(x)
    y_store = get_store_from_cunumeric_array(y)

    if domain_part:
        # In this case, we use a partition of the column vector
        # to create partitions of the rest of the matrix.
        # We'll start with an equal partition of the domain vector,
        # and take preimages all the way back out to the range vector,
        # similar to LegionSolvers. Importantly we use DensePreimages,
        # which densify the tight preimage computed by Realm, as our
        # iteration methods like to iterate over data that might not
        # be actually present in the tight partitions.
        procs = runtime.num_procs
        tile_shape = (x_store.shape[0] + procs - 1) // procs
        x_tiling = Tiling(
            Shape(
                tile_shape,
            ),
            Shape(
                procs,
            ),
        )
        x_part = x_store.partition(x_tiling)
        # Create the partition of crd from x. We'll also use
        # this partition for vals.
        crd_part = DensePreimage(
            A.crd,
            x_store,
            x_part.partition,
            ctx.mapper_id,
            range=False,
            disjoint=False,
            complete=False,
        )
        # Preimage again up from crd into pos. We'll also use
        # this partition for y.
        pos_part = DensePreimage(
            A.pos,
            A.crd,
            crd_part,
            ctx.mapper_id,
            range=True,
            disjoint=False,
            complete=False,
        )
        launch_domain = Rect(
            hi=Shape(
                procs,
            )
        )
        # We're using a manual task right now. In the future,
        # we can add preimage support to the solver so that
        # this can fit nicer with the existing infra.
        task = ctx.create_manual_task(
            SparseOpCode.CSR_SPMV_COL_SPLIT, launch_domain=launch_domain
        )
        task.add_reduction(y_store.partition(pos_part), ReductionOp.ADD)
        task.add_input(A.pos.partition(pos_part))
        task.add_input(A.crd.partition(crd_part))
        task.add_input(A.vals.partition(crd_part))
        task.add_input(x_part)
        task.execute()
    else:
        # An auto-parallelized version of the kernel.
        task = ctx.create_task(SparseOpCode.CSR_SPMV_ROW_SPLIT)
        task.add_output(y_store)
        task.add_input(A.pos)
        task.add_input(A.crd)
        task.add_input(A.vals)
        task.add_input(x_store)
        task.add_alignment(y_store, A.pos)
        task.add_image_constraint(
            A.pos,
            A.crd,
            range=True,
            functor=CompressedImagePartition,
        )
        task.add_alignment(A.crd, A.vals)
        # TODO (rohany): Both adding an image constraint explicitly and an
        #  alignment constraint between vals and crd works now. Adding the
        #  image is explicit though, while adding the alignment is more in
        #  line with the DISTAL way of doing things.
        #
        # task.add_image_constraint(self.pos, self.vals, range=True)
        #
        # An important optimization is to use an image operation to request
        # only the necessary pieces of data from the x vector in y = Ax. We
        # don't make an attempt to use a sparse instance, so we allocate
        # the full vector x in each task, but by using the sparse instance
        # we ensure that only the necessary pieces of data are
        # communicated. In many common sparse matrix patterns, this can
        # result in an asymptotic decrease in the amount of communication.
        # The image of the selected coordinates into other vector is
        # not complete or disjoint.
        task.add_image_constraint(
            A.crd,
            x_store,
            range=False,
            disjoint=False,
            complete=False,
            functor=MinMaxImagePartition,
        )
        task.execute()


# add computes A = B + C and returns A.
def add(B: csr_array, C: csr_array) -> csr_array:
    # Create the assemble query result array.
    shape = B.shape
    q_nnz = ctx.create_store(nnz_ty, shape=(shape[0]))
    task = ctx.create_task(SparseOpCode.ADD_CSR_CSR_NNZ)
    task.add_output(q_nnz)
    task.add_input(B.pos)
    task.add_input(B.crd)
    task.add_input(C.pos)
    task.add_input(C.crd)
    task.add_scalar_arg(shape[1], types.int64)
    # Partitioning.
    task.add_alignment(q_nnz, B.pos)
    task.add_alignment(B.pos, C.pos)
    task.add_image_constraint(
        B.pos, B.crd, range=True, functor=CompressedImagePartition
    )
    task.add_image_constraint(
        C.pos, C.crd, range=True, functor=CompressedImagePartition
    )
    task.execute()

    pos, nnz = CompressedBase.nnz_to_pos_cls(q_nnz)
    crd = ctx.create_store(coord_ty, shape=(nnz,))
    vals = ctx.create_store(B.dtype, shape=(nnz,))

    task = ctx.create_task(SparseOpCode.ADD_CSR_CSR)
    task.add_output(pos)
    task.add_output(crd)
    task.add_output(vals)
    task.add_input(B.pos)
    task.add_input(B.crd)
    task.add_input(B.vals)
    task.add_input(C.pos)
    task.add_input(C.crd)
    task.add_input(C.vals)
    task.add_scalar_arg(shape[1], types.int64)
    # Partitioning.
    task.add_alignment(pos, B.pos)
    task.add_alignment(B.pos, C.pos)
    task.add_image_constraint(
        pos, crd, range=True, functor=CompressedImagePartition
    )
    task.add_image_constraint(
        B.pos, B.crd, range=True, functor=CompressedImagePartition
    )
    task.add_image_constraint(
        C.pos, C.crd, range=True, functor=CompressedImagePartition
    )
    task.add_alignment(crd, vals)
    task.add_alignment(B.crd, B.vals)
    task.add_alignment(C.crd, C.vals)
    # Make sure that we get pos in READ_WRITE mode.
    task.add_input(pos)
    task.execute()
    return csr_array((vals, crd, pos), shape=shape, dtype=B.dtype)


# mult computes A = B + C and returns A when C is sparse.
def mult(B: csr_array, C: csr_array) -> csr_array:
    q_nnz = ctx.create_store(nnz_ty, shape=(B.shape[0]))
    task = ctx.create_task(SparseOpCode.ELEM_MULT_CSR_CSR_NNZ)
    task.add_output(q_nnz)
    task.add_input(B.pos)
    task.add_input(B.crd)
    task.add_input(C.pos)
    task.add_input(C.crd)
    task.add_scalar_arg(B.shape[1], types.int64)
    task.add_alignment(q_nnz, B.pos)
    task.add_alignment(B.pos, C.pos)
    task.add_image_constraint(
        B.pos,
        B.crd,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_image_constraint(
        C.pos,
        C.crd,
        range=True,
        functor=CompressedImagePartition,
    )
    task.execute()

    pos, nnz = CompressedBase.nnz_to_pos_cls(q_nnz)
    crd = ctx.create_store(coord_ty, shape=(nnz))
    vals = ctx.create_store(B.dtype, shape=(nnz))

    task = ctx.create_task(SparseOpCode.ELEM_MULT_CSR_CSR)
    task.add_output(pos)
    task.add_output(crd)
    task.add_output(vals)
    task.add_input(B.pos)
    task.add_input(B.crd)
    task.add_input(B.vals)
    task.add_input(C.pos)
    task.add_input(C.crd)
    task.add_input(C.vals)
    task.add_scalar_arg(B.shape[1], types.int64)
    task.add_alignment(pos, B.pos)
    task.add_alignment(B.pos, C.pos)
    task.add_image_constraint(
        pos, crd, range=True, functor=CompressedImagePartition
    )
    task.add_image_constraint(
        B.pos,
        B.crd,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_image_constraint(
        C.pos,
        C.crd,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_alignment(crd, vals)
    task.add_alignment(B.crd, B.vals)
    task.add_alignment(C.crd, C.vals)
    # Make sure that pos is in READ_WRITE mode.
    task.add_input(pos)
    task.execute()
    return csr_array((vals, crd, pos), shape=B.shape, dtype=B.dtype)


# mult_dense computes A = B * C when C is dense.
def mult_dense(B: csr_array, C: cunumeric.ndarray) -> csr_array:
    # This is an operation that preserves the non-zero structure of
    # the output, so we'll just allocate a new store of values for
    # the output matrix and share the existing pos and crd arrays.
    C_store = get_store_from_cunumeric_array(C)
    result_vals = ctx.create_store(B.dtype, shape=B.vals.shape)
    promoted_pos = B.pos.promote(1, B.shape[1])
    task = ctx.create_task(SparseOpCode.ELEM_MULT_CSR_DENSE)
    task.add_output(result_vals)
    task.add_input(promoted_pos)
    task.add_input(B.crd)
    task.add_input(B.vals)
    task.add_input(C_store)
    # Partition the rows.
    task.add_broadcast(promoted_pos, 1)
    task.add_alignment(promoted_pos, C_store)
    task.add_image_constraint(
        promoted_pos,
        result_vals,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_image_constraint(
        promoted_pos,
        B.crd,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_image_constraint(
        promoted_pos,
        B.vals,
        range=True,
        functor=CompressedImagePartition,
    )
    task.execute()
    return csr_array.make_with_same_nnz_structure(
        B, (result_vals, B.crd, B.pos)
    )


# spmm computes C = A @ B.
def spmm(A: csr_array, B: cunumeric.ndarray, C: cunumeric.ndarray) -> None:
    B_store = get_store_from_cunumeric_array(B)
    C_store = get_store_from_cunumeric_array(C)
    # TODO (rohany): In an initial implementation, we'll partition
    # things only by the `i` dimension of the computation. However, the
    # leaf kernel as written allows for both the `i` and `j` dimensions
    # of the computation to be partitioned. I'm avoiding doing the
    # multi-dimensional parallelism here because I'm not sure how to
    # express it within the solver's partitioning constraints. This is
    # the first example of needing affine projections the partitioning
    # -- we need to partition the output into tiles, and project the
    # first dimension onto self.pos, and project the second dimension
    # onto other.
    promoted_pos = A.pos.promote(1, C_store.shape[1])
    task = ctx.create_task(SparseOpCode.SPMM_CSR_DENSE)
    task.add_output(C_store)
    task.add_input(promoted_pos)
    task.add_input(A.crd)
    task.add_input(A.vals)
    task.add_input(B_store)
    # Partitioning.
    task.add_broadcast(promoted_pos, 1)
    task.add_alignment(C_store, promoted_pos)
    task.add_image_constraint(
        promoted_pos,
        A.crd,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_image_constraint(
        promoted_pos,
        A.vals,
        range=True,
        functor=CompressedImagePartition,
    )

    # In order to do the image from the coordinates into the
    # corresponding rows of other, we have to apply an AffineProjection
    # from the coordinates to cast them up to reference rows of other,
    # rather than single points. The API for this is a bit restrictive,
    # so we have to pass a staged MinMaxImagePartition functor through
    # to the image constraint.
    def partFunc(*args, **kwargs):
        return MinMaxImagePartition(*args, proj_dims=[0], **kwargs)

    task.add_image_constraint(
        A.crd,
        B_store,
        range=False,
        disjoint=False,
        complete=False,
        functor=partFunc,
    )
    task.add_scalar_arg(A.shape[1], types.int64)
    task.execute()


# rspmm computes C = A @ B.
def rspmm(A: cunumeric.ndarray, B: csr_array, C: cunumeric.ndarray) -> None:
    A_store = get_store_from_cunumeric_array(A)
    C_store = get_store_from_cunumeric_array(C)
    # TODO (rohany): As with the other SpMM case, we could do a 2-D
    #  distribution, but I'll just do a 1-D distribution right now,
    #  along the k-dimension of the computation.
    promoted_pos = B.pos.promote(0, A_store.shape[0])
    task = ctx.create_task(SparseOpCode.SPMM_DENSE_CSR)
    task.add_reduction(C_store, ReductionOp.ADD)
    task.add_input(A_store)
    task.add_input(promoted_pos)
    task.add_input(B.crd)
    task.add_input(B.vals)
    # Partition the rows of the sparse matrix.
    task.add_broadcast(promoted_pos, 0)
    task.add_alignment(promoted_pos, A_store)
    task.add_image_constraint(
        promoted_pos,
        B.crd,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_image_constraint(
        promoted_pos,
        B.vals,
        range=True,
        functor=CompressedImagePartition,
    )
    # Initially, only k is partitioned, so we'll be reducing into the
    # full output.
    task.add_broadcast(C_store)
    task.execute()


# sddmm_impl computes A = B * (C @ D), returning A.
def sddmm_impl(
    B: csr_array, C: cunumeric.ndarray, D: cunumeric.ndarray
) -> csr_array:
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
    C_store = get_store_from_cunumeric_array(C)
    D_store = get_store_from_cunumeric_array(D)
    result_vals = ctx.create_store(B.dtype, shape=B.vals.shape)
    task = ctx.create_task(SparseOpCode.CSR_SDDMM)
    task.add_output(result_vals)
    promoted_pos = B.pos.promote(1, C_store.shape[1])
    task.add_input(promoted_pos)
    task.add_input(B.crd)
    task.add_input(B.vals)
    task.add_input(C_store)
    task.add_input(D_store)
    # Partition the rows of the sparse matrix and C.
    task.add_broadcast(promoted_pos, 1)
    task.add_alignment(promoted_pos, C_store)
    task.add_image_constraint(
        promoted_pos,
        result_vals,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_image_constraint(
        promoted_pos,
        B.crd,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_image_constraint(
        promoted_pos,
        B.vals,
        range=True,
        functor=CompressedImagePartition,
    )
    task.add_broadcast(D_store)
    task.execute()
    return csr_array.make_with_same_nnz_structure(
        B, (result_vals, B.crd, B.pos)
    )


# spgemm_csr_csr_csr computes A = B @ C when B and C and
# both csr matrices, and returns the result A as a csr matrix.
def spgemm_csr_csr_csr(B: csr_array, C: csr_array) -> csr_array:
    # Due to limitations in cuSPARSE, we cannot use a uniform task
    # implementation for CSRxCSRxCSR SpGEMM across CPUs, OMPs and GPUs.
    # The GPU implementation will create a set of local CSR matrices
    # that will be aggregated into a global CSR.
    if runtime.num_gpus > 0:
        pos = ctx.create_store(rect1, shape=B.shape[0])
        crd = ctx.create_store(coord_ty, ndim=1)
        vals = ctx.create_store(B.dtype, ndim=1)
        num_procs = runtime.num_procs
        tile_shape = (B.shape[0] + num_procs - 1) // num_procs
        tiling = Tiling(Shape(tile_shape), Shape(num_procs))
        task = ctx.create_manual_task(
            SparseOpCode.SPGEMM_CSR_CSR_CSR_GPU,
            launch_domain=Rect(hi=(num_procs,)),
        )
        pos_part = pos.partition(tiling)
        task.add_output(pos_part)
        task.add_output(crd)
        task.add_output(vals)
        my_pos_part = B.pos.partition(tiling)
        task.add_input(my_pos_part)
        image = CompressedImagePartition(
            B.pos,
            my_pos_part.partition,
            ctx.mapper_id,
            range=True,
            disjoint=True,
            complete=True,
        )
        crd_part = B.crd.partition(image)
        task.add_input(crd_part)
        task.add_input(B.vals.partition(image))
        # The C matrix is unfortunately replicated in this algorithm.
        # However, we can make the world a little better for us by
        # gathering only the rows of C that are referenced by each
        # partition using Image operations.
        crd_image = MinMaxImagePartition(
            B.crd,
            crd_part.partition,
            ctx.mapper_id,
            range=False,
            disjoint=False,
            complete=False,
        )
        other_pos_part = C.pos.partition(crd_image)
        task.add_input(other_pos_part)
        other_pos_image = CompressedImagePartition(
            C.pos,
            other_pos_part.partition,
            ctx.mapper_id,
            range=True,
            disjoint=False,
            complete=False,
        )
        task.add_input(C.crd.partition(other_pos_image))
        task.add_input(C.vals.partition(other_pos_image))
        task.add_scalar_arg(C.shape[1], types.uint64)
        task.add_scalar_arg(C.shape[0], types.uint64)
        task.execute()
        # Build the global CSR array by performing a scan across the
        # individual CSR results. Due to a recent change in legate.core
        # that doesn't do future map reductions on launches of size
        # one, we have to gaurd this operation as crd might not have a
        # key partition.
        if num_procs > 1:
            scan_local_results_and_scale_pos(
                crd.get_key_partition()._weights,
                pos,
                pos_part,
                num_procs,
            )
        return csr_array((vals, crd, pos), shape=(B.shape[0], C.shape[1]))
    else:
        # Create the query result.
        q_nnz = ctx.create_store(nnz_ty, shape=B.shape[0])
        task = ctx.create_task(SparseOpCode.SPGEMM_CSR_CSR_CSR_NNZ)
        task.add_output(q_nnz)
        task.add_input(B.pos)
        task.add_input(B.crd)
        task.add_input(C.pos)
        task.add_input(C.crd)
        task.add_alignment(q_nnz, B.pos)
        task.add_image_constraint(
            B.pos,
            B.crd,
            range=True,
            functor=CompressedImagePartition,
        )
        # We'll only ask for the rows used by each partition by
        # following an image of pos through crd. We'll then use that
        # partition to declare the pieces of crd and vals of other that
        # are needed by the matmul. The resulting image of coordinates
        # into rows of other is not necessarily complete or disjoint.
        task.add_image_constraint(
            B.crd,
            C.pos,
            range=False,
            disjoint=False,
            complete=False,
            functor=MinMaxImagePartition,
        )
        # Since the target partition of pos is likely not contiguous,
        # we can't use the CompressedImagePartition functor and have to
        # fall back to a standard functor. Since the source partition
        # of the rows is not complete or disjoint, the images into crd
        # and vals are not disjoint either.
        task.add_image_constraint(
            C.pos,
            C.crd,
            range=True,
            disjoint=False,
            complete=False,
        )
        task.add_image_constraint(
            C.pos,
            C.vals,
            range=True,
            disjoint=False,
            complete=False,
        )
        task.add_scalar_arg(C.shape[1], types.uint64)
        task.execute()

        pos, nnz = CompressedBase.nnz_to_pos_cls(q_nnz)
        crd = ctx.create_store(coord_ty, shape=(nnz))
        vals = ctx.create_store(B.dtype, shape=(nnz))

        task = ctx.create_task(SparseOpCode.SPGEMM_CSR_CSR_CSR)
        task.add_output(pos)
        task.add_output(crd)
        task.add_output(vals)
        task.add_input(B.pos)
        task.add_input(B.crd)
        task.add_input(B.vals)
        task.add_input(C.pos)
        task.add_input(C.crd)
        task.add_input(C.vals)
        task.add_alignment(B.pos, pos)
        task.add_image_constraint(
            B.pos,
            B.crd,
            range=True,
            functor=CompressedImagePartition,
        )
        task.add_alignment(B.crd, B.vals)
        task.add_image_constraint(
            pos, crd, range=True, functor=CompressedImagePartition
        )
        task.add_alignment(crd, vals)
        task.add_broadcast(C.pos)
        task.add_broadcast(C.crd)
        task.add_broadcast(C.vals)
        # Add pos to the inputs as well so that we get READ_WRITE
        # privileges.
        task.add_input(pos)
        task.add_scalar_arg(C.shape[1], types.uint64)
        task.execute()
        return csr_array(
            (vals, crd, pos),
            shape=Shape((B.shape[0], C.shape[1])),
        )


# spgemm_csr_csr_csc computes A = B @ C when B is a csr_array,
# C is a csc_array, and returns A as a csr_array.
def spgemm_csr_csr_csc(B: csr_array, C: Any) -> csr_array:
    # Here, we want to enable partitioning the i and j dimensions of
    # A(i, j) = B(i, k) * C(k, j).  To do this, we'll first logically
    # organize our processors into a 2-D grid, and partition the pos
    # region of B along the i dimension of the processor grid,
    # replicated onto the j dimension, and partition C along the j
    # dimension, replicated onto the i dimension.
    num_procs = runtime.num_procs
    grid = Shape(factor_int(num_procs))

    rows_proj_fn = runtime.get_1d_to_2d_functor_id(grid[0], grid[1], True)
    cols_proj_fn = runtime.get_1d_to_2d_functor_id(grid[0], grid[1], False)

    # To create a tiling on a 2-D color space of a 1-D region, we first
    # promote the region into a 2-D region, and then apply a tiling to
    # it with the correct color shape. In particular, we want to
    # broadcast the colorings over the i dimension across the j
    # dimension of the grid, which the promotion does for us.
    my_promoted_pos = B.pos.promote(1)
    my_pos_tiling = Tiling(
        Shape(((B.pos.shape[0] + grid[0] - 1) // grid[0], 1)), grid
    )
    my_pos_partition = my_promoted_pos.partition(my_pos_tiling)
    other_promoted_pos = C.pos.promote(0)
    other_pos_tiling = (C.pos.shape[0] + grid[1] - 1) // grid[1]
    other_pos_partition = other_promoted_pos.partition(
        Tiling(Shape((1, other_pos_tiling)), grid)
    )

    # TODO (rohany): There's a really weird interaction here that needs
    # help from wonchan.  If I'm deriving a partition from another, I
    # need the partition pos before all of the transformations that are
    # applied to it. However, the normal partition.partition is the
    # partition before any transformations. I'm not sure what's the
    # best way to untangle this.
    my_pos_image = ImagePartition(
        B.pos,
        my_pos_partition._storage_partition._partition,
        ctx.mapper_id,
        range=True,
    )
    other_pos_image = ImagePartition(
        C.pos,
        other_pos_partition._storage_partition._partition,
        ctx.mapper_id,
        range=True,
    )

    # First, we launch a task that tiles the output matrix and creates
    # a local csr matrix result for each tile.
    task = ctx.create_manual_task(
        SparseOpCode.SPGEMM_CSR_CSR_CSC_LOCAL_TILES,
        launch_domain=Rect(hi=(num_procs,)),
    )
    # Note that while we colored the region in a 2-D color space,
    # legate does some smart things to essentially reduce the coloring
    # to a 1-D coloring and uses projection functors to send the right
    # subregions to the right tasks.  So, we use a special functor that
    # we register that understands the logic for this particular type
    # of partitioning.
    task.add_input(my_pos_partition, proj=rows_proj_fn)
    task.add_input(B.crd.partition(my_pos_image), proj=rows_proj_fn)
    task.add_input(B.vals.partition(my_pos_image), proj=rows_proj_fn)
    task.add_input(other_pos_partition, proj=cols_proj_fn)
    task.add_input(C.crd.partition(other_pos_image), proj=cols_proj_fn)
    task.add_input(C.vals.partition(other_pos_image), proj=cols_proj_fn)
    # Finally, create some unbound stores that will represent the
    # logical components of each sub-csr matrix that are created by the
    # launched tasks.
    pos = ctx.create_store(rect1, ndim=1)
    crd = ctx.create_store(coord_ty, ndim=1)
    vals = ctx.create_store(B.dtype, ndim=1)
    task.add_output(pos)
    task.add_output(crd)
    task.add_output(vals)
    task.add_scalar_arg(C.shape[0], types.int64)
    task.execute()

    # Due to recent changes in the legate core, we don't get a future
    # map back if the size of the launch is 1, meaning that we won't
    # have key partitions the launch over. Luckily if the launch domain
    # has size 1 then all of the data structures we have created are a
    # valid CSR array, so we can return early.
    if num_procs == 1:
        return csr_array(
            (vals, crd, pos), shape=(B.shape[0], C.shape[1]), dtype=B.dtype
        )

    # After the local execution, we need to start building a global CSR
    # array.  First, we offset all of the local pos pieces with indices
    # in the global crd and vals arrays. Since each local pos region is
    # already valid, we just need to perform a scan over the final size
    # of each local crd region and offset each pos region by the
    # result.
    pos_part = pos.partition(pos.get_key_partition())
    scan_local_results_and_scale_pos(
        crd.get_key_partition()._weights, pos, pos_part, num_procs
    )

    # Now it gets trickier. We have to massage the local tiles of csr
    # matrices into one global CSR matrix. To do this, we will consider
    # each row of the processor grid independently. Within each row of
    # the grid, each tile contains a CSR matrix over the same set of
    # rows of the output, but different columns. So, we will distribute
    # those rows across the processors in the current row. We construct
    # a store that describes how the communication should occur between
    # the processors in the current row. Each processor records the
    # pieces of their local pos regions that correspond to the rows
    # assigned to each other processor in the grid. We can then apply
    # image operations to this store to collect the slices of the local
    # results that should be sent to each processor. Precisely, we
    # create a 3-D region that describes for each processor in the
    # processor grid, the range of entries to be sent to the other
    # processors in that row.
    partitioner_store = ctx.create_store(
        rect1, shape=Shape((grid[0], grid[1], grid[1]))
    )
    # TODO (rohany): This operation _should_ be possible with tiling
    # operations, but I can't get the API to do what I want -- the
    # problem appears to be trying to match a 2-D coloring onto a 3-D
    # region.
    partitioner_store_coloring = {}
    for i in range(grid[0]):
        for j in range(grid[1]):
            rect = Rect(
                lo=(i, j, 0),
                hi=(i, j, grid[1] - 1),
                dim=3,
                exclusive=False,
            )
            partitioner_store_coloring[Point((i, j))] = rect
    partitioner_store_partition = partitioner_store.partition(
        DomainPartition(
            partitioner_store.shape, grid, partitioner_store_coloring
        )
    )
    task = ctx.create_manual_task(
        SparseOpCode.SPGEMM_CSR_CSR_CSC_COMM_COMPUTE,
        launch_domain=Rect(hi=(num_procs,)),
    )
    promote_1d_to_2d = ctx.get_projection_id(
        SparseProjectionFunctor.PROMOTE_1D_TO_2D
    )
    task.add_output(partitioner_store_partition, proj=promote_1d_to_2d)
    task.add_input(my_pos_partition, proj=rows_proj_fn)
    task.add_input(pos_part)
    # Scalar arguments must use the legate core type system.
    task.add_scalar_arg(grid[0], types.int32)
    task.add_scalar_arg(grid[1], types.int32)
    task.execute()

    # We now create a transposed partition of the store to get sets of
    # ranges assigned to each processor. This partition selects for
    # each processor the ranges created for that processor by all of
    # the other processor.
    transposed_partitioner_store_coloring = {}
    for i in range(grid[0]):
        for j in range(grid[1]):
            rect = Rect(
                lo=(i, 0, j),
                hi=(i, grid[1] - 1, j),
                dim=3,
                exclusive=False,
            )
            transposed_partitioner_store_coloring[Point((i, j))] = rect
    transposed_partition = partitioner_store.partition(
        DomainPartition(
            partitioner_store.shape,
            grid,
            transposed_partitioner_store_coloring,
        )
    )
    # Cascade images down to the global pos, crd and vals regions.
    global_pos_partition = pos.partition(
        ImagePartition(
            partitioner_store,
            transposed_partition.partition,
            ctx.mapper_id,
            range=True,
        )
    )
    global_crd_partition = crd.partition(
        ImagePartition(
            pos,
            global_pos_partition.partition,
            ctx.mapper_id,
            range=True,
        )
    )
    global_vals_partition = vals.partition(
        ImagePartition(
            pos,
            global_pos_partition.partition,
            ctx.mapper_id,
            range=True,
        )
    )
    # This next task utilizes the pieces computed by the transposed
    # partition and gathers them into contiguous pieces to form the
    # result csr matrix.
    task = ctx.create_manual_task(
        SparseOpCode.SPGEMM_CSR_CSR_CSC_SHUFFLE,
        launch_domain=Rect(hi=(num_procs,)),
    )
    task.add_input(global_pos_partition, proj=promote_1d_to_2d)
    task.add_input(global_crd_partition, proj=promote_1d_to_2d)
    task.add_input(global_vals_partition, proj=promote_1d_to_2d)
    # We could compute this up front with a 2 pass algorithm, but it
    # seems expedient to just use Legion's output region support for
    # now.
    final_pos = ctx.create_store(rect1, ndim=1)
    final_crd = ctx.create_store(coord_ty, ndim=1)
    final_vals = ctx.create_store(B.dtype, ndim=1)
    task.add_output(final_pos)
    task.add_output(final_crd)
    task.add_output(final_vals)
    task.execute()

    # At this point, we have an almost valid csr array. The only thing
    # missing is that again each pos array created by the grouping task
    # is not globally offset. We adjust this with one more scan over
    # all of the output sizes.
    weights = final_crd.get_key_partition()._weights
    scan_local_results_and_scale_pos(
        weights,
        final_pos,
        final_pos.partition(final_pos.get_key_partition()),
        num_procs,
    )
    return csr_array(
        (final_vals, final_crd, final_pos),
        shape=Shape((B.shape[0], C.shape[1])),
        dtype=B.dtype,
    )


csr_matrix = csr_array

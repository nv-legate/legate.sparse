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
from legate.core import Rect, types
from legate.core.launcher import TaskLauncher
from legate.core.partition import Broadcast, DomainPartition, Tiling
from legate.core.shape import Shape
from legate.core.types import ReductionOp

import sparse

from .base import CompressedBase
from .config import SparseOpCode, domain_ty
from .coverage import clone_scipy_arr_kind
from .runtime import ctx, runtime
from .types import coord_ty, nnz_ty
from .utils import (
    cast_arr,
    get_store_from_cunumeric_array,
    store_to_cunumeric_array,
)


@clone_scipy_arr_kind(scipy.sparse.coo_array)
class coo_array(CompressedBase):
    # TODO (rohany): For simplicity, we'll assume duplicate free COO.  These
    # should probably be arguments provided by the user, as computing these
    # statistics about the data is likely as intensive as actually performing
    # the conversions to other formats.
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        if isinstance(arg, numpy.ndarray):
            arg = cunumeric.array(arg)
        if isinstance(arg, cunumeric.ndarray):
            assert arg.ndim == 2
            # As a simple workaround for now, just load the matrix as a CSR
            # matrix and convert it to COO.
            # TODO (rohany): I basically want to do self = result, but I don't
            #  think that this is a valid python thing to do.
            arr = sparse.csr_array(arg).tocoo()
            shape = arr.shape
            data = cast_arr(arr._vals, dtype=arg.dtype)
            row = cast_arr(arr._i, dtype=coord_ty)
            col = cast_arr(arr._j, dtype=coord_ty)
        elif isinstance(arg, scipy.sparse.coo_array):
            # TODO (rohany): Not sure yet how to handle duplicates.
            shape = arg.shape
            data = cunumeric.array(arg.data, dtype=arg.dtype)
            row = cunumeric.array(arg.row, dtype=coord_ty)
            col = cunumeric.array(arg.col, dtype=coord_ty)
        else:
            (data, (row, col)) = arg
            data = cast_arr(data)
            row = cast_arr(row, dtype=coord_ty)
            col = cast_arr(col, dtype=coord_ty)

        # Extract stores from the cunumeric arrays.
        self._i = get_store_from_cunumeric_array(row)
        self._j = get_store_from_cunumeric_array(col)
        self._vals = get_store_from_cunumeric_array(data)

        if shape is None:
            # TODO (rohany): Perform a max over the rows and cols to estimate
            # the shape.
            raise NotImplementedError
        self.shape = shape

        # Use the user's dtype if requested, otherwise infer it from
        # the input data.
        if dtype is None:
            dtype = self.data.dtype
        else:
            self.data = self.data.astype(dtype)
        assert dtype is not None
        if not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)
        self.dtype = dtype

        # Ensure that we distribute operations on COO arrays across the entire
        # machine.
        # TODO (rohany): The sort routine is slightly broken when there are
        # more processors than potential split points in the samplesort. Work
        # around this by not distributing operations on very small numbers of
        # processors.
        if self._i.shape[0] >= runtime.num_procs:
            tile_size = (
                self._i.shape[0] + runtime.num_procs - 1
            ) // runtime.num_procs
            i_part = Tiling(Shape(tile_size), Shape(runtime.num_procs))
            self._i.set_key_partition(i_part)

    # TODO (rohany): Maybe use the individual setters and getters to handle
    #  when writes are performed to this field.
    @property
    def row(self):
        return store_to_cunumeric_array(self._i)

    @property
    def col(self):
        return store_to_cunumeric_array(self._j)

    # Enable direct operation on the values array.
    def get_data(self):
        return store_to_cunumeric_array(self._vals)

    def set_data(self, data):
        if isinstance(data, numpy.ndarray):
            data = cunumeric.array(data)
        assert isinstance(data, cunumeric.ndarray)
        self._vals = get_store_from_cunumeric_array(data)

    data = property(fget=get_data, fset=set_data)

    @property
    def nnz(self):
        return self._vals.shape[0]

    @property
    def format(self):
        return "coo"

    def astype(self, dtype, casting="unsafe", copy=True):
        row = self.row.copy() if copy else self.row
        col = self.col.copy() if copy else self.col
        data = self.data.astype(dtype, casting=casting, copy=copy)
        return coo_array((data, (row, col)), shape=self.shape, dtype=dtype)

    def diagonal(self, k=0):
        if k != 0:
            raise NotImplementedError
        # This function is lifted almost directly from scipy.sparse's
        # implementation.
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cunumeric.empty(0, dtype=self.data.dtype)
        diag = cunumeric.zeros(
            min(rows + min(k, 0), cols - max(k, 0)), dtype=self.dtype
        )
        diag_mask = (self.row + k) == self.col
        # I'll assert that we don't have duplicate coordinates, so this
        # operation is safe.
        row = self.row[diag_mask]
        data = self.data[diag_mask]
        diag[row + min(k, 0)] = data
        return diag

    def copy(self):
        rows = cunumeric.copy(self.row)
        cols = cunumeric.copy(self.col)
        data = cunumeric.copy(self.data)
        return coo_array(
            (data, (rows, cols)), dtype=self.dtype, shape=self.shape
        )

    def transpose(self, copy=False):
        if copy:
            raise NotImplementedError
        return coo_array(
            (self.data, (self.col, self.row)),
            dtype=self.dtype,
            shape=(self.shape[1], self.shape[0]),
        )

    T = property(transpose)

    def tocoo(self, copy=False):
        if copy:
            raise NotImplementedError
        return self

    def tocsr(self, copy=False):
        if copy:
            raise NotImplementedError

        # We'll try a different (and parallel) approach here. First, we'll sort
        # the data using key (row, column), and sort the values accordingly.
        # The result of this operation is that the columns and values arrays
        # suffice as crd and vals arrays for a csr array. We just need to
        # compress the rows array into a valid pos array.
        rows_store = ctx.create_store(self._i.type, ndim=1)
        cols_store = ctx.create_store(self._j.type, ndim=1)
        vals_store = ctx.create_store(self._vals.type, ndim=1)
        # Now sort the regions.
        task = ctx.create_task(SparseOpCode.SORT_BY_KEY)
        # Add all of the unbounded outputs.
        task.add_output(rows_store)
        task.add_output(cols_store)
        task.add_output(vals_store)
        # Add all of the input stores.
        task.add_input(self._i)
        task.add_input(self._j)
        task.add_input(self._vals)
        # The input stores need to all be aligned.
        task.add_alignment(self._i, self._j)
        task.add_alignment(self._i, self._vals)
        if runtime.num_gpus > 1:
            task.add_nccl_communicator()
        elif runtime.num_gpus == 0:
            task.add_cpu_communicator()
        task.execute()

        # Explicitly partition the rows into equal components to get the number
        # of non-zeros per row. We'll then partition up the non-zeros array
        # according to the per-partition ranges given by the min and max of
        # each partition.
        num_procs = runtime.num_procs
        # TODO (rohany): If I try to partition this on really small inputs
        #  (like size 0 or 1 stores) across multiple processors, I see some
        #  sparse non-deterministic failures. I haven't root caused these, and
        #  I'm running out of time to figure them out. It seems just not
        #  partitioning the input on these really small matrices side-steps the
        #  underlying issue.
        if rows_store.shape[0] <= num_procs:
            num_procs = 1
        row_tiling = (rows_store.shape[0] + num_procs - 1) // num_procs
        rows_part = rows_store.partition(
            Tiling(Shape(row_tiling), Shape(num_procs))
        )
        # In order to bypass legate.core's current inability to handle
        # representing stores as FutureMaps, we drop below the ManualTask API
        # to launch as task ourselves and use the returned future map directly
        # instead of letting the core try and extract data from it.
        launcher = TaskLauncher(
            ctx,
            SparseOpCode.BOUNDS_FROM_PARTITIONED_COORDINATES,
            error_on_interference=False,
            tag=ctx.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG,
            provenance=runtime.legate_context.provenance,
        )
        launcher.add_input(
            rows_store, rows_part.get_requirement(1, 0), tag=1
        )  # LEGATE_CORE_KEY_STORE_TAG
        bounds_store = ctx.create_store(
            domain_ty, shape=(1,), optimize_scalar=True
        )
        launcher.add_output(bounds_store, Broadcast(None, 0), tag=0)
        result = launcher.execute(Rect(hi=(num_procs,)))

        q_nnz = get_store_from_cunumeric_array(
            cunumeric.zeros((self.shape[0],), dtype=nnz_ty)
        )
        task = ctx.create_manual_task(
            SparseOpCode.SORTED_COORDS_TO_COUNTS,
            launch_domain=Rect(hi=(num_procs,)),
        )
        task.add_input(rows_part)
        task.add_reduction(
            q_nnz.partition(
                DomainPartition(q_nnz.shape, Shape(num_procs), result)
            ),
            ReductionOp.ADD,
        )
        task.add_scalar_arg(self.shape[0], types.int64)
        task.execute()
        # TODO (rohany): On small inputs, it appears that I get a
        #  non-deterministic failure, which appears either as a segfault or an
        #  incorrect output. This appears to show up only an OpenMP processors,
        #  and appears when running the full test suite with 4 opemps and 2
        #  openmp threads. My notes from debugging this are as follows:
        #   * The result of the sort appears to be correct.
        #   * We start with a valid COO matrix.
        #   * Adding print statements make the bug much harder to reproduce.
        #   * In particular, the bug is harder to reproduce when q_nnz is
        #     printed out before the `self.nnz_to_pos(q_nnz)` line here.
        #   * If q_nnz is printed out after the `self.nnz_to_pos(q_nnz)` line,
        #     then the computation looks correct but an incorrect pos array is
        #     generated.

        pos, _ = self.nnz_to_pos(q_nnz)
        return sparse.csr_array(
            (vals_store, cols_store, pos), shape=self.shape, dtype=self.dtype
        )

    def tocsc(self, copy=False):
        if copy:
            raise NotImplementedError
        # The strategy for CSC conversion is the same as COO conversion, we'll
        # just sort by the columns and then the rows.
        rows_store = ctx.create_store(self._i.type, ndim=1)
        cols_store = ctx.create_store(self._j.type, ndim=1)
        vals_store = ctx.create_store(self._vals.type, ndim=1)
        # Now sort the regions.
        task = ctx.create_task(SparseOpCode.SORT_BY_KEY)
        # Add all of the unbounded outputs.
        task.add_output(cols_store)
        task.add_output(rows_store)
        task.add_output(vals_store)
        # Add all of the input stores.
        task.add_input(self._j)
        task.add_input(self._i)
        task.add_input(self._vals)
        # The input stores need to all be aligned.
        task.add_alignment(self._i, self._j)
        task.add_alignment(self._i, self._vals)
        if runtime.num_gpus > 1:
            task.add_nccl_communicator()
        elif runtime.num_gpus == 0:
            task.add_cpu_communicator()
        task.execute()

        # Explicitly partition the cols into equal components to get the number
        # of non-zeros per row. We'll then partition up the non-zeros array
        # according to the per-partition ranges given by the min and max of
        # each partition.
        num_procs = runtime.num_procs
        # TODO (rohany): If I try to partition this on really small inputs
        #  (like size 0 or 1 stores) across multiple processors, I see some
        #  sparse non-deterministic failures. I haven't root caused these, and
        #  I'm running out of time to figure them out. It seems just not
        #  partitioning the input on these really small matrices side-steps the
        #  underlying issue.
        if cols_store.shape[0] <= num_procs:
            num_procs = 1
        col_tiling = (cols_store.shape[0] + num_procs - 1) // num_procs
        cols_part = cols_store.partition(
            Tiling(Shape(col_tiling), Shape(num_procs))
        )
        # In order to bypass legate.core's current inability to handle
        # representing stores as FutureMaps, we drop below the ManualTask API
        # to launch as task ourselves and use the returned future map directly
        # instead of letting the core try and extract data from it.
        launcher = TaskLauncher(
            ctx,
            SparseOpCode.BOUNDS_FROM_PARTITIONED_COORDINATES,
            error_on_interference=False,
            tag=ctx.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG,
            provenance=runtime.legate_context.provenance,
        )
        launcher.add_input(
            cols_store, cols_part.get_requirement(1, 0), tag=1
        )  # LEGATE_CORE_KEY_STORE_TAG
        bounds_store = ctx.create_store(
            domain_ty, shape=(1,), optimize_scalar=True
        )
        launcher.add_output(bounds_store, Broadcast(None, 0), tag=0)
        result = launcher.execute(Rect(hi=(num_procs,)))

        q_nnz = get_store_from_cunumeric_array(
            cunumeric.zeros((self.shape[1],), dtype=nnz_ty)
        )
        task = ctx.create_manual_task(
            SparseOpCode.SORTED_COORDS_TO_COUNTS,
            launch_domain=Rect(hi=(num_procs,)),
        )
        task.add_input(cols_part)
        task.add_reduction(
            q_nnz.partition(
                DomainPartition(q_nnz.shape, Shape(num_procs), result)
            ),
            ReductionOp.ADD,
        )
        task.add_scalar_arg(self.shape[1], types.int64)
        task.execute()

        pos, _ = self.nnz_to_pos(q_nnz)
        return sparse.csc_array(
            (vals_store, rows_store, pos), shape=self.shape, dtype=self.dtype
        )

    def todense(self):
        result = cunumeric.zeros(self.shape, dtype=self.dtype)
        result_store = get_store_from_cunumeric_array(result)
        task = ctx.create_task(SparseOpCode.COO_TO_DENSE)
        task.add_output(result_store)
        task.add_input(self._i)
        task.add_input(self._j)
        task.add_input(self._vals)
        task.add_input(result_store)
        task.add_broadcast(result_store)
        task.execute()
        # We have to return the store casted back to a cunumeric array,
        # because if the cunumeric array is small, it will get represented
        # as a future, and the changes won't get propogated back to the
        # container cunumeric array.
        final = store_to_cunumeric_array(result_store)
        return final

    def __matmul__(self, other):
        return self.tocsr() @ other

    def __rmatmul__(self, other):
        return self.tocsr().__rmatmul__(other)

    def __mul__(self, other):
        return self.tocsr() * other

    def dot(self, other):
        return self.tocsr().dot(other)

    def __str__(self):
        return (
            f"{store_to_cunumeric_array(self._i)}, "
            f"{store_to_cunumeric_array(self._j)}, "
            f"{store_to_cunumeric_array(self._vals)}"
        )


coo_matrix = coo_array

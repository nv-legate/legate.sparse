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

from .runtime import ctx, runtime
from .config import rect1, domain_ty, SparseOpCode, SparseProjectionFunctor, _sparse
from .coverage import clone_scipy_arr_kind
from .partition import CompressedImagePartition, MinMaxImagePartition
from .utils import find_last_user_stacklevel

import cunumeric
from cunumeric import eager, deferred

from cffi import FFI
ffi = FFI()

from legate.core import Rect, Store, Array, Point, FutureMap, Future, types, track_provenance
from legate.core.launcher import Broadcast, TaskLauncher
from legate.core.partition import ImagePartition, Tiling, DomainPartition, PreimagePartition
from legate.core.shape import Shape
from legate.core.store import StorePartition
from legate.core.types import ReductionOp

import pyarrow
import numpy
import math
import scipy.sparse
import warnings

# TODO (rohany): Notes and TODOs...
#  1) We'll have to implement our own copy routines, as we can't directly use cunumeric's
#   copy rountines since they aren't going to accept data like Rect<1>'s. Since we have
#   to implement our own anyway, we might as well use it for everything.
#  2) Can look here for ideas about doing parallel transposes (i.e.) CSR->CSC
#     https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf.
#     The "simple" option is to convert to COO, sort by column coordinate (unclear exactly
#     how to do this), and then deduplicate (also unclear how to do this). Actually,
#     we know that there aren't any duplicates because it's already a valid matrix. So
#     it boils down to calling the COO->CSR transformation function (which we can hopefully
#     parallelize) efficiently. My initial thoughts for parallelization of this function is
#     that the data distributions don't line up -- we can distribute the coordinates evenly
#     for the first pass, but then once we find the nnz-per-row we have to distribute by
#     the rows and take an image there to parallelize construction. This could potentially
#     result in alot of data being shuffled around if the row-wise distribution does not
#     align well with the non-zero distribution.
#     Another distributed transposition algorithm: https://arxiv.org/pdf/2012.06012.pdf.
#     This paper also suggests an algorithm to perform specifically CSR->CSC operations
#     with a single task that utilizes a communicator to perform a transpose by assigning
#     the rows and columns to each task, then having each task scan over the assigned rows
#     and bucket them per output rank, and then do an AllToAll communication operation to
#     shuffle the coordinates to the destination nodes.
#     I wonder if such an approach can also be used for COO->CSR etc.

# TODO (rohany): For the class of tensor formats representable by TACO, it makes sense
#  to have a superclass that maintains the indices and values arrays.

# TODO (rohany): It makes sense (for simplicities sake) to allow for lifting from raw
#  scipy sparse matrices into legate types. I can utilize cunumeric to dispatch and handle
#  internal details of getting the stores etc.

# TODO (rohany): Move this into an encapsulated runtime.
dynamic_projection_functor_id = 1
def get_projection_functor_id():
    global dynamic_projection_functor_id
    retval = dynamic_projection_functor_id
    dynamic_projection_functor_id += 1
    return retval + SparseProjectionFunctor.LAST_STATIC_PROJ_FN


def get_store_from_cunumeric_array(arr: cunumeric.ndarray, allow_future = False) -> Store:
    # TODO (rohany): It's unclear how to actually get stores from the __legate_data_interface__
    #  for cunumeric arrays. It seems to depend on whether they are eager/deferred etc. Have to
    #  ask Mike and Wonchan about this.
    # data = arr.__legate_data_interface__["data"]
    # (_, array) = list(data.items())[0]
    # target = array.stores()[1]
    # if isinstance(target, Store):
    #     store = target
    # else:
    #     if isinstance(target, cunumeric.eager.EagerArray):
    #         target = target.to_deferred_array()
    #     store = target.base
    # Because of https://github.com/nv-legate/cunumeric/issues/595, we can't access
    # stores of cunumeric arrays through the `__legate_data_interface__` if the stores
    # happen to have a complex type. So, we'll do something hackier and just reach
    # into the array's thunk and extract the store.
    target = arr._thunk
    if isinstance(target, cunumeric.eager.EagerArray):
        target = target.to_deferred_array()
    assert(isinstance(target, cunumeric.deferred.DeferredArray))
    store = target.base
    assert(isinstance(store, Store))

    # Our implementation can't handle future backed stores when we use this, as we
    # expect to be able to partition things up. If we have a future backed store,
    # create a normal store and issue a copy from the backed store to the new store.
    if store.kind == Future and not allow_future:
        store_copy = ctx.create_store(store.type, shape=store.shape)
        task = ctx.create_task(SparseOpCode.UPCAST_FUTURE_TO_REGION)
        task.add_output(store_copy)
        task.add_input(store)
        task.add_broadcast(store_copy)
        task.add_scalar_arg(store.type.size, types.uint64)
        task.execute()
        store = store_copy
    return store


def unpack_rect1_store(pos):
    out1 = ctx.create_store(int64, shape=pos.shape)
    out2 = ctx.create_store(int64, shape=pos.shape)
    task = ctx.create_task(SparseOpCode.UNZIP_RECT1)
    task.add_output(out1)
    task.add_output(out2)
    task.add_input(pos)
    task.add_alignment(out1, out2)
    task.add_alignment(out2, pos)
    task.execute()
    return out1, out2

def pack_to_rect1_store(lo, hi, output=None):
    if output is None:
        output = ctx.create_store(rect1, shape=(lo.shape[0]))
    task = ctx.create_task(SparseOpCode.ZIP_TO_RECT1)
    task.add_output(output)
    task.add_input(lo)
    task.add_input(hi)
    task.add_alignment(lo, output)
    task.add_alignment(output, hi)
    task.execute()
    return output

def cast_arr(arr, dtype=None):
    if isinstance(arr, Store):
        arr = store_to_cunumeric_array(arr)
    elif not isinstance(arr, cunumeric.ndarray):
        arr = cunumeric.array(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr

def cast_to_store(arr):
    if isinstance(arr, Store):
        return arr
    if isinstance(arr, numpy.ndarray):
        arr = cunumeric.array(arr)
    if isinstance(arr, cunumeric.ndarray):
        return get_store_from_cunumeric_array(arr)
    raise NotImplementedError


# Common types so that this is a bit more modular to change later.
coord_ty = numpy.dtype(numpy.int64)
nnz_ty = numpy.dtype(numpy.uint64)
float64 = numpy.dtype(numpy.float64)
int32 = numpy.dtype(numpy.int32)
int64 = numpy.dtype(numpy.int64)
uint64 = numpy.dtype(numpy.uint64)


class WrappedStore:
    def __init__(self, store: Store):
        self.store = store

        # Set up the __legate_data_interface__. For some reason
        # @property isn't working...
        arrow_type = self.store.type.type
        if isinstance(arrow_type, numpy.dtype):
            arrow_type = pyarrow.from_numpy_dtype(arrow_type)

        # We don't have nullable data for the moment
        # until we support masked arrays
        array = Array(arrow_type, [None, self.store])
        data = dict()
        field = pyarrow.field(
            "cuNumeric Array", arrow_type, nullable=False
        )
        data[field] = array
        self.__legate_data_interface__ = dict()
        self.__legate_data_interface__["data"] = data
        self.__legate_data_interface__["version"] = 1


    # @property
    # def __legate_data_interface__(self):
    #     if self._legate_data is None:
    #         # All of our thunks implement the Legate Store interface
    #         # so we just need to convert our type and stick it in
    #         # a Legate Array
    #         arrow_type = pyarrow.from_numpy_dtype(self.store.dtype)
    #         # We don't have nullable data for the moment
    #         # until we support masked arrays
    #         array = Array(arrow_type, [None, self.store])
    #         self._legate_data = dict()
    #         self._legate_data["version"] = 1
    #         data = dict()
    #         field = pyarrow.field(
    #             "cuNumeric Array", arrow_type, nullable=False
    #         )
    #         data[field] = array
    #         self._legate_data["data"] = data
    #     return self._legate_data


def store_to_cunumeric_array(store : Store):
    # ws = WrappedStore(store)
    # print("WS:", hasattr(ws, "__legate_data_interface__"))
    # print("WS:", hasattr(ws, "_legate_data"))
    return cunumeric.array(WrappedStore(store))


def print_rect_1_store(store):
    out1, out2 = unpack_rect1_store(store)
    print("To rect1:", store_to_cunumeric_array(out1), store_to_cunumeric_array(out2))


def print_store(store):
    print(store_to_cunumeric_array(store))


def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)
    return val, val2


def scan_local_results_and_scale_pos(weights: FutureMap, pos: Store, pos_part: StorePartition, num_procs: int):
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
        scanFutures.append(Future.from_buffer(runtime.legate_runtime.legion_runtime, buf))
    scanFutureMap = FutureMap.from_list(runtime.legate_runtime.legion_context, runtime.legate_runtime.legion_runtime, scanFutures)
    assert(pos_part.partition.color_shape.ndim == 1 and pos_part.partition.color_shape[0] == num_procs)
    task = ctx.create_manual_task(SparseOpCode.SCALE_RECT_1, launch_domain=Rect(hi=(num_procs,)))
    task.add_output(pos_part)
    task.add_input(pos_part)
    task._scalar_future_maps.append(scanFutureMap)
    task.execute()


# DenseSparseBase is a base class for sparse matrices that have a TACO
# format of {Dense, Sparse}. For our purposes, that means CSC and CSR
# matrices.
class DenseSparseBase:
    def __init__(self):
        self._balanced_pos_partition = None

    def balance(self):
        num_procs = runtime.num_procs
        # First, partition the non-zero coordinates into equal pieces.
        crd_tiling = (self.crd.shape[0] + num_procs - 1) // num_procs
        crd_reg = self.crd.storage.region
        crd_part = Tiling(Shape(crd_tiling), Shape(num_procs))
        # Next, construct a preimage of the coordinates into the rows.
        preimage_legate = PreimagePartition(
            self.pos,
            self.crd,
            crd_part,
            ctx.mapper_id,
            range=True,
            # TODO (rohany): Do we want to ask the runtime to infer this for us?
            #  If so, we can short circuit later.
            disjoint=False,
            complete=True,
        )
        preimage = preimage_legate.construct(self.pos.storage.region)
        # If the resulting preimage partition is disjoint (we could be lucky!)
        # then use it as is. Otherwise, we have to manually make it a disjoint
        # partition of the rows.
        if preimage.disjoint:
            self.pos.set_key_partition(preimage_legate)
            self._balanced_row_partition = preimage_legate
            return

        preimage_bounds = {}
        for i in range(num_procs):
            subspace = preimage.index_partition.get_child(Point(i))
            preimage_bounds[i] = subspace.domain.rect
        balanced_row_bounds = {}
        for i in range(num_procs):
            # If we're the first point in the color space, just take the bounds
            # of the first preimage partition as is.
            if i == 0:
                bounds = preimage_bounds[i]
                # Make sure that the bounds include the full pos array.
                if bounds.lo[0] != 0:
                    bounds = Rect(lo=(0,), hi=bounds.hi, exclusive=False)
                balanced_row_bounds[Point(i)] = bounds
            else:
                # Now, our job is to construct a disjoint partition of the rows
                # from the (possibly aliased) input pieces.
                bounds = preimage_bounds[i]
                prev_bounds = balanced_row_bounds[Point(i - 1)]
                lo, hi = bounds.lo, bounds.hi
                # If our bounds intersect with the previous boundary, bump our lower
                # bound up past the previous boundary.
                if bounds.lo[0] <= prev_bounds.hi[0]:
                    lo = Point(prev_bounds.hi[0] + 1)
                # Do the same thing for our upper bound.
                if bounds.hi[0] <= prev_bounds.hi[0]:
                    hi = Point(prev_bounds.hi[0] + 1)
                # Next, make sure that holes in the pos region coloring are filled so that
                # we get a complete partition of the pos region.
                if lo[0] >= prev_bounds.hi[0] + 1:
                    lo = Point(prev_bounds.hi[0] + 1)
                # We've been doing all of these bounds computations with inclusive
                # indexing, so make sure that the Rect constructor doesn't reverse
                # that for us.
                balanced_row_bounds[Point(i)] = Rect(lo=lo, hi=hi, exclusive=False)
        # Do the same normalization for the final color.
        if balanced_row_bounds[Point(num_procs - 1)].hi[0] + 1 != self.pos.shape[0]:
            rect = balanced_row_bounds[Point(num_procs - 1)]
            balanced_row_bounds[Point(num_procs - 1)] = Rect(lo=rect.lo, hi=(self.pos.shape[0] - 1,), exclusive=False)
        # Use our adjusted bounds to construct the resulting partition.
        balanced_legate_part = DomainPartition(
            self.pos.shape,
            Shape(num_procs),
            balanced_row_bounds
        )
        # Actually construct the partition to force it to get cached and analyzed.
        balanced_legion_part = balanced_legate_part.construct(self.pos.storage.region)
        assert(balanced_legion_part.disjoint)
        assert(balanced_legion_part.complete)
        self.pos.set_key_partition(balanced_legate_part)
        self._balanced_pos_partition = balanced_legate_part

    @classmethod
    def make_with_same_nnz_structure(cls, mat, arg, shape=None):
        if shape is None:
            shape = mat.shape
        result = cls(arg, shape=shape, dtype=mat.dtype)
        # Copy over all cached dats structures that depend on the same non-zero structure.
        result._balanced_pos_partition = mat._balanced_pos_partition
        if (result._balanced_pos_partition is not None):
            result.pos.set_key_partition(result._balanced_pos_partition)
        return result


class CompressedBase:
    @classmethod
    def nnz_to_pos_cls(cls, q_nnz: Store):
        cs = cunumeric.array(cunumeric.cumsum(store_to_cunumeric_array(q_nnz)))
        cs_store = get_store_from_cunumeric_array(cs)
        cs_shifted = cunumeric.append(cunumeric.array([0], nnz_ty), cs[:-1])
        cs_shifted_store = get_store_from_cunumeric_array(cs_shifted)
        # Zip the scan result into a rect1 region for the pos.
        pos = ctx.create_store(rect1, shape=(q_nnz.shape[0]), optimize_scalar=False)
        task = ctx.create_task(SparseOpCode.ZIP_TO_RECT1)
        task.add_output(pos)
        task.add_input(cs_shifted_store)
        task.add_input(cs_store)
        task.add_alignment(pos, cs_shifted_store)
        task.add_alignment(cs_shifted_store, cs_store)
        task.execute()
        return pos, int(cs[-1])

    def nnz_to_pos(self, q_nnz: Store):
        return CompressedBase.nnz_to_pos_cls(q_nnz)

    def asformat(self, format, copy=False):
        if format is None or format == self.format:
            if copy:
                raise NotImplementedError
            else:
                return self
        else:
            try:
                convert_method = getattr(self, 'to' + format)
            except AttributeError as e:
                raise ValueError('Format {} is unknown.'.format(format)) from e

            # Forward the copy kwarg, if it's accepted.
            try:
                return convert_method(copy=copy)
            except TypeError:
                return convert_method()

    # The implementation of sum is mostly lifted from scipy.sparse.
    def sum(self, axis=None, dtype=None, out=None):
        """
        Sum the matrix elements over a given axis.
        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the sum of all the matrix elements, returning a scalar
            (i.e., `axis` = `None`).
        dtype : dtype, optional
            The type of the returned matrix and of the accumulator in which
            the elements are summed.  The dtype of `a` is used by default
            unless `a` has an integer dtype of less precision than the default
            platform integer.  In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.
            .. versionadded:: 0.18.0
        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.
            .. versionadded:: 0.18.0
        Returns
        -------
        sum_along_axis : np.matrix
            A matrix with the same shape as `self`, with the specified
            axis removed.
        See Also
        --------
        numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
        """

        # We use multiplication by a matrix of ones to achieve this.
        # For some sparse matrix formats more efficient methods are
        # possible -- these should override this function.
        m, n = self.shape

        # Mimic numpy's casting.
        res_dtype = self.dtype

        if axis is None:
            return self.data.sum(dtype=res_dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            # sum over columns
            ret = self.__rmatmul__(cunumeric.ones((1, m), dtype=res_dtype))
        else:
            # sum over rows
            ret = self @ cunumeric.ones((n, 1), dtype=res_dtype)

        if out is not None and out.shape != ret.shape:
            raise ValueError("dimensions do not match")

        return ret.sum(axis=axis, dtype=dtype, out=out)


@clone_scipy_arr_kind(scipy.sparse.csr_array)
class csr_array(CompressedBase, DenseSparseBase):
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        if copy:
            raise NotImplementedError

        if dtype is not None:
            assert dtype == numpy.float64
            dtype = numpy.dtype(dtype)
        else:
            dtype = float64
        self.dtype = dtype
        self.ndim = 2
        super().__init__()

        if isinstance(arg, numpy.ndarray):
            arg = cunumeric.array(arg)
        if isinstance(arg, cunumeric.ndarray):
            assert(arg.ndim == 2)
            self.shape = arg.shape
            # Conversion from dense arrays is pretty easy. We'll do a row-wise distribution
            # and use a two-pass algorithm that first counts the non-zeros per row and then
            # fills them in.
            arg_store = get_store_from_cunumeric_array(arg)
            q_nnz = ctx.create_store(nnz_ty, shape=(arg.shape[0]))
            task = ctx.create_task(SparseOpCode.DENSE_TO_CSR_NNZ)
            promoted_q_nnz = q_nnz.promote(1, self.shape[1])
            task.add_output(promoted_q_nnz)
            task.add_input(arg_store)
            task.add_broadcast(promoted_q_nnz, 1)
            task.add_alignment(promoted_q_nnz, arg_store)
            task.execute()

            # Assemble the output CSR array using the non-zeros per row.
            self.pos, nnz = self.nnz_to_pos(q_nnz)
            self.crd = ctx.create_store(coord_ty, shape=(nnz))
            self.vals = ctx.create_store(self.dtype, shape=(nnz))
            task = ctx.create_task(SparseOpCode.DENSE_TO_CSR)
            promoted_pos = self.pos.promote(1, self.shape[1])
            task.add_output(promoted_pos)
            task.add_output(self.crd)
            task.add_output(self.vals)
            task.add_input(arg_store)
            task.add_input(promoted_pos)
            # Partition the rows.
            task.add_broadcast(promoted_pos, 1)
            task.add_alignment(promoted_pos, arg_store)
            task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
            task.add_alignment(self.crd, self.vals)
            task.execute()
        elif isinstance(arg, scipy.sparse.csr_array) or isinstance(arg, scipy.sparse.csr_matrix):
            self.vals = get_store_from_cunumeric_array(cunumeric.array(arg.data, dtype=cunumeric.float64))
            self.crd = get_store_from_cunumeric_array(cunumeric.array(arg.indices, dtype=coord_ty))
            # Cast the indptr array in the scipy.csr_matrix into our Rect<1> based pos array.
            indptr = cunumeric.array(arg.indptr, dtype=cunumeric.int64)
            los = indptr[:-1]
            his = indptr[1:]
            self.pos = pack_to_rect1_store(get_store_from_cunumeric_array(los), get_store_from_cunumeric_array(his))
            self.shape = arg.shape
        elif isinstance(arg, tuple):
            if len(arg) == 2:
                # If the tuple has two arguments, then it must be of the form
                # (data, (row, col)), so just pass it to the COO constructor
                # and transform it into a CSR matrix.
                data, (row, col) = arg
                result = coo_array((data, (row, col)), shape=shape, dtype=dtype).tocsr()
                self.pos = result.pos
                self.crd = result.crd
                self.vals = result.vals
            elif len(arg) == 3:
                (data, indices, indptr) = arg
                if isinstance(indptr, cunumeric.ndarray):
                    assert indptr.shape[0] == shape[0] + 1
                    los = indptr[:-1]
                    his = indptr[1:]
                    self.pos = pack_to_rect1_store(get_store_from_cunumeric_array(los), get_store_from_cunumeric_array(his))
                else:
                    assert(isinstance(indptr, Store))
                    self.pos = indptr
                # TODO (rohany): We need to ensure that the input types here are correct (i.e.
                #  the crd array is indeed an int64).
                self.crd = cast_to_store(indices)
                self.vals = cast_to_store(data)
            else:
                raise AssertionError
            assert shape is not None
            self.shape = shape
        else:
            raise NotImplementedError

        # Manually adjust the key partition of the pos array to distribute the
        # sparse matrix by the rows across all processors. This makes the solver
        # understand that not everything should be replicated just because the
        # matrix construction is not parallelized.
        tile_size = (self.pos.shape[0] + runtime.num_procs - 1) // runtime.num_procs
        pos_part = self.pos.partition_by_tiling(Shape(tile_size))
        self.pos.set_key_partition(pos_part.partition)
        # Override the volume calculation on pos regions, since repartitioning
        # the pos region will most definitely cause moving around the crd and
        # vals arrays. Importantly, we need to compute the volume here and then
        # return it in the closure. If we compute it inside the closure, we end
        # up creating a reference cycle between self.pos._storage and self,
        # leading us to never collect self, leaking futures stored here.
        volume = self.crd.comm_volume() + self.vals.comm_volume() + self.pos.extents.volume()
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
        q = cunumeric.zeros((M,), dtype=coord_ty)
        pos, _ = CompressedBase.nnz_to_pos_cls(get_store_from_cunumeric_array(q))
        crd = ctx.create_store(coord_ty, (0, ), optimize_scalar=False)
        vals = ctx.create_store(dtype, (0, ), optimize_scalar=False)
        return cls((vals, crd, pos), shape=shape, dtype=dtype)

    @property
    def nnz(self):
        return self.vals.shape[0]

    def copy(self):
        # copy = ctx.create_copy()
        pos = ctx.create_store(rect1, shape=self.pos.shape)
        # Issue a copy from the old pos to the new pos. We can't do this
        # with cunumeric because cunumeric doesn't support the Rect<1> type.
        copy = ctx.create_copy()
        copy.add_input(self.pos)
        copy.add_output(pos)
        copy.execute()
        crd = cunumeric.array(store_to_cunumeric_array(self.crd))
        vals = cunumeric.array(store_to_cunumeric_array(self.vals))
        return csr_array.make_with_same_nnz_structure(self, (vals, crd, pos))

    def conj(self, copy=True):
        if copy:
            raise NotImplementedError
        if self.dtype != float64:
            raise NotImplementedError
        return self

    @track_provenance(runtime.legate_context, nested=True)
    def tropical_spmv(self, other, out=None):
        if not isinstance(other, cunumeric.ndarray):
            other = cunumeric.array(other)
        assert(len(other.shape) == 2)
        # TODO (rohany): Add checks around dtypes.
        assert(self.shape[1] == other.shape[0])
        if out is None:
            output = ctx.create_store(coord_ty, shape=(self.shape[0], other.shape[1]))
        else:
            assert isinstance(out, cunumeric.ndarray)
            assert out.shape[0] == self.shape[0] and out.shape[1] == other.shape[1]
            output = get_store_from_cunumeric_array(out)
        other_store = get_store_from_cunumeric_array(other)

        # An auto-parallelized version of the kernel.
        promoted_pos = self.pos.promote(1, dim_size=output.shape[1])
        task = ctx.create_task(SparseOpCode.CSR_SPMV_ROW_SPLIT_TROPICAL_SEMIRING)
        task.add_output(output)
        task.add_input(promoted_pos)
        task.add_input(self.crd)
        task.add_input(other_store)
        # Add partitioning. We make sure the field dimensions aren't partitioned.
        task.add_broadcast(output, 1)
        task.add_alignment(output, promoted_pos)
        task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
        task.add_alignment(self.crd, self.vals)
        # In order to do the image from the coordinates into the corresponding rows of other,
        # we have to apply an AffineProjection from the coordinates to cast them up to reference
        # rows of other, rather than single points. The API for this is a bit restrictive, so we
        # have to pass a staged MinMaxImagePartition functor through to the image constraint.
        def partFunc(*args, **kwargs):
            return MinMaxImagePartition(*args, proj_dims=[0], **kwargs)
        task.add_image_constraint(self.crd, other_store, range=False, disjoint=False, complete=False, functor=partFunc)
        task.execute()
        return store_to_cunumeric_array(output)

    def to_scipy_sparse_csr(self):
        import scipy.sparse
        los, _ = unpack_rect1_store(self.pos)
        los = store_to_cunumeric_array(los)
        indptr = cunumeric.append(los, [self.crd.shape[0]])
        return scipy.sparse.csr_array((store_to_cunumeric_array(self.vals), store_to_cunumeric_array(self.crd), indptr), shape=self.shape, dtype=self.dtype)

    def dot(self, other, out=None):
        if out is not None:
            assert isinstance(out, cunumeric.ndarray)
        if isinstance(other, numpy.ndarray):
            other = cunumeric.array(other)
        # TODO (rohany): Rewrite this to dispatch to dot, and have dot handle matmul,spmv
        #  as the different cases of GEMM.
        # We're doing a SpMV (or maybe an SpMSpV)?
        if len(other.shape) == 1 or (len(other.shape) == 2 and other.shape[1] == 1):
            # We don't have an SpMSpV implementation, so just convert the input
            # sparse matrix into a dense vector first.
            other_originally_sparse = False
            if is_sparse_matrix(other):
                other = other.todense()
                other_originally_sparse = True
            if not isinstance(other, cunumeric.ndarray):
                other = cunumeric.array(other)
            assert(self.shape[1] == other.shape[0])
            other_originally_2d = False
            if len(other.shape) == 2 and other.shape[1] == 1:
                other = other.squeeze(1)
                other_originally_2d = True
            # Ensure that we write into the output when possible.
            if out is not None:
                if other_originally_2d:
                    assert out.shape == (self.shape[0], 1)
                    out = out.squeeze(1)
                else:
                    assert out.shape == (self.shape[0],)
                output = get_store_from_cunumeric_array(out)
            else:
                output = ctx.create_store(self.dtype, shape=self.pos.shape)
            other_store = get_store_from_cunumeric_array(other)
            # An auto-parallelized version of the kernel.
            task = ctx.create_task(SparseOpCode.CSR_SPMV_ROW_SPLIT)
            task.add_output(output)
            task.add_input(self.pos)
            task.add_input(self.crd)
            task.add_input(self.vals)
            task.add_input(other_store)
            task.add_alignment(output, self.pos)
            task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
            task.add_alignment(self.crd, self.vals)
            # TODO (rohany): Both adding an image constraint explicitly and an alignment
            #  constraint between vals and crd works now. Adding the image is explicit though,
            #  while adding the alignment is more in line with the DISTAL way of doing things.
            # task.add_image_constraint(self.pos, self.vals, range=True)
            # An important optimization is to use an image operation to request only
            # the necessary pieces of data from the x vector in y = Ax. We don't make
            # an attempt to use a sparse instance, so we allocate the full vector x in
            # each task, but by using the sparse instance we ensure that only the
            # necessary pieces of data are communicated. In many common sparse matrix
            # patterns, this can result in an asymptotic decrease in the amount of
            # communication. This is a bit of a hassle (requires some data copying and
            # reorganization to get the coordinates correct) when the input store has
            # been transformed, so we'll just avoid this case for now.
            if other_store.transformed:
                # So we don't get blind-sided by this again, issue a warning
                level = find_last_user_stacklevel()
                warnings.warn(
                    "SpMV not using image optimization due to reshaped x in y=Ax."
                    "This will cause performance and memory usage to suffer as you scale.",
                    category=RuntimeWarning,
                    stacklevel=level
                )
                task.add_broadcast(other_store)
            else:
                # The image of the selected coordinates into other vector is not
                # complete or disjoint.
                task.add_image_constraint(self.crd, other_store, range=False, disjoint=False, complete=False, functor=MinMaxImagePartition)
            task.execute()
            output = store_to_cunumeric_array(output)
            if other_originally_2d:
                output = output.reshape((-1, 1))
            if other_originally_sparse:
                output = csr_array(output, shape=output.shape, dtype=output.dtype)
            return output
        # TODO (rohany): Let's just worry about handling CSR output for now. It looks like
        #  TACO can do CSC outputs, but let's think about that in the future.
        # TODO (rohany): See how we can leverage the runtime to help us do the communication
        #  to construct the CSC array in parallel if we partition i and j.
        elif isinstance(other, csc_array):
            if out is not None:
                raise ValueError("Cannot specify out for CSRxCSC matmul.")
            assert(self.shape[1] == other.shape[0])
            # Here, we want to enable partitioning the i and j dimensions of A(i, j) = B(i, k) * C(k, j).
            # To do this, we'll first logically organize our processors into a 2-D grid, and partition
            # the pos region of B along the i dimension of the processor grid, replicated onto the j
            # dimension, and partition C along the j dimension, replicated onto the i dimension.
            num_procs = runtime.num_procs
            grid = Shape(factor_int(num_procs))

            rows_proj_fn = ctx.get_projection_id(get_projection_functor_id())
            cols_proj_fn = ctx.get_projection_id(get_projection_functor_id())
            # TODO (rohany): Cache these in a sparse runtime.
            _sparse.register_legate_sparse_1d_to_2d_functor(rows_proj_fn, grid[0], grid[1], True) # True is for the rows.
            _sparse.register_legate_sparse_1d_to_2d_functor(cols_proj_fn, grid[0], grid[1], False) # False is for the cols.

            # To create a tiling on a 2-D color space of a 1-D region, we first promote the region
            # into a 2-D region, and then apply a tiling to it with the correct color shape. In
            # particular, we want to broadcast the colorings over the i dimension across the j
            # dimension of the grid, which the promotion does for us.
            my_promoted_pos = self.pos.promote(1)
            my_pos_tiling = Tiling(Shape(((self.pos.shape[0] + grid[0] - 1) // grid[0], 1)), grid)
            my_pos_partition = my_promoted_pos.partition(my_pos_tiling)
            other_promoted_pos = other.pos.promote(0)
            other_pos_tiling = (other.pos.shape[0] + grid[1] - 1) // grid[1]
            other_pos_partition = other_promoted_pos.partition(Tiling(Shape((1, other_pos_tiling)), grid))

            # TODO (rohany): There's a really weird interaction here that needs help from wonchan.
            #  If I'm deriving a partition from another, I need the partition pos all of the transformations
            #  that are applied to it. However, the normal partition.partition is the partition before
            #  any transformations. I'm not sure what's the best way to untangle this.
            my_pos_image = ImagePartition(self.pos, my_pos_partition._storage_partition._partition, ctx.mapper_id, range=True)
            other_pos_image = ImagePartition(other.pos, other_pos_partition._storage_partition._partition, ctx.mapper_id, range=True)

            # First, we launch a task that tiles the output matrix and creates a local
            # csr matrix result for each tile.
            task = ctx.create_manual_task(SparseOpCode.SPGEMM_CSR_CSR_CSC_LOCAL_TILES, launch_domain=Rect(hi=(num_procs,)))
            # Note that while we colored the region in a 2-D color space, legate does
            # some smart things to essentially reduce the coloring to a 1-D coloring
            # and uses projection functors to send the right subregions to the right tasks.
            # So, we use a special functor that we register that understands the logic for
            # this particular type of partitioning.
            task.add_input(my_pos_partition, proj=rows_proj_fn)
            task.add_input(self.crd.partition(my_pos_image), proj=rows_proj_fn)
            task.add_input(self.vals.partition(my_pos_image), proj=rows_proj_fn)
            task.add_input(other_pos_partition, proj=cols_proj_fn)
            task.add_input(other.crd.partition(other_pos_image), proj=cols_proj_fn)
            task.add_input(other.vals.partition(other_pos_image), proj=cols_proj_fn)
            # Finally, create some unbound stores that will represent the logical components
            # of each sub-csr matrix that are created by the launched tasks.
            pos = ctx.create_store(rect1, ndim=1)
            crd = ctx.create_store(coord_ty, ndim=1)
            vals = ctx.create_store(self.dtype, ndim=1)
            task.add_output(pos)
            task.add_output(crd)
            task.add_output(vals)
            task.add_scalar_arg(other.shape[0], types.int64)
            task.execute()

            # Due to recent changes in the legate core, we don't get a future map
            # back if the size of the launch is 1, meaning that we won't have key
            # partitions the launch over. Luckily if the launch domain has size
            # 1 then all of the data structures we have created are a valid
            # CSR array, so we can return early.
            if num_procs == 1:
                return csr_array((vals, crd, pos), shape=(self.shape[0], other.shape[1]))


            # After the local execution, we need to start building a global CSR array.
            # First, we offset all of the local pos pieces with indices in the global
            # crd and vals arrays. Since each local pos region is already valid, we
            # just need to perform a scan over the final size of each local crd region
            # and offset each pos region by the result.
            pos_part = pos.partition(pos.get_key_partition())
            scan_local_results_and_scale_pos(crd.get_key_partition()._weights, pos, pos_part, num_procs)

            # Now it gets trickier. We have to massage the local tiles of csr matrices into
            # one global CSR matrix. To do this, we will consider each row of the processor
            # grid independently. Within each row of the grid, each tile contains a CSR matrix
            # over the same set of rows of the output, but different columns. So, we will
            # distribute those rows across the processors in the current row. We construct
            # a store that describes how the communication should occur between the processors
            # in the current row. Each processor records the pieces of their local pos regions
            # that correspond to the rows assigned to each other processor in the grid. We can
            # then apply image operations to this store to collect the slices of the local results
            # that should be sent to each processor. Precisely, we create a 3-D region that describes
            # for each processor in the processor grid, the range of entries to be sent to the other
            # processors in that row.
            partitioner_store = ctx.create_store(rect1, shape=Shape((grid[0], grid[1], grid[1])))
            # TODO (rohany): This operation _should_ be possible with tiling operations, but I can't
            #  get the API to do what I want -- the problem appears to be trying to match a 2-D coloring
            #  onto a 3-D region.
            partitioner_store_coloring = {}
            for i in range(grid[0]):
                for j in range(grid[1]):
                    rect = Rect(lo=(i, j, 0), hi=(i, j, grid[1] - 1), dim=3, exclusive=False)
                    partitioner_store_coloring[Point((i, j))] = rect
            partitioner_store_partition = partitioner_store.partition(DomainPartition(partitioner_store.shape, grid, partitioner_store_coloring))
            task = ctx.create_manual_task(SparseOpCode.SPGEMM_CSR_CSR_CSC_COMM_COMPUTE, launch_domain=Rect(hi=(num_procs,)))
            promote_1d_to_2d = ctx.get_projection_id(SparseProjectionFunctor.PROMOTE_1D_TO_2D)
            task.add_output(partitioner_store_partition, proj=promote_1d_to_2d)
            task.add_input(my_pos_partition, proj=rows_proj_fn)
            task.add_input(pos_part)
            # Scalar arguments must use the legate core type system.
            task.add_scalar_arg(grid[0], types.int32)
            task.add_scalar_arg(grid[1], types.int32)
            task.execute()

            # We now create a transposed partition of the store to get sets of ranges assigned
            # to each processor. This partition selects for each processor the ranges created
            # for that processor by all of the other processor.
            transposed_partitioner_store_coloring = {}
            for i in range(grid[0]):
                for j in range(grid[1]):
                    rect = Rect(lo=(i, 0, j), hi=(i, grid[1] - 1, j), dim=3, exclusive=False)
                    transposed_partitioner_store_coloring[Point((i, j))] = rect
            transposed_partition = partitioner_store.partition(DomainPartition(partitioner_store.shape, grid, transposed_partitioner_store_coloring))
            # Cascade images down to the global pos, crd and vals regions.
            global_pos_partition = pos.partition(ImagePartition(partitioner_store, transposed_partition.partition, ctx.mapper_id, range=True))
            global_crd_partition = crd.partition(ImagePartition(pos, global_pos_partition.partition, ctx.mapper_id, range=True))
            global_vals_partition = vals.partition(ImagePartition(pos, global_pos_partition.partition, ctx.mapper_id, range=True))
            # This next task utilizes the pieces computed by the transposed partition
            # and gathers them into contiguous pieces to form the result csr matrix.
            task = ctx.create_manual_task(SparseOpCode.SPGEMM_CSR_CSR_CSC_SHUFFLE, launch_domain=Rect(hi=(num_procs,)))
            task.add_input(global_pos_partition, proj=promote_1d_to_2d)
            task.add_input(global_crd_partition, proj=promote_1d_to_2d)
            task.add_input(global_vals_partition, proj=promote_1d_to_2d)
            # We could compute this up front with a 2 pass algorithm, but it seems expedient
            # to just use Legion's output region support for now.
            final_pos = ctx.create_store(rect1, ndim=1)
            final_crd = ctx.create_store(coord_ty, ndim=1)
            final_vals = ctx.create_store(self.dtype, ndim=1)
            task.add_output(final_pos)
            task.add_output(final_crd)
            task.add_output(final_vals)
            task.execute()

            # At this point, we have an almost valid csr array. The only thing missing
            # is that again each pos array created by the grouping task is not globally
            # offset. We adjust this with one more scan over all of the output sizes.
            weights = final_crd.get_key_partition()._weights
            scan_local_results_and_scale_pos(weights, final_pos, final_pos.partition(final_pos.get_key_partition()), num_procs)
            return csr_array((final_vals, final_crd, final_pos), shape=Shape((self.shape[0], other.shape[1])))
        elif isinstance(other, csr_array):
            if out is not None:
                raise ValueError("Cannot provide out for CSRxCSC matmul.")
            assert(self.shape[1] == other.shape[0])
            # Due to limitations in cuSPARSE, we cannot use a uniform task implementation
            # for CSRxCSRxCSR SpGEMM across CPUs, OMPs and GPUs. The GPU implementation will
            # create a set of local CSR matrices that will be aggregated into a global CSR.
            if runtime.num_gpus > 0:
                pos = ctx.create_store(rect1, shape=self.shape[0])
                crd = ctx.create_store(coord_ty, ndim=1)
                vals = ctx.create_store(float64, ndim=1)
                num_procs = runtime.num_procs
                tile_shape = (self.shape[0] + num_procs - 1) // num_procs
                tiling = Tiling(Shape(tile_shape), Shape(num_procs))
                task = ctx.create_manual_task(SparseOpCode.SPGEMM_CSR_CSR_CSR_GPU, launch_domain=Rect(hi=(num_procs,)))
                pos_part = pos.partition(tiling)
                task.add_output(pos_part)
                task.add_output(crd)
                task.add_output(vals)
                my_pos_part = self.pos.partition(tiling)
                task.add_input(my_pos_part)
                image = CompressedImagePartition(self.pos, my_pos_part.partition, ctx.mapper_id, range=True)
                crd_part = self.crd.partition(image)
                task.add_input(self.crd.partition(image))
                task.add_input(self.vals.partition(image))
                # The C matrix is unfortunately replicated in this algorithm. However,
                # we can make the world a little better for us by gathering only the
                # rows of C that are referenced by each partition using Image operations.
                task.add_input(other.pos)
                crd_image = MinMaxImagePartition(self.crd, crd_part.partition, ctx.mapper_id, range=False)
                other_pos_part = other.pos.partition(crd_image)
                task.add_input(other.crd.partition(ImagePartition(other.pos, other_pos_part.partition, ctx.mapper_id, range=True)))
                task.add_input(other.vals.partition(ImagePartition(other.pos, other_pos_part.partition, ctx.mapper_id, range=True)))
                task.add_scalar_arg(other.shape[1], types.uint64)
                task.execute()
                # Build the global CSR array by performing a scan across the individual CSR results. Due
                # to a recent change in legate.core that doesn't do future map reductions on launches of
                # size one, we have to gaurd this operation as crd might not have a key partition.
                if num_procs > 1:
                    scan_local_results_and_scale_pos(crd.get_key_partition()._weights, pos, pos_part, num_procs)
                return csr_array((vals, crd, pos), shape=(self.shape[0], other.shape[1]))
            else:
                # Create the query result.
                q_nnz = ctx.create_store(nnz_ty, shape=self.shape[0])
                task = ctx.create_task(SparseOpCode.SPGEMM_CSR_CSR_CSR_NNZ)
                task.add_output(q_nnz)
                self._add_to_task(task, vals=False)
                other._add_to_task(task, vals=False)
                task.add_alignment(q_nnz, self.pos)
                task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
                # We'll only ask for the rows used by each partition by following an image
                # of pos through crd. We'll then use that partition to declare the pieces of
                # crd and vals of other that are needed by the matmul. The resulting image
                # of coordinates into rows of other is not necessarily complete or disjoint.
                task.add_image_constraint(self.crd, other.pos, range=False, disjoint=False, complete=False, functor=MinMaxImagePartition)
                # Since the target partition of pos is likely not contiguous, we can't
                # use the CompressedImagePartition functor and have to fall back to a
                # standard functor. Since the source partition of the rows is not
                # complete or disjoint, the images into crd and vals are not disjoint either.
                task.add_image_constraint(other.pos, other.crd, range=True, disjoint=False, complete=False)
                task.add_image_constraint(other.pos, other.vals, range=True, disjoint=False, complete=False)
                task.add_scalar_arg(other.shape[1], types.uint64)
                task.execute()

                pos, nnz = self.nnz_to_pos(q_nnz)
                crd = ctx.create_store(coord_ty, shape=(nnz))
                vals = ctx.create_store(self.dtype, shape=(nnz))

                task = ctx.create_task(SparseOpCode.SPGEMM_CSR_CSR_CSR)
                task.add_output(pos)
                task.add_output(crd)
                task.add_output(vals)
                self._add_to_task(task)
                other._add_to_task(task)
                task.add_alignment(self.pos, pos)
                task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
                task.add_alignment(self.crd, self.vals)
                task.add_image_constraint(pos, crd, range=True, functor=CompressedImagePartition)
                task.add_alignment(crd, vals)
                task.add_broadcast(other.pos)
                task.add_broadcast(other.crd)
                task.add_broadcast(other.vals)
                # Add pos to the inputs as well so that we get READ_WRITE privileges.
                task.add_input(pos)
                task.add_scalar_arg(other.shape[1], types.uint64)
                task.execute()
                return csr_array((vals, crd, pos), shape=Shape((self.shape[0], other.shape[1])))
        elif isinstance(other, cunumeric.ndarray):
            assert(self.shape[1] == other.shape[0])
            # We can dispatch to SpMM here. There are different implementations that one can
            # go for, like the 2-D distribution, or the 1-D non-zero balanced distribution.
            other_store = get_store_from_cunumeric_array(other)
            if out is not None:
                assert out.shape == (self.shape[0], other.shape[1])
                output_store = get_store_from_cunumeric_array(out)
            else:
                output_store = ctx.create_store(self.dtype, shape=Shape((self.shape[0], other.shape[1])))
            # TODO (rohany): In an initial implementation, we'll partition things only by the
            #  `i` dimension of the computation. However, the leaf kernel as written allows for
            #  both the `i` and `j` dimensions of the computation to be partitioned. I'm avoiding
            #  doing the multi-dimensional parallelism here because I'm not sure how to express it
            #  within the solver's partitioning constraints. This is the first example of needing
            #  affine projections the partitioning -- we need to partition the output into tiles,
            #  and project the first dimension onto self.pos, and project the second dimension
            #  onto other.
            promoted_pos = self.pos.promote(1, output_store.shape[1])
            task = ctx.create_task(SparseOpCode.SPMM_CSR_DENSE)
            task.add_output(output_store)
            task.add_input(promoted_pos)
            task.add_input(self.crd)
            task.add_input(self.vals)
            task.add_input(other_store)
            # Partitioning.
            task.add_broadcast(promoted_pos, 1)
            task.add_alignment(output_store, promoted_pos)
            task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
            task.add_image_constraint(promoted_pos, self.vals, range=True, functor=CompressedImagePartition)
            # In order to do the image from the coordinates into the corresponding rows of other,
            # we have to apply an AffineProjection from the coordinates to cast them up to reference
            # rows of other, rather than single points. The API for this is a bit restrictive, so we
            # have to pass a staged MinMaxImagePartition functor through to the image constraint.
            def partFunc(*args, **kwargs):
                return MinMaxImagePartition(*args, proj_dims=[0], **kwargs)
            task.add_image_constraint(self.crd, other_store, range=False, disjoint=False, complete=False, functor=partFunc)
            task.add_scalar_arg(self.shape[1], types.int64)
            task.execute()
            return store_to_cunumeric_array(output_store)
        else:
            raise NotImplementedError

    def matvec(self, other):
        return self @ other

    def tocsr(self, copy=False):
        if copy:
            raise NotImplementedError
        return self

    def tocsc(self, copy=False):
        return self.tocoo().tocsc()

    def tocoo(self, copy=False):
        if copy:
            raise NotImplementedError
        # The conversion to COO is pretty straightforward. The crd and values arrays are already
        # set up for COO, we just need to expand the pos array into coordinates.
        rows_expanded = ctx.create_store(coord_ty, shape=self.crd.shape)
        task = ctx.create_task(SparseOpCode.EXPAND_POS_TO_COORDINATES)
        task.add_input(self.pos)
        task.add_output(rows_expanded)
        task.add_image_constraint(self.pos, rows_expanded, range=True, functor=CompressedImagePartition)
        task.execute()
        return coo_array((self.vals, (rows_expanded, self.crd)), shape=self.shape, dtype=self.dtype)

    def transpose(self, copy=False):
        if copy:
            raise NotImplementedError
        return csc_array.make_with_same_nnz_structure(self, (self.vals, self.crd, self.pos), shape=Shape((self.shape[1], self.shape[0])))

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cunumeric.empty(0, dtype=self.dtype)
        output = ctx.create_store(self.dtype, shape=(min(rows + min(k, 0), cols - max(k, 0))))
        # TODO (rohany): Just to get things working for the AMG example, we'll just
        #  support k == 0.
        if k != 0:
            raise NotImplementedError
        task = ctx.create_task(SparseOpCode.CSR_DIAGONAL)
        task.add_output(output)
        self._add_to_task(task)
        task.add_alignment(output, self.pos)
        task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
        task.add_alignment(self.crd, self.vals)
        task.execute()
        return store_to_cunumeric_array(output)

    T = property(transpose)

    def todense(self, order=None, out=None):
        if order is not None:
            raise NotImplementedError
        if out is not None:
            out = cunumeric.array(out)
            out = get_store_from_cunumeric_array(out)
        elif out is None:
            out = ctx.create_store(self.dtype, shape=self.shape)
        # TODO (rohany): We'll do a row-based distribution for now, but a non-zero based
        #  distribution of this operation with overlapping output regions being reduced
        #  into also seems possible.
        promoted_pos = self.pos.promote(1, self.shape[1])
        task = ctx.create_task(SparseOpCode.CSR_TO_DENSE)
        task.add_output(out)
        task.add_input(promoted_pos)
        task.add_input(self.crd)
        task.add_input(self.vals)
        # We aren't partitioning the columns.
        task.add_broadcast(out, 1)
        task.add_alignment(out, promoted_pos)
        task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
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
        # We'll start out with a row-based distribution of the CSR matrix.
        # In the future, we can look into doing a non-zero based distribution
        # of the computation, as there aren't really any downsides of doing it
        # versus a row-based distribution. The problem with both is that they
        # require replicating the D matrix onto all processors. Doing a partitioning
        # strategy that partitions up the j dimension of the computation is harder.
        # This operation is also non-zero structure preserving, so we'll just write
        # into an output array of values and share the pos and crd arrays.
        # TODO (rohany): An option is also partitioning up the `k` dimension of the
        #  computation (allows for partitioning C twice and D once), but requires
        #  reducing into the output.
        if isinstance(C, numpy.ndarray):
            C = cunumeric.array(C)
        if isinstance(D, numpy.ndarray):
            D = cunumeric.array(D)
        assert(len(C.shape) == 2 and len(D.shape) == 2)
        assert(self.shape[0] == C.shape[0] and self.shape[1] == D.shape[1] and C.shape[1] == D.shape[0])
        C_store = get_store_from_cunumeric_array(C)
        D_store = get_store_from_cunumeric_array(D)
        result_vals = ctx.create_store(self.dtype, shape=self.vals.shape)
        task = ctx.create_task(SparseOpCode.CSR_SDDMM)
        task.add_output(result_vals)
        promoted_pos = self.pos.promote(1, C_store.shape[1])
        task.add_input(promoted_pos)
        task.add_input(self.crd)
        task.add_input(self.vals)
        task.add_input(C_store)
        task.add_input(D_store)
        # Partition the rows of the sparse matrix and C.
        task.add_broadcast(promoted_pos, 1)
        task.add_alignment(promoted_pos, C_store)
        task.add_image_constraint(promoted_pos, result_vals, range=True, functor=CompressedImagePartition)
        task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
        task.add_image_constraint(promoted_pos, self.vals, range=True, functor=CompressedImagePartition)
        task.add_broadcast(D_store)
        task.execute()
        return csr_array.make_with_same_nnz_structure(self, (result_vals, self.crd, self.pos))

    def multiply(self, other):
        return self * other

    # other / mat is defined to be the element-wise division of a other by mat over
    # the non-zero coordinates of mat. For now, we restrict this operation to be on
    # scalars only.
    def __rtruediv__(self, other):
        if not cunumeric.isscalar(other):
            raise NotImplementedError
        vals_arr = store_to_cunumeric_array(self.vals)
        new_vals = other / vals_arr
        return csr_array.make_with_same_nnz_structure(self, (get_store_from_cunumeric_array(new_vals), self.crd, self.pos))

    # This is an element-wise operation now.
    def __mul__(self, other):
        if isinstance(other, csr_array):
            assert(self.shape == other.shape)
            # TODO (rohany): This common pattern could probably be deduplicated somehow.
            # Create the assemble query result array.
            q_nnz = ctx.create_store(nnz_ty, shape=(self.shape[0]))
            task = ctx.create_task(SparseOpCode.ELEM_MULT_CSR_CSR_NNZ)
            task.add_output(q_nnz)
            self._add_to_task(task, input=True, vals=False)
            other._add_to_task(task, input=True, vals=False)
            task.add_scalar_arg(self.shape[1], types.int64)
            task.add_alignment(q_nnz, self.pos)
            task.add_alignment(self.pos, other.pos)
            task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
            task.add_image_constraint(other.pos, other.crd, range=True, functor=CompressedImagePartition)
            task.execute()

            pos, nnz = self.nnz_to_pos(q_nnz)
            crd = ctx.create_store(coord_ty, shape=(nnz))
            vals = ctx.create_store(self.dtype, shape=(nnz))

            task = ctx.create_task(SparseOpCode.ELEM_MULT_CSR_CSR)
            task.add_output(pos)
            task.add_output(crd)
            task.add_output(vals)
            self._add_to_task(task)
            other._add_to_task(task)
            task.add_scalar_arg(self.shape[1], types.int64)
            task.add_alignment(pos, self.pos)
            task.add_alignment(self.pos, other.pos)
            task.add_image_constraint(pos, crd, range=True, functor=CompressedImagePartition)
            task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
            task.add_image_constraint(other.pos, other.crd, range=True, functor=CompressedImagePartition)
            task.add_alignment(crd, vals)
            task.add_alignment(self.crd, self.vals)
            task.add_alignment(other.crd, other.vals)
            # Make sure that pos is in READ_WRITE mode.
            task.add_input(pos)
            task.execute()
            return csr_array((vals, crd, pos), shape=self.shape)
        # If we got a sparse matrix type that we know about, try and convert it
        # to csr to complete the addition.
        if is_sparse_matrix(other):
            return self * other.tocsr()

        # At this point, we have objects that we might not understand. Case to try
        # and figure out what they are.
        if isinstance(other, numpy.ndarray):
            other = cunumeric.ndarray(other)
        if cunumeric.ndim(other) == 0:
            # If we have a scalar, then do an element-wise multiply on the values array.
            new_vals = store_to_cunumeric_array(self.vals) * other
            return csr_array.make_with_same_nnz_structure(self, (get_store_from_cunumeric_array(new_vals), self.crd, self.pos))
        elif isinstance(other, cunumeric.ndarray):
            assert(self.shape == other.shape)
            # This is an operation that preserves the non-zero structure of
            # the output, so we'll just allocate a new store of values for
            # the output matrix and share the existing pos and crd arrays.
            other_store = get_store_from_cunumeric_array(other)
            result_vals = ctx.create_store(self.dtype, shape=self.vals.shape)
            promoted_pos = self.pos.promote(1, self.shape[1])
            task = ctx.create_task(SparseOpCode.ELEM_MULT_CSR_DENSE)
            task.add_output(result_vals)
            task.add_input(promoted_pos)
            task.add_input(self.crd)
            task.add_input(self.vals)
            task.add_input(other_store)
            # Partition the rows.
            task.add_broadcast(promoted_pos, 1)
            task.add_alignment(promoted_pos, other_store)
            task.add_image_constraint(promoted_pos, result_vals, range=True, functor=CompressedImagePartition)
            task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
            task.add_image_constraint(promoted_pos, self.vals, range=True, functor=CompressedImagePartition)
            task.execute()
            return csr_array.make_with_same_nnz_structure(self, (result_vals, self.crd, self.pos))
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        # A not-too-optimized implementation of subtract: multiply by -1 then add.
        return self + (other * -1.0)

    def __add__(self, other):
        if isinstance(other, numpy.ndarray):
            other = cunumeric.array(other)
        # If we're being added against a dense matrix, there's no point of doing anything
        # smart. Convert ourselves to a dense matrix and do the addition.
        if isinstance(other, cunumeric.ndarray):
            return self.todense() + other
        # If the other operand is sparse, then cast it into the format we know how to deal with.
        if is_sparse_matrix(other):
            other = other.tocsr()
        else:
            raise NotImplementedError
        assert(self.shape == other.shape)
        # Create the assemble query result array.
        q_nnz = ctx.create_store(nnz_ty, shape=(self.shape[0]))
        task = ctx.create_task(SparseOpCode.ADD_CSR_CSR_NNZ)
        task.add_output(q_nnz)
        self._add_to_task(task, input=True, vals=False)
        other._add_to_task(task, input=True, vals=False)
        task.add_scalar_arg(self.shape[1], types.int64)
        # Partitioning.
        task.add_alignment(q_nnz, self.pos)
        task.add_alignment(self.pos, other.pos)
        task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
        task.add_image_constraint(other.pos, other.crd, range=True, functor=CompressedImagePartition)
        task.execute()

        pos, nnz = self.nnz_to_pos(q_nnz)
        crd = ctx.create_store(coord_ty, shape=(nnz))
        vals = ctx.create_store(self.dtype, shape=(nnz))

        task = ctx.create_task(SparseOpCode.ADD_CSR_CSR)
        task.add_output(pos)
        task.add_output(crd)
        task.add_output(vals)
        self._add_to_task(task)
        other._add_to_task(task)
        task.add_scalar_arg(self.shape[1], types.int64)
        # Partitioning.
        task.add_alignment(pos, self.pos)
        task.add_alignment(self.pos, other.pos)
        task.add_image_constraint(pos, crd, range=True, functor=CompressedImagePartition)
        task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
        task.add_image_constraint(other.pos, other.crd, range=True, functor=CompressedImagePartition)
        task.add_alignment(crd, vals)
        task.add_alignment(self.crd, self.vals)
        task.add_alignment(other.crd, other.vals)
        # Make sure that we get pos in READ_WRITE mode.
        task.add_input(pos)
        task.execute()
        return csr_array((vals, crd, pos), shape=self.shape)

    # rmatmul represents the operation other @ self.
    def __rmatmul__(self, other):
        if isinstance(other, numpy.ndarray):
            other = cunumeric.array(other)
        # TODO (rohany): We'll just support a dense RHS matrix right now.
        if len(other.shape) == 1:
            raise NotImplementedError
        elif len(other.shape) == 2:
            assert other.shape[1] == self.shape[0]
            # TODO (rohany): As with the other SpMM case, we could do a 2-D distribution, but
            #  I'll just do a 1-D distribution right now, along the k-dimension of the computation.
            other_store = get_store_from_cunumeric_array(other)
            output = cunumeric.zeros((other.shape[0], self.shape[1]), dtype=self.dtype)
            output_store = get_store_from_cunumeric_array(output)
            promoted_pos = self.pos.promote(0, other_store.shape[0])
            task = ctx.create_task(SparseOpCode.SPMM_DENSE_CSR)
            task.add_reduction(output_store, ReductionOp.ADD)
            task.add_input(other_store)
            task.add_input(promoted_pos)
            task.add_input(self.crd)
            task.add_input(self.vals)
            # Partition the rows of the sparse matrix.
            task.add_broadcast(promoted_pos, 0)
            task.add_alignment(promoted_pos, other_store)
            task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
            task.add_image_constraint(promoted_pos, self.vals, range=True, functor=CompressedImagePartition)
            # Initially, only k is partitioned, so we'll be reducing into the full output.
            task.add_broadcast(output_store)
            task.execute()
            return store_to_cunumeric_array(output_store)
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        return self.dot(other)

    def __str__(self):
        los, his = self._unpack_pos()
        crdarr = store_to_cunumeric_array(self.crd)
        valsarr = store_to_cunumeric_array(self.vals)
        return f"{store_to_cunumeric_array(los)}, {store_to_cunumeric_array(his)}, {crdarr}, {valsarr}"

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


@clone_scipy_arr_kind(scipy.sparse.csc_array)
class csc_array(CompressedBase, DenseSparseBase):
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        if copy:
            raise NotImplementedError
        super().__init__()

        if dtype is not None:
            assert dtype == numpy.float64
            dtype = numpy.dtype(dtype)
        else:
            dtype = float64
        self.dtype = dtype

        if isinstance(arg, numpy.ndarray):
            arg = cunumeric.array(arg)
        if isinstance(arg, cunumeric.ndarray):
            assert(arg.ndim == 2)
            self.shape = arg.shape
            # Similarly to the CSR from dense case, we'll do a column based distribution.
            arg_store = get_store_from_cunumeric_array(arg)
            q_nnz = ctx.create_store(nnz_ty, shape=(arg.shape[1]))
            promoted_q_nnz = q_nnz.promote(0, arg.shape[0])
            task = ctx.create_task(SparseOpCode.DENSE_TO_CSC_NNZ)
            task.add_output(promoted_q_nnz)
            task.add_input(arg_store)
            task.add_broadcast(promoted_q_nnz, 0)
            task.add_alignment(promoted_q_nnz, arg_store)
            task.execute()
            # Assemble the output CSC array using the non-zeros per column.
            self.pos, nnz = self.nnz_to_pos(q_nnz)
            self.crd = ctx.create_store(coord_ty, shape=(nnz))
            self.vals = ctx.create_store(self.dtype, shape=(nnz))
            promoted_pos = self.pos.promote(0, arg.shape[0])
            task = ctx.create_task(SparseOpCode.DENSE_TO_CSC)
            task.add_output(promoted_pos)
            task.add_output(self.crd)
            task.add_output(self.vals)
            task.add_input(arg_store)
            task.add_input(promoted_pos)
            # Partition the columns.
            task.add_broadcast(promoted_pos, 0)
            task.add_alignment(promoted_pos, arg_store)
            task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
            task.add_image_constraint(promoted_pos, self.vals, range=True, functor=CompressedImagePartition)
            task.execute()
        elif isinstance(arg, tuple):
            (data, indices, indptr) = arg
            # Handle when someone passes a CSC indptr array as input.
            if isinstance(indptr, cunumeric.ndarray):
                assert indptr.shape[0] == shape[1] + 1
                los = indptr[:-1]
                his = indptr[1:]
                self.pos = pack_to_rect1_store(get_store_from_cunumeric_array(los), get_store_from_cunumeric_array(his))
            else:
                assert(isinstance(indptr, Store))
                self.pos = indptr
            self.crd = cast_to_store(indices)
            self.vals = cast_to_store(data)
            assert shape is not None
            self.shape = shape
        else:
            raise NotImplementedError

        # Manually adjust the key partition of the pos array to distribute the
        # sparse matrix by the rows across all processors. This makes the solver
        # understand that not everything should be replicated just because the
        # matrix construction is not parallelized.
        tile_size = (self.pos.shape[0] + runtime.num_procs - 1) // runtime.num_procs
        pos_part = self.pos.partition_by_tiling(Shape(tile_size))
        self.pos.set_key_partition(pos_part.partition)
        # Override the volume calculation on pos regions, since repartitioning
        # the pos region will most definitely cause moving around the crd and
        # vals arrays. Importantly, we need to compute the volume here and then
        # return it in the closure. If we compute it inside the closure, we end
        # up creating a reference cycle between self.pos._storage and self,
        # leading us to never collect self, leaking futures stored here.
        volume = self.crd.comm_volume() + self.vals.comm_volume() + self.pos.extents.volume()
        def compute_volume():
            return volume
        self.pos._storage.volume = compute_volume

    @classmethod
    def make_empty(cls, shape, dtype):
        M, N = shape
        # Make an empty pos array.
        q = cunumeric.zeros((N,), dtype=coord_ty)
        pos, _ = CompressedBase.nnz_to_pos_cls(get_store_from_cunumeric_array(q))
        crd = ctx.create_store(coord_ty, (0, ), optimize_scalar=False)
        vals = ctx.create_store(dtype, (0, ), optimize_scalar=False)
        return cls((vals, crd, pos), shape=shape, dtype=dtype)

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
        # The conversion to COO is pretty straightforward. The crd and values arrays are already
        # set up for COO, we just need to expand the pos array into coordinates.
        cols_expanded = ctx.create_store(coord_ty, shape=self.crd.shape)
        task = ctx.create_task(SparseOpCode.EXPAND_POS_TO_COORDINATES)
        task.add_input(self.pos)
        task.add_output(cols_expanded)
        task.add_image_constraint(self.pos, cols_expanded, range=True, functor=CompressedImagePartition)
        task.execute()
        return coo_array((self.vals, (self.crd, cols_expanded)), shape=self.shape, dtype=self.dtype)

    def __mul__(self, other):
        if not isinstance(other, csc_array):
            raise NotImplementedError
        # We can actually re-use the CSR * CSR method to add two CSC matrices.
        this_csr = csr_array((self.vals, self.crd, self.pos), shape=(self.shape[1], self.shape[0]))
        other_csr = csr_array((other.vals, other.crd, other.pos), shape=(other.shape[1], other.shape[0]))
        result = this_csr * other_csr
        # Now unpack it into a CSC matrix.
        return csc_array((result.vals, result.crd, result.pos), shape=self.shape)

    def __add__(self, other):
        if not isinstance(other, csc_array):
            raise NotImplementedError
        # We can actually re-use the CSR + CSR method to add two CSC matrices.
        this_csr = csr_array((self.vals, self.crd, self.pos), shape=(self.shape[1], self.shape[0]))
        other_csr = csr_array((other.vals, other.crd, other.pos), shape=(other.shape[1], other.shape[0]))
        result = this_csr + other_csr
        # Now unpack it into a CSC matrix.
        return csc_array((result.vals, result.crd, result.pos), shape=self.shape)

    def diagonal(self, k=0):
        return csr_array((self.vals, self.crd, self.pos), shape=(self.shape[1], self.shape[0])).diagonal(k=-k)

    def transpose(self, copy=False):
        if copy:
            raise NotImplementedError
        return csr_array.make_with_same_nnz_structure(self, (self.vals, self.crd, self.pos), shape=Shape((self.shape[1], self.shape[0])))

    T = property(transpose)

    # other / mat is defined to be the element-wise division of a other by mat over
    # the non-zero coordinates of mat. For now, we restrict this operation to be on
    # scalars only.
    def __rtruediv__(self, other):
        if not cunumeric.isscalar(other):
            raise NotImplementedError
        vals_arr = store_to_cunumeric_array(self.vals)
        new_vals = other / vals_arr
        return csc_array.make_with_same_nnz_structure(self, (get_store_from_cunumeric_array(new_vals), self.crd, self.pos))

    # TODO (rohany): This is a diversion from the API, but we'll set copy to be false for now.
    def conj(self, copy=False):
        if copy:
            raise NotImplementedError
        if self.dtype != float64:
            raise NotImplementedError
        return self

    def dot(self, other, out=None):
        if out is not None:
            assert isinstance(out, cunumeric.ndarray)
        if len(other.shape) == 1 or (len(other.shape) == 2 and other.shape[1] == 1):
            if not isinstance(other, cunumeric.ndarray):
                other = cunumeric.array(other)
            other_originally_2d = False
            if len(other.shape) == 2 and other.shape[1] == 1:
                other = other.squeeze(1)
                other_originally_2d = True
            assert(self.shape[1] == other.shape[0])
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
            task.add_image_constraint(self.pos, self.crd, range=True, functor=CompressedImagePartition)
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
        if order is not None:
            raise NotImplementedError
        if out is not None:
            out = cunumeric.array(out)
            out = get_store_from_cunumeric_array(out)
        elif out is None:
            out = ctx.create_store(self.dtype, shape=self.shape)
        # TODO (rohany): We'll do a col-based distribution for now, but a non-zero based
        #  distribution of this operation with overlapping output regions being reduced
        #  into also seems possible.
        promoted_pos = self.pos.promote(0, self.shape[0])
        task = ctx.create_task(SparseOpCode.CSC_TO_DENSE)
        task.add_output(out)
        task.add_input(promoted_pos)
        task.add_input(self.crd)
        task.add_input(self.vals)
        # We aren't partitioning the rows.
        task.add_broadcast(out, 0)
        task.add_alignment(out, promoted_pos)
        task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
        task.add_alignment(self.crd, self.vals)
        task.execute()
        return store_to_cunumeric_array(out)

    # sddmm computes a sampled dense-dense matrix multiplication operation
    # by fusing the element-wise multiply by the sparse matrix into the
    # dense matrix multiplication of the C and D operands. This function
    # is _not_ part of the scipy.sparse package but is prudent to add as
    # a kernel in many emerging workloads.
    def sddmm(self, C, D):
        # We'll start out with a row-based distribution of the CSR matrix.
        # In the future, we can look into doing a non-zero based distribution
        # of the computation, as there aren't really any downsides of doing it
        # versus a row-based distribution. The problem with both is that they
        # require replicating the D matrix onto all processors. Doing a partitioning
        # strategy that partitions up the j dimension of the computation is harder.
        # This operation is also non-zero structure preserving, so we'll just write
        # into an output array of values and share the pos and crd arrays.
        # TODO (rohany): An option is also partitioning up the `k` dimension of the
        #  computation (allows for partitioning C twice and D once), but requires
        #  reducing into the output.
        if isinstance(C, numpy.ndarray):
            C = cunumeric.array(C)
        if isinstance(D, numpy.ndarray):
            D = cunumeric.array(D)
        assert(len(C.shape) == 2 and len(D.shape) == 2)
        assert(self.shape[0] == C.shape[0] and self.shape[1] == D.shape[1] and C.shape[1] == D.shape[0])
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
        task.add_image_constraint(promoted_pos, result_vals, range=True, functor=CompressedImagePartition)
        task.add_image_constraint(promoted_pos, self.crd, range=True, functor=CompressedImagePartition)
        task.add_image_constraint(promoted_pos, self.vals, range=True, functor=CompressedImagePartition)
        task.add_broadcast(C_store)
        task.execute()
        return csc_array.make_with_same_nnz_structure(self, (result_vals, self.crd, self.pos))

    # TODO (rohany): Deduplicate these methods against csr_matrix.
    def __str__(self):
        los, his = self._unpack_pos()
        crdarr = store_to_cunumeric_array(self.crd)
        valsarr = store_to_cunumeric_array(self.vals)
        return f"{store_to_cunumeric_array(los)}, {store_to_cunumeric_array(his)}, {crdarr}, {valsarr}"

    def _unpack_pos(self):
        return unpack_rect1_store(self.pos)


@clone_scipy_arr_kind(scipy.sparse.coo_array)
class coo_array(CompressedBase):
    # TODO (rohany): For simplicity, we'll assume duplicate free COO.
    #  These should probably be arguments provided by the user, as computing
    #  these statistics about the data is likely as intensive as actually
    #  performing the conversions to other formats.
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        if isinstance(arg, numpy.ndarray):
            arg = cunumeric.array(arg)
        if isinstance(arg, cunumeric.ndarray):
            assert(arg.ndim == 2)
            # As a simple workaround for now, just load the matrix as a CSR
            # matrix and convert it to COO.
            # TODO (rohany): I basically want to do self = result, but I don't
            #  think that this is a valid python thing to do.
            arr = csr_array(arg).tocoo()
            data = cast_arr(arr._vals, dtype=float64)
            row = cast_arr(arr._i, dtype=coord_ty)
            col = cast_arr(arr._j, dtype=coord_ty)
            shape = arr.shape
            dtype = arr.dtype
        elif isinstance(arg, scipy.sparse.coo_array):
            # TODO (rohany): Handle the types here...
            data = cunumeric.array(arg.data, dtype=cunumeric.float64)
            row = cunumeric.array(arg.row, dtype=coord_ty)
            col = cunumeric.array(arg.col, dtype=coord_ty)
            shape = arg.shape
            dtype = cunumeric.float64
            # TODO (rohany): Not sure yet how to handle duplicates.
        else:
            (data, (row, col)) = arg
            data = cast_arr(data, dtype=float64)
            row = cast_arr(row, dtype=coord_ty)
            col = cast_arr(col, dtype=coord_ty)

        if shape is None:
            # TODO (rohany): Perform a max over the rows and cols to estimate the shape.
            raise NotImplementedError
        self.shape = shape

        # Not handling copies just yet.
        if copy:
            raise NotImplementedError

        if dtype is not None:
            assert dtype == numpy.float64
            dtype = numpy.dtype(dtype)
        else:
            dtype = float64
        self.dtype = dtype

        # Extract stores from the cunumeric arrays.
        self._i = get_store_from_cunumeric_array(row)
        self._j = get_store_from_cunumeric_array(col)
        self._vals = get_store_from_cunumeric_array(data)

        # Ensure that we distribute operations on COO arrays across the entire machine.
        # TODO (rohany): The sort routine is slightly broken when there are more processors
        #  than potential split points in the samplesort. Work around this by not distributing
        #  operations on very small numbers of processors.
        if self._i.shape[0] >= runtime.num_procs:
            tile_size = (self._i.shape[0] + runtime.num_procs - 1) // runtime.num_procs
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

    def diagonal(self, k=0):
        if k != 0:
            raise NotImplementedError
        # This function is lifted almost directly from scipy.sparse's implementation.
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cunumeric.empty(0, dtype=self.data.dtype)
        diag = cunumeric.zeros(min(rows + min(k, 0), cols - max(k, 0)), dtype=self.dtype)
        diag_mask = (self.row + k) == self.col
        # I'll assert that we don't have duplicate coordinates, so this operation is safe.
        row = self.row[diag_mask]
        data = self.data[diag_mask]
        diag[row + min(k, 0)] = data
        return diag

    def copy(self):
        rows = cunumeric.copy(self.row)
        cols = cunumeric.copy(self.col)
        data = cunumeric.copy(self.data)
        return coo_array((data, (rows, cols)), dtype=self.dtype, shape=self.shape)

    def transpose(self, copy=False):
        if copy:
            raise NotImplementedError
        return coo_array((self.data, (self.col, self.row)), dtype=self.dtype, shape=(self.shape[1], self.shape[0]))

    T = property(transpose)

    def tocoo(self, copy=False):
        if copy:
            raise NotImplementedError
        return self

    def tocsr(self, copy=False):
        if copy:
            raise NotImplementedError

        # We'll try a different (and parallel) approach here. First, we'll sort
        # the data using key (row, column), and sort the values accordingly. The
        # result of this operation is that the columns and values arrays suffice
        # as crd and vals arrays for a csr array. We just need to compress the
        # rows array into a valid pos array.
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
        # TODO (rohany): Change this once we have a GPU implementation.
        if runtime.num_gpus > 1:
             task.add_nccl_communicator()
        elif runtime.num_gpus == 0:
             task.add_cpu_communicator()
        task.execute()

        # Explicitly partition the rows into equal components to get the number
        # of non-zeros per row. We'll then partition up the non-zeros array according
        # to the per-partition ranges given by the min and max of each partition.
        num_procs = runtime.num_procs
        # TODO (rohany): If I try to partition this on really small inputs (like size 0 or 1 stores)
        #  across multiple processors, I see some sparse non-deterministic failures. I haven't root
        #  caused these, and I'm running out of time to figure them out. It seems just not partitioning
        #  the input on these really small matrices side-steps the underlying issue.
        if rows_store.shape[0] <= num_procs:
            num_procs = 1
        row_tiling = (rows_store.shape[0] + num_procs - 1) // num_procs
        rows_part = rows_store.partition(Tiling(Shape(row_tiling), Shape(num_procs)))
        # In order to bypass legate.core's current inability to handle representing
        # stores as FutureMaps, we drop below the ManualTask API to launch as task
        # ourselves and use the returned future map directly instead of letting the
        # core try and extract data from it.
        launcher = TaskLauncher(
            ctx,
            SparseOpCode.BOUNDS_FROM_PARTITIONED_COORDINATES,
            error_on_interference=False,
            tag=ctx.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG,
            provenance=runtime.legate_context.provenance,
        )
        launcher.add_input(rows_store, rows_part.get_requirement(1, 0), tag=1) # LEGATE_CORE_KEY_STORE_TAG
        bounds_store = ctx.create_store(domain_ty, shape=(1,), optimize_scalar=True)
        launcher.add_output(bounds_store, Broadcast(None, 0), tag=0)
        result = launcher.execute(Rect(hi=(num_procs,)))

        q_nnz = get_store_from_cunumeric_array(cunumeric.zeros((self.shape[0],), dtype=nnz_ty))
        task = ctx.create_manual_task(SparseOpCode.SORTED_COORDS_TO_COUNTS, launch_domain=Rect(hi=(num_procs,)))
        task.add_input(rows_part)
        task.add_reduction(q_nnz.partition(DomainPartition(q_nnz.shape, Shape(num_procs), result)), ReductionOp.ADD)
        task.add_scalar_arg(self.shape[0], types.int64)
        task.execute()
        # TODO (rohany): On small inputs, it appears that I get a non-deterministic failure, which appears either
        #  as a segfault or an incorrect output. This appears to show up only an OpenMP processors, and appears
        #  when running the full test suite with 4 opemps and 2 openmp threads. My notes from debugging this are
        #  as follows:
        #  * The result of the sort appears to be correct.
        #  * We start with a valid COO matrix.
        #  * Adding print statements make the bug much harder to reproduce.
        #  * In particular, the bug is harder to reproduce when q_nnz is printed
        #    out before the `self.nnz_to_pos(q_nnz)` line here.
        #  * If q_nnz is printed out after the `self.nnz_to_pos(q_nnz)` line, then
        #    the computation looks correct but an incorrect pos array is generated.

        pos, _ = self.nnz_to_pos(q_nnz)
        return csr_array((vals_store, cols_store, pos), shape=self.shape, dtype=self.dtype)

    def tocsc(self, copy=False):
        if copy:
            raise NotImplementedError
        # The strategy for CSC conversion is the same as COO conversion, we'll just
        # sort by the columns and then the rows.
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
        # of non-zeros per row. We'll then partition up the non-zeros array according
        # to the per-partition ranges given by the min and max of each partition.
        num_procs = runtime.num_procs
        # TODO (rohany): If I try to partition this on really small inputs (like size 0 or 1 stores)
        #  across multiple processors, I see some sparse non-deterministic failures. I haven't root
        #  caused these, and I'm running out of time to figure them out. It seems just not partitioning
        #  the input on these really small matrices side-steps the underlying issue.
        if cols_store.shape[0] <= num_procs:
            num_procs = 1
        col_tiling = (cols_store.shape[0] + num_procs - 1) // num_procs
        cols_part = cols_store.partition(Tiling(Shape(col_tiling), Shape(num_procs)))
        # In order to bypass legate.core's current inability to handle representing
        # stores as FutureMaps, we drop below the ManualTask API to launch as task
        # ourselves and use the returned future map directly instead of letting the
        # core try and extract data from it.
        launcher = TaskLauncher(
            ctx,
            SparseOpCode.BOUNDS_FROM_PARTITIONED_COORDINATES,
            error_on_interference=False,
            tag=ctx.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG,
            provenance=runtime.legate_context.provenance,
        )
        launcher.add_input(cols_store, cols_part.get_requirement(1, 0), tag=1) # LEGATE_CORE_KEY_STORE_TAG
        bounds_store = ctx.create_store(domain_ty, shape=(1,), optimize_scalar=True)
        launcher.add_output(bounds_store, Broadcast(None, 0), tag=0)
        result = launcher.execute(Rect(hi=(num_procs,)))

        q_nnz = get_store_from_cunumeric_array(cunumeric.zeros((self.shape[1],), dtype=nnz_ty))
        task = ctx.create_manual_task(SparseOpCode.SORTED_COORDS_TO_COUNTS, launch_domain=Rect(hi=(num_procs,)))
        task.add_input(cols_part)
        task.add_reduction(q_nnz.partition(DomainPartition(q_nnz.shape, Shape(num_procs), result)), ReductionOp.ADD)
        task.add_scalar_arg(self.shape[1], types.int64)
        task.execute()

        pos, _ = self.nnz_to_pos(q_nnz)
        return csc_array((vals_store, rows_store, pos), shape=self.shape, dtype=self.dtype)

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
        return f"{store_to_cunumeric_array(self._i)}, {store_to_cunumeric_array(self._j)}, {store_to_cunumeric_array(self._vals)}"


@clone_scipy_arr_kind(scipy.sparse.dia_array)
class dia_array(CompressedBase):
    def __init__(self, arg, shape=None, dtype=None, copy=False):
        if copy:
            raise NotImplementedError
        if shape is None:
            raise NotImplementedError
        assert(isinstance(arg, tuple))
        data, offsets = arg
        if isinstance(offsets, int):
            offsets = cunumeric.full((1,), offsets)
        data, offsets = cast_to_store(data), cast_to_store(offsets)
        if dtype is not None:
            assert dtype == numpy.float64
            dtype = numpy.dtype(dtype)
        else:
            dtype = float64
        self.dtype = dtype
        self.shape = shape
        self._offsets = offsets
        self._data = data

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

    # This implementation of nnz on DIA matrices is lifted from scipy.sparse.
    @property
    def nnz(self):
        M,N = self.shape
        nnz = 0
        for k in self.offsets:
            if k > 0:
                nnz += min(M,N-k)
            else:
                nnz += min(M+k,N)
        return int(nnz)

    # This implementation of diagonal() on DIA matrices is lifted from scipy.sparse.
    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cunumeric.empty(0, dtype=self.data.dtype)
        idx, = cunumeric.nonzero(self.offsets == k)
        first_col = max(0, k)
        last_col = min(rows + k, cols)
        result_size = last_col - first_col
        if idx.size == 0:
            return cunumeric.zeros(result_size, dtype=self.data.dtype)
        result = self.data[int(idx[0]), first_col:last_col]
        padding = result_size - len(result)
        if padding > 0:
            result = cunumeric.pad(result, (0, padding), mode='constant')
        return result

    # This implementation of tocoo() is lifted from scipy.sparse.
    def tocoo(self, copy=False):
        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = cunumeric.arange(offset_len)

        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        mask &= (self.data != 0)
        row = cunumeric.array(row[mask])
        col = cunumeric.array(cunumeric.tile(offset_inds, num_offsets)[mask.ravel()])
        data = cunumeric.array(self.data[mask])

        return coo_array((data, (row, col)), shape=self.shape, dtype=self.dtype)

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
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))
        if copy:
            raise AssertionError

        num_rows, num_cols = self.shape
        max_dim = max(self.shape)

        # flip diagonal offsets
        offsets = -self.offsets

        # re-align the data matrix
        r = cunumeric.arange(len(offsets), dtype=coord_ty)[:, None]
        c = cunumeric.arange(num_rows, dtype=coord_ty) - (offsets % max_dim)[:, None]
        pad_amount = max(0, max_dim-self.data.shape[1])
        data = cunumeric.hstack((self.data, cunumeric.zeros((self.data.shape[0], pad_amount),
                                              dtype=self.data.dtype)))
        data = data[r, c]
        return dia_array((data, offsets), shape=(num_cols, num_rows), copy=copy)

    T = property(transpose)

    # This routine is lifted from scipy.sparse's converter.
    def tocsc(self, copy=False):
        if copy:
            raise AssertionError
        if self.nnz == 0:
            return csc_array.make_empty(self.shape, self.dtype)

        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = cunumeric.arange(offset_len)

        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        mask &= (self.data != 0)

        idx_dtype = coord_ty
        indptr = cunumeric.zeros(num_cols + 1, dtype=idx_dtype)
        indptr[1:offset_len+1] = cunumeric.cumsum(mask.sum(axis=0)[:num_cols])
        if offset_len < num_cols:
            indptr[offset_len+1:] = indptr[offset_len]
        indices = row.T[mask.T].astype(idx_dtype, copy=False)
        data = self.data.T[mask.T]
        return csc_array((data, indices, indptr), shape=self.shape, dtype=self.dtype)

    def todense(self):
        return self.tocoo().todense()


# Include common type aliases.
csc_matrix = csc_array
csr_matrix = csr_array
coo_matrix = coo_array
dia_matrix = dia_array


def is_sparse_matrix(o):
    return any((
        isinstance(o, csr_array),
        isinstance(o, csc_array),
        isinstance(o, coo_array),
        isinstance(o, dia_array)
    ))

# TODO (rohany): We don't have sparse vectors, so not sure how mike garlands vision of SpMSpV will pan out.
#  I could hack this up right now though. I also don't know if dot product with a n-by-1 matrix counts.

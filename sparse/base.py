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

import cunumeric
from legate.core import Point, Rect, Store
from legate.core.partition import DomainPartition, PreimagePartition, Tiling
from legate.core.shape import Shape

from .config import SparseOpCode, rect1
from .runtime import ctx, runtime
from .types import int64, nnz_ty
from .utils import get_store_from_cunumeric_array, store_to_cunumeric_array


# CompressedBase is a base class for several different kinds of sparse
# matrices, such as CSR, CSC, COO and DIA.
class CompressedBase:
    @classmethod
    def nnz_to_pos_cls(cls, q_nnz: Store):
        cs = cunumeric.array(cunumeric.cumsum(store_to_cunumeric_array(q_nnz)))
        cs_store = get_store_from_cunumeric_array(cs)
        cs_shifted = cunumeric.append(cunumeric.array([0], nnz_ty), cs[:-1])
        cs_shifted_store = get_store_from_cunumeric_array(cs_shifted)
        # Zip the scan result into a rect1 region for the pos.
        pos = ctx.create_store(
            rect1, shape=(q_nnz.shape[0]), optimize_scalar=False
        )
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
                convert_method = getattr(self, "to" + format)
            except AttributeError as e:
                raise ValueError("Format {} is unknown.".format(format)) from e

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

    def copy_pos(self):
        # Issue a copy from the old pos to the new pos. We can't do this
        # with cunumeric because cunumeric doesn't support the Rect<1> type.
        pos = ctx.create_store(rect1, shape=self.pos.shape)
        copy = ctx.create_copy()
        copy.add_input(self.pos)
        copy.add_output(pos)
        copy.execute()
        return pos


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
        crd_part = Tiling(Shape(crd_tiling), Shape(num_procs))
        # Next, construct a preimage of the coordinates into the rows.
        preimage_legate = PreimagePartition(
            self.pos,
            self.crd,
            crd_part,
            ctx.mapper_id,
            range=True,
            # TODO (rohany): Do we want to ask the runtime to infer this for
            # us?  If so, we can short circuit later.
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
                # If our bounds intersect with the previous boundary, bump our
                # lower bound up past the previous boundary.
                if bounds.lo[0] <= prev_bounds.hi[0]:
                    lo = Point(prev_bounds.hi[0] + 1)
                # Do the same thing for our upper bound.
                if bounds.hi[0] <= prev_bounds.hi[0]:
                    hi = Point(prev_bounds.hi[0] + 1)
                # Next, make sure that holes in the pos region coloring are
                # filled so that we get a complete partition of the pos region.
                if lo[0] >= prev_bounds.hi[0] + 1:
                    lo = Point(prev_bounds.hi[0] + 1)
                # We've been doing all of these bounds computations with
                # inclusive indexing, so make sure that the Rect constructor
                # doesn't reverse that for us.
                balanced_row_bounds[Point(i)] = Rect(
                    lo=lo, hi=hi, exclusive=False
                )
        # Do the same normalization for the final color.
        if (
            balanced_row_bounds[Point(num_procs - 1)].hi[0] + 1
            != self.pos.shape[0]
        ):
            rect = balanced_row_bounds[Point(num_procs - 1)]
            balanced_row_bounds[Point(num_procs - 1)] = Rect(
                lo=rect.lo, hi=(self.pos.shape[0] - 1,), exclusive=False
            )
        # Use our adjusted bounds to construct the resulting partition.
        balanced_legate_part = DomainPartition(
            self.pos.shape, Shape(num_procs), balanced_row_bounds
        )
        # Actually construct the partition to force it to get cached and
        # analyzed.
        balanced_legion_part = balanced_legate_part.construct(
            self.pos.storage.region
        )
        assert balanced_legion_part.disjoint
        assert balanced_legion_part.complete
        self.pos.set_key_partition(balanced_legate_part)
        self._balanced_pos_partition = balanced_legate_part

    @classmethod
    def make_with_same_nnz_structure(cls, mat, arg, shape=None, dtype=None):
        if shape is None:
            shape = mat.shape
        if dtype is None:
            dtype = mat.dtype
        result = cls(arg, shape=shape, dtype=dtype)
        # Copy over all cached dats structures that depend on the same non-zero
        # structure.
        result._balanced_pos_partition = mat._balanced_pos_partition
        if result._balanced_pos_partition is not None:
            result.pos.set_key_partition(result._balanced_pos_partition)
        return result


# unpack_rect1_store unpacks a rect1 store into two int64 stores.
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


# pack_to_rect1_store packs two int64 stores into a rect1 store.
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

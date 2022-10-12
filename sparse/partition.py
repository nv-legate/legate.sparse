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

from .config import SparseOpCode, domain_ty
from .runtime import ctx, runtime as sparse_runtime

from legate.core import Partition as LegionPartition, Rect, Region, Transform, Point, track_provenance
from legate.core.launcher import TaskLauncher
from legate.core.partition import Broadcast, DomainPartition, ImagePartition, AffineProjection
from legate.core.runtime import runtime
from legate.core.shape import Shape

from typing import Optional


# CompressedImagePartition is a special implementation of
# taking an image partition that utilizes some domain knowledge
# about the kinds of partitions in these sparse computations.
# In particular, the standard encoding for CSR/CSC matrices with
# a pos array does not require a full-image. Since the pos array
# represents sorted and non-overlapping components of the crd and
# values arrays, we can simply take the upper and lower bounds of
# the input partition to create an image of the target crd or
# values region. This is key for avoiding the extra work of reading
# every element in the pos array and copying data from the GPU
# to the CPU, as dependent partitioning operations don't currently
# have GPU implementations.
class CompressedImagePartition(ImagePartition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct(
        self,
        region: Region,
        complete: bool = False,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        # TODO (rohany): We also can't use this when the input partition
        #  contains sparse index spaces. I'm not going to attempt to detect
        #  this case right now though.
        if (len(self._store.shape) != 1 and not self._store.transformed) or not self._range:
            assert False
            # TODO (rohany): I think that we could seamlessly handle this case,
            #  but I think that it makes more sense for this to be a user-error,
            #  since otherwise they would silently not be getting the behavior
            #  that they expect.
            # return super().construct(region, complete)

        index_partition = runtime.find_partition(region.index_space, self)
        if index_partition is None:
            source_part = self._store.partition(self._part)
            launch_shape = self._part.color_shape
            if self._store.transformed:
                orig_launch_shape = Shape(self._store._transform.get_inverse_color_transform(launch_shape.ndim).apply(Point(launch_shape)))
                assert orig_launch_shape.ndim == 1
            # Similarly as in the COO->CS* conversions, we have to bypass legate.core
            # and launch a lower level task to get the behavior that we want.
            launcher = TaskLauncher(
                ctx,
                SparseOpCode.FAST_IMAGE_RANGE,
                error_on_interference=False,
                tag=ctx.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG,
                provenance=sparse_runtime.legate_context.provenance,
            )
            launcher.add_input(self._store, source_part.get_requirement(launch_shape.ndim, None), tag=1) # LEGATE_CORE_KEY_STORE_TAG
            bounds_store = ctx.create_store(domain_ty, shape=(1,), optimize_scalar=True)
            launcher.add_output(bounds_store, Broadcast(None, 0), tag=0)
            domains = launcher.execute(Rect(hi=launch_shape))
            # We'll return a partition by domain using the resulting FutureMap.
            result = DomainPartition(Shape(ispace=region.index_space), self.color_shape, domains).construct(region)
            runtime.record_partition(region.index_space, self, result.index_partition)
            return result
        else:
            return region.get_child(index_partition)

    def __str__(self) -> str:
        return f"CompressedImage({self._store}, {self._part}, range={self._range})"


# MinMaxImagePartition is a partitioning functor used for projecting
# images through partitions of the crd array for kernels that restrict
# the indices they read of another tensor based on the coordinates.
# Instead of paying for a general image, we can short circuit a bit
# and access the full range of accessed coordinates in an image instead
# of restricting the image to just the present coordinates. This shortcut
# will allow us to avoid copying data back to the host and performing
# expensive image operations.
class MinMaxImagePartition(ImagePartition):
    def __init__(self, *args, proj_dims = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._proj_dims = proj_dims

    def construct(
            self,
            region: Region,
            complete: bool = False,
            color_shape: Optional[Shape] = None,
            color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        assert len(self._store.shape) == 1 and not self._range
        index_partition = runtime.find_partition(region.index_space, self)
        if index_partition is None:
            source_part = self._store.partition(self._part)
            launch_shape = self._part.color_shape
            # Similarly as in the COO->CS* conversions, we have to bypass legate.core
            # and launch a lower level task to get the behavior that we want.
            launcher = TaskLauncher(
                ctx,
                SparseOpCode.BOUNDS_FROM_PARTITIONED_COORDINATES,
                error_on_interference=False,
                tag=ctx.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG,
                provenance=sparse_runtime.legate_context.provenance,
            )
            launcher.add_input(self._store, source_part.get_requirement(launch_shape.ndim, None), tag=1) # LEGATE_CORE_KEY_STORE_TAG
            bounds_store = ctx.create_store(domain_ty, shape=(1,), optimize_scalar=True)
            launcher.add_output(bounds_store, Broadcast(None, 0), tag=0)
            domains = launcher.execute(Rect(hi=launch_shape))
            # We'll return a partition by domain using the resulting FutureMap.
            part = DomainPartition(Shape(ispace=region.index_space), self.color_shape, domains)
            if self._proj_dims is not None:
                projection = AffineProjection(self._proj_dims)
                # TODO (rohany): Might need to also do a projection on the point.
                part = projection.project_partition(part, Rect(hi=Shape(ispace=region.index_space)))
            result = part.construct(region)
            runtime.record_partition(region.index_space, self, result.index_partition)
            return result
        else:
            return region.get_child(index_partition)

    def __str__(self) -> str:
        return f"MinMaxImage({self._store}, {self._part}, range={self._range})"

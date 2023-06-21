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
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from legate.core import Rect, Store, get_legate_runtime, types

from .config import (
    SparseOpCode,
    SparseProjectionFunctor,
    SparseTunable,
    _sparse,
    sparse_ctx,
)

if TYPE_CHECKING:
    from typing import Optional, Union

    import numpy.typing as npt
    from legate.core import Shape


TO_CORE_DTYPES = {
    np.dtype(np.bool_): types.bool_,
    np.dtype(np.int8): types.int8,
    np.dtype(np.int16): types.int16,
    np.dtype(np.int32): types.int32,
    np.dtype(np.int64): types.int64,
    np.dtype(np.uint8): types.uint8,
    np.dtype(np.uint16): types.uint16,
    np.dtype(np.uint32): types.uint32,
    np.dtype(np.uint64): types.uint64,
    np.dtype(np.float16): types.float16,
    np.dtype(np.float32): types.float32,
    np.dtype(np.float64): types.float64,
    np.dtype(np.complex64): types.complex64,
    np.dtype(np.complex128): types.complex128,
}


class Runtime:
    def __init__(self, legate_context):
        self.legate_context = legate_context
        self.legate_runtime = get_legate_runtime()

        self.num_procs = int(
            self.legate_context.get_tunable(
                SparseTunable.NUM_PROCS,
                types.int32,
            )
        )
        if "LEGATE_SPARSE_NUM_PROCS" in os.environ:
            self.num_procs = int(os.environ["LEGATE_SPARSE_NUM_PROCS"])
            print(f"Overriding LEGATE_SPARSE_NUM_PROCS to {self.num_procs}")

        self.num_gpus = int(
            self.legate_context.get_tunable(
                SparseTunable.NUM_GPUS,
                types.int32,
            )
        )
        self.dynamic_projection_functor_id = 1
        self.proj_fn_1d_to_2d_cache = {}

        # Load all the necessary CUDA libraries if we have GPUs.
        if self.num_gpus > 0:
            # TODO (rohany): Also handle destroying the cuda libraries when the
            #  runtime is torn down.
            task = self.legate_context.create_manual_task(
                SparseOpCode.LOAD_CUDALIBS,
                launch_domain=Rect(lo=(0,), hi=(self.num_gpus,)),
            )
            task.execute()
            self.legate_runtime.issue_execution_fence(block=True)
        # Also initialize NCCL eagerly since we will most likely use it when
        # converting objects from COO.
        if self.num_gpus > 1:
            self.legate_runtime.get_nccl_communicator().initialize(
                self.num_gpus
            )

    def get_1d_to_2d_functor_id(self, xdim: int, ydim: int, rows: bool) -> int:
        key = (xdim, ydim, rows)
        if key in self.proj_fn_1d_to_2d_cache:
            return self.proj_fn_1d_to_2d_cache[key]
        result = self.get_projection_functor_id()
        _sparse.register_legate_sparse_1d_to_2d_functor(
            result, xdim, ydim, rows
        )
        self.proj_fn_1d_to_2d_cache[key] = result
        return result

    def get_projection_functor_id(self) -> int:
        result = self.dynamic_projection_functor_id
        self.dynamic_projection_functor_id += 1
        return sparse_ctx.get_projection_id(
            result + SparseProjectionFunctor.LAST_STATIC_PROJ_FN
        )

    def create_store(
        self,
        ty: Union[npt.DTypeLike, types.Dtype],
        shape: Optional[tuple[int, ...], Shape] = None,
        optimize_scalar: bool = False,
        ndim: Optional[int] = None,
    ) -> Store:
        core_ty = TO_CORE_DTYPES[ty] if isinstance(ty, np.dtype) else ty
        return self.legate_context.create_store(
            core_ty, shape=shape, optimize_scalar=optimize_scalar, ndim=ndim
        )


ctx = sparse_ctx
runtime = Runtime(ctx)

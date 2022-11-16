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

import os

import numpy as np
from legate.core import Rect, get_legate_runtime, types

from .config import (
    SparseOpCode,
    SparseProjectionFunctor,
    SparseTunable,
    _supported_dtypes,
    sparse_ctx,
)


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

        # Register type aliases for all of the common numpy types.
        for np_type, core_type in _supported_dtypes.items():
            self.legate_context.type_system.make_alias(
                np.dtype(np_type), core_type
            )

        # Load all the necessary CUDA libraries if we have GPUs.
        if self.num_gpus > 0:
            # TODO (rohany): Also handle destroying the cuda libraries when the
            #  runtime is torn down.
            task = self.legate_context.create_task(
                SparseOpCode.LOAD_CUDALIBS,
                manual=True,
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

    def get_1d_to_2d_functor_id(self, xdim: int, ydim: int, rows: bool):
        result = self.dynamic_projection_functor_id
        self.dynamic_projection_functor_id += 1
        return result + SparseProjectionFunctor.LAST_STATIC_PROJ_FN


ctx = sparse_ctx
runtime = Runtime(ctx)

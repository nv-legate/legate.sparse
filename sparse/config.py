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
import pyarrow as pa
from enum import IntEnum, unique
import subprocess

from legate.core import Library, ResourceConfig, get_legate_runtime, types, ffi
import numpy as np

class LegateSparseLib(Library):
    def __init__(self, name):
        self.name = name
        self.runtime = None
        self.shared_object = None

    def get_name(self):
        return self.name

    def get_shared_library(self):
        from sparse.install_info import libpath
        return os.path.join(
            libpath, "liblegate_sparse" + self.get_library_extension()
        )

    def get_c_header(self):
        from sparse.install_info import header
        return header

    def get_registration_callback(self):
        return "perform_registration"

    def initialize(self, shared_object):
        assert self.runtime is None
        self.shared_object = shared_object

    def set_runtime(self, runtime):
        assert self.runtime is None
        assert self.shared_object is not None
        self.runtime = runtime

    def get_resource_configuration(self):
        assert self.shared_object is not None
        # TODO (rohany): Make these line up with the configuration in the
        #  registration callback.
        # TODO (rohany): How can I make these match up with the enums in sparse_c.h??
        config = ResourceConfig()
        config.max_tasks = 100
        config.max_mappers = 1
        config.max_reduction_ops = 0
        config.max_projections = 1000
        config.max_shardings = 0
        return config

    def destroy(self):
        if self.runtime is not None:
            self.runtime.destroy()


SPARSE_LIB_NAME = "legate.sparse"
sparse_lib = LegateSparseLib(SPARSE_LIB_NAME)
sparse_ctx = get_legate_runtime().register_library(sparse_lib)
_sparse = sparse_lib.shared_object


@unique
class SparseOpCode(IntEnum):
    CSR_TO_DENSE = _sparse.LEGATE_SPARSE_CSR_TO_DENSE
    CSC_TO_DENSE = _sparse.LEGATE_SPARSE_CSC_TO_DENSE
    COO_TO_DENSE = _sparse.LEGATE_SPARSE_COO_TO_DENSE
    DENSE_TO_CSR_NNZ = _sparse.LEGATE_SPARSE_DENSE_TO_CSR_NNZ
    DENSE_TO_CSR = _sparse.LEGATE_SPARSE_DENSE_TO_CSR
    DENSE_TO_CSC_NNZ = _sparse.LEGATE_SPARSE_DENSE_TO_CSC_NNZ
    DENSE_TO_CSC = _sparse.LEGATE_SPARSE_DENSE_TO_CSC
    DIA_TO_CSR_NNZ = _sparse.LEGATE_SPARSE_DIA_TO_CSR_NNZ
    DIA_TO_CSR = _sparse.LEGATE_SPARSE_DIA_TO_CSR
    BOUNDS_FROM_PARTITIONED_COORDINATES = _sparse.LEGATE_SPARSE_BOUNDS_FROM_PARTITIONED_COORDINATES
    SORTED_COORDS_TO_COUNTS = _sparse.LEGATE_SPARSE_SORTED_COORDS_TO_COUNTS
    EXPAND_POS_TO_COORDINATES = _sparse.LEGATE_SPARSE_EXPAND_POS_TO_COORDINATES

    CSR_SPMV_ROW_SPLIT = _sparse.LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT
    CSR_SPMV_ROW_SPLIT_TROPICAL_SEMIRING = _sparse.LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT_TROPICAL_SEMIRING
    CSC_SPMV_COL_SPLIT = _sparse.LEGATE_SPARSE_CSC_SPMV_COL_SPLIT

    SPGEMM_CSR_CSR_CSR_NNZ = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ
    SPGEMM_CSR_CSR_CSR = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR
    SPGEMM_CSR_CSR_CSR_GPU = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU

    SPGEMM_CSR_CSR_CSC_LOCAL_TILES = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_LOCAL_TILES
    SPGEMM_CSR_CSR_CSC_COMM_COMPUTE = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_COMM_COMPUTE
    SPGEMM_CSR_CSR_CSC_SHUFFLE = _sparse.LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_SHUFFLE

    SPMM_CSR_DENSE = _sparse.LEGATE_SPARSE_SPMM_CSR_DENSE
    SPMM_DENSE_CSR = _sparse.LEGATE_SPARSE_SPMM_DENSE_CSR

    ADD_CSR_CSR_NNZ = _sparse.LEGATE_SPARSE_ADD_CSR_CSR_NNZ
    ADD_CSR_CSR = _sparse.LEGATE_SPARSE_ADD_CSR_CSR

    ELEM_MULT_CSR_CSR_NNZ = _sparse.LEGATE_SPARSE_ELEM_MULT_CSR_CSR_NNZ
    ELEM_MULT_CSR_CSR = _sparse.LEGATE_SPARSE_ELEM_MULT_CSR_CSR
    ELEM_MULT_CSR_DENSE = _sparse.LEGATE_SPARSE_ELEM_MULT_CSR_DENSE

    CSR_SDDMM = _sparse.LEGATE_SPARSE_CSR_SDDMM
    CSC_SDDMM = _sparse.LEGATE_SPARSE_CSC_SDDMM

    ZIP_TO_RECT1 = _sparse.LEGATE_SPARSE_ZIP_TO_RECT_1
    UNZIP_RECT1 = _sparse.LEGATE_SPARSE_UNZIP_RECT_1
    SCALE_RECT_1 = _sparse.LEGATE_SPARSE_SCALE_RECT_1
    UPCAST_FUTURE_TO_REGION = _sparse.LEGATE_SPARSE_UPCAST_FUTURE_TO_REGION
    FAST_IMAGE_RANGE = _sparse.LEGATE_SPARSE_FAST_IMAGE_RANGE

    READ_MTX_TO_COO = _sparse.LEGATE_SPARSE_READ_MTX_TO_COO

    EUCLIDEAN_CDIST = _sparse.LEGATE_SPARSE_EUCLIDEAN_CDIST
    CSR_DIAGONAL = _sparse.LEGATE_SPARSE_CSR_DIAGONAL
    SORT_BY_KEY = _sparse.LEGATE_SPARSE_SORT_BY_KEY
    VEC_MULT_ADD = _sparse.LEGATE_SPARSE_VEC_MULT_ADD

    LOAD_CUDALIBS = _sparse.LEGATE_SPARSE_LOAD_CUDALIBS
    UNLOAD_CUDALIBS = _sparse.LEGATE_SPARSE_UNLOAD_CUDALIBS

    RK_CALC_DY = _sparse.LEGATE_SPARSE_RK_CALC_DY

    ENUMERATE_INDEPENDENT_SETS = _sparse.LEGATE_QUANTUM_ENUMERATE_INDEP_SETS
    SETS_TO_SIZES = _sparse.LEGATE_QUANTUM_SETS_TO_SIZES
    CREATE_HAMILTONIANS = _sparse.LEGATE_QUANTUM_CREATE_HAMILTONIANS

@unique
class SparseProjectionFunctor(IntEnum):
    PROMOTE_1D_TO_2D = _sparse.LEGATE_SPARSE_PROJ_FN_1D_TO_2D
    LAST_STATIC_PROJ_FN = _sparse.LEGATE_SPARSE_LAST_PROJ_FN

@unique
class SparseTunable(IntEnum):
    NUM_PROCS = _sparse.LEGATE_SPARSE_TUNABLE_NUM_PROCS
    HAS_NUMAMEM = _sparse.LEGATE_SPARSE_TUNABLE_HAS_NUMAMEM
    NUM_GPUS = _sparse.LEGATE_SPARSE_TUNABLE_NUM_GPUS

@unique
class SparseTypeCode(IntEnum):
    SPARSE_TYPE_DOMAIN = _sparse.LEGATE_SPARSE_TYPE_DOMAIN
    SPARSE_TYPE_RECT1 = _sparse.LEGATE_SPARSE_TYPE_RECT1


# Register some types for us to use.
rect1 = pa.struct([('lo', types.int64), ('hi', types.int64)])
sparse_ctx.type_system.add_type(rect1, 16, SparseTypeCode.SPARSE_TYPE_RECT1)
domain_ty = "legion_domain_t"
sparse_ctx.type_system.add_type(domain_ty, ffi.sizeof(domain_ty), SparseTypeCode.SPARSE_TYPE_DOMAIN)
# Similarly to cunumeric, we'll register aliases for all of the
# numpy types into our type system so that we don't need to worry
# about implicitly converting types between numpy types and legate
# core store types. We set up the supported types here, and add them
# to the type system in the runtime initialization.
_supported_dtypes = {
    np.bool_: types.bool_,
    np.int8: types.int8,
    np.int16: types.int16,
    np.int32: types.int32,
    int: types.int64,
    np.int64: types.int64,
    np.uint8: types.uint8,
    np.uint16: types.uint16,
    np.uint32: types.uint32,
    np.uint: types.uint64,
    np.uint64: types.uint64,
    np.float16: types.float16,
    np.float32: types.float32,
    float: types.float64,
    np.float64: types.float64,
    np.complex64: types.complex64,
    np.complex128: types.complex128,
}

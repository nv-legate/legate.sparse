/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef __SPARSE_C_H
#define __SPARSE_C_H

#include "legate_preamble.h"

#ifndef LEGATE_USE_PYTHON_CFFI
#include "core/legate_c.h"
#endif

enum LegateSparseOpCode {
  _LEGATE_SPARSE_OP_CODE_BASE = 0,
  LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT,
  LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT_TROPICAL_SEMIRING,
  LEGATE_SPARSE_CSC_SPMV_COL_SPLIT,
  LEGATE_SPARSE_CSR_SPMV_POS_SPLIT, // Unused for now.

  // GEMMs. Need tasks that compute the NNZ and tasks
  // that actually assemble the output. Need to worry
  // about output matrices types as well... For now,
  // we'll just assume CSR output and rely on conversions
  // for CSC outputs. Hopefully, we can make CSR->CSC
  // conversions parallel and distributed?
  // TODO (rohany): Is there some easy way of deduplicating
  //  the code between the NNZ and Compute loops (other than
  //  by having a compiler)? I can probably utilize the format
  //  interface idea.
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ,
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR,
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU,

  // CSRxCSRxCSC is more complicated and requires more phases.
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_LOCAL_TILES,
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_COMM_COMPUTE,
  LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_SHUFFLE,

  // Sparse-Dense matmul.
  LEGATE_SPARSE_SPMM_CSR_DENSE,
  // Dense-Sparse matmul.
  LEGATE_SPARSE_SPMM_DENSE_CSR,

  // SDDMM on CSR matrices.
  LEGATE_SPARSE_CSR_SDDMM,
  LEGATE_SPARSE_CSC_SDDMM,

  // Addition.
  LEGATE_SPARSE_ADD_CSR_CSR_NNZ,
  LEGATE_SPARSE_ADD_CSR_CSR,

  // Element-wise multiplication.
  LEGATE_SPARSE_ELEM_MULT_CSR_CSR_NNZ,
  LEGATE_SPARSE_ELEM_MULT_CSR_CSR,
  LEGATE_SPARSE_ELEM_MULT_CSR_DENSE,

  // Conversions.
  LEGATE_SPARSE_CSR_TO_CSC_NNZ,
  LEGATE_SPARSE_CSR_TO_CSC,
  LEGATE_SPARSE_CSC_TO_CSR_NNZ,
  LEGATE_SPARSE_CSC_TO_CSR,
  LEGATE_SPARSE_CSR_TO_DENSE,
  LEGATE_SPARSE_CSC_TO_DENSE,
  LEGATE_SPARSE_COO_TO_DENSE,
  LEGATE_SPARSE_DENSE_TO_CSR_NNZ,
  LEGATE_SPARSE_DENSE_TO_CSR,
  LEGATE_SPARSE_DENSE_TO_CSC_NNZ,
  LEGATE_SPARSE_DENSE_TO_CSC,
  LEGATE_SPARSE_DIA_TO_CSR_NNZ,
  LEGATE_SPARSE_DIA_TO_CSR,
  LEGATE_SPARSE_BOUNDS_FROM_PARTITIONED_COORDINATES,
  LEGATE_SPARSE_SORTED_COORDS_TO_COUNTS,
  LEGATE_SPARSE_EXPAND_POS_TO_COORDINATES,

  // File IO.
  LEGATE_SPARSE_READ_MTX_TO_COO,

  // Operations on matrices that aren't quite tensor algebra related.
  LEGATE_SPARSE_CSR_DIAGONAL,

  // Utility tasks.
  LEGATE_SPARSE_ZIP_TO_RECT_1,
  LEGATE_SPARSE_UNZIP_RECT_1,
  LEGATE_SPARSE_SCALE_RECT_1,
  LEGATE_SPARSE_UPCAST_FUTURE_TO_REGION,
  LEGATE_SPARSE_FAST_IMAGE_RANGE,

  // Random other tasks that aren't really sparse...
  LEGATE_SPARSE_EUCLIDEAN_CDIST,
  LEGATE_SPARSE_SORT_BY_KEY,
  LEGATE_SPARSE_VEC_MULT_ADD,

  // Utility tasks for loading cuda libraries.
  LEGATE_SPARSE_LOAD_CUDALIBS,
  LEGATE_SPARSE_UNLOAD_CUDALIBS,

  LEGATE_SPARSE_LAST_TASK, // must be last
};

enum LegateSparseProjectionFunctors {
  _LEGATE_SPARSE_PROJ_FN_BASE = 0,
  LEGATE_SPARSE_PROJ_FN_1D_TO_2D,
  LEGATE_SPARSE_LAST_PROJ_FN, // must be last
};

enum LegateSparseTunable {
  LEGATE_SPARSE_TUNABLE_NUM_PROCS = 1,
  LEGATE_SPARSE_TUNABLE_HAS_NUMAMEM,
  LEGATE_SPARSE_TUNABLE_NUM_GPUS,
};

enum LegateSparseTypes {
  LEGATE_SPARSE_TYPE_DOMAIN = MAX_TYPE_NUMBER + 1,
  LEGATE_SPARSE_TYPE_RECT1,
};

#ifdef __cplusplus
extern "C" {
#endif

void perform_registration();

void register_legate_sparse_1d_to_2d_functor(legion_projection_id_t proj_id, int32_t gx, int32_t gy, bool rows);

#ifdef __cplusplus
}
#endif

#endif // __SPARSE_C_H

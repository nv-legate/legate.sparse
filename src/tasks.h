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

#pragma once

#include "sparse.h"
#include "sparse_c.h"
#include "legate.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

namespace sparse {

// Extract common types into typedefs to make the code easier to change later.
typedef uint64_t nnz_ty;
typedef int64_t coord_ty;
typedef double val_ty;

class SpGEMMCSRxCSRxCSCLocalTiles : public SparseTask<SpGEMMCSRxCSRxCSCLocalTiles> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_LOCAL_TILES;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
};

class SpGEMMCSRxCSRxCSCCommCompute : public SparseTask<SpGEMMCSRxCSRxCSCCommCompute> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_COMM_COMPUTE;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
  // TODO (rohany): I don't think that this task needs a GPU implementation, as it
  //  shouldn't be moving a large amount of data around. If it ends up appearing
  //  that the cost of moving the data it uses is relatively large then we can
  //  add a dummy GPU implementation.
};

class SpGEMMCSRxCSRxCSCShuffle : public SparseTask<SpGEMMCSRxCSRxCSCShuffle> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_SHUFFLE;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
};

}  // namespace sparse

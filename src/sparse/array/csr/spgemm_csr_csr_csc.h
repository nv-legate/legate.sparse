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

#include "sparse/sparse.h"
#include "sparse/sparse_c.h"
#include "legate.h"

namespace sparse {

struct SpGEMMCSRxCSRxCSCLocalTilesArgs {
  // The A stores are not const because we will use
  // create_output_buffer on them.
  legate::Store& A_pos;
  legate::Store& A_crd;
  legate::Store& A_vals;
  const legate::Store& B_pos;
  const legate::Store& B_crd;
  const legate::Store& B_vals;
  const legate::Store& C_pos;
  const legate::Store& C_crd;
  const legate::Store& C_vals;
  const int64_t C_rows;
};

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

struct SpGEMMCSRxCSRxCSCCommComputeArgs {
  const legate::Store& out;
  const legate::Store& pos;
  const legate::Store& global_pos;
  const int32_t gx;
  const int32_t gy;
};

class SpGEMMCSRxCSRxCSCCommCompute : public SparseTask<SpGEMMCSRxCSRxCSCCommCompute> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSC_COMM_COMPUTE;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
};

struct SpGEMMCSRxCSRxCSCShuffleArgs {
  // We'll use create_output_buffer on these stores.
  legate::Store& out_pos;
  legate::Store& out_crd;
  legate::Store& out_vals;
  const legate::Store& global_pos;
  const legate::Store& global_crd;
  const legate::Store& global_vals;
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

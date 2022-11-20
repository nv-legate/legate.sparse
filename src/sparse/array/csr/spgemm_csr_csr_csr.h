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

namespace sparse {

struct SpGEMMCSRxCSRxCSRNNZArgs {
  const legate::Store& nnz;
  const legate::Store& B_pos;
  const legate::Store& B_crd;
  const legate::Store& C_pos;
  const legate::Store& C_crd;
  const uint64_t A2_dim;
};

class SpGEMMCSRxCSRxCSRNNZ : public SparseTask<SpGEMMCSRxCSRxCSRNNZ> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
};

struct SpGEMMCSRxCSRxCSRArgs {
  const legate::Store& A_pos;
  const legate::Store& A_crd;
  const legate::Store& A_vals;
  const legate::Store& B_pos;
  const legate::Store& B_crd;
  const legate::Store& B_vals;
  const legate::Store& C_pos;
  const legate::Store& C_crd;
  const legate::Store& C_vals;
  const uint64_t A2_dim;
};

class SpGEMMCSRxCSRxCSR : public SparseTask<SpGEMMCSRxCSRxCSR> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
};

struct SpGEMMCSRxCSRxCSRGPUArgs {
  const legate::Store& A_pos;
  // A_crd and A_vals are not const because we will call
  // create_output_buffer on them.
  legate::Store& A_crd;
  legate::Store& A_vals;
  const legate::Store& B_pos;
  const legate::Store& B_crd;
  const legate::Store& B_vals;
  const legate::Store& C_pos;
  const legate::Store& C_crd;
  const legate::Store& C_vals;
  const uint64_t A2_dim;
};

// CSRxCSRxCSR SpGEMM for NVIDIA GPUs. Due to limitations with cuSPARSE,
// we take a different approach than on CPUs and OMPs.
class SpGEMMCSRxCSRxCSRGPU : public SparseTask<SpGEMMCSRxCSRxCSRGPU> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_GPU;
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
};

}  // namespace sparse

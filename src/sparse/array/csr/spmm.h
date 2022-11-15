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

struct SpMMCSRArgs {
  const legate::Store& A_vals;
  const legate::Store& B_pos;
  const legate::Store& B_crd;
  const legate::Store& B_vals;
  const legate::Store& C_vals;
};

class SpMMCSR : public SparseTask<SpMMCSR> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPMM_CSR_DENSE;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

struct SpMMDenseCSRArgs {
  const legate::Store& A_vals;
  const legate::Store& B_vals;
  const legate::Store& C_pos;
  const legate::Store& C_crd;
  const legate::Store& C_vals;
};

class SpMMDenseCSR : public SparseTask<SpMMDenseCSR> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPMM_DENSE_CSR;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace sparse

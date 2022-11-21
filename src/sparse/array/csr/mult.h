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

struct ElemwiseMultCSRCSRNNZArgs {
  const legate::Store& nnz;
  const legate::Store& B_pos;
  const legate::Store& B_crd;
  const legate::Store& C_pos;
  const legate::Store& C_crd;
};

class ElemwiseMultCSRCSRNNZ : public SparseTask<ElemwiseMultCSRCSRNNZ> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_ELEM_MULT_CSR_CSR_NNZ;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

struct ElemwiseMultCSRCSRArgs {
  const legate::Store& A_pos;
  const legate::Store& A_crd;
  const legate::Store& A_vals;
  const legate::Store& B_pos;
  const legate::Store& B_crd;
  const legate::Store& B_vals;
  const legate::Store& C_pos;
  const legate::Store& C_crd;
  const legate::Store& C_vals;
};

class ElemwiseMultCSRCSR : public SparseTask<ElemwiseMultCSRCSR> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_ELEM_MULT_CSR_CSR;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace sparse

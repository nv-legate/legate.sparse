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

// Compute kernels.

// A row-based SpMV over the tropical semiring instead of +,*.
class CSRSpMVRowSplitTropicalSemiring : public SparseTask<CSRSpMVRowSplitTropicalSemiring> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT_TROPICAL_SEMIRING;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

// SpGEMM kernels.
class SpGEMMCSRxCSRxCSRNNZ : public SparseTask<SpGEMMCSRxCSRxCSRNNZ> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR_NNZ;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
};

class SpGEMMCSRxCSRxCSR : public SparseTask<SpGEMMCSRxCSRxCSR> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SPGEMM_CSR_CSR_CSR;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
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

// SpMM Kernels.
// TODO (rohany): Implement the statically load-balanced SpMM as well.
//  However, there's a caveat that we would only want to pick it if
//  we are using GPUs and the C matrix fits into memory.
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

// SpMM Dense * CSR.
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

// Element-wise multiplication.
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

class ElemwiseMultCSRDense : public SparseTask<ElemwiseMultCSRDense> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_ELEM_MULT_CSR_DENSE;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class CSRSDDMM : public SparseTask<CSRSDDMM> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_CSR_SDDMM;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class CSCSDDMM : public SparseTask<CSCSDDMM> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_CSC_SDDMM;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

// Tasks for conversion to dense arrays.

class CSCToDense : public SparseTask<CSCToDense> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_CSC_TO_DENSE;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace sparse

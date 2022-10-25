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
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

namespace sparse {

// Extract common types into typedefs to make the code easier to change later.
typedef uint64_t nnz_ty;
typedef int64_t coord_ty;
typedef double val_ty;

// Compute kernels.

// SpMV kernels.
class CSRSpMVRowSplit : public SparseTask<CSRSpMVRowSplit> {
public:
  static const int TASK_ID = LEGATE_SPARSE_CSR_SPMV_ROW_SPLIT;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

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

class CSCSpMVColSplit : public SparseTask<CSCSpMVColSplit> {
public:
  static const int TASK_ID = LEGATE_SPARSE_CSC_SPMV_COL_SPLIT;
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

// Addition.
// TODO (rohany): It seems like addition is not something supported by CuSparse. We'll
//  have to utilize DISTAL to get the fastest implementation of GPU sparse matrix addition.
class AddCSRCSRNNZ : public SparseTask<AddCSRCSRNNZ> {
public:
  static const int TASK_ID = LEGATE_SPARSE_ADD_CSR_CSR_NNZ;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class AddCSRCSR : public SparseTask<AddCSRCSR> {
public:
  static const int TASK_ID = LEGATE_SPARSE_ADD_CSR_CSR;
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

// Tasks for conversion between formats.
class BoundsFromPartitionedCoordinates : public SparseTask<BoundsFromPartitionedCoordinates> {
public:
  static const int TASK_ID = LEGATE_SPARSE_BOUNDS_FROM_PARTITIONED_COORDINATES;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
};

class SortedCoordsToCounts : public SparseTask<SortedCoordsToCounts> {
public:
  static const int TASK_ID = LEGATE_SPARSE_SORTED_COORDS_TO_COUNTS;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
};

class ExpandPosToCoordinates : public SparseTask<ExpandPosToCoordinates> {
public:
  static const int TASK_ID = LEGATE_SPARSE_EXPAND_POS_TO_COORDINATES;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
public:
  // We make this struct templated so that we can define it here and have it
  // instantiated (with the cuda annotations) where it is needed. Unfortunately
  // we have to make this struct public for NVCC to be happy.
  template<typename T>
  struct volume : public thrust::unary_function<T, size_t> {
    #if defined(__CUDACC__)
    __host__ __device__
    #endif
    size_t operator()(Legion::Rect<1> x) { return x.volume(); }
  };

private:
  // We'll use thrust for the body of this function (for all variants),
  // so we'll define a helper to do the heavy lifting of the task.
  template<typename Policy>
  static void expand_pos_impl(const Policy& policy,
                              legate::AccessorRO<Legion::Rect<1>, 1> pos,
                              Legion::Domain pos_domain,
                              legate::AccessorWO<coord_ty, 1> result,
                              Legion::Domain result_domain,
                              Legion::Memory::Kind tempMemKind) {
    // Return early if there isn't any work to do. Entering this code
    // with an empty domain results in CUDA errors for the thrust backend.
    if (pos_domain.empty() || result_domain.empty()) return;

    // This implementation of expand was inspired from
    // https://huggingface.co/spaces/ma-xu/LIVE/blob/main/thrust/examples/expand.cu.
    Legion::DeferredBuffer<size_t, 1> volumes({0, pos_domain.get_volume() - 1}, tempMemKind);
    Legion::DeferredBuffer<size_t, 1> offsets({0, pos_domain.get_volume() - 1}, tempMemKind);
    // Initialize all of our arrays.
    thrust::fill(
      policy,
      volumes.ptr(0),
      volumes.ptr(0) + pos_domain.get_volume(),
      size_t(0)
    );
    thrust::fill(
      policy,
      offsets.ptr(0),
      offsets.ptr(0) + pos_domain.get_volume(),
      size_t(0)
    );
    thrust::fill(
      policy,
      result.ptr(result_domain.lo()),
      result.ptr(result_domain.lo()) + result_domain.get_volume(),
      coord_ty(0)
    );
    // Transform each pos rectangle into its volume. We have to make a
    // temporary here because not all of the thrust functions accept a
    // transform.
    thrust::transform(
      policy,
      pos.ptr(pos_domain.lo()),
      pos.ptr(pos_domain.lo()) + pos_domain.get_volume(),
      volumes.ptr(0),
      volume<Legion::Rect<1>>{}
    );
    // Perform an exclusive scan to find the offsets to write coordinates into.
    thrust::exclusive_scan(
      policy,
      volumes.ptr(0),
      volumes.ptr(0) + pos_domain.get_volume(),
      offsets.ptr(0)
    );
    // Scatter the non-zero counts into their output indices.
    thrust::scatter_if(
      policy,
      thrust::counting_iterator<coord_ty>(0),
      thrust::counting_iterator<coord_ty>(pos_domain.get_volume()),
      offsets.ptr(0),
      volumes.ptr(0),
      result.ptr(result_domain.lo())
    );
    // Compute a max-scan over the output indices, filling in holes.
    thrust::inclusive_scan(
      policy,
      result.ptr(result_domain.lo()),
      result.ptr(result_domain.lo()) + result_domain.get_volume(),
      result.ptr(result_domain.lo()),
      thrust::maximum<coord_ty>{}
    );
    // Gather input values according to the computed indices.
    thrust::gather(
      policy,
      result.ptr(result_domain.lo()),
      result.ptr(result_domain.lo()) + result_domain.get_volume(),
      thrust::counting_iterator<coord_ty>(pos_domain.lo()[0]),
      result.ptr(result_domain.lo())
    );
  }
};

// Tasks for conversion to dense arrays.
class CSRToDense : public SparseTask<CSRToDense> {
public:
  static const int TASK_ID = LEGATE_SPARSE_CSR_TO_DENSE;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

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

class COOToDense : public SparseTask<COOToDense> {
public:
  static const int TASK_ID = LEGATE_SPARSE_COO_TO_DENSE;
  static void cpu_variant(legate::TaskContext& ctx);
};

class DenseToCSRNNZ : public SparseTask<DenseToCSRNNZ> {
public:
  static const int TASK_ID = LEGATE_SPARSE_DENSE_TO_CSR_NNZ;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class DenseToCSR : public SparseTask<DenseToCSR> {
public:
  static const int TASK_ID = LEGATE_SPARSE_DENSE_TO_CSR;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class DenseToCSCNNZ : public SparseTask<DenseToCSCNNZ> {
public:
  static const int TASK_ID = LEGATE_SPARSE_DENSE_TO_CSC_NNZ;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class DenseToCSC : public SparseTask<DenseToCSC> {
public:
  static const int TASK_ID = LEGATE_SPARSE_DENSE_TO_CSC;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class DIAToCSRNNZ : public SparseTask<DIAToCSRNNZ> {
public:
  static const int TASK_ID = LEGATE_SPARSE_DIA_TO_CSR_NNZ;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx) {
    DIAToCSRNNZ::cpu_variant(ctx);
  }
#endif
};

class DIAToCSR : public SparseTask<DIAToCSR> {
public:
  static const int TASK_ID = LEGATE_SPARSE_DIA_TO_CSR;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx) {
    DIAToCSR::cpu_variant(ctx);
  }
#endif
};

// Utility tasks.
class ZipToRect1 : public SparseTask<ZipToRect1> {
public:
  static const int TASK_ID = LEGATE_SPARSE_ZIP_TO_RECT_1;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class UnZipRect1 : public SparseTask<UnZipRect1> {
public:
  static const int TASK_ID = LEGATE_SPARSE_UNZIP_RECT_1;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class ScaleRect1 : public SparseTask<ScaleRect1> {
public:
  static const int TASK_ID = LEGATE_SPARSE_SCALE_RECT_1;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class UpcastFutureToRegion : public SparseTask<UpcastFutureToRegion> {
public:
  static const int TASK_ID = LEGATE_SPARSE_UPCAST_FUTURE_TO_REGION;
  static void cpu_variant(legate::TaskContext& ctx);
private:
  template<typename T>
  static void cpu_variant_impl(legate::TaskContext& ctx);
};

class FastImageRange : public SparseTask<FastImageRange> {
public:
  static const int TASK_ID = LEGATE_SPARSE_FAST_IMAGE_RANGE;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class ReadMTXToCOO : public SparseTask<ReadMTXToCOO> {
public:
  static const int TASK_ID = LEGATE_SPARSE_READ_MTX_TO_COO;
  static void cpu_variant(legate::TaskContext& ctx);
};

class EuclideanCDist : public SparseTask<EuclideanCDist> {
public:
  static const int TASK_ID = LEGATE_SPARSE_EUCLIDEAN_CDIST;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

class GetCSRDiagonal : public SparseTask<GetCSRDiagonal> {
public:
  static const int TASK_ID = LEGATE_SPARSE_CSR_DIAGONAL;
  // TODO (rohany): We could rewrite this having each implementation just make
  //  a call to thrust::transform, but the implementations are simple enough
  //  anyway.
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

// SortByKey sorts a set of key regions and a value region.
// Out of an input `n` regions, the first `n-1` regions are
// zipped together to be a key to sort the value region `n`.
// SortByKey operates in place.
class SortByKey : public SparseTask<SortByKey> {
public:
  static const int TASK_ID = LEGATE_SPARSE_SORT_BY_KEY;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
};

class VecMultAdd : public SparseTask<VecMultAdd> {
public:
  static const int TASK_ID = LEGATE_SPARSE_VEC_MULT_ADD;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& ctx);
#endif
};

} // namespace sparse

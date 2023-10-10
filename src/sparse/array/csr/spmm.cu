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

#include "sparse/array/csr/spmm.h"
#include "sparse/array/csr/spmm_template.inl"
#include "sparse/util/cusparse_utils.h"
#include "sparse/util/distal_cuda_utils.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename DST, typename SRC>
__global__ void cast_and_offset(size_t elems, DST* dst, const SRC* src, int64_t offset)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) return;
  dst[idx] = static_cast<DST>(src[idx] - offset);
}

template <>
struct SpMMCSRImpl<VariantKind::GPU> {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpMMCSRArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    const auto& A_vals = args.A_vals;
    const auto& B_pos  = args.B_pos;
    const auto& B_crd  = args.B_crd;
    const auto& B_vals = args.B_vals;
    const auto& C_vals = args.C_vals;
    const auto B1_dim  = args.B1_dim;

    // Break out early if the iteration space partition is empty.
    if (A_vals.domain().empty() || B_pos.domain().empty() || B_vals.domain().empty() ||
        C_vals.domain().empty()) {
      return;
    }

    // Get context sensitive objects.
    auto handle = get_cusparse();
    auto stream = get_cached_stream();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    // Construct the CUSPARSE objects from individual regions.
    auto cusparse_A = makeCuSparseDenseMat<VAL_TY>(A_vals);

    cusparseSpMatDescr_t cusparse_B;
    cusparseDnMatDescr_t cusparse_C;
    // TODO (rohany): Comment.
    // Based on whether the input store is transposed or not,
    // we need to handle the SpMM differently. At a high level,
    // for a row-major matrix we can do the image optimization.
    // For column-major matrix, we have to just offset the crd
    // array of the input sparse matrix down by the minimum value.
    auto C_domain   = args.C_vals.domain();
    auto C_vals_acc = args.C_vals.read_accessor<VAL_TY, 2>();
    auto C_vals_ptr = C_vals_acc.ptr(C_domain.lo());
    auto x_stride   = C_vals_acc.accessor.strides[0] / sizeof(VAL_TY);
    auto y_stride   = C_vals_acc.accessor.strides[1] / sizeof(VAL_TY);
    cusparseSpMMAlg_t alg;
    if (x_stride >= y_stride) {
      // Because we are doing the same optimization as in SpMV to minimize
      // the communication instead of replicating the C matrix, we have to
      // offset the pointer into C down to the "base" of the region (which
      // may be invalid). We can rely on cuSPARSE not accessing this invalid
      // region because it is not referenced by any coordinates of B.
      auto ld    = x_stride;
      C_vals_ptr = C_vals_ptr - size_t(ld * C_domain.lo()[0]);
      CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse_C,
                                         B1_dim,
                                         C_domain.hi()[1] - C_domain.lo()[1] + 1, /* columns */
                                         ld,
                                         (void*)C_vals_ptr,
                                         cusparseDataType<VAL_TY>(),
                                         CUSPARSE_ORDER_ROW));
      cusparse_B = makeCuSparseCSR<INDEX_TY, VAL_TY>(B_pos, B_crd, B_vals, B1_dim);
      alg        = CUSPARSE_SPMM_CSR_ALG2;
    } else {
      std::cout << "Handling a transpose case." << std::endl;
      auto B_rows = B_pos.domain().get_volume();
      DeferredBuffer<INDEX_TY, 1> B_indptr({0, B_rows}, Memory::GPU_FB_MEM);
      {
        auto blocks = get_num_blocks_1d(B_rows);
        convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          B_rows, B_pos.read_accessor<Rect<1>, 1>().ptr(B_pos.domain().lo()), B_indptr.ptr(0));
      }
      DeferredBuffer<INDEX_TY, 1> B_crd_int({0, B_crd.domain().get_volume() - 1},
                                            Memory::GPU_FB_MEM);
      auto B_min_coord = C_vals.domain().lo()[0];
      auto B_max_coord = C_vals.domain().hi()[0];
      auto C_rows      = B_max_coord - B_min_coord + 1;
      {
        auto dom    = B_crd.domain();
        auto elems  = dom.get_volume();
        auto blocks = get_num_blocks_1d(elems);
        cast_and_offset<INDEX_TY, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          elems, B_crd_int.ptr(0), B_crd.read_accessor<INDEX_TY, 1>().ptr(dom.lo()), B_min_coord);
      }
      CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_B,
                                       B_rows,
                                       C_rows /* cols */,
                                       B_crd.domain().get_volume() /* nnz */,
                                       B_indptr.ptr(0),
                                       B_crd_int.ptr(0),
                                       getPtrFromStore<VAL_TY, 1>(B_vals),
                                       cusparseIndexType<INDEX_TY>(),
                                       cusparseIndexType<INDEX_TY>(),
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       cusparseDataType<VAL_TY>()));
      CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse_C,
                                         C_rows,
                                         C_domain.hi()[1] - C_domain.lo()[1] + 1, /* columns */
                                         y_stride,
                                         (void*)C_vals_ptr,
                                         cusparseDataType<VAL_TY>(),
                                         CUSPARSE_ORDER_COL));
      alg = CUSPARSE_SPMM_CSR_ALG1;
    }

    // Call CUSPARSE.
    VAL_TY alpha   = static_cast<VAL_TY>(1);
    VAL_TY beta    = static_cast<VAL_TY>(0);
    size_t bufSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           cusparse_B,
                                           cusparse_C,
                                           &beta,
                                           cusparse_A,
                                           cusparseDataType<VAL_TY>(),
                                           alg,
                                           &bufSize));
    // Allocate a buffer if we need to.
    void* workspacePtr = nullptr;
    if (bufSize > 0) {
      DeferredBuffer<char, 1> buf({0, bufSize - 1}, Memory::GPU_FB_MEM);
      workspacePtr = buf.ptr(0);
    }
    // Do the SpMM.
    CHECK_CUSPARSE(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                cusparse_B,
                                cusparse_C,
                                &beta,
                                cusparse_A,
                                cusparseDataType<VAL_TY>(),
                                CUSPARSE_SPMM_CSR_ALG2,
                                workspacePtr));
    // Destroy the created objects.
    CHECK_CUSPARSE(cusparseDestroyDnMat(cusparse_A));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_B));
    CHECK_CUSPARSE(cusparseDestroyDnMat(cusparse_C));
    CHECK_CUDA_STREAM(stream);
  }
};

template <typename INDEX_TY, typename VAL_TY>
__global__ void spmm_dense_csr_kernel(const size_t nnzs,
                                      const size_t pos_offset,
                                      const int64_t* block_starts,
                                      const int64_t idim,
                                      const AccessorRD<SumReduction<VAL_TY>, false, 2> A_vals,
                                      const AccessorRO<VAL_TY, 2> B_vals,
                                      const AccessorRO<Rect<1>, 1> C_pos,
                                      const AccessorRO<INDEX_TY, 1> C_crd,
                                      const AccessorRO<VAL_TY, 1> C_vals)
{
  size_t block   = blockIdx.x;
  const auto idx = global_tid_1d();
  if (idx >= nnzs) return;
  auto nnz_idx = idx + pos_offset;
  // Search for the current coordinate. I've written this right now as processing
  // a single non-zero per thread. In the future, the cost of the binary search
  // can be amortized so that we process more values after the search.
  auto p_start = block_starts[block];
  auto p_end   = block_starts[block + 1];
  auto k       = taco_binarySearchBefore(C_pos, p_start, p_end, nnz_idx);
  auto j       = C_crd[nnz_idx];
  for (int64_t i = 0; i < idim; i++) { A_vals[{i, j}] <<= B_vals[{i, k}] * C_vals[nnz_idx]; }
}

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename ACC>
struct SpMMDenseCSRImplBody<VariantKind::GPU, INDEX_CODE, VAL_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(ACC A_vals,
                  const AccessorRO<VAL_TY, 2>& B_vals,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorRO<INDEX_TY, 1>& C_crd,
                  const AccessorRO<VAL_TY, 1>& C_vals,
                  const Rect<2>& rect,
                  const Rect<1>& nnz_rect)
  {
    // Use a DISTAL approach. Position space split on the non-zeros.
    auto stream = get_cached_stream();
    auto nnzs   = nnz_rect.volume();
    auto blocks = get_num_blocks_1d(nnzs);
    DeferredBuffer<int64_t, 1> buf({0, blocks}, Memory::GPU_FB_MEM);
    taco_binarySearchBeforeBlockLaunch(stream,
                                       C_pos,
                                       buf.ptr(0),
                                       rect.lo[1],
                                       rect.hi[1],
                                       THREADS_PER_BLOCK,
                                       THREADS_PER_BLOCK,
                                       blocks,
                                       nnz_rect.lo /* offset */
    );
    spmm_dense_csr_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(nnzs,
                                                                    nnz_rect.lo,
                                                                    buf.ptr(0),
                                                                    rect.hi[0] - rect.lo[0] + 1,
                                                                    A_vals,
                                                                    B_vals,
                                                                    C_pos,
                                                                    C_crd,
                                                                    C_vals);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void SpMMCSR::gpu_variant(TaskContext& context)
{
  spmm_template<VariantKind::GPU>(context);
}

/*static*/ void SpMMDenseCSR::gpu_variant(TaskContext& context)
{
  spmm_dense_csr_template<VariantKind::GPU>(context);
}

}  // namespace sparse

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

#include "sparse/array/conv/dense_to_csr.h"
#include "sparse/array/conv/dense_to_csr_template.inl"
#include "sparse/util/cusparse_utils.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename VAL_TY>
__global__ void denseToCSRNNZKernel(size_t rows,
                                    Rect<2> bounds,
                                    AccessorWO<nnz_ty, 1> A_nnz_acc,
                                    AccessorRO<VAL_TY, 2> B_vals_acc)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  coord_t i        = idx + bounds.lo[0];
  nnz_ty nnz_count = 0;
  for (coord_t j = bounds.lo[1]; j < bounds.hi[1] + 1; j++) {
    if (B_vals_acc[{i, j}] != static_cast<VAL_TY>(0.0)) { nnz_count++; }
  }
  A_nnz_acc[i] = nnz_count;
}

template <>
struct DenseToCSRNNZImpl<VariantKind::GPU> {
  template <LegateTypeCode VAL_CODE>
  void operator()(DenseToCSRNNZArgs& args) const
  {
    using VAL_TY = legate_type_of<VAL_CODE>;

    auto& nnz    = args.nnz;
    auto& B_vals = args.B_vals;

    // Break out early if the iteration space partition is empty.
    if (B_vals.domain().empty()) { return; }

    auto stream = get_cached_stream();

#if (CUSPARSE_VER_MAJOR < 11 || CUSPARSE_VER_MINOR < 2)
    auto B_domain = B_vals.domain();
    auto rows     = B_domain.hi()[0] - B_domain.lo()[0] + 1;
    auto blocks   = get_num_blocks_1d(rows);
    denseToCSRNNZKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      rows,
      Rect<2>(B_domain.lo(), B_domain.hi()),
      nnz.write_accessor<nnz_ty, 1>(),
      B_vals.read_accessor<VAL_TY, 2>());
    CHECK_CUDA_STREAM(stream);
#else
    // Get context sensitive objects.
    auto handle = get_cusparse();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    auto B_domain = B_vals.domain();
    auto rows     = B_domain.hi()[0] - B_domain.lo()[0] + 1;
    auto cols     = B_domain.hi()[1] - B_domain.lo()[1] + 1;
    // Allocate an output buffer for the offsets.
    DeferredBuffer<int64_t, 1> A_indptr({0, rows}, Memory::GPU_FB_MEM);

    // Construct the cuSPARSE objects from individual regions.
    auto cusparse_B = makeCuSparseDenseMat<VAL_TY>(B_vals);
    // We will construct the sparse matrix explicitly due to not having
    // all of the components right now.
    cusparseSpMatDescr_t cusparse_A;
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_A,
                                     rows,
                                     cols,
                                     0 /* nnz */,
                                     A_indptr.ptr(0),
                                     nullptr,
                                     nullptr,
                                     cusparseIndexType<int64_t>(),
                                     cusparseIndexType<int64_t>(),
                                     index_base,
                                     cusparseDataType<VAL_TY>()));
    // Now make cuSPARSE calls.
    size_t bufSize = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
      handle, cusparse_B, cusparse_A, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufSize));
    // Allocate a buffer if we need to.
    void* workspacePtr = nullptr;
    if (bufSize > 0) {
      DeferredBuffer<char, 1> buf({0, bufSize - 1}, Memory::GPU_FB_MEM);
      workspacePtr = buf.ptr(0);
    }
    // Do the analysis only to compute the indptr array.
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(
      handle, cusparse_B, cusparse_A, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, workspacePtr));
    // Destroy the created cuSPARSE objects.
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_A));
    CHECK_CUSPARSE(cusparseDestroyDnMat(cusparse_B));
    // Finally, convert the computed indptr array into an nnz array.
    {
      auto blocks = get_num_blocks_1d(rows);
      localIndptrToNnz<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, nnz.write_accessor<nnz_ty, 1>().ptr(nnz.domain().lo()), A_indptr.ptr(0));
    }
#endif

    CHECK_CUDA_STREAM(stream);
  }
};

template <typename INDEX_TY, typename VAL_TY>
__global__ void denseToCSRKernel(size_t rows,
                                 Rect<2> bounds,
                                 AccessorRW<Rect<1>, 1> A_pos_acc,
                                 AccessorWO<INDEX_TY, 1> A_crd_acc,
                                 AccessorWO<VAL_TY, 1> A_vals_acc,
                                 AccessorRO<VAL_TY, 2> B_vals_acc)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  coord_t i       = idx + bounds.lo[0];
  int64_t nnz_pos = A_pos_acc[i].lo;
  for (coord_t j = bounds.lo[1]; j < bounds.hi[1] + 1; j++) {
    if (B_vals_acc[{i, j}] != static_cast<VAL_TY>(0.0)) {
      A_crd_acc[nnz_pos]  = static_cast<INDEX_TY>(j);
      A_vals_acc[nnz_pos] = B_vals_acc[{i, j}];
      nnz_pos++;
    }
  }
}

template <>
struct DenseToCSRImpl<VariantKind::GPU> {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(DenseToCSRArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto& A_pos  = args.A_pos;
    auto& A_crd  = args.A_crd;
    auto& A_vals = args.A_vals;
    auto& B_vals = args.B_vals;

    // Break out early if the iteration space partition is empty.
    if (B_vals.domain().empty()) { return; }

    // Get context sensitive objects.
    auto stream = get_cached_stream();

    auto B_domain = B_vals.domain();
    auto rows     = B_domain.hi()[0] - B_domain.lo()[0] + 1;
    auto cols     = B_domain.hi()[1] - B_domain.lo()[1] + 1;
    auto blocks   = get_num_blocks_1d(rows);
    denseToCSRKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      rows,
      Rect<2>(B_domain.lo(), B_domain.hi()),
      A_pos.read_write_accessor<Rect<1>, 1>(),
      A_crd.write_accessor<INDEX_TY, 1>(),
      A_vals.write_accessor<VAL_TY, 1>(),
      B_vals.read_accessor<VAL_TY, 2>());
    CHECK_CUDA_STREAM(stream);

    // TODO (rohany): The below cuSPARSE code is buggy. In particular, it results
    //  in some column segments of the resulting CSR array to be unsorted, which is a
    //  violation of the CSR data structure.
    // // Construct the cuSPARSE objects from individual regions.
    // auto cusparse_A = makeCuSparseCSR(A_pos, A_crd, A_vals, cols);
    // auto cusparse_B = makeCuSparseDenseMat(B_vals);

    // // Now make cuSPARSE calls.
    // size_t bufSize = 0;
    // CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
    //     handle,
    //     cusparse_B,
    //     cusparse_A,
    //     CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
    //     &bufSize
    // ));
    // // Allocate a buffer if we need to.
    // void* workspacePtr = nullptr;
    // if (bufSize > 0) {
    //   DeferredBuffer<char, 1> buf({0, bufSize - 1}, Memory::GPU_FB_MEM);
    //   workspacePtr = buf.ptr(0);
    // }
    // // Do the analysis only to compute the indptr array.
    // CHECK_CUSPARSE(cusparseDenseToSparse_convert(
    //     handle,
    //     cusparse_B,
    //     cusparse_A,
    //     CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
    //     workspacePtr
    // ));
    // // Destroy the created cuSPARSE objects.
    // CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_A));
    // CHECK_CUSPARSE(cusparseDestroyDnMat(cusparse_B));
  }
};

/*static*/ void DenseToCSRNNZ::gpu_variant(TaskContext& context)
{
  dense_to_csr_nnz_template<VariantKind::GPU>(context);
}

/*static*/ void DenseToCSR::gpu_variant(TaskContext& context)
{
  dense_to_csr_template<VariantKind::GPU>(context);
}

}  // namespace sparse

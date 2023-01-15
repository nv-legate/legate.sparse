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

#include "sparse/array/csr/spgemm_csr_csr_csr.h"
#include "sparse/util/cusparse_utils.h"
#include "sparse/util/dispatch.h"

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

struct SpGEMMCSRxCSRxCSRGPUImplCuSparse {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpGEMMCSRxCSRxCSRGPUArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto& A_pos  = args.A_pos;
    auto& A_crd  = args.A_crd;
    auto& A_vals = args.A_vals;
    auto& B_pos  = args.B_pos;
    auto& B_crd  = args.B_crd;
    auto& B_vals = args.B_vals;
    auto& C_pos  = args.C_pos;
    auto& C_crd  = args.C_crd;
    auto& C_vals = args.C_vals;
    auto& A2_dim = args.A2_dim;

    // Due to limitations around the cuSPARSE SpGEMM API, we can't do the standard
    // symbolic and actual execution phases of SpGEMM. Instead, we'll have each GPU
    // task output a local CSR matrix, and then we'll collapse the results of each
    // task into a global CSR matrix in Python land. The computation here and
    // interaction with cuSPARSE has gone through several iterations, and has
    // settled on an implementation that avoids all pointer offsetting to be
    // non-trusting of what cuSPARSE may do when reading pointers. In this task,
    // we have a row-partitioned B matrix, and use an image from the coordinates
    // in each partition of B to construct a row partition of the C matrix. Instead
    // of offsetting any pointers, we'll attempt to construct two new local matrices
    // that we can pass to cuSPARSE that are themselves valid. In particular, we use
    // the fact that we took an image from B to construct a matrix B', where each
    // coordinate in B' has been offset from the minimum coordinate in each partition
    // of B. The range of min and max coordinates in B is exactly equal to the number
    // of rows of C. We use this to construct a related matrix of C named C' that
    // doesn't offset the arrays at all, but uses the results of the images directly,
    // as the referencing coordinates from B' have been offset already.

    // Get context sensitive objects.
    auto handle = get_cusparse();
    auto stream = get_cached_stream();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    auto B_rows      = B_pos.domain().get_volume();
    auto B_min_coord = C_pos.domain().lo()[0];
    auto B_max_coord = C_pos.domain().hi()[0];
    auto C_rows      = B_max_coord - B_min_coord + 1;

    // If there are no rows to process, then return empty output instances.
    if (B_rows == 0 || C_rows == 0 || B_crd.domain().empty() || C_crd.domain().empty()) {
      A_crd.create_output_buffer<INDEX_TY, 1>(0, true /* return_data */);
      A_vals.create_output_buffer<VAL_TY, 1>(0, true /* return_data */);
      return;
    }

    // Convert the pos arrays into local indptr arrays.
    DeferredBuffer<int32_t, 1> B_indptr({0, B_rows}, Memory::GPU_FB_MEM);
    DeferredBuffer<int32_t, 1> C_indptr({0, C_rows}, Memory::GPU_FB_MEM);
    {
      auto blocks = get_num_blocks_1d(B_rows);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        B_rows, B_pos.read_accessor<Rect<1>, 1>().ptr(B_pos.domain().lo()), B_indptr.ptr(0));
    }
    {
      auto blocks = get_num_blocks_1d(C_rows);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        C_rows, C_pos.read_accessor<Rect<1>, 1>().ptr(C_pos.domain().lo()), C_indptr.ptr(0));
    }

    DeferredBuffer<int32_t, 1> B_crd_int({0, B_crd.domain().get_volume() - 1}, Memory::GPU_FB_MEM);
    // Importantly, don't use the volume for C, as the image optimization
    // is being applied. Compute an upper bound on the volume directly.
    auto C_nnz = C_crd.domain().hi()[0] - C_crd.domain().lo()[0] + 1;
    DeferredBuffer<int32_t, 1> C_crd_int({0, C_nnz - 1}, Memory::GPU_FB_MEM);
    {
      auto dom    = B_crd.domain();
      auto elems  = dom.get_volume();
      auto blocks = get_num_blocks_1d(elems);
      cast_and_offset<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        elems, B_crd_int.ptr(0), B_crd.read_accessor<INDEX_TY, 1>().ptr(dom.lo()), B_min_coord);
    }
    {
      auto blocks = get_num_blocks_1d(C_nnz);
      cast<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        C_nnz, C_crd_int.ptr(0), C_crd.read_accessor<INDEX_TY, 1>().ptr(C_crd.domain().lo()));
    }

    // Initialize the cuSPARSE matrices.
    cusparseSpMatDescr_t cusparse_A, cusparse_B, cusparse_C;
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_B,
                                     B_rows,
                                     C_rows /* cols */,
                                     B_crd.domain().get_volume() /* nnz */,
                                     B_indptr.ptr(0),
                                     B_crd_int.ptr(0),
                                     getPtrFromStore<VAL_TY, 1>(B_vals),
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cusparseDataType<VAL_TY>()));
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_C,
                                     C_rows,
                                     A2_dim /* cols */,
                                     C_nnz,
                                     C_indptr.ptr(0),
                                     C_crd_int.ptr(0),
                                     (VAL_TY*)getPtrFromStore<VAL_TY, 1>(C_vals),
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cusparseDataType<VAL_TY>()));
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_A,
                                     B_rows /* rows */,
                                     A2_dim /* cols */,
                                     0 /* nnz */,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cusparseDataType<VAL_TY>()));

    // Allocate the SpGEMM descriptor.
    cusparseSpGEMMDescr_t descr;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&descr));

    VAL_TY alpha       = static_cast<VAL_TY>(1);
    VAL_TY beta        = static_cast<VAL_TY>(0);
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 cusparse_B,
                                                 cusparse_C,
                                                 &beta,
                                                 cusparse_A,
                                                 cusparseDataType<VAL_TY>(),
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 descr,
                                                 &bufferSize1,
                                                 nullptr));
    void* buffer1 = nullptr;
    if (bufferSize1 > 0) {
      DeferredBuffer<char, 1> buf({0, bufferSize1 - 1}, Memory::GPU_FB_MEM);
      buffer1 = buf.ptr(0);
    }
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 cusparse_B,
                                                 cusparse_C,
                                                 &beta,
                                                 cusparse_A,
                                                 cusparseDataType<VAL_TY>(),
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 descr,
                                                 &bufferSize1,
                                                 buffer1));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          cusparse_B,
                                          cusparse_C,
                                          &beta,
                                          cusparse_A,
                                          cusparseDataType<VAL_TY>(),
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          descr,
                                          &bufferSize2,
                                          nullptr));
    void* buffer2 = nullptr;
    if (bufferSize2 > 0) {
      DeferredBuffer<char, 1> buf({0, bufferSize2 - 1}, Memory::GPU_FB_MEM);
      buffer2 = buf.ptr(0);
    }
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          cusparse_B,
                                          cusparse_C,
                                          &beta,
                                          cusparse_A,
                                          cusparseDataType<VAL_TY>(),
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          descr,
                                          &bufferSize2,
                                          buffer2));
    // Allocate buffers for the 32-bit version of the A matrix.
    int64_t A_rows, A_cols, A_nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(cusparse_A, &A_rows, &A_cols, &A_nnz));
    DeferredBuffer<int32_t, 1> A_indptr({0, A_rows}, Memory::GPU_FB_MEM);
    // Handle the creation of the A_crd buffer depending on whether the result
    // type is the type of data we are supposed to create.
    DeferredBuffer<int32_t, 1> A_crd_int;
    if constexpr (INDEX_CODE == LegateTypeCode::INT32_LT) {
      A_crd_int = A_crd.create_output_buffer<INDEX_TY, 1>(A_nnz, true /* return_buffer */);
    } else {
      A_crd_int = DeferredBuffer<int32_t, 1>({0, A_nnz - 1}, Memory::GPU_FB_MEM);
    }
    auto A_vals_acc = A_vals.create_output_buffer<VAL_TY, 1>(A_nnz, true /* return_buffer */);
    CHECK_CUSPARSE(
      cusparseCsrSetPointers(cusparse_A, A_indptr.ptr(0), A_crd_int.ptr(0), A_vals_acc.ptr(0)));
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha,
                                       cusparse_B,
                                       cusparse_C,
                                       &beta,
                                       cusparse_A,
                                       cusparseDataType<VAL_TY>(),
                                       CUSPARSE_SPGEMM_DEFAULT,
                                       descr));

    // Convert the A_indptr array into a pos array.
    {
      auto blocks = get_num_blocks_1d(A_rows);
      localIndptrToPos<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        A_rows, A_pos.write_accessor<Rect<1>, 1>().ptr(A_pos.domain().lo()), A_indptr.ptr(0));
    }
    // Cast the A coordinates back into 64 bits, if that is the desired
    // data type.
    if constexpr (INDEX_CODE != LegateTypeCode::INT32_LT) {
      auto blocks = get_num_blocks_1d(A_nnz);
      auto buf    = A_crd.create_output_buffer<INDEX_TY, 1>(A_nnz, true /* return_buffer */);
      cast<INDEX_TY, int32_t>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(A_nnz, buf.ptr(0), A_crd_int.ptr(0));
    }

    // Destroy all of the resources that we allocated.
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(descr));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_A));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_B));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_C));
    CHECK_CUDA_STREAM(stream);
  }
};

struct SpGEMMCSRxCSRxCSRGPUImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpGEMMCSRxCSRxCSRGPUArgs& args) const
  {
    if constexpr (cusparseSupportsType<legate_type_of<VAL_CODE>>()) {
      SpGEMMCSRxCSRxCSRGPUImplCuSparse{}.template operator()<INDEX_CODE, VAL_CODE>(args);
    } else {
      assert(false && "Type unsupported for GPU execution.");
    }
  }
};

/*static*/ void SpGEMMCSRxCSRxCSRGPU::gpu_variant(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  SpGEMMCSRxCSRxCSRGPUArgs args{
    outputs[0],
    outputs[1],
    outputs[2],
    inputs[0],
    inputs[1],
    inputs[2],
    inputs[3],
    inputs[4],
    inputs[5],
    context.scalars()[0].value<uint64_t>(),
    context.scalars()[1].value<uint64_t>(),
  };
  index_type_value_type_dispatch(
    args.A_crd.code(), args.A_vals.code(), SpGEMMCSRxCSRxCSRGPUImpl{}, args);
}

}  // namespace sparse

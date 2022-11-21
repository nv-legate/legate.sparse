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

#include "sparse/array/csr/spgemm_csr_csr_csc.h"
#include "sparse/array/csr/spgemm_csr_csr_csc_template.inl"
#include "sparse/util/cusparse_utils.h"
#include "thrust_allocator.h"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename INDEX_TY>
__global__ void offset_coordinates_to_global(size_t elems, coord_t offset, INDEX_TY* coords)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) return;
  coords[idx] += offset;
}

template <>
struct SpGEMMCSRxCSRxCSCLocalTilesImpl<VariantKind::GPU> {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpGEMMCSRxCSRxCSCLocalTilesArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto& A_pos    = args.A_pos;
    auto& A_crd    = args.A_crd;
    auto& A_vals   = args.A_vals;
    auto& B_pos    = args.B_pos;
    auto& B_crd    = args.B_crd;
    auto& B_vals   = args.B_vals;
    auto& C_pos    = args.C_pos;
    auto& C_crd    = args.C_crd;
    auto& C_vals   = args.C_vals;
    int64_t C_rows = args.C_rows;

    // Get context sensitive objects.
    auto handle = get_cusparse();
    auto stream = get_cached_stream();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    // Par for the course, neither the conversion routines or SpGEMM computations
    // allow for 64-bit integer coordinates. So we have to convert everything
    // into 64-bit integers. Next, the SpGEMM algorithm only supports CSR matrices!
    // So we have to convert the C CSC matrix into a local tile of CSR before we can
    // even do the multiply.
    auto B_rows = B_pos.domain().get_volume();
    auto C_cols = C_pos.domain().get_volume();

    // Cast the pos stores to indptr arrays.
    DeferredBuffer<int32_t, 1> B_indptr({0, B_rows}, Memory::GPU_FB_MEM);
    DeferredBuffer<int32_t, 1> C_indptr({0, C_cols}, Memory::GPU_FB_MEM);
    {
      auto blocks = get_num_blocks_1d(B_rows);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        B_rows, B_pos.read_accessor<Rect<1>, 1>().ptr(B_pos.domain().lo()), B_indptr.ptr(0));
    }
    {
      auto blocks = get_num_blocks_1d(C_cols);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        C_cols, C_pos.read_accessor<Rect<1>, 1>().ptr(C_pos.domain().lo()), C_indptr.ptr(0));
    }

    // Optionally cast the coordinates if they are already not already 32-bit.
    int32_t* B_crd_int = nullptr;
    int32_t* C_crd_int = nullptr;
    if constexpr (INDEX_CODE == LegateTypeCode::INT32_LT) {
      B_crd_int = const_cast<int32_t*>(B_crd.read_accessor<INDEX_TY, 1>().ptr(B_crd.domain().lo()));
      C_crd_int = const_cast<int32_t*>(C_crd.read_accessor<INDEX_TY, 1>().ptr(C_crd.domain().lo()));
    } else {
      DeferredBuffer<int32_t, 1> B_crd_int_buf({0, B_crd.domain().get_volume() - 1},
                                               Memory::GPU_FB_MEM);
      DeferredBuffer<int32_t, 1> C_crd_int_buf({0, C_crd.domain().get_volume() - 1},
                                               Memory::GPU_FB_MEM);
      {
        auto dom    = B_crd.domain();
        auto elems  = dom.get_volume();
        auto blocks = get_num_blocks_1d(elems);
        cast<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          elems, B_crd_int_buf.ptr(0), B_crd.read_accessor<INDEX_TY, 1>().ptr(dom.lo()));
      }
      {
        auto dom    = C_crd.domain();
        auto elems  = dom.get_volume();
        auto blocks = get_num_blocks_1d(elems);
        cast<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          elems, C_crd_int_buf.ptr(0), C_crd.read_accessor<INDEX_TY, 1>().ptr(dom.lo()));
      }
      B_crd_int = B_crd_int_buf.ptr(0);
      C_crd_int = C_crd_int_buf.ptr(0);
    }

    // Now, we can start the conversion of C from CSC to CSR. The method is called
    // CSR2CSC, so we can use it in the reverse way also by doing CSC2CSR.
    // First, allocate buffers for the resulting C CSR data.
    DeferredBuffer<int32_t, 1> C_CSR_indptr({0, C_rows}, Memory::GPU_FB_MEM);
    DeferredBuffer<int32_t, 1> C_CSR_crd({0, C_crd.domain().get_volume() - 1}, Memory::GPU_FB_MEM);
    DeferredBuffer<VAL_TY, 1> C_CSR_vals({0, C_crd.domain().get_volume() - 1}, Memory::GPU_FB_MEM);
    size_t convBufSize = 0;
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(handle,
                                                 // Because we're treating the CSC matrix as a CSR
                                                 // matrix, we have to reverse the columns and rows.
                                                 C_cols,
                                                 C_rows,
                                                 C_crd.domain().get_volume(),
                                                 getPtrFromStore<VAL_TY, 1>(C_vals),
                                                 C_indptr.ptr(0),
                                                 C_crd_int,
                                                 C_CSR_vals.ptr(0),
                                                 C_CSR_indptr.ptr(0),
                                                 C_CSR_crd.ptr(0),
                                                 cusparseDataType<VAL_TY>(),
                                                 CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG1,
                                                 &convBufSize));
    DeferredBuffer<char*, 1> convBuffer({0, convBufSize - 1}, Memory::GPU_FB_MEM);
    CHECK_CUSPARSE(cusparseCsr2cscEx2(handle,
                                      // Look above for reasoning about the size reversal.
                                      C_cols,
                                      C_rows,
                                      C_crd.domain().get_volume(),
                                      getPtrFromStore<VAL_TY, 1>(C_vals),
                                      C_indptr.ptr(0),
                                      C_crd_int,
                                      C_CSR_vals.ptr(0),
                                      C_CSR_indptr.ptr(0),
                                      C_CSR_crd.ptr(0),
                                      cusparseDataType<VAL_TY>(),
                                      CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1,
                                      convBuffer.ptr(0)));
    // We don't need this instance anymore.
    convBuffer.destroy();
    // Now we can do the SpGEMM. First, create all of the cusparse matrices.
    cusparseSpMatDescr_t cusparse_A, cusparse_B, cusparse_C;
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_B,
                                     B_rows,
                                     C_rows /* cols */,
                                     B_crd.domain().get_volume() /* nnz */,
                                     B_indptr.ptr(0),
                                     B_crd_int,
                                     getPtrFromStore<VAL_TY, 1>(B_vals),
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cusparseDataType<VAL_TY>()));
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_C,
                                     C_rows,
                                     C_cols,
                                     C_crd.domain().get_volume() /* nnz */,
                                     C_CSR_indptr.ptr(0),
                                     C_CSR_crd.ptr(0),
                                     C_CSR_vals.ptr(0),
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     cusparseDataType<VAL_TY>()));
    CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_A,
                                     B_rows /* rows */,
                                     C_cols /* cols */,
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
      auto buf    = A_pos.create_output_buffer<Rect<1>, 1>(A_rows, true /* return_buffer */);
      localIndptrToPos<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        A_rows, buf.ptr(0), A_indptr.ptr(0));
    }
    // Optionally cast the A_crd coordinates into the appropriate type.
    // We need to offset the resulting coordinates into the global space.
    // cuSPARSE is going to compute a resulting matrix where all the coordinates
    // are zero-based, but we need the coordinates to be global addressable, so we
    // offset them by the partition of the column space that we are in.
    if constexpr (INDEX_CODE != LegateTypeCode::INT32_LT) {
      auto blocks = get_num_blocks_1d(A_nnz);
      auto buf    = A_crd.create_output_buffer<INDEX_TY, 1>(A_nnz, true /* return_buffer */);
      cast<INDEX_TY, int32_t>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(A_nnz, buf.ptr(0), A_crd_int.ptr(0));
      offset_coordinates_to_global<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        A_nnz, C_pos.domain().lo()[0], buf.ptr(0));
    } else {
      auto blocks = get_num_blocks_1d(A_nnz);
      offset_coordinates_to_global<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        A_nnz, C_pos.domain().lo()[0], A_crd_int.ptr(0));
    }
    CHECK_CUDA_STREAM(stream);
  }
};

template <>
struct SpGEMMCSRxCSRxCSCCommComputeImplBody<VariantKind::GPU> {
  void operator()(const AccessorWO<Rect<1>, 3>& out,
                  const Store& pos,
                  const Store& global_pos,
                  const int32_t gx,
                  const int32_t gy,
                  const Point<3>& output_point)
  {
    auto oi                = output_point[0];
    auto oj                = output_point[1];
    auto global_pos_domain = global_pos.domain();
    auto i_lo              = pos.domain().lo()[0];
    auto i_hi              = pos.domain().hi()[0] + 1;
    auto i_tile_size       = i_hi - i_lo;
    auto tile_size         = (i_hi - i_lo + gy - 1) / gy;
#pragma omp parallel for schedule(static)
    for (int32_t j = 0; j < gy; j++) {
      auto sub_tile_start = j * tile_size;
      auto sub_tile_end   = std::min((j + 1) * tile_size, i_tile_size);
      auto lo             = global_pos_domain.lo()[0] + sub_tile_start;
      auto hi             = global_pos_domain.lo()[0] + sub_tile_end;
      out[{oi, oj, j}]    = Rect<1>{lo, hi - 1};
    }
  }
};

__global__ void calculate_copy_sizes(size_t total_rows,
                                     size_t num_rects,
                                     DeferredBuffer<Rect<1>, 1> rects,
                                     DeferredBuffer<size_t, 1> row_offsets,
                                     const AccessorRO<Rect<1>, 1> global_pos_acc)
{
  const auto idx = global_tid_1d();
  if (idx >= total_rows) return;
  size_t elems = 0;
  for (size_t i = 0; i < num_rects; i++) {
    auto rect           = rects[i];
    auto global_pos_idx = rect.lo + idx;
    if (rect.contains(global_pos_idx)) { elems += global_pos_acc[global_pos_idx].volume(); }
  }
  row_offsets[idx] = elems;
}

template <typename INDEX_TY, typename VAL_TY>
__global__ void scatter_rows(size_t total_rows,
                             size_t num_rects,
                             DeferredBuffer<Rect<1>, 1> rects,
                             DeferredBuffer<size_t, 1> row_offsets,
                             const AccessorRO<Rect<1>, 1> global_pos_acc,
                             const AccessorRO<INDEX_TY, 1> global_crd_acc,
                             const AccessorRO<VAL_TY, 1> global_vals_acc,
                             DeferredBuffer<Rect<1>, 1> pos_acc,
                             DeferredBuffer<INDEX_TY, 1> crd_acc,
                             DeferredBuffer<VAL_TY, 1> vals_acc)
{
  const auto idx = global_tid_1d();
  if (idx >= total_rows) return;
  auto offset = row_offsets[idx];
  auto lo     = offset;
  for (size_t i = 0; i < num_rects; i++) {
    auto rect           = rects[i];
    auto global_pos_idx = rect.lo + idx;
    if (rect.contains(global_pos_idx)) {
      for (int64_t pos = global_pos_acc[global_pos_idx].lo;
           pos < global_pos_acc[global_pos_idx].hi + 1;
           pos++) {
        crd_acc[offset]  = global_crd_acc[pos];
        vals_acc[offset] = global_vals_acc[pos];
        offset++;
      }
    }
  }
  auto hi      = offset - 1;
  pos_acc[idx] = {lo, hi};
}

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpGEMMCSRxCSRxCSCShuffleImplBody<VariantKind::GPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(Store& out_pos,
                  Store& out_crd,
                  Store& out_vals,
                  const AccessorRO<Rect<1>, 1>& global_pos,
                  const AccessorRO<INDEX_TY, 1>& global_crd,
                  const AccessorRO<VAL_TY, 1>& global_vals,
                  const Domain& global_pos_domain)
  {
    auto stream = get_cached_stream();
    // Collect all rectangles in the global_pos domain.
    std::vector<Rect<1>> rects_cpu;
    size_t total_nnzs = 0;
    size_t total_rows = 0;
    for (RectInDomainIterator<1> itr(global_pos_domain); itr(); itr++) {
      rects_cpu.push_back(*itr);
      total_rows = std::max(itr->volume(), total_rows);
      if (itr->empty()) continue;
      Rect<1> lo, hi;
      cudaMemcpy(&lo, global_pos.ptr(itr->lo), sizeof(Rect<1>), cudaMemcpyDeviceToHost);
      cudaMemcpy(&hi, global_pos.ptr(itr->hi), sizeof(Rect<1>), cudaMemcpyDeviceToHost);
      total_nnzs += hi.hi[0] - lo.lo[0] + 1;
    }
    // Allocate our output buffers.
    auto pos_acc  = out_pos.create_output_buffer<Rect<1>, 1>(total_rows, true /* return_buffer */);
    auto crd_acc  = out_crd.create_output_buffer<INDEX_TY, 1>(total_nnzs, true /* return_buffer */);
    auto vals_acc = out_vals.create_output_buffer<VAL_TY, 1>(total_nnzs, true /* return_buffer */);

    // We'll start with a simple row-based parallelization for our copies. If/when performance
    // suffers due to this, we can think about algorithms for a full-data based parallelization.
    DeferredBuffer<size_t, 1> row_offsets({0, total_rows - 1}, Memory::GPU_FB_MEM);
    DeferredBuffer<Rect<1>, 1> rects({0, rects_cpu.size() - 1}, Memory::GPU_FB_MEM);
    cudaMemcpy(
      rects.ptr(0), rects_cpu.data(), sizeof(Rect<1>) * rects_cpu.size(), cudaMemcpyHostToDevice);
    auto blocks = get_num_blocks_1d(total_rows);
    calculate_copy_sizes<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      total_rows, rects_cpu.size(), rects, row_offsets, global_pos);
    // Scan over the counts to find the offsets for each row.
    ThrustAllocator alloc(Memory::GPU_FB_MEM);
    auto policy = thrust::cuda::par(alloc).on(stream);
    thrust::exclusive_scan(
      policy, row_offsets.ptr(0), row_offsets.ptr(0) + total_rows, row_offsets.ptr(0));
    // Perform the final scatter/gather.
    scatter_rows<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(total_rows,
                                                           rects_cpu.size(),
                                                           rects,
                                                           row_offsets,
                                                           global_pos,
                                                           global_crd,
                                                           global_vals,
                                                           pos_acc,
                                                           crd_acc,
                                                           vals_acc);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void SpGEMMCSRxCSRxCSCLocalTiles::gpu_variant(TaskContext& context)
{
  spgemm_csr_csr_csc_local_tiles_template<VariantKind::GPU>(context);
}

/*static*/ void SpGEMMCSRxCSRxCSCCommCompute::gpu_variant(TaskContext& context)
{
  spgemm_csr_csr_csc_comm_compute_template<VariantKind::GPU>(context);
}

/*static*/ void SpGEMMCSRxCSRxCSCShuffle::gpu_variant(TaskContext& context)
{
  spgemm_csr_csr_csc_shuffle_template<VariantKind::GPU>(context);
}

}  // namespace sparse

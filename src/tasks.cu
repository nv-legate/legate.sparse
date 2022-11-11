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

#include "sparse.h"
#include "tasks.h"
#include "cuda_help.h"
#include "pitches.h"
#include "distal_cuda_utils.h"
#include "sparse/util/cusparse_utils.h"

#include "thrust_allocator.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace sparse {

using namespace legate;
using namespace Legion;

const cudaDataType_t cuda_val_ty   = CUDA_R_64F;
const cusparseIndexType_t index_ty = CUSPARSE_INDEX_64I;

// We also include an overload that does not specify the values.
cusparseSpMatDescr_t makeCuSparseCSR(Store& pos, Store& crd, size_t cols)
{
  cusparseSpMatDescr_t matDescr;
  auto stream = get_cached_stream();

  auto pos_domain = pos.domain();
  auto crd_domain = crd.domain();

  auto pos_acc = pos.read_accessor<Rect<1>, 1>();
  size_t rows  = pos_domain.get_volume();
  DeferredBuffer<int64_t, 1> indptr({0, rows}, Memory::GPU_FB_MEM);
  auto blocks = get_num_blocks_1d(rows);
  convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    rows, pos_acc.ptr(pos_domain.lo()), indptr.ptr(0));

  CHECK_CUSPARSE(cusparseCreateCsr(&matDescr,
                                   rows,
                                   cols,
                                   crd_domain.get_volume(), /* nnz */
                                   (void*)indptr.ptr(0),
                                   crd_domain.empty() ? nullptr : getPtrFromStore<coord_ty, 1>(crd),
                                   nullptr,
                                   index_ty,
                                   index_ty,
                                   index_base,
                                   cuda_val_ty));

  return matDescr;
}

__global__ void tropical_spmv_kernel(size_t rows,
                                     coord_ty offset,
                                     coord_ty num_fields,
                                     AccessorWO<coord_ty, 2> y,
                                     AccessorRO<Rect<1>, 1> pos,
                                     AccessorRO<coord_ty, 1> crd,
                                     AccessorRO<coord_ty, 2> x)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  coord_ty i = idx + offset;
  // Initialize the output.
  for (coord_ty f = 0; f < num_fields; f++) { y[{i, f}] = 0; }
  for (size_t jpos = pos[i].lo; jpos < pos[i].hi + 1; jpos++) {
    auto j         = crd[jpos];
    bool y_greater = true;
    for (coord_ty f = 0; f < num_fields; f++) {
      if (y[{i, f}] > x[{j, f}]) {
        y_greater = true;
        break;
      } else if (y[{i, f}] < x[{j, f}]) {
        y_greater = false;
        break;
      }
      // Else the fields are equal, so move onto the next field.
    }
    if (!y_greater) {
      for (coord_ty f = 0; f < num_fields; f++) { y[{i, f}] = x[{j, f}]; }
    }
  }
}

void CSRSpMVRowSplitTropicalSemiring::gpu_variant(legate::TaskContext& ctx)
{
  auto& y   = ctx.outputs()[0];
  auto& pos = ctx.inputs()[0];
  auto& crd = ctx.inputs()[1];
  auto& x   = ctx.inputs()[2];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (pos.transformed()) { pos.remove_transform(); }
  auto stream = get_cached_stream();

  // Break out if there aren't any rows.
  if (pos.domain().empty()) { return; }

  auto num_fields = x.domain().hi()[1] - x.domain().lo()[1] + 1;
  auto blocks     = get_num_blocks_1d(pos.domain().get_volume());
  // Since we can't use cuSPARSE over this semiring, we'll implement
  // a simple row-based kernel.
  tropical_spmv_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(pos.domain().get_volume(),
                                                                 pos.domain().lo()[0],
                                                                 num_fields,
                                                                 y.write_accessor<coord_ty, 2>(),
                                                                 pos.read_accessor<Rect<1>, 1>(),
                                                                 crd.read_accessor<coord_ty, 1>(),
                                                                 x.read_accessor<coord_ty, 2>());
  CHECK_CUDA_STREAM(stream);
}

template <typename T>
__global__ void localIndptrToPos(size_t rows, Rect<1>* out, T* in)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  out[idx] = {in[idx], in[idx + 1] - 1};
}

// Due to limitations around the cuSPARSE SpGEMM API, we can't do the standard
// symbolic and actual execution phases of SpGEMM. Instead, we'll have each GPU
// task output a local CSR matrix, and then we'll collapse the results of each
// task into a global CSR matrix in Python land.
void SpGEMMCSRxCSRxCSRGPU::gpu_variant(legate::TaskContext& ctx)
{
  auto A2_dim  = ctx.scalars()[0].value<size_t>();
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos  = ctx.inputs()[3];
  auto& C_crd  = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  // Get context sensitive objects.
  auto handle = get_cusparse();
  auto stream = get_cached_stream();
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  auto B_rows = B_pos.domain().get_volume();
  auto C_rows = C_pos.domain().get_volume();

  // If there are no rows to process, then return empty output instances.
  if (B_rows == 0) {
    A_crd.create_output_buffer<coord_ty, 1>(0, true /* return_data */);
    A_vals.create_output_buffer<val_ty, 1>(0, true /* return_data */);
    return;
  }

  // We have to cast our coordinates into 32 bit integers, since cuSPARSE does not
  // support 64 bit coordinates. While we're doing it, we also must convert the pos
  // regions into cuSPARSE indptr arrays.
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
  auto C_nnz = C_crd.domain().hi()[0] - C_crd.domain().lo()[0] + 1;
  DeferredBuffer<int32_t, 1> C_crd_int({0, C_nnz - 1}, Memory::GPU_FB_MEM);
  {
    auto dom    = B_crd.domain();
    auto elems  = dom.get_volume();
    auto blocks = get_num_blocks_1d(elems);
    cast<int, coord_ty><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      elems, B_crd_int.ptr(0), B_crd.read_accessor<coord_ty, 1>().ptr(dom.lo()));
  }
  {
    auto blocks = get_num_blocks_1d(C_nnz);
    cast<int, coord_ty><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      C_nnz, C_crd_int.ptr(0), C_crd.read_accessor<coord_ty, 1>().ptr(C_crd.domain().lo()));
  }
  cusparseSpMatDescr_t cusparse_A, cusparse_B, cusparse_C;
  CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_B,
                                   B_rows,
                                   C_rows /* cols */,
                                   B_crd.domain().get_volume() /* nnz */,
                                   B_indptr.ptr(0),
                                   B_crd_int.ptr(0),
                                   getPtrFromStore<val_ty, 1>(B_vals),
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_64F));
  CHECK_CUSPARSE(
    cusparseCreateCsr(&cusparse_C,
                      C_rows,
                      A2_dim /* cols */,
                      C_nnz,
                      C_indptr.ptr(0),
                      C_crd_int.ptr(0) - C_crd.domain().lo()[0],
                      (val_ty*)getPtrFromStore<val_ty, 1>(C_vals) - C_vals.domain().lo()[0],
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_A,
                                   B_rows /* rows */,
                                   A2_dim /* cols */,
                                   0 /* nnz */,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   index_base,
                                   cuda_val_ty));

  // Allocate the SpGEMM descriptor.
  cusparseSpGEMMDescr_t descr;
  CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&descr));

  val_ty alpha = 1.0, beta = 0.0;
  size_t bufferSize1 = 0, bufferSize2 = 0;
  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               cusparse_B,
                                               cusparse_C,
                                               &beta,
                                               cusparse_A,
                                               cuda_val_ty,
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
                                               cuda_val_ty,
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
                                        cuda_val_ty,
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
                                        cuda_val_ty,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        descr,
                                        &bufferSize2,
                                        buffer2));
  // Allocate buffers for the 32-bit version of the A matrix.
  int64_t A_rows, A_cols, A_nnz;
  CHECK_CUSPARSE(cusparseSpMatGetSize(cusparse_A, &A_rows, &A_cols, &A_nnz));
  DeferredBuffer<int32_t, 1> A_indptr({0, A_rows}, Memory::GPU_FB_MEM);
  DeferredBuffer<int32_t, 1> A_crd_int({0, A_nnz - 1}, Memory::GPU_FB_MEM);
  auto A_vals_acc = A_vals.create_output_buffer<val_ty, 1>(A_nnz, true /* return_buffer */);
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
                                     cuda_val_ty,
                                     CUSPARSE_SPGEMM_DEFAULT,
                                     descr));

  // Convert the A_indptr array into a pos array.
  {
    auto blocks = get_num_blocks_1d(A_rows);
    localIndptrToPos<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      A_rows, A_pos.write_accessor<Rect<1>, 1>().ptr(A_pos.domain().lo()), A_indptr.ptr(0));
  }
  // Cast the A coordinates back into 64 bits.
  {
    auto blocks = get_num_blocks_1d(A_nnz);
    auto buf    = A_crd.create_output_buffer<coord_ty, 1>(A_nnz, true /* return_buffer */);
    cast<coord_ty, int>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(A_nnz, buf.ptr(0), A_crd_int.ptr(0));
  }

  // Destroy all of the resources that we allocated.
  CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(descr));
  CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_A));
  CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_B));
  CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_C));
  CHECK_CUDA_STREAM(stream);
}

__global__ void offset_coordinates_to_global(size_t elems, coord_ty offset, coord_ty* coords)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) return;
  coords[idx] += offset;
}

void SpGEMMCSRxCSRxCSCLocalTiles::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];

  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos  = ctx.inputs()[3];
  auto& C_crd  = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  // TODO (rohany): Make sure to package these scalars in to the task.
  int64_t C_rows = ctx.scalars()[0].value<int64_t>();

  // Get context sensitive objects.
  auto handle = get_cusparse();
  auto stream = get_cached_stream();
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  // Remove the transformations on the B_pos and C_pos stores.
  B_pos.remove_transform();
  C_pos.remove_transform();

  // Par for the course, neither the conversion routines or SpGEMM computations
  // allow for 64-bit integer coordinates. So we have to convert everything
  // into 64-bit integers. Next, the SpGEMM algorithm only supports CSR matrices!
  // So we have to convert the C CSC matrix into a local tile of CSR before we can
  // even do the multiply.
  auto B_rows = B_pos.domain().get_volume();
  auto C_cols = C_pos.domain().get_volume();

  // Start doing the casts.
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
  DeferredBuffer<int32_t, 1> B_crd_int({0, B_crd.domain().get_volume() - 1}, Memory::GPU_FB_MEM);
  DeferredBuffer<int32_t, 1> C_crd_int({0, C_crd.domain().get_volume() - 1}, Memory::GPU_FB_MEM);
  {
    auto dom    = B_crd.domain();
    auto elems  = dom.get_volume();
    auto blocks = get_num_blocks_1d(elems);
    cast<int, coord_ty><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      elems, B_crd_int.ptr(0), B_crd.read_accessor<coord_ty, 1>().ptr(dom.lo()));
  }
  {
    auto dom    = C_crd.domain();
    auto elems  = dom.get_volume();
    auto blocks = get_num_blocks_1d(elems);
    cast<int, coord_ty><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      elems, C_crd_int.ptr(0), C_crd.read_accessor<coord_ty, 1>().ptr(dom.lo()));
  }
  // Now, we can start the conversion of C from CSC to CSR. The method is called
  // CSR2CSC, so we can use it in the reverse way also by doing CSC2CSR.
  // First, allocate buffers for the resulting C CSR data.
  DeferredBuffer<int32_t, 1> C_CSR_indptr({0, C_rows}, Memory::GPU_FB_MEM);
  DeferredBuffer<int32_t, 1> C_CSR_crd({0, C_crd.domain().get_volume() - 1}, Memory::GPU_FB_MEM);
  DeferredBuffer<val_ty, 1> C_CSR_vals({0, C_crd.domain().get_volume() - 1}, Memory::GPU_FB_MEM);
  size_t convBufSize = 0;
  CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(handle,
                                               // Because we're treating the CSC matrix as a CSR
                                               // matrix, we have to reverse the columns and rows.
                                               C_cols,
                                               C_rows,
                                               C_crd.domain().get_volume(),
                                               getPtrFromStore<val_ty, 1>(C_vals),
                                               C_indptr.ptr(0),
                                               C_crd_int.ptr(0),
                                               C_CSR_vals.ptr(0),
                                               C_CSR_indptr.ptr(0),
                                               C_CSR_crd.ptr(0),
                                               cuda_val_ty,
                                               CUSPARSE_ACTION_NUMERIC,
                                               index_base,
                                               CUSPARSE_CSR2CSC_ALG1,
                                               &convBufSize));
  DeferredBuffer<char*, 1> convBuffer({0, convBufSize - 1}, Memory::GPU_FB_MEM);
  CHECK_CUSPARSE(cusparseCsr2cscEx2(handle,
                                    // Look above for reasoning about the size reversal.
                                    C_cols,
                                    C_rows,
                                    C_crd.domain().get_volume(),
                                    getPtrFromStore<val_ty, 1>(C_vals),
                                    C_indptr.ptr(0),
                                    C_crd_int.ptr(0),
                                    C_CSR_vals.ptr(0),
                                    C_CSR_indptr.ptr(0),
                                    C_CSR_crd.ptr(0),
                                    cuda_val_ty,
                                    CUSPARSE_ACTION_NUMERIC,
                                    index_base,
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
                                   B_crd_int.ptr(0),
                                   getPtrFromStore<val_ty, 1>(B_vals),
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   index_base,
                                   cuda_val_ty));
  CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_C,
                                   C_rows,
                                   C_cols,
                                   C_crd.domain().get_volume() /* nnz */,
                                   C_CSR_indptr.ptr(0),
                                   C_CSR_crd.ptr(0),
                                   C_CSR_vals.ptr(0),
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   index_base,
                                   cuda_val_ty));
  CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_A,
                                   B_rows /* rows */,
                                   C_cols /* cols */,
                                   0 /* nnz */,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   index_base,
                                   cuda_val_ty));

  // Allocate the SpGEMM descriptor.
  cusparseSpGEMMDescr_t descr;
  CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&descr));

  val_ty alpha = 1.0, beta = 0.0;
  size_t bufferSize1 = 0, bufferSize2 = 0;
  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               cusparse_B,
                                               cusparse_C,
                                               &beta,
                                               cusparse_A,
                                               cuda_val_ty,
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
                                               cuda_val_ty,
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
                                        cuda_val_ty,
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
                                        cuda_val_ty,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        descr,
                                        &bufferSize2,
                                        buffer2));
  // Allocate buffers for the 32-bit version of the A matrix.
  int64_t A_rows, A_cols, A_nnz;
  CHECK_CUSPARSE(cusparseSpMatGetSize(cusparse_A, &A_rows, &A_cols, &A_nnz));
  DeferredBuffer<int32_t, 1> A_indptr({0, A_rows}, Memory::GPU_FB_MEM);
  DeferredBuffer<int32_t, 1> A_crd_int({0, A_nnz - 1}, Memory::GPU_FB_MEM);
  auto A_vals_acc = A_vals.create_output_buffer<val_ty, 1>(A_nnz, true /* return_buffer */);
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
                                     cuda_val_ty,
                                     CUSPARSE_SPGEMM_DEFAULT,
                                     descr));

  // Convert the A_indptr array into a pos array.
  {
    auto blocks = get_num_blocks_1d(A_rows);
    auto buf    = A_pos.create_output_buffer<Rect<1>, 1>(A_rows, true /* return_buffer */);
    localIndptrToPos<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(A_rows, buf.ptr(0), A_indptr.ptr(0));
  }
  // Cast the A coordinates back into 64 bits.
  {
    auto blocks = get_num_blocks_1d(A_nnz);
    auto buf    = A_crd.create_output_buffer<coord_ty, 1>(A_nnz, true /* return_buffer */);
    cast<coord_ty, int>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(A_nnz, buf.ptr(0), A_crd_int.ptr(0));
    // Finally, we need to offset the resulting coordinates into the global space.
    // cuSPARSE is going to compute a resulting matrix where all the coordinates
    // are zero-based, but we need the coordinates to be global addressable, so we
    // offset them by the partition of the column space that we are in.
    offset_coordinates_to_global<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      A_nnz, C_pos.domain().lo()[0], buf.ptr(0));
  }
  CHECK_CUDA_STREAM(stream);
}

__global__ void calculate_copy_sizes(size_t total_rows,
                                     size_t num_rects,
                                     DeferredBuffer<Rect<1>, 1> rects,
                                     DeferredBuffer<size_t, 1> row_offsets,
                                     AccessorRO<Rect<1>, 1> global_pos_acc)
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

__global__ void scatter_rows(size_t total_rows,
                             size_t num_rects,
                             DeferredBuffer<Rect<1>, 1> rects,
                             DeferredBuffer<size_t, 1> row_offsets,
                             AccessorRO<Rect<1>, 1> global_pos_acc,
                             AccessorRO<coord_ty, 1> global_crd_acc,
                             AccessorRO<val_ty, 1> global_vals_acc,
                             DeferredBuffer<Rect<1>, 1> pos_acc,
                             DeferredBuffer<coord_ty, 1> crd_acc,
                             DeferredBuffer<val_ty, 1> vals_acc)
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

void SpGEMMCSRxCSRxCSCShuffle::gpu_variant(legate::TaskContext& ctx)
{
  auto& global_pos  = ctx.inputs()[0];
  auto& global_crd  = ctx.inputs()[1];
  auto& global_vals = ctx.inputs()[2];

  auto& out_pos  = ctx.outputs()[0];
  auto& out_crd  = ctx.outputs()[1];
  auto& out_vals = ctx.outputs()[2];

  // TODO (rohany): I want a sparse instance here.
  auto global_pos_acc  = global_pos.read_accessor<Rect<1>, 1>();
  auto global_crd_acc  = global_crd.read_accessor<coord_ty, 1>();
  auto global_vals_acc = global_vals.read_accessor<val_ty, 1>();
  auto stream          = get_cached_stream();

  // Collect all rectangles in the global_pos domain.
  std::vector<Rect<1>> rects_cpu;
  size_t total_nnzs = 0;
  size_t total_rows = 0;
  for (RectInDomainIterator<1> itr(global_pos.domain()); itr(); itr++) {
    rects_cpu.push_back(*itr);
    total_rows = std::max(itr->volume(), total_rows);
    if (itr->empty()) continue;
    Rect<1> lo, hi;
    cudaMemcpy(&lo, global_pos_acc.ptr(itr->lo), sizeof(Rect<1>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hi, global_pos_acc.ptr(itr->hi), sizeof(Rect<1>), cudaMemcpyDeviceToHost);
    total_nnzs += hi.hi[0] - lo.lo[0] + 1;
  }

  // Allocate our output buffers.
  auto pos_acc  = out_pos.create_output_buffer<Rect<1>, 1>(total_rows, true /* return_buffer */);
  auto crd_acc  = out_crd.create_output_buffer<coord_ty, 1>(total_nnzs, true /* return_buffer */);
  auto vals_acc = out_vals.create_output_buffer<val_ty, 1>(total_nnzs, true /* return_buffer */);

  // We'll start with a simple row-based parallelization for our copies. If/when performance
  // suffers due to this, we can think about algorithms for a full-data based parallelization.
  DeferredBuffer<size_t, 1> row_offsets({0, total_rows - 1}, Memory::GPU_FB_MEM);
  DeferredBuffer<Rect<1>, 1> rects({0, rects_cpu.size() - 1}, Memory::GPU_FB_MEM);
  cudaMemcpy(
    rects.ptr(0), rects_cpu.data(), sizeof(Rect<1>) * rects_cpu.size(), cudaMemcpyHostToDevice);
  auto blocks = get_num_blocks_1d(total_rows);
  calculate_copy_sizes<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    total_rows, rects_cpu.size(), rects, row_offsets, global_pos_acc);
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
                                                         global_pos_acc,
                                                         global_crd_acc,
                                                         global_vals_acc,
                                                         pos_acc,
                                                         crd_acc,
                                                         vals_acc);
  CHECK_CUDA_STREAM(stream);
}

void SpMMCSR::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];
  auto B1_dim  = ctx.scalars()[0].value<int64_t>();

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  // Get context sensitive objects.
  auto handle = get_cusparse();
  auto stream = get_cached_stream();
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  // Break out early if the iteration space partition is empty.
  if (A_vals.domain().empty() || B_pos.domain().empty() || C_vals.domain().empty()) { return; }

  // Construct the CUSPARSE objects from individual regions.
  auto cusparse_A = makeCuSparseDenseMat(A_vals);
  // Because we are doing the same optimization as in SpMV to minimize
  // the communication instead of replicating the C matrix, we have to
  // offset the pointer into C down to the "base" of the region (which
  // may be invalid). We can rely on cuSPARSE not accessing this invalid
  // region because it is not referenced by any coordinates of B.
  auto C_domain   = C_vals.domain();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();
  auto ld         = C_vals_acc.accessor.strides[0] / sizeof(val_ty);
  auto C_vals_ptr = C_vals_acc.ptr(C_domain.lo());
  C_vals_ptr      = C_vals_ptr - size_t(ld * C_domain.lo()[0]);
  cusparseDnMatDescr_t cusparse_C;
  CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse_C,
                                     B1_dim,
                                     C_domain.hi()[1] - C_domain.lo()[1] + 1, /* columns */
                                     ld,
                                     (void*)C_vals_ptr,
                                     cuda_val_ty,
                                     CUSPARSE_ORDER_ROW));
  auto cusparse_B = makeCuSparseCSR(B_pos, B_crd, B_vals, B1_dim);

  // Call CUSPARSE.
  val_ty alpha   = 1.0;
  val_ty beta    = 0.0;
  size_t bufSize = 0;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         cusparse_B,
                                         cusparse_C,
                                         &beta,
                                         cusparse_A,
                                         cuda_val_ty,
                                         CUSPARSE_SPMM_CSR_ALG2,
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
                              cuda_val_ty,
                              CUSPARSE_SPMM_CSR_ALG2,
                              workspacePtr));
  // Destroy the created objects.
  CHECK_CUSPARSE(cusparseDestroyDnMat(cusparse_A));
  CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_B));
  CHECK_CUSPARSE(cusparseDestroyDnMat(cusparse_C));
  CHECK_CUDA_STREAM(stream);
}

__global__ void spmm_dense_csr_kernel(size_t nnzs,
                                      size_t pos_offset,
                                      int64_t* block_starts,
                                      int64_t idim,
                                      AccessorRD<SumReduction<val_ty>, false, 2> A_vals,
                                      AccessorRO<val_ty, 2> B_vals,
                                      AccessorRO<Rect<1>, 1> C_pos,
                                      AccessorRO<coord_ty, 1> C_crd,
                                      AccessorRO<val_ty, 1> C_vals)
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

void SpMMDenseCSR::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.reductions()[0];
  auto& B_vals = ctx.inputs()[0];
  auto& C_pos  = ctx.inputs()[1];
  auto& C_crd  = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (C_pos.transformed()) { C_pos.remove_transform(); }

  // Get context sensitive objects.
  auto handle = get_cusparse();
  auto stream = get_cached_stream();

  auto A_domain     = A_vals.domain();
  auto B_domain     = B_vals.domain();
  auto C_domain     = C_pos.domain();
  auto C_crd_domain = C_crd.domain();
  if (C_domain.empty() || C_crd_domain.empty()) { return; }

  auto A_vals_acc = A_vals.reduce_accessor<SumReduction<val_ty>, false /* exclusive */, 2>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();
  auto C_pos_acc  = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc  = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  // Use a DISTAL approach. Position space split on the non-zeros.
  auto nnzs   = C_crd_domain.get_volume();
  auto blocks = get_num_blocks_1d(nnzs);
  DeferredBuffer<int64_t, 1> buf({0, blocks}, Memory::GPU_FB_MEM);
  taco_binarySearchBeforeBlockLaunch(stream,
                                     C_pos_acc,
                                     buf.ptr(0),
                                     Point<1>(C_domain.lo()),
                                     Point<1>(C_domain.hi()),
                                     THREADS_PER_BLOCK,
                                     THREADS_PER_BLOCK,
                                     blocks,
                                     C_crd_domain.lo()[0] /* offset */
  );
  spmm_dense_csr_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    nnzs,
    C_crd_domain.lo()[0],
    buf.ptr(0),
    B_domain.hi()[0] - B_domain.lo()[0] + 1,
    A_vals_acc,
    B_vals_acc,
    C_pos_acc,
    C_crd_acc,
    C_vals_acc);
  CHECK_CUDA_STREAM(stream);
}

__global__ void elemwiseMult(size_t elems, val_ty* out, const val_ty* in)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) return;
  out[idx] *= in[idx];
}

__global__ void sddmm_csr_kernel(size_t nnzs,
                                 size_t pos_offset,
                                 int64_t* block_starts,
                                 int64_t kdim,
                                 AccessorWO<val_ty, 1> A_vals,
                                 AccessorRO<Rect<1>, 1> B_pos,
                                 AccessorRO<coord_ty, 1> B_crd,
                                 AccessorRO<val_ty, 1> B_vals,
                                 AccessorRO<val_ty, 2> C_vals,
                                 AccessorRO<val_ty, 2> D_vals)
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
  auto i       = taco_binarySearchBefore(B_pos, p_start, p_end, nnz_idx);
  auto j       = B_crd[nnz_idx];
  val_ty sum   = 0.0;
  for (int64_t k = 0; k < kdim; k++) { sum += C_vals[{i, k}] * D_vals[{k, j}]; }
  A_vals[nnz_idx] = sum * B_vals[nnz_idx];
}

void CSRSDDMM::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];
  auto& D_vals = ctx.inputs()[4];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  // Get context sensitive objects.
  auto handle = get_cusparse();
  auto stream = get_cached_stream();

  // Break out early if the iteration space partition is empty.
  if (A_vals.domain().empty() || B_pos.domain().empty() || B_crd.domain().empty() ||
      C_vals.domain().empty() || D_vals.domain().empty()) {
    return;
  }

  // Do the SDDMM, DISTAL style.
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();
  auto D_vals_acc = D_vals.read_accessor<val_ty, 2>();
  // The data has been distributed row-wise across the machine, and we can
  // do a non-zero based distribution among the local GPU threads.
  auto B_domain     = B_pos.domain();
  auto B_crd_domain = B_crd.domain();
  auto C_domain     = C_vals.domain();
  auto nnzs         = B_crd_domain.get_volume();
  // TODO (rohany): Can play around with the number of blocks here...
  //  DISTAL used 256 threads per block or something...
  // TODO (rohany): We can also attempt to chunk up the non-zeros by some
  //  amount so that each thread handles more than one nonzero.
  auto blocks = get_num_blocks_1d(nnzs);
  DeferredBuffer<int64_t, 1> buf({0, blocks}, Memory::GPU_FB_MEM);
  taco_binarySearchBeforeBlockLaunch(stream,
                                     B_pos_acc,
                                     buf.ptr(0),
                                     Point<1>(B_domain.lo()),
                                     Point<1>(B_domain.hi()),
                                     THREADS_PER_BLOCK,
                                     THREADS_PER_BLOCK,
                                     blocks,
                                     B_crd_domain.lo()[0] /* offset */
  );
  sddmm_csr_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    nnzs,
    B_crd_domain.lo()[0],
    buf.ptr(0),
    C_domain.hi()[1] - C_domain.lo()[1] + 1,
    A_vals_acc,
    B_pos_acc,
    B_crd_acc,
    B_vals_acc,
    C_vals_acc,
    D_vals_acc);
  CHECK_CUDA_STREAM(stream);
}

__global__ void sddmm_csc_kernel(size_t nnzs,
                                 size_t pos_offset,
                                 int64_t* block_starts,
                                 int64_t kdim,
                                 AccessorWO<val_ty, 1> A_vals,
                                 AccessorRO<Rect<1>, 1> B_pos,
                                 AccessorRO<coord_ty, 1> B_crd,
                                 AccessorRO<val_ty, 1> B_vals,
                                 AccessorRO<val_ty, 2> C_vals,
                                 AccessorRO<val_ty, 2> D_vals)
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
  auto j       = taco_binarySearchBefore(B_pos, p_start, p_end, nnz_idx);
  auto i       = B_crd[nnz_idx];
  val_ty sum   = 0.0;
  for (int64_t k = 0; k < kdim; k++) { sum += C_vals[{i, k}] * D_vals[{k, j}]; }
  A_vals[nnz_idx] = sum * B_vals[nnz_idx];
}

void CSCSDDMM::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];
  auto& D_vals = ctx.inputs()[4];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  // Get context sensitive objects.
  auto handle = get_cusparse();
  auto stream = get_cached_stream();

  // Break out early if the iteration space partition is empty.
  if (A_vals.domain().empty() || B_pos.domain().empty() || B_crd.domain().empty() ||
      C_vals.domain().empty() || D_vals.domain().empty()) {
    return;
  }

  // Do the SDDMM, DISTAL style.
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();
  auto D_vals_acc = D_vals.read_accessor<val_ty, 2>();
  // The data has been distributed column-wise across the machine, and we can
  // do a non-zero based distribution among the local GPU threads.
  auto B_domain     = B_pos.domain();
  auto B_crd_domain = B_crd.domain();
  auto C_domain     = C_vals.domain();
  auto nnzs         = B_crd_domain.get_volume();
  // TODO (rohany): Can play around with the number of blocks here...
  //  DISTAL used 256 threads per block or something...
  // TODO (rohany): We can also attempt to chunk up the non-zeros by some
  //  amount so that each thread handles more than one nonzero.
  auto blocks = get_num_blocks_1d(nnzs);
  DeferredBuffer<int64_t, 1> buf({0, blocks}, Memory::GPU_FB_MEM);
  taco_binarySearchBeforeBlockLaunch(stream,
                                     B_pos_acc,
                                     buf.ptr(0),
                                     Point<1>(B_domain.lo()),
                                     Point<1>(B_domain.hi()),
                                     THREADS_PER_BLOCK,
                                     THREADS_PER_BLOCK,
                                     blocks,
                                     B_crd_domain.lo()[0] /* offset */
  );
  sddmm_csc_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    nnzs,
    B_crd_domain.lo()[0],
    buf.ptr(0),
    C_domain.hi()[1] - C_domain.lo()[1] + 1,
    A_vals_acc,
    B_pos_acc,
    B_crd_acc,
    B_vals_acc,
    C_vals_acc,
    D_vals_acc);
  CHECK_CUDA_STREAM(stream);
}

__global__ void elementwise_mult_csr_csr_nnz_kernel(size_t rows,
                                                    Rect<1> iterBounds,
                                                    AccessorWO<nnz_ty, 1> nnz,
                                                    AccessorRO<Rect<1>, 1> B_pos,
                                                    AccessorRO<coord_ty, 1> B_crd,
                                                    AccessorRO<Rect<1>, 1> C_pos,
                                                    AccessorRO<coord_ty, 1> C_crd)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  auto i         = idx + iterBounds.lo;
  size_t num_nnz = 0;
  size_t jB      = B_pos[i].lo;
  size_t pB2_end = B_pos[i].hi + 1;
  size_t jC      = C_pos[i].lo;
  size_t pC2_end = C_pos[i].hi + 1;
  while (jB < pB2_end && jC < pC2_end) {
    coord_ty jB0 = B_crd[jB];
    coord_ty jC0 = C_crd[jC];
    coord_ty j   = std::min(jB0, jC0);
    if (jB0 == j && jC0 == j) { num_nnz++; }
    jB += (size_t)(jB0 == j);
    jC += (size_t)(jC0 == j);
  }
  nnz[i] = num_nnz;
}

// This implementation uses a row-wise distribution across GPU threads.
// cuSPARSE does not have an element-wise multiply, so we have to hand-write one.
void ElemwiseMultCSRCSRNNZ::gpu_variant(legate::TaskContext& ctx)
{
  auto& nnz   = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& C_pos = ctx.inputs()[2];
  auto& C_crd = ctx.inputs()[3];

  // Break out early if the iteration space partition is empty.
  if (B_pos.domain().empty() || C_pos.domain().empty()) { return; }

  auto nnz_acc   = nnz.write_accessor<nnz_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();

  auto stream   = get_cached_stream();
  auto B_domain = B_pos.domain();
  auto rows     = B_domain.get_volume();
  auto blocks   = get_num_blocks_1d(rows);
  elementwise_mult_csr_csr_nnz_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    rows,
    Rect<1>(B_domain.lo(), B_domain.hi()),
    nnz_acc,
    B_pos_acc,
    B_crd_acc,
    C_pos_acc,
    C_crd_acc);
  CHECK_CUDA_STREAM(stream);
}

__global__ void elementwise_mult_csr_csr_kernel(size_t rows,
                                                Rect<1> iterBounds,
                                                AccessorRW<Rect<1>, 1> A_pos,
                                                AccessorWO<coord_ty, 1> A_crd,
                                                AccessorWO<val_ty, 1> A_vals,
                                                AccessorRO<Rect<1>, 1> B_pos,
                                                AccessorRO<coord_ty, 1> B_crd,
                                                AccessorRO<val_ty, 1> B_vals,
                                                AccessorRO<Rect<1>, 1> C_pos,
                                                AccessorRO<coord_ty, 1> C_crd,
                                                AccessorRO<val_ty, 1> C_vals)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  auto i = idx + iterBounds.lo;
  // Similarly to other codes, since we're doing the inserts for a particular
  // value of i at a time, we don't need to mutate the pos array.
  size_t nnz_pos = A_pos[i].lo;
  size_t jB      = B_pos[i].lo;
  size_t pB2_end = B_pos[i].hi + 1;
  size_t jC      = C_pos[i].lo;
  size_t pC2_end = C_pos[i].hi + 1;
  while (jB < pB2_end && jC < pC2_end) {
    size_t jB0 = B_crd[jB];
    size_t jC0 = C_crd[jC];
    coord_ty j = std::min(jB0, jC0);
    if (jB0 == j && jC0 == j) {
      A_crd[nnz_pos]  = j;
      A_vals[nnz_pos] = B_vals[jB] * C_vals[jC];
      nnz_pos++;
    }
    jB += (size_t)(jB0 == j);
    jC += (size_t)(jC0 == j);
  }
}

void ElemwiseMultCSRCSR::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos  = ctx.inputs()[3];
  auto& C_crd  = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  // Break out early if the iteration space partition is empty.
  if (B_pos.domain().empty() || C_pos.domain().empty()) { return; }

  auto A_pos_acc  = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc  = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_pos_acc  = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc  = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  auto stream   = get_cached_stream();
  auto B_domain = B_pos.domain();
  auto rows     = B_domain.get_volume();
  auto blocks   = get_num_blocks_1d(rows);

  elementwise_mult_csr_csr_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    rows,
    Rect<1>(B_domain.lo(), B_domain.hi()),
    A_pos_acc,
    A_crd_acc,
    A_vals_acc,
    B_pos_acc,
    B_crd_acc,
    B_vals_acc,
    C_pos_acc,
    C_crd_acc,
    C_vals_acc);
  CHECK_CUDA_STREAM(stream);
}

__global__ void elementwise_mult_csr_dense_kernel(size_t nnzs,
                                                  size_t pos_offset,
                                                  int64_t* block_starts,
                                                  AccessorWO<val_ty, 1> A_vals,
                                                  AccessorRO<Rect<1>, 1> B_pos,
                                                  AccessorRO<coord_ty, 1> B_crd,
                                                  AccessorRO<val_ty, 1> B_vals,
                                                  AccessorRO<val_ty, 2> C_vals)
{
  size_t block   = blockIdx.x;
  const auto idx = global_tid_1d();
  if (idx >= nnzs) return;
  auto nnz_idx = idx + pos_offset;

  // Search for the current coordinate. I've written this right now as processing
  // a single non-zero per thread. In the future, the cost of the binary search
  // can be amortized so that we process more values after the search.
  auto p_start    = block_starts[block];
  auto p_end      = block_starts[block + 1];
  auto i          = taco_binarySearchBefore(B_pos, p_start, p_end, nnz_idx);
  auto j          = B_crd[nnz_idx];
  A_vals[nnz_idx] = B_vals[nnz_idx] * C_vals[{i, j}];
}

void ElemwiseMultCSRDense::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  // Break out early if the iteration space partition is empty.
  if (B_pos.domain().empty() || C_vals.domain().empty()) { return; }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();

  // The data has been distributed row-wise across the machine, and we can
  // do a non-zero based distribution among the local GPU threads.
  auto B_domain     = B_pos.domain();
  auto B_crd_domain = B_crd.domain();
  auto rows         = B_domain.get_volume();
  auto nnzs         = B_crd_domain.get_volume();
  auto blocks       = get_num_blocks_1d(nnzs);
  auto stream       = get_cached_stream();

  // Find the offsets within the pos array that each coordinate should search for.
  DeferredBuffer<int64_t, 1> buf({0, blocks}, Memory::GPU_FB_MEM);
  taco_binarySearchBeforeBlockLaunch(stream,
                                     B_pos_acc,
                                     buf.ptr(0),
                                     Point<1>(B_domain.lo()),
                                     Point<1>(B_domain.hi()),
                                     THREADS_PER_BLOCK,
                                     THREADS_PER_BLOCK,
                                     blocks,
                                     B_crd_domain.lo()[0] /* offset */
  );
  // Use these offsets to execute the kernel.
  elementwise_mult_csr_dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(nnzs,
                                                                              B_crd_domain.lo()[0],
                                                                              buf.ptr(0),
                                                                              A_vals_acc,
                                                                              B_pos_acc,
                                                                              B_crd_acc,
                                                                              B_vals_acc,
                                                                              C_vals_acc);
  CHECK_CUDA_STREAM(stream);
}

__global__ void CSCtoDenseKernel(size_t cols,
                                 Rect<1> bounds,
                                 AccessorRW<val_ty, 2> A_vals_acc,
                                 AccessorRO<Rect<1>, 1> B_pos_acc,
                                 AccessorRO<coord_ty, 1> B_crd_acc,
                                 AccessorRO<val_ty, 1> B_vals_acc)
{
  const auto idx = global_tid_1d();
  if (idx >= cols) return;
  coord_ty j = idx + bounds.lo[0];
  for (coord_ty i_pos = B_pos_acc[j].lo; i_pos < B_pos_acc[j].hi + 1; i_pos++) {
    auto i             = B_crd_acc[i_pos];
    A_vals_acc[{i, j}] = B_vals_acc[i_pos];
  }
}

void CSCToDense::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  // Break out early if the iteration space partition is empty.
  if (B_pos.domain().empty()) { return; }

  auto stream = get_cached_stream();

  // If we are running on an old cuSPARSE version, then we don't
  // have access to many cuSPARSE functions. In that case, use
  // a hand-written version.
#if (CUSPARSE_VER_MAJOR < 11 || CUSPARSE_VER_MINOR < 2)
  auto B_domain = B_pos.domain();
  auto cols     = B_domain.hi()[0] - B_domain.lo()[0] + 1;
  auto blocks   = get_num_blocks_1d(cols);
  CSCtoDenseKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    cols,
    Rect<1>(B_domain.lo(), B_domain.hi()),
    A_vals.read_write_accessor<val_ty, 2>(),
    B_pos.read_accessor<Rect<1>, 1>(),
    B_crd.read_accessor<coord_ty, 1>(),
    B_vals.read_accessor<val_ty, 1>());
#else
  // Get context sensitive objects.
  auto handle = get_cusparse();
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  // Construct our cuSPARSE matrices.
  auto A_domain   = A_vals.domain();
  auto cusparse_A = makeCuSparseDenseMat(A_vals);
  auto cusparse_B =
    makeCuSparseCSC(B_pos, B_crd, B_vals, A_domain.hi()[0] - A_domain.lo()[0] + 1 /* rows */);

  // Finally make the cuSPARSE calls.
  size_t bufSize = 0;
  CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(
    handle, cusparse_B, cusparse_A, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufSize));
  // Allocate a buffer if we need to.
  void* workspacePtr = nullptr;
  if (bufSize > 0) {
    DeferredBuffer<char, 1> buf({0, bufSize - 1}, Memory::GPU_FB_MEM);
    workspacePtr = buf.ptr(0);
  }
  // Finally do the conversion.
  CHECK_CUSPARSE(cusparseSparseToDense(
    handle, cusparse_B, cusparse_A, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, workspacePtr));
  // Destroy the created objects.
  CHECK_CUSPARSE(cusparseDestroyDnMat(cusparse_A));
  CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_B));
#endif
  CHECK_CUDA_STREAM(stream);
}

__global__ void denseToCSCNNZKernel(size_t cols,
                                    Rect<2> bounds,
                                    AccessorWO<nnz_ty, 1> A_nnz_acc,
                                    AccessorRO<val_ty, 2> B_vals_acc)
{
  const auto idx = global_tid_1d();
  if (idx >= cols) return;
  coord_ty j       = idx + bounds.lo[1];
  nnz_ty nnz_count = 0;
  for (coord_ty i = bounds.lo[0]; i < bounds.hi[0] + 1; i++) {
    if (B_vals_acc[{i, j}] != 0.0) { nnz_count++; }
  }
  A_nnz_acc[j] = nnz_count;
}

void DenseToCSCNNZ::gpu_variant(legate::TaskContext& ctx)
{
  auto& nnz    = ctx.outputs()[0];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the nnz region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (nnz.transformed()) { nnz.remove_transform(); }

  // Break out early if the iteration space partition is empty.
  if (B_vals.domain().empty()) { return; }

  auto stream = get_cached_stream();

#if (CUSPARSE_VER_MAJOR < 11 || CUSPARSE_VER_MINOR < 2)
  auto B_domain = B_vals.domain();
  auto cols     = B_domain.hi()[1] - B_domain.lo()[1] + 1;
  auto blocks   = get_num_blocks_1d(cols);
  denseToCSCNNZKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    cols,
    Rect<2>(B_domain.lo(), B_domain.hi()),
    nnz.write_accessor<nnz_ty, 1>(),
    B_vals.read_accessor<val_ty, 2>());
#else
  // Get context sensitive objects.
  auto handle = get_cusparse();
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  auto B_domain = B_vals.domain();
  auto rows     = B_domain.hi()[0] - B_domain.lo()[0] + 1;
  auto cols     = B_domain.hi()[1] - B_domain.lo()[1] + 1;
  // Allocate an output buffer for the offsets.
  DeferredBuffer<int64_t, 1> A_indptr({0, cols}, Memory::GPU_FB_MEM);

  // Construct the cuSPARSE objects from individual regions.
  auto cusparse_B = makeCuSparseDenseMat(B_vals);
  // We will construct the sparse matrix explicitly due to not having
  // all of the components right now.
  cusparseSpMatDescr_t cusparse_A;
  CHECK_CUSPARSE(cusparseCreateCsc(&cusparse_A,
                                   rows,
                                   cols,
                                   0 /* nnz */,
                                   A_indptr.ptr(0),
                                   nullptr,
                                   nullptr,
                                   index_ty,
                                   index_ty,
                                   index_base,
                                   cuda_val_ty));
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
    auto blocks = get_num_blocks_1d(cols);
    localIndptrToNnz<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      cols, nnz.write_accessor<uint64_t, 1>().ptr(nnz.domain().lo()), A_indptr.ptr(0));
  }
#endif

  CHECK_CUDA_STREAM(stream);
}

__global__ void denseToCSCKernel(size_t cols,
                                 Rect<2> bounds,
                                 AccessorRW<Rect<1>, 1> A_pos_acc,
                                 AccessorWO<coord_ty, 1> A_crd_acc,
                                 AccessorWO<val_ty, 1> A_vals_acc,
                                 AccessorRO<val_ty, 2> B_vals_acc)
{
  const auto idx = global_tid_1d();
  if (idx >= cols) return;
  coord_ty j      = idx + bounds.lo[1];
  int64_t nnz_pos = A_pos_acc[j].lo;
  for (coord_ty i = bounds.lo[0]; i < bounds.hi[0] + 1; i++) {
    if (B_vals_acc[{i, j}] != 0.0) {
      A_crd_acc[nnz_pos]  = i;
      A_vals_acc[nnz_pos] = B_vals_acc[{i, j}];
      nnz_pos++;
    }
  }
}

void DenseToCSC::gpu_variant(legate::TaskContext& ctx)
{
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (A_pos.transformed()) { A_pos.remove_transform(); }

  // Break out early if the iteration space partition is empty.
  if (B_vals.domain().empty()) { return; }

  // Get context sensitive objects.
  auto handle = get_cusparse();
  auto stream = get_cached_stream();
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  auto B_domain = B_vals.domain();
  auto rows     = B_domain.hi()[0] - B_domain.lo()[0] + 1;
  auto cols     = B_domain.hi()[1] - B_domain.lo()[1] + 1;
  auto blocks   = get_num_blocks_1d(cols);
  denseToCSCKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    cols,
    Rect<2>(B_domain.lo(), B_domain.hi()),
    A_pos.read_write_accessor<Rect<1>, 1>(),
    A_crd.write_accessor<coord_ty, 1>(),
    A_vals.write_accessor<val_ty, 1>(),
    B_vals.read_accessor<val_ty, 2>());
  CHECK_CUDA_STREAM(stream);

  // TODO (rohany): The below cuSPARSE code is buggy. In particular, it results
  //  in some row segments of the resulting CSC array to be unsorted, which is a
  //  violation of the CSC data structure.
  // // Construct the cuSPARSE objects from individual regions.
  // auto cusparse_A = makeCuSparseCSC(A_pos, A_crd, A_vals, rows);
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

}  // namespace sparse

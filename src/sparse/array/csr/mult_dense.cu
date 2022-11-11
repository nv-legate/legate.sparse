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

#include "sparse/array/csr/mult_dense.h"
#include "sparse/array/csr/mult_dense_template.inl"
#include "cuda_help.h"
#include "distal_cuda_utils.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename INDEX_TY, typename VAL_TY>
__global__ void elementwise_mult_csr_dense_kernel(size_t nnzs,
                                                  size_t pos_offset,
                                                  int64_t* block_starts,
                                                  const AccessorWO<VAL_TY, 1> A_vals,
                                                  const AccessorRO<Rect<1>, 1> B_pos,
                                                  const AccessorRO<INDEX_TY, 1> B_crd,
                                                  const AccessorRO<VAL_TY, 1> B_vals,
                                                  const AccessorRO<VAL_TY, 2> C_vals)
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

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct ElemwiseMultCSRDenseArgsImplBody<VariantKind::GPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<VAL_TY, 2>& C_vals,
                  const Rect<1>& B_pos_rect,
                  const Rect<1>& B_crd_rect)
  {
    // The data has been distributed row-wise across the machine, and we can
    // do a non-zero based distribution among the local GPU threads.
    auto rows   = B_pos_rect.volume();
    auto nnzs   = B_crd_rect.volume();
    auto blocks = get_num_blocks_1d(nnzs);
    auto stream = get_cached_stream();

    // Find the offsets within the pos array that each coordinate should search for.
    DeferredBuffer<int64_t, 1> buf({0, blocks}, Memory::GPU_FB_MEM);
    taco_binarySearchBeforeBlockLaunch(stream,
                                       B_pos,
                                       buf.ptr(0),
                                       B_pos_rect.lo,
                                       B_pos_rect.hi,
                                       THREADS_PER_BLOCK,
                                       THREADS_PER_BLOCK,
                                       blocks,
                                       B_crd_rect.lo[0] /* offset */
    );
    // Use these offsets to execute the kernel.
    elementwise_mult_csr_dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      nnzs, B_crd_rect.lo[0], buf.ptr(0), A_vals, B_pos, B_crd, B_vals, C_vals);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ElemwiseMultCSRDense::gpu_variant(TaskContext& context)
{
  elemwise_mult_dense_template<VariantKind::GPU>(context);
}

}  // namespace sparse

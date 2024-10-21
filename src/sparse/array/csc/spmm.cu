/* Copyright 2023 NVIDIA Corporation
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

#include "sparse/array/csc/spmm.h"
#include "sparse/array/csc/spmm_template.inl"
#include "sparse/util/distal_cuda_utils.h"

namespace sparse {

using namespace legate;

template <typename INDEX_TY, typename VAL_TY>
__global__ void spmm_csc_kernel(const size_t nnzs,
                                const size_t pos_offset,
                                const int64_t* block_starts,
                                const Rect<2> A_rect,
                                const AccessorRD<SumReduction<VAL_TY>, false, 2> A_vals,
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
  auto p_start = block_starts[block];
  auto p_end   = block_starts[block + 1];
  auto k       = taco_binarySearchBefore(B_pos, p_start, p_end, nnz_idx);
  auto i       = B_crd[nnz_idx];
  for (int64_t j = A_rect.lo[1]; j < A_rect.hi[1] + 1; j++) {
    A_vals[{i, j}] <<= B_vals[nnz_idx] * C_vals[{k, j}];
  }
}

template <Type::Code INDEX_CODE, Type::Code VAL_CODE, typename ACC>
struct SpMMCSCImplBody<VariantKind::GPU, INDEX_CODE, VAL_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(ACC A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<VAL_TY, 2>& C_vals,
                  const Rect<2>& A_rect,
                  const Rect<2>& C_rect,
                  const Rect<1>& nnz_rect)
  {
    // Use a DISTAL approach. Position space split on the non-zeros.
    auto stream = get_cached_stream();
    auto nnzs   = nnz_rect.volume();
    auto blocks = get_num_blocks_1d(nnzs);
    Buffer<int64_t, 1> buf({0, blocks}, Memory::GPU_FB_MEM);
    taco_binarySearchBeforeBlockLaunch(stream,
                                       B_pos,
                                       buf.ptr(0),
                                       C_rect.lo[0],
                                       C_rect.hi[0],
                                       THREADS_PER_BLOCK,
                                       THREADS_PER_BLOCK,
                                       blocks,
                                       nnz_rect.lo /* offset */
    );
    spmm_csc_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      nnzs, nnz_rect.lo, buf.ptr(0), A_rect, A_vals, B_pos, B_crd, B_vals, C_vals);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void SpMMCSC::gpu_variant(TaskContext& context)
{
  spmm_template<VariantKind::GPU>(context);
}

}  // namespace sparse

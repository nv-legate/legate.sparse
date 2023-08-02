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

#include "sparse/array/csr/sddmm.h"
#include "sparse/array/csr/sddmm_template.inl"
#include "sparse/util/cuda_help.h"
#include "sparse/util/distal_cuda_utils.h"

namespace sparse {

using namespace legate;

template <typename INDEX_TY, typename VAL_TY>
__global__ void sddmm_csr_kernel(size_t nnzs,
                                 size_t pos_offset,
                                 int64_t* block_starts,
                                 int64_t kdim,
                                 AccessorWO<VAL_TY, 1> A_vals,
                                 AccessorRO<Rect<1>, 1> B_pos,
                                 AccessorRO<INDEX_TY, 1> B_crd,
                                 AccessorRO<VAL_TY, 1> B_vals,
                                 AccessorRO<VAL_TY, 2> C_vals,
                                 AccessorRO<VAL_TY, 2> D_vals)
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
  VAL_TY sum   = static_cast<VAL_TY>(0);
  for (int64_t k = 0; k < kdim; k++) { sum += C_vals[{i, k}] * D_vals[{k, j}]; }
  A_vals[nnz_idx] = sum * B_vals[nnz_idx];
}

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct CSRSDDMMImplBody<VariantKind::GPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<VAL_TY, 2>& C_vals,
                  const AccessorRO<VAL_TY, 2>& D_vals,
                  const Rect<2>& rect,
                  const Rect<1>& vals_rect)
  {
    auto stream = get_cached_stream();
    // The data has been distributed row-wise across the machine, and we can
    // do a non-zero based distribution among the local GPU threads.
    auto nnzs = vals_rect.volume();
    // TODO (rohany): Can play around with the number of blocks here...
    //  DISTAL used 256 threads per block or something...
    // TODO (rohany): We can also attempt to chunk up the non-zeros by some
    //  amount so that each thread handles more than one nonzero.
    auto blocks = get_num_blocks_1d(nnzs);
    Buffer<int64_t, 1> buf({0, blocks}, Memory::GPU_FB_MEM);
    taco_binarySearchBeforeBlockLaunch(stream,
                                       B_pos,
                                       buf.ptr(0),
                                       rect.lo[0],
                                       rect.hi[0],
                                       THREADS_PER_BLOCK,
                                       THREADS_PER_BLOCK,
                                       blocks,
                                       vals_rect.lo[0] /* offset */
    );
    sddmm_csr_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(nnzs,
                                                               vals_rect.lo[0],
                                                               buf.ptr(0),
                                                               rect.hi[1] - rect.lo[1] + 1,
                                                               A_vals,
                                                               B_pos,
                                                               B_crd,
                                                               B_vals,
                                                               C_vals,
                                                               D_vals);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void CSRSDDMM::gpu_variant(TaskContext& context)
{
  csr_sddmm_template<VariantKind::GPU>(context);
}

}  // namespace sparse

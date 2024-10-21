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

#include "sparse/array/csr/get_diagonal.h"
#include "sparse/array/csr/get_diagonal_template.inl"
#include "sparse/util/cuda_help.h"

namespace sparse {

using namespace legate;

template <typename INDEX_TY, typename VAL_TY>
__global__ void compute_diag_kernel(size_t rows,
                                    int64_t offset,
                                    AccessorWO<VAL_TY, 1> diag,
                                    AccessorRO<Rect<1>, 1> pos,
                                    AccessorRO<INDEX_TY, 1> crd,
                                    AccessorRO<VAL_TY, 1> vals)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  auto i  = idx + offset;
  diag[i] = 0.0;
  for (size_t j_pos = pos[i].lo; j_pos < pos[i].hi + 1; j_pos++) {
    if (crd[j_pos] == i) { diag[i] = vals[j_pos]; }
  }
}

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct GetCSRDiagonalImplBody<VariantKind::GPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& diag,
                  const AccessorRO<Rect<1>, 1>& pos,
                  const AccessorRO<INDEX_TY, 1>& crd,
                  const AccessorRO<VAL_TY, 1>& vals,
                  const Rect<1>& rect)
  {
    auto stream = get_cached_stream();
    auto blocks = get_num_blocks_1d(rect.volume());
    compute_diag_kernel<INDEX_TY, VAL_TY>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(rect.volume(), rect.lo[0], diag, pos, crd, vals);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void GetCSRDiagonal::gpu_variant(TaskContext& context)
{
  get_csr_diagonal_template<VariantKind::GPU>(context);
}

}  // namespace sparse

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

#include "sparse/linalg/axpby.h"
#include "sparse/linalg/axpby_template.inl"
#include "cuda_help.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename VAL_TY>
__global__ void axpby_kernel(size_t elems,
                             coord_ty offset,
                             AccessorRW<VAL_TY, 1> y,
                             AccessorRO<VAL_TY, 1> x,
                             AccessorRO<VAL_TY, 1> alpha,
                             AccessorRO<VAL_TY, 1> beta)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) return;
  auto i = idx + offset;
  y[i]   = alpha[0] * x[i] + beta[0] * y[i];
}

template <LegateTypeCode VAL_CODE>
struct AXPBYImplBody<VariantKind::GPU, VAL_CODE> {
  using VAL_TY = legate_type_of<VAL_CODE>;

  void operator()(const AccessorRW<VAL_TY, 1>& y,
                  const AccessorRO<VAL_TY, 1>& x,
                  const AccessorRO<VAL_TY, 1>& alpha,
                  const AccessorRO<VAL_TY, 1>& beta,
                  const Rect<1>& rect)
  {
    auto elems  = rect.volume();
    auto blocks = get_num_blocks_1d(elems);
    auto stream = get_cached_stream();
    axpby_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(elems, rect.lo[0], y, x, alpha, beta);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void AXPBY::gpu_variant(TaskContext& context)
{
  axpby_template<VariantKind::GPU>(context);
}

}  // namespace sparse

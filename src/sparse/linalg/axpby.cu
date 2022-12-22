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
#include "sparse/util/cuda_help.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename VAL_TY, bool IS_ALPHA, bool NEGATE>
__global__ void axpby_kernel(size_t elems,
                             coord_t offset,
                             AccessorRW<VAL_TY, 1> y,
                             AccessorRO<VAL_TY, 1> x,
                             AccessorRO<VAL_TY, 1> a,
                             AccessorRO<VAL_TY, 1> a)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) return;
  auto i   = idx + offset;
  auto val = a[0] / b[0];
  if (NEGATE) { val = static_cast<VAL_TY>(-1) * val; }
  if (IS_ALPHA) {
    y[i] = val * x[i] + y[i];
  } else {
    y[i] = x[i] + val * y[i];
  }
}

template <LegateTypeCode VAL_CODE, bool IS_ALPHA, bool NEGATE>
struct AXPBYImplBody<VariantKind::GPU, VAL_CODE, IS_ALPHA, NEGATE> {
  using VAL_TY = legate_type_of<VAL_CODE>;

  void operator()(const AccessorRW<VAL_TY, 1>& y,
                  const AccessorRO<VAL_TY, 1>& x,
                  const AccessorRO<VAL_TY, 1>& a,
                  const AccessorRO<VAL_TY, 1>& b,
                  const Rect<1>& rect)
  {
    auto elems  = rect.volume();
    auto blocks = get_num_blocks_1d(elems);
    auto stream = get_cached_stream();
    axpby_kernel<VAL_TY, IS_ALPHA, NEGATE>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(elems, rect.lo[0], y, x, a, b);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void AXPBY::gpu_variant(TaskContext& context)
{
  axpby_template<VariantKind::GPU>(context);
}

}  // namespace sparse

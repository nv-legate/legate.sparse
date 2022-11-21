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

#include "sparse/integrate/runge_kutta.h"
#include "sparse/integrate/runge_kutta_template.inl"
#include "sparse/util/cuda_help.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename K_TY, typename A_TY>
__global__ void rk_kernel(const size_t elems,
                          const coord_t offset,
                          const AccessorWO<K_TY, 1> dy,
                          const AccessorRO<K_TY, 2> K,
                          const AccessorRO<A_TY, 1> a,
                          const int32_t s,
                          const double h)
{
  const auto tid = global_tid_1d();
  if (tid >= elems) return;
  auto i   = tid + offset;
  K_TY acc = 0;
#pragma unroll
  for (int32_t j = 0; j < s; j++) { acc += K[{j, i}] * a[j]; }
  dy[i] = acc * h;
}

template <LegateTypeCode K_CODE, LegateTypeCode A_CODE>
struct RKCalcDyImplBody<VariantKind::GPU, K_CODE, A_CODE> {
  using K_TY = legate_type_of<K_CODE>;
  using A_TY = legate_type_of<A_CODE>;

  void operator()(const AccessorWO<K_TY, 1>& dy,
                  const AccessorRO<K_TY, 2>& K,
                  const AccessorRO<A_TY, 1>& a,
                  const int32_t s,
                  const double h,
                  const Rect<2>& rect)
  {
    auto stream = get_cached_stream();
    auto elems  = rect.hi[1] - rect.lo[1] + 1;
    auto blocks = get_num_blocks_1d(elems);
    rk_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(elems, rect.lo[1], dy, K, a, s, h);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void RKCalcDy::gpu_variant(TaskContext& context)
{
  rk_calc_dy_template<VariantKind::GPU>(context);
}

}  // namespace sparse

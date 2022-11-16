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

#include "sparse/array/util/scale_rect.h"
#include "sparse/array/util/scale_rect_template.inl"
#include "cuda_help.h"

namespace sparse {

using namespace Legion;
using namespace legate;

__global__ void scale_rect1_kernel(size_t elems, const AccessorRW<Rect<1>, 1> out, int64_t scale)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) return;
  out[idx].lo = out[idx].lo + scale;
  out[idx].hi = out[idx].hi + scale;
}

template <>
struct ScaleRect1ImplBody<VariantKind::GPU> {
  void operator()(const AccessorRW<Rect<1>, 1>& output, const int64_t scale, const Rect<1>& rect)
  {
    auto elems  = rect.volume();
    auto blocks = get_num_blocks_1d(elems);
    auto stream = get_cached_stream();
    scale_rect1_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(elems, output, scale);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ScaleRect1::gpu_variant(TaskContext& context)
{
  scale_rect_1_template<VariantKind::GPU>(context);
}

}  // namespace sparse

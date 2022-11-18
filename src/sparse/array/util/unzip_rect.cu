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

#include "sparse/array/util/unzip_rect.h"
#include "sparse/array/util/unzip_rect_template.inl"
#include "cuda_help.h"

namespace sparse {

using namespace Legion;
using namespace legate;

__global__ void unzip_rect1_kernel(size_t elems,
                                   coord_t offset,
                                   const AccessorWO<int64_t, 1> lo,
                                   const AccessorWO<int64_t, 1> hi,
                                   const AccessorRO<Rect<1>, 1> in)
{
  const auto tid = global_tid_1d();
  if (tid >= elems) return;
  const auto idx = tid + offset;
  lo[idx]        = in[idx].lo;
  hi[idx]        = in[idx].hi;
}

template <>
struct UnZipRect1ImplBody<VariantKind::GPU> {
  void operator()(const AccessorWO<int64_t, 1>& out1,
                  const AccessorWO<int64_t, 1>& out2,
                  const AccessorRO<Rect<1>, 1>& in,
                  const Rect<1>& rect)
  {
    auto elems  = rect.volume();
    auto blocks = get_num_blocks_1d(elems);
    auto stream = get_cached_stream();
    unzip_rect1_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(elems, rect.lo, out1, out2, in);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void UnZipRect1::gpu_variant(TaskContext& context)
{
  unzip_rect_1_template<VariantKind::GPU>(context);
}

}  // namespace sparse

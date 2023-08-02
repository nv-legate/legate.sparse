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

#include "sparse/array/util/zip_to_rect.h"
#include "sparse/array/util/zip_to_rect_template.inl"
#include "sparse/util/cuda_help.h"

namespace sparse {

using namespace legate;

__global__ void zip_rect1_kernel(size_t elems,
                                 coord_t offset,
                                 const AccessorWO<Rect<1>, 1> out,
                                 const AccessorRO<uint64_t, 1> lo,
                                 const AccessorRO<uint64_t, 1> hi)
{
  const auto tid = global_tid_1d();
  if (tid >= elems) return;
  const auto idx = tid + offset;
  out[idx]       = {lo[idx], hi[idx] - 1};
}

template <>
struct ZipToRect1ImplBody<VariantKind::GPU> {
  void operator()(const AccessorWO<Rect<1>, 1>& output,
                  const AccessorRO<uint64_t, 1>& lo,
                  const AccessorRO<uint64_t, 1>& hi,
                  const Rect<1>& rect)
  {
    auto stream = get_cached_stream();
    auto elems  = rect.volume();
    auto blocks = get_num_blocks_1d(elems);
    zip_rect1_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(elems, rect.lo, output, lo, hi);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ZipToRect1::gpu_variant(TaskContext& context)
{
  zip_to_rect_1_template<VariantKind::GPU>(context);
}

}  // namespace sparse

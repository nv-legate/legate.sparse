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

#include "sparse/partition/fast_image_range.h"
#include "sparse/partition/fast_image_range_template.inl"
#include "sparse/util/cuda_help.h"

namespace sparse {

using namespace legate;

template <>
struct FastImageRangeImplBody<VariantKind::GPU> {
  void operator()(const AccessorWO<Domain, 1>& out,
                  const AccessorRO<Rect<1>, 1>& in,
                  const Domain& dom)
  {
    auto stream = get_cached_stream();
    if (dom.empty()) {
      Domain result{Rect<1>::make_empty()};
      auto ptr = out.ptr(0);
#ifdef LEGATE_NO_FUTURES_ON_FB
      *ptr = result;
#else
      CHECK_CUDA(cudaMemcpyAsync(ptr, &result, sizeof(Domain), cudaMemcpyHostToDevice, stream));
#endif
    } else {
      Rect<1> lo, hi;
      CHECK_CUDA(
        cudaMemcpyAsync(&lo, in.ptr(dom.lo()), sizeof(Rect<1>), cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(
        cudaMemcpyAsync(&hi, in.ptr(dom.hi()), sizeof(Rect<1>), cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
      Domain result{Rect<1>{lo.lo, hi.hi}};
      auto ptr = out.ptr(0);
#ifdef LEGATE_NO_FUTURES_ON_FB
      *ptr = result;
#else
      CHECK_CUDA(cudaMemcpyAsync(ptr, &result, sizeof(Domain), cudaMemcpyHostToDevice, stream));
#endif
    }
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void FastImageRange::gpu_variant(TaskContext& context)
{
  fast_image_range_template<VariantKind::GPU>(context);
}

}  // namespace sparse

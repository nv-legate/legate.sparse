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

#include "sparse/partition/bounds_from_partitioned_coordinates.h"
#include "sparse/partition/bounds_from_partitioned_coordinates_template.inl"
#include "cuda_help.h"
#include "thrust_allocator.h"

#include <thrust/extrema.h>

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE>
struct BoundsFromPartitionedCoordinatesImplBody<VariantKind::GPU, INDEX_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  void operator()(const AccessorWO<Domain, 1> output,
                  AccessorRO<INDEX_TY, 1> input,
                  const Domain& dom)
  {
    auto stream = get_cached_stream();
    if (dom.empty()) {
      auto result = Domain(Rect<1>::make_empty());
      auto ptr    = output.ptr(0);
#ifdef LEGATE_NO_FUTURES_ON_FB
      *ptr = result;
#else
      cudaMemcpy(ptr, &result, sizeof(Domain), cudaMemcpyHostToDevice);
#endif
    } else {
      ThrustAllocator alloc(Memory::GPU_FB_MEM);
      auto policy = thrust::cuda::par(alloc).on(stream);
      auto ptr    = input.ptr(dom.lo());
      auto result = thrust::minmax_element(policy, ptr, ptr + dom.get_volume());
      INDEX_TY lo, hi;
      cudaMemcpy(&lo, result.first, sizeof(INDEX_TY), cudaMemcpyDeviceToHost);
      cudaMemcpy(&hi, result.second, sizeof(INDEX_TY), cudaMemcpyDeviceToHost);
      Domain output_dom({lo, hi});
      auto output_ptr = output.ptr(0);
#ifdef LEGATE_NO_FUTURES_ON_FB
      *output_ptr = output_dom;
#else
      cudaMemcpy(output_ptr, &output_dom, sizeof(Domain), cudaMemcpyHostToDevice);
#endif
    }
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void BoundsFromPartitionedCoordinates::gpu_variant(TaskContext& context)
{
  bounds_from_partitioned_coordinates_template<VariantKind::GPU>(context);
}

}  // namespace sparse

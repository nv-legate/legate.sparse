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

#include "sparse/array/conv/sorted_coords_to_counts.h"
#include "sparse/array/conv/sorted_coords_to_counts_template.inl"
#include "cuda_help.h"
#include "thrust_allocator.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename ACC, typename INDEX_TY>
__global__ void scatter_reduce(size_t elems, ACC out, INDEX_TY* keys, uint64_t* counts)
{
  auto idx = global_tid_1d();
  if (idx >= elems) return;
  out[keys[idx]] <<= counts[idx];
}

template <LegateTypeCode INDEX_CODE, typename ACC>
struct SortedCoordsToCountsImplBody<VariantKind::GPU, INDEX_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  void operator()(ACC out, AccessorRO<INDEX_TY, 1> in, int64_t max_vals, const Domain& dom)
  {
    // TODO (rohany): We could make a call to unique before this to avoid allocating
    //  O(rows/cols) space. For now this shouldn't be too much of a problem. Unfortunately
    //  unique counting has just recently landed, and I'm not sure when our thrust version
    //  will pick up the change: https://github.com/NVIDIA/thrust/issues/1612.
    auto kind   = Memory::GPU_FB_MEM;
    auto stream = get_cached_stream();
    ThrustAllocator alloc(kind);
    auto policy = thrust::cuda::par(alloc).on(stream);
    DeferredBuffer<coord_t, 1> keys({0, max_vals - 1}, kind);
    DeferredBuffer<uint64_t, 1> counts({0, max_vals - 1}, kind);
    auto result = thrust::reduce_by_key(policy,
                                        in.ptr(dom.lo()),
                                        in.ptr(dom.lo()) + dom.get_volume(),
                                        thrust::make_constant_iterator(1),
                                        keys.ptr(0),
                                        counts.ptr(0));
    auto elems  = result.first - keys.ptr(0);
    auto blocks = get_num_blocks_1d(elems);
    scatter_reduce<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      elems, out, keys.ptr(0), counts.ptr(0));
    CHECK_CUDA_STREAM(stream);
  };
};

/*static*/ void SortedCoordsToCounts::gpu_variant(TaskContext& context)
{
  sorted_coords_to_counts_template<VariantKind::GPU>(context);
}

}  // namespace sparse

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
#include "sparse/util/cuda_help.h"
#include "sparse/util/thrust_allocator.h"

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

namespace sparse {

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
  void operator()(ACC out, AccessorRO<INDEX_TY, 1> in, const Domain& dom)
  {
    auto kind   = Memory::GPU_FB_MEM;
    auto stream = get_cached_stream();
    ThrustAllocator alloc(kind);
    auto policy = thrust::cuda::par(alloc).on(stream);
    // Estimate the maximum space we'll need to store the unique elements in the
    // reduce-by-key operation. To get an estimate here, we take the difference
    // between the min and max coordinate in the input region. In the future,
    // we could do this with a unique count, but that functionality has not yet
    // made it into the thrust repository yet.
    auto in_ptr = in.ptr(dom.lo());
    auto minmax = thrust::minmax_element(policy, in_ptr, in_ptr + dom.get_volume());
    INDEX_TY min, max;
    CHECK_CUDA(
      cudaMemcpyAsync(&min, minmax.first, sizeof(INDEX_TY), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(
      cudaMemcpyAsync(&max, minmax.second, sizeof(INDEX_TY), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    INDEX_TY max_vals = max - min + 1;
    Buffer<coord_t, 1> keys({0, max_vals - 1}, kind);
    Buffer<uint64_t, 1> counts({0, max_vals - 1}, kind);
    auto result = thrust::reduce_by_key(policy,
                                        in_ptr,
                                        in_ptr + dom.get_volume(),
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

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

#include "sparse/spatial/euclidean_distance.h"
#include "sparse/spatial/euclidean_distance_template.inl"
#include "sparse/util/cuda_help.h"
#include "sparse/util/distal_cuda_utils.h"
#include "sparse/util/pitches.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename VAL_TY>
__global__ void accumulate_euclidean_diffs(size_t volume,
                                           Pitches<2> pitches,
                                           Rect<3> iterBounds,
                                           AccessorWO<VAL_TY, 2> out,
                                           AccessorRO<VAL_TY, 2> XA,
                                           AccessorRO<VAL_TY, 2> XB)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) return;

  Point<3> point = pitches.unflatten(idx, iterBounds.lo);
  auto diff      = XA[{point.x, point.z}] - XB[{point.y, point.z}];
  diff           = diff * diff;
  atomicAddWarp(out.ptr({point.x, point.y}), flattenPoint(out, Point<2>{point.x, point.y}), diff);
}

template <typename VAL_TY>
__global__ void elementwise_sqrt(size_t volume,
                                 Pitches<1> pitches,
                                 Rect<2> iterBounds,
                                 AccessorWO<VAL_TY, 2> out)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) return;

  Point<2> point = pitches.unflatten(idx, iterBounds.lo);
  out[point]     = sqrt(out[point]);
}

template <LegateTypeCode VAL_CODE>
struct EuclideanCDistImplBody<VariantKind::GPU, VAL_CODE> {
  using VAL_TY = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 2>& out,
                  const AccessorRO<VAL_TY, 2>& XA,
                  const AccessorRO<VAL_TY, 2>& XB,
                  const Rect<2>& out_rect,
                  const Rect<2>& XA_rect)
  {
    auto stream = get_cached_stream();
    // We'll parallelize over all 3 dimensions and do thread-level
    // reductions into the output.
    {
      Rect<3> iterBounds({out_rect.lo[0], out_rect.lo[1], XA_rect.lo[1]},
                         {out_rect.hi[0], out_rect.hi[1], XA_rect.hi[1]});
      Pitches<2> pitches;
      auto volume = pitches.flatten(iterBounds);
      auto blocks = get_num_blocks_1d(volume);
      accumulate_euclidean_diffs<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, pitches, iterBounds, out, XA, XB);
    }
    {
      Pitches<1> pitches;
      auto volume = pitches.flatten(out_rect);
      auto blocks = get_num_blocks_1d(volume);
      elementwise_sqrt<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, pitches, out_rect, out);
    }
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void EuclideanCDist::gpu_variant(TaskContext& context)
{
  euclidean_distance_template<VariantKind::GPU>(context);
}

}  // namespace sparse

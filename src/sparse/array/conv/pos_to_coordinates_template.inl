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

#pragma once

// Useful for IDEs.
#include "sparse/array/conv/pos_to_coordinates.h"
#include "sparse/util/dispatch.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

namespace sparse {
using namespace legate;

template <typename T>
struct volume : public thrust::unary_function<T, size_t> {
#if defined(__CUDACC__)
  __host__ __device__
#endif
    size_t
    operator()(Legion::Rect<1> x)
  {
    return x.volume();
  }
};

template <typename Policy>
struct ExpandPosToCoordinatesImpl {
  ExpandPosToCoordinatesImpl(const Policy& policy, Memory::Kind tempMemKind)
    : policy(policy), tempMemKind(tempMemKind)
  {
  }

  template <Type::Code INDEX_CODE>
  void operator()(ExpandPosToCoordinatesArgs& args) const
  {
    using INDEX_TY     = legate_type_of<INDEX_CODE>;
    auto pos           = args.pos.read_accessor<Rect<1>, 1>();
    auto result        = args.result.write_accessor<INDEX_TY, 1>();
    auto pos_domain    = args.pos.domain();
    auto result_domain = args.result.domain();

    // Return early if there isn't any work to do. Entering this code
    // with an empty domain results in CUDA errors for the thrust backend.
    if (pos_domain.empty() || result_domain.empty()) return;

    // This implementation of expand was inspired from
    // https://huggingface.co/spaces/ma-xu/LIVE/blob/main/thrust/examples/expand.cu.
    Legion::DeferredBuffer<size_t, 1> volumes({0, pos_domain.get_volume() - 1}, tempMemKind);
    Legion::DeferredBuffer<size_t, 1> offsets({0, pos_domain.get_volume() - 1}, tempMemKind);
    // Initialize all of our arrays.
    thrust::fill(policy, volumes.ptr(0), volumes.ptr(0) + pos_domain.get_volume(), size_t(0));
    thrust::fill(policy, offsets.ptr(0), offsets.ptr(0) + pos_domain.get_volume(), size_t(0));
    thrust::fill(policy,
                 result.ptr(result_domain.lo()),
                 result.ptr(result_domain.lo()) + result_domain.get_volume(),
                 INDEX_TY(0));
    // Transform each pos rectangle into its volume. We have to make a
    // temporary here because not all of the thrust functions accept a
    // transform.
    thrust::transform(policy,
                      pos.ptr(pos_domain.lo()),
                      pos.ptr(pos_domain.lo()) + pos_domain.get_volume(),
                      volumes.ptr(0),
                      volume<Legion::Rect<1>>{});
    // Perform an exclusive scan to find the offsets to write coordinates into.
    thrust::exclusive_scan(
      policy, volumes.ptr(0), volumes.ptr(0) + pos_domain.get_volume(), offsets.ptr(0));
    // Scatter the non-zero counts into their output indices.
    thrust::scatter_if(policy,
                       thrust::counting_iterator<INDEX_TY>(0),
                       thrust::counting_iterator<INDEX_TY>(pos_domain.get_volume()),
                       offsets.ptr(0),
                       volumes.ptr(0),
                       result.ptr(result_domain.lo()));
    // Compute a max-scan over the output indices, filling in holes.
    thrust::inclusive_scan(policy,
                           result.ptr(result_domain.lo()),
                           result.ptr(result_domain.lo()) + result_domain.get_volume(),
                           result.ptr(result_domain.lo()),
                           thrust::maximum<INDEX_TY>{});
    // Gather input values according to the computed indices.
    thrust::gather(policy,
                   result.ptr(result_domain.lo()),
                   result.ptr(result_domain.lo()) + result_domain.get_volume(),
                   thrust::counting_iterator<INDEX_TY>(pos_domain.lo()[0]),
                   result.ptr(result_domain.lo()));
  }

 private:
  const Policy& policy;
  Memory::Kind tempMemKind;
};

template <typename Policy>
static void pos_to_coordinates_template(TaskContext& context,
                                        const Policy& policy,
                                        Memory::Kind tempMemKind)
{
  ExpandPosToCoordinatesArgs args{
    context.outputs()[0],
    context.inputs()[0],
  };
  index_type_dispatch(args.result.code(), ExpandPosToCoordinatesImpl(policy, tempMemKind), args);
}

}  // namespace sparse

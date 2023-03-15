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

#include <thrust/extrema.h>

namespace sparse {

using namespace legate;

template <LegateTypeCode INDEX_CODE>
struct BoundsFromPartitionedCoordinatesImplBody<VariantKind::CPU, INDEX_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  void operator()(const AccessorWO<Domain, 1> output,
                  AccessorRO<INDEX_TY, 1> input,
                  const Domain& dom)
  {
    if (dom.empty()) {
      output[0] = Rect<1>::make_empty();
    } else {
      auto ptr    = input.ptr(dom.lo());
      auto result = thrust::minmax_element(thrust::host, ptr, ptr + dom.get_volume());
      output[0]   = {*result.first, *result.second};
    }
  }
};

/*static*/ void BoundsFromPartitionedCoordinates::cpu_variant(TaskContext& context)
{
  bounds_from_partitioned_coordinates_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  BoundsFromPartitionedCoordinates::register_variants();
}
}  // namespace

}  // namespace sparse

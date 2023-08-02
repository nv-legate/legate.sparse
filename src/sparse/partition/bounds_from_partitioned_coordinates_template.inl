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
#include "sparse/partition/bounds_from_partitioned_coordinates.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, LegateTypeCode INDEX_CODE>
struct BoundsFromPartitionedCoordinatesImplBody;

template <VariantKind KIND>
struct BoundsFromPartitionedCoordinatesImpl {
  template <LegateTypeCode INDEX_CODE>
  void operator()(BoundsFromPartitionedCoordinatesArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    auto output    = args.output.write_accessor<Domain, 1>();
    auto input     = args.input.read_accessor<INDEX_TY, 1>();
    auto dom       = args.input.domain();
    BoundsFromPartitionedCoordinatesImplBody<KIND, INDEX_CODE>()(output, input, dom);
  }
};

template <VariantKind KIND>
static void bounds_from_partitioned_coordinates_template(TaskContext& context)
{
  BoundsFromPartitionedCoordinatesArgs args{
    context.outputs()[0],
    context.inputs()[0],
  };
  index_type_dispatch(args.input.code(), BoundsFromPartitionedCoordinatesImpl<KIND>{}, args);
}

}  // namespace sparse

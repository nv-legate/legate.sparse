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
#include "sparse/array/conv/sorted_coords_to_counts.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, LegateTypeCode INDEX_CODE, typename ACC>
struct SortedCoordsToCountsImplBody;

template <VariantKind KIND>
struct SortedCoordsToCountsImpl {
  template <LegateTypeCode INDEX_CODE>
  void operator()(SortedCoordsToCountsArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    auto output    = args.output.reduce_accessor<SumReduction<uint64_t>, true /* exclusive */, 1>();
    auto input     = args.input.read_accessor<INDEX_TY, 1>();
    if (args.output.domain().empty() || args.input.domain().empty()) return;
    SortedCoordsToCountsImplBody<KIND, INDEX_CODE, decltype(output)>()(
      output, input, args.input.domain());
  }
};

template <VariantKind KIND>
static void sorted_coords_to_counts_template(TaskContext& context)
{
  SortedCoordsToCountsArgs args{
    context.reductions()[0],
    context.inputs()[0],
  };
  index_type_dispatch(args.input.code(), SortedCoordsToCountsImpl<KIND>{}, args);
}

}  // namespace sparse

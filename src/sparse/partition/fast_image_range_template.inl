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
#include "sparse/partition/fast_image_range.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <VariantKind KIND>
struct FastImageRangeImplBody;

template <VariantKind KIND>
struct FastImageRangeImpl {
  void operator()(FastImageRangeArgs& args) const
  {
    auto output = args.output.write_accessor<Domain, 1>();
    auto input  = args.input.read_accessor<Rect<1>, 1>();
    FastImageRangeImplBody<KIND>()(output, input, args.input.domain());
  }
};

template <VariantKind KIND>
static void fast_image_range_template(TaskContext& context)
{
  auto& input = context.inputs()[0];
  if (input.transformed()) { input.remove_transform(); }
  FastImageRangeArgs args{context.outputs()[0], input};
  FastImageRangeImpl<KIND>{}(args);
}

}  // namespace sparse

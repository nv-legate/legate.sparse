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
#include "sparse/array/util/scale_rect.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND>
struct ScaleRect1ImplBody;

template <VariantKind KIND>
struct ScaleRect1Impl {
  void operator()(ScaleRect1Args& args) const
  {
    auto output = args.out.read_write_accessor<Rect<1>, 1>();
    if (args.out.domain().empty()) return;
    ScaleRect1ImplBody<KIND>()(output, args.scale, args.out.shape<1>());
  }
};

template <VariantKind KIND>
static void scale_rect_1_template(TaskContext& context)
{
  auto task  = context.task_;
  auto scale = task->futures[0].get_result<int64_t>();
  ScaleRect1Args args{context.outputs()[0], scale};
  ScaleRect1Impl<KIND>{}(args);
}

}  // namespace sparse

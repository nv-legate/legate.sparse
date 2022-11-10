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
#include "sparse/array/util/unzip_rect.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <VariantKind KIND>
struct UnZipRect1ImplBody;

template <VariantKind KIND>
struct UnZipRect1Impl {
  void operator()(UnZipRect1Args& args) const
  {
    auto out1 = args.out1.write_accessor<int64_t, 1>();
    auto out2 = args.out2.write_accessor<int64_t, 1>();
    auto in   = args.in.read_accessor<Rect<1>, 1>();
    if (args.in.domain().empty()) return;
    UnZipRect1ImplBody<KIND>()(out1, out2, in, args.in.shape<1>());
  }
};

template <VariantKind KIND>
static void unzip_rect_1_template(TaskContext& context)
{
  auto& outputs = context.outputs();
  UnZipRect1Args args{outputs[0], outputs[1], context.inputs()[0]};
  UnZipRect1Impl<KIND>{}(args);
}

}  // namespace sparse

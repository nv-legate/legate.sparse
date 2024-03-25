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
#include "sparse/array/util/zip_to_rect.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, typename VAL>
struct ZipToRect1ImplBody;

template <VariantKind KIND, typename VAL>
struct ZipToRect1Impl {
  void operator()(ZipToRect1Args& args) const
  {
    auto output = args.out.write_accessor<Rect<1>, 1>();
    auto lo     = args.lo.read_accessor<VAL, 1>();
    auto hi     = args.hi.read_accessor<VAL, 1>();
    if (args.out.domain().empty()) return;
    ZipToRect1ImplBody<KIND, VAL>()(output, lo, hi, args.out.shape<1>());
  }
};

template <VariantKind KIND>
static void zip_to_rect_1_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  ZipToRect1Args args{context.outputs()[0], inputs[0], inputs[1]};
  if (inputs[0].type().code == legate::Type::Code::INT64)
    ZipToRect1Impl<KIND, int64_t>{}(args);
  else {
    assert(inputs[0].type().code == legate::Type::Code::UINT64);
    ZipToRect1Impl<KIND, uint64_t>{}(args);
  }
}

}  // namespace sparse

/* Copyright 2021-2022 NVIDIA Corporation
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
#include "sparse/spatial/euclidean_distance.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, Type::Code VAL_CODE>
struct EuclideanCDistImplBody;

template <VariantKind KIND>
struct EuclideanCDistImpl {
  template <Type::Code VAL_CODE>
  void operator()(EuclideanCDistArgs& args) const
  {
    using VAL_TY = legate_type_of<VAL_CODE>;
    auto out     = args.out.write_accessor<VAL_TY, 2>();
    auto XA      = args.XA.read_accessor<VAL_TY, 2>();
    auto XB      = args.XB.read_accessor<VAL_TY, 2>();
    if (args.out.domain().empty() || args.XA.domain().empty() || args.XB.domain().empty()) return;
    EuclideanCDistImplBody<KIND, VAL_CODE>()(out, XA, XB, args.out.shape<2>(), args.XA.shape<2>());
  }
};

template <VariantKind KIND>
static void euclidean_distance_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  EuclideanCDistArgs args{
    context.outputs()[0],
    inputs[0],
    inputs[1],
  };
  value_type_dispatch_no_complex(args.out.code(), EuclideanCDistImpl<KIND>{}, args);
}

}  // namespace sparse

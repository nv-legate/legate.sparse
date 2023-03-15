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
#include "sparse/integrate/runge_kutta.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, LegateTypeCode K_CODE, LegateTypeCode A_CODE>
struct RKCalcDyImplBody;

template <VariantKind KIND>
struct RKCalcDyImpl {
  template <LegateTypeCode K_CODE, LegateTypeCode A_CODE>
  void operator()(RKCalcDyArgs& args) const
  {
    using K_TY = legate_type_of<K_CODE>;
    using A_TY = legate_type_of<A_CODE>;
    auto dy    = args.dy.write_accessor<K_TY, 1>();
    auto K     = args.K.read_accessor<K_TY, 2>();
    auto a     = args.a.read_accessor<A_TY, 1>();
    if (args.K.domain().empty()) return;
    RKCalcDyImplBody<KIND, K_CODE, A_CODE>()(dy, K, a, args.s, args.h, args.K.shape<2>());
  }
};

template <VariantKind KIND>
static void rk_calc_dy_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();
  if (outputs[0].transformed()) { outputs[0].remove_transform(); }
  RKCalcDyArgs args{
    context.outputs()[0],
    inputs[0],
    inputs[1],
    scalars[0].value<int32_t>(),
    scalars[1].value<double>(),
  };
  // TODO (rohany): I'm not sure how this type casting works out right now, so
  //  we'll just use the existing implementation and just instantiate for complex
  //  and double.
  RKCalcDyImpl<KIND> impl;
  impl.template operator()<LegateTypeCode::COMPLEX128_LT, LegateTypeCode::DOUBLE_LT>(args);
}

}  // namespace sparse

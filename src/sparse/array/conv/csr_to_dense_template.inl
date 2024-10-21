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
#include "sparse/array/conv/csr_to_dense.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct CSRToDenseImplBody;

template <VariantKind KIND>
struct CSRToDenseImpl {
  template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
  void operator()(CSRToDenseArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto A_vals = args.A_vals.write_accessor<VAL_TY, 2>();
    auto B_pos  = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd  = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 1>();

    if (args.A_vals.domain().empty()) return;
    CSRToDenseImplBody<KIND, INDEX_CODE, VAL_CODE>()(
      A_vals, B_pos, B_crd, B_vals, args.A_vals.shape<2>());
  }
};

template <VariantKind KIND>
static void csr_to_dense_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (inputs[0].transformed()) { inputs[0].remove_transform(); }
  CSRToDenseArgs args{context.outputs()[0], inputs[0], inputs[1], inputs[2]};
  index_type_value_type_dispatch(
    args.B_crd.code(), args.A_vals.code(), CSRToDenseImpl<KIND>{}, args);
}

}  // namespace sparse

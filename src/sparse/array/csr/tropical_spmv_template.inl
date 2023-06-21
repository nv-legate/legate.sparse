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
#include "sparse/array/csr/tropical_spmv.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, Type::Code INDEX_CODE>
struct CSRTropicalSpMVImplBody;

template <VariantKind KIND>
struct CSRTropicalSpMVImpl {
  template <Type::Code INDEX_CODE>
  void operator()(CSRTropicalSpMVArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;

    auto y     = args.y.write_accessor<INDEX_TY, 2>();
    auto A_pos = args.A_pos.read_accessor<Rect<1>, 1>();
    auto A_crd = args.A_crd.read_accessor<INDEX_TY, 1>();
    auto x     = args.x.read_accessor<INDEX_TY, 2>();

    assert(args.y.domain().dense());
    if (args.y.domain().empty()) return;

    CSRTropicalSpMVImplBody<KIND, INDEX_CODE>()(y, A_pos, A_crd, x, args.y.shape<2>());
  }
};

template <VariantKind KIND>
static void csr_tropical_spmv_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (inputs[0].transformed()) { inputs[0].remove_transform(); }
  CSRTropicalSpMVArgs args{context.outputs()[0], inputs[0], inputs[1], inputs[2]};
  index_type_dispatch(args.A_crd.code(), CSRTropicalSpMVImpl<KIND>{}, args);
}

}  // namespace sparse

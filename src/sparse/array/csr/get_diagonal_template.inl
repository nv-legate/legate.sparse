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
#include "sparse/array/csr/get_diagonal.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct GetCSRDiagonalImplBody;

template <VariantKind KIND>
struct GetCSRDiagonalImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(GetCSRDiagonalArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto diag = args.diag.write_accessor<VAL_TY, 1>();
    auto pos  = args.pos.read_accessor<Rect<1>, 1>();
    auto crd  = args.crd.read_accessor<INDEX_TY, 1>();
    auto vals = args.vals.read_accessor<VAL_TY, 1>();

    assert(args.diag.domain().dense());
    if (args.diag.domain().empty()) return;

    GetCSRDiagonalImplBody<KIND, INDEX_CODE, VAL_CODE>()(
      diag, pos, crd, vals, args.diag.shape<1>());
  }
};

template <VariantKind KIND>
static void get_csr_diagonal_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  GetCSRDiagonalArgs args{context.outputs()[0], inputs[0], inputs[1], inputs[2]};
  index_type_value_type_dispatch(
    args.crd.code(), args.diag.code(), GetCSRDiagonalImpl<KIND>{}, args);
}
}  // namespace sparse

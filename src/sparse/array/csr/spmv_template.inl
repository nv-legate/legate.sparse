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
#include "sparse/array/csr/spmv.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct CSRSpMVRowSplitImplBody;

template <VariantKind KIND>
struct CSRSpMVRowSplitImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(CSRSpMVRowSplitArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto y      = args.y.write_accessor<VAL_TY, 1>();
    auto A_pos  = args.A_pos.read_accessor<Rect<1>, 1>();
    auto A_crd  = args.A_crd.read_accessor<INDEX_TY, 1>();
    auto A_vals = args.A_vals.read_accessor<VAL_TY, 1>();
    auto x      = args.x.read_accessor<VAL_TY, 1>();

    assert(args.y.domain().dense());
    if (args.y.domain().empty()) return;

    CSRSpMVRowSplitImplBody<KIND, INDEX_CODE, VAL_CODE>()(
      y, A_pos, A_crd, A_vals, x, args.y.shape<1>());
  }
};

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename ACC>
struct CSRSpMVColSplitImplBody;

template <VariantKind KIND>
struct CSRSpMVColSplitImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(CSRSpMVColSplitArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto y      = args.y.reduce_accessor<SumReduction<VAL_TY>, true /* exclusive */, 1>();
    auto A_pos  = args.A_pos.read_accessor<Rect<1>, 1>();
    auto A_crd  = args.A_crd.read_accessor<INDEX_TY, 1>();
    auto A_vals = args.A_vals.read_accessor<VAL_TY, 1>();
    auto x      = args.x.read_accessor<VAL_TY, 1>();

    if (args.y.domain().empty()) return;

    CSRSpMVColSplitImplBody<KIND, INDEX_CODE, VAL_CODE, decltype(y)>()(
      y, A_pos, A_crd, A_vals, x, args.y.shape<1>(), args.A_crd.shape<1>(), args.x.shape<1>());
  }
};

template <VariantKind KIND>
static void csr_spmv_row_split_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  CSRSpMVRowSplitArgs args{context.outputs()[0], inputs[0], inputs[1], inputs[2], inputs[3]};
  index_type_value_type_dispatch(
    args.A_crd.code(), args.y.code(), CSRSpMVRowSplitImpl<KIND>{}, args);
}

template <VariantKind KIND>
static void csr_spmv_col_split_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  CSRSpMVColSplitArgs args{context.reductions()[0], inputs[0], inputs[1], inputs[2], inputs[3]};
  index_type_value_type_dispatch(
    args.A_crd.code(), args.y.code(), CSRSpMVColSplitImpl<KIND>{}, args);
}

}  // namespace sparse

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
#include "sparse/array/csr/spmm.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpMMCSRImplBody;

template <VariantKind KIND>
struct SpMMCSRImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpMMCSRArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto A_vals = args.A_vals.write_accessor<VAL_TY, 2>();
    auto B_pos  = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd  = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 1>();
    auto C_vals = args.C_vals.read_accessor<VAL_TY, 2>();

    if (args.A_vals.domain().empty() || args.C_vals.domain().empty()) { return; }
    SpMMCSRImplBody<KIND, INDEX_CODE, VAL_CODE>()(
      A_vals, B_pos, B_crd, B_vals, C_vals, args.A_vals.shape<2>(), args.C_vals.shape<2>());
  }
};

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename ACC>
struct SpMMDenseCSRImplBody;

template <VariantKind KIND>
struct SpMMDenseCSRImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpMMDenseCSRArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto A_vals = args.A_vals.reduce_accessor<SumReduction<VAL_TY>, true /* exclusive */, 2>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 2>();
    auto C_pos  = args.C_pos.read_accessor<Rect<1>, 1>();
    auto C_crd  = args.C_crd.read_accessor<INDEX_TY, 1>();
    auto C_vals = args.C_vals.read_accessor<VAL_TY, 1>();

    if (args.A_vals.domain().empty() || args.B_vals.domain().empty()) { return; }
    SpMMDenseCSRImplBody<KIND, INDEX_CODE, VAL_CODE, decltype(A_vals)>()(
      A_vals, B_vals, C_pos, C_crd, C_vals, args.B_vals.shape<2>());
  }
};

template <VariantKind KIND>
static void spmm_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (inputs[0].transformed()) { inputs[0].remove_transform(); }
  SpMMCSRArgs args{context.outputs()[0], inputs[0], inputs[1], inputs[2], inputs[3]};
  index_type_value_type_dispatch(args.B_crd.code(), args.A_vals.code(), SpMMCSRImpl<KIND>{}, args);
}

template <VariantKind KIND>
static void spmm_dense_csr_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (inputs[1].transformed()) { inputs[1].remove_transform(); }
  SpMMDenseCSRArgs args{context.reductions()[0], inputs[0], inputs[1], inputs[2], inputs[3]};
  index_type_value_type_dispatch(
    args.C_crd.code(), args.A_vals.code(), SpMMDenseCSRImpl<KIND>{}, args);
}

}  // namespace sparse

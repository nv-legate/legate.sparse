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
#include "sparse/array/conv/dense_to_csr.h"
#include "sparse/util/dispatch.h"
#include "sparse/util/typedefs.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, Type::Code VAL_CODE>
struct DenseToCSRNNZImplBody;

template <VariantKind KIND>
struct DenseToCSRNNZImpl {
  template <Type::Code VAL_CODE>
  void operator()(DenseToCSRNNZArgs& args) const
  {
    using VAL_TY = legate_type_of<VAL_CODE>;

    auto nnz    = args.nnz.write_accessor<nnz_ty, 1>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 2>();

    if (args.nnz.domain().empty()) return;
    DenseToCSRNNZImplBody<KIND, VAL_CODE>()(nnz, B_vals, args.B_vals.shape<2>());
  }
};

template <VariantKind KIND, Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct DenseToCSRImplBody;

template <VariantKind KIND>
struct DenseToCSRImpl {
  template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
  void operator()(DenseToCSRArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto A_pos  = args.A_pos.read_write_accessor<Rect<1>, 1>();
    auto A_crd  = args.A_crd.write_accessor<INDEX_TY, 1>();
    auto A_vals = args.A_vals.write_accessor<VAL_TY, 1>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 2>();

    if (args.A_pos.domain().empty()) return;
    DenseToCSRImplBody<KIND, INDEX_CODE, VAL_CODE>()(
      A_pos, A_crd, A_vals, B_vals, args.B_vals.shape<2>());
  }
};

template <VariantKind KIND>
static void dense_to_csr_nnz_template(TaskContext& context)
{
  auto& outputs = context.outputs();
  // We have to promote the nnz region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (outputs[0].transformed()) { outputs[0].remove_transform(); }
  DenseToCSRNNZArgs args{outputs[0], context.inputs()[0]};
  value_type_dispatch(args.B_vals.code(), DenseToCSRNNZImpl<KIND>{}, args);
}

template <VariantKind KIND>
static void dense_to_csr_template(TaskContext& context)
{
  auto& outputs = context.outputs();
  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (outputs[0].transformed()) { outputs[0].remove_transform(); }
  DenseToCSRArgs args{outputs[0], outputs[1], outputs[2], context.inputs()[0]};
  index_type_value_type_dispatch(
    args.A_crd.code(), args.B_vals.code(), DenseToCSRImpl<KIND>{}, args);
}

}  // namespace sparse

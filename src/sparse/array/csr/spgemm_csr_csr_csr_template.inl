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
#include "sparse/array/csr/spgemm_csr_csr_csr.h"
#include "sparse/util/dispatch.h"
#include "sparse/util/typedefs.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, LegateTypeCode INDEX_CODE>
struct SpGEMMCSRxCSRxCSRNNZImplBody;

template <VariantKind KIND>
struct SpGEMMCSRxCSRxCSRNNZImpl {
  template <LegateTypeCode INDEX_CODE>
  void operator()(SpGEMMCSRxCSRxCSRNNZArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;

    auto nnz   = args.nnz.write_accessor<nnz_ty, 1>();
    auto B_pos = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto C_pos = args.C_pos.read_accessor<Rect<1>, 1>();
    auto C_crd = args.C_crd.read_accessor<INDEX_TY, 1>();

    SpGEMMCSRxCSRxCSRNNZImplBody<KIND, INDEX_CODE>()(
      nnz, B_pos, B_crd, C_pos, C_crd, args.B_pos.shape<1>(), args.C_crd.shape<1>());
  }
};

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpGEMMCSRxCSRxCSRImplBody;

template <VariantKind KIND>
struct SpGEMMCSRxCSRxCSRImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpGEMMCSRxCSRxCSRArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto A_pos  = args.A_pos.read_write_accessor<Rect<1>, 1>();
    auto A_crd  = args.A_crd.write_accessor<INDEX_TY, 1>();
    auto A_vals = args.A_vals.write_accessor<VAL_TY, 1>();
    auto B_pos  = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd  = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 1>();
    auto C_pos  = args.C_pos.read_accessor<Rect<1>, 1>();
    auto C_crd  = args.C_crd.read_accessor<INDEX_TY, 1>();
    auto C_vals = args.C_vals.read_accessor<VAL_TY, 1>();

    SpGEMMCSRxCSRxCSRImplBody<KIND, INDEX_CODE, VAL_CODE>()(A_pos,
                                                            A_crd,
                                                            A_vals,
                                                            B_pos,
                                                            B_crd,
                                                            B_vals,
                                                            C_pos,
                                                            C_crd,
                                                            C_vals,
                                                            args.B_pos.shape<1>(),
                                                            args.C_crd.shape<1>());
  }
};

template <VariantKind KIND>
static void spgemm_csr_csr_csr_nnz_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  SpGEMMCSRxCSRxCSRNNZArgs args{
    context.outputs()[0],
    inputs[0],
    inputs[1],
    inputs[2],
    inputs[3],
  };
  index_type_dispatch(args.B_crd.code(), SpGEMMCSRxCSRxCSRNNZImpl<KIND>{}, args);
}

template <VariantKind KIND>
static void spgemm_csr_csr_csr_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  SpGEMMCSRxCSRxCSRArgs args{
    outputs[0],
    outputs[1],
    outputs[2],
    inputs[0],
    inputs[1],
    inputs[2],
    inputs[3],
    inputs[4],
    inputs[5],
  };
  index_type_value_type_dispatch(
    args.A_crd.code(), args.A_vals.code(), SpGEMMCSRxCSRxCSRImpl<KIND>{}, args);
}

}  // namespace sparse

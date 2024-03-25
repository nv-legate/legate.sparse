/* Copyright 2021-2023 NVIDIA Corporation
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
#include "sparse/array/csc/spmm.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, Type::Code INDEX_CODE, Type::Code VAL_CODE, typename ACC>
struct SpMMCSCImplBody;

template <VariantKind KIND>
struct SpMMCSCImpl {
  template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
  void operator()(SpMMCSCArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    // If we're running with OMP's, then we need to use an non-exclusive
    // accessor instead of an exclusive one.
    auto A_vals = [&args]() -> auto {
      if constexpr (KIND == VariantKind::CPU) {
        return args.A_vals.reduce_accessor<SumReduction<VAL_TY>, true, 2>();
      } else {
        return args.A_vals.reduce_accessor<SumReduction<VAL_TY>, false, 2>();
      }
    }();
    auto B_pos  = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd  = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 1>();
    auto C_vals = args.C_vals.read_accessor<VAL_TY, 2>();

    if (args.A_vals.domain().empty() || args.C_vals.domain().empty() ||
        args.B_vals.domain().empty()) {
      return;
    }
    SpMMCSCImplBody<KIND, INDEX_CODE, VAL_CODE, decltype(A_vals)>()(A_vals,
                                                                    B_pos,
                                                                    B_crd,
                                                                    B_vals,
                                                                    C_vals,
                                                                    args.A_vals.shape<2>(),
                                                                    args.C_vals.shape<2>(),
                                                                    args.B_vals.shape<1>());
  }
};

template <VariantKind KIND>
static void spmm_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (inputs[0].transformed()) { inputs[0].remove_transform(); }
  SpMMCSCArgs args{context.reductions()[0],
                   inputs[0],
                   inputs[1],
                   inputs[2],
                   inputs[3],
                   context.scalars()[0].value<int64_t>()};
  index_type_value_type_dispatch(args.B_crd.code(), args.A_vals.code(), SpMMCSCImpl<KIND>{}, args);
}

}  // namespace sparse

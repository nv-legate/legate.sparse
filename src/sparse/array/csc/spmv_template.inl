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
#include "sparse/array/csc/spmv.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename ACC>
struct CSCSpMVColSplitImplBody;

template <VariantKind KIND>
struct CSCSpMVColSplitImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(CSCSpMVColSplitArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto y = [&args]() -> auto
    {
      if constexpr (KIND == VariantKind::CPU) {
        return args.y.reduce_accessor<SumReduction<VAL_TY>, true /* exclusive */, 1>();
      } else {
        // Since we're scattering into the output vector y, we'll have an initial
        // implementation that just does atomic reductions into the output. I think
        // that is a better approach than blowing up the output memory by a factor
        // equal to the number of threads. It's possible that a factor of threads
        // blowup isn't even _that_ bad, it's what the SpGEMM implementation does
        // as well. We could even use an index set list to only reduce over the
        // set entries. However, it might better in the end to just convert matrices
        // back into CSR up front.
        return args.y.reduce_accessor<SumReduction<VAL_TY>, false /* exclusive */, 1>();
      }
    }
    ();
    auto A_pos  = args.A_pos.read_accessor<Rect<1>, 1>();
    auto A_crd  = args.A_crd.read_accessor<INDEX_TY, 1>();
    auto A_vals = args.A_vals.read_accessor<VAL_TY, 1>();
    auto x      = args.x.read_accessor<VAL_TY, 1>();

    assert(args.x.domain().dense());
    if (args.x.domain().empty()) return;

    CSCSpMVColSplitImplBody<KIND, INDEX_CODE, VAL_CODE, decltype(y)>()(
      y, A_pos, A_crd, A_vals, x, args.x.shape<1>());
  }
};

template <VariantKind KIND>
static void csc_spmv_col_split_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  CSCSpMVColSplitArgs args{context.reductions()[0], inputs[0], inputs[1], inputs[2], inputs[3]};
  index_type_value_type_dispatch(
    args.A_crd.code(), args.y.code(), CSCSpMVColSplitImpl<KIND>{}, args);
}

}  // namespace sparse

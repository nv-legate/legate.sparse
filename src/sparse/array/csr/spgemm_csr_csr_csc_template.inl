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
#include "sparse/array/csr/spgemm_csr_csr_csc.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace legate;

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpGEMMCSRxCSRxCSCLocalTilesImplBody;

template <VariantKind KIND>
struct SpGEMMCSRxCSRxCSCLocalTilesImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpGEMMCSRxCSRxCSCLocalTilesArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;
    auto B_pos     = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd     = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto B_vals    = args.B_vals.read_accessor<VAL_TY, 1>();
    auto C_pos     = args.C_pos.read_accessor<Rect<1>, 1>();
    auto C_crd     = args.C_crd.read_accessor<INDEX_TY, 1>();
    auto C_vals    = args.C_vals.read_accessor<VAL_TY, 1>();
    SpGEMMCSRxCSRxCSCLocalTilesImplBody<KIND, INDEX_CODE, VAL_CODE>()(args.A_pos,
                                                                      args.A_crd,
                                                                      args.A_vals,
                                                                      B_pos,
                                                                      B_crd,
                                                                      B_vals,
                                                                      C_pos,
                                                                      C_crd,
                                                                      C_vals,
                                                                      args.B_pos.shape<1>(),
                                                                      args.C_pos.shape<1>());
  }
};

template <VariantKind KIND>
struct SpGEMMCSRxCSRxCSCCommComputeImplBody;

template <VariantKind KIND>
struct SpGEMMCSRxCSRxCSCCommComputeImpl {
  void operator()(SpGEMMCSRxCSRxCSCCommComputeArgs&& args) const
  {
    auto out = args.out.write_accessor<Rect<1>, 3>();
    SpGEMMCSRxCSRxCSCCommComputeImplBody<KIND>()(
      out, args.pos, args.global_pos, args.gx, args.gy, args.out.shape<3>().lo);
  }
};

template <VariantKind KIND, LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpGEMMCSRxCSRxCSCShuffleImplBody;

template <VariantKind KIND>
struct SpGEMMCSRxCSRxCSCShuffleImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SpGEMMCSRxCSRxCSCShuffleArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;
    // TODO (rohany): I want a sparse instance here.
    auto global_pos  = args.global_pos.read_accessor<Rect<1>, 1>();
    auto global_crd  = args.global_crd.read_accessor<INDEX_TY, 1>();
    auto global_vals = args.global_vals.read_accessor<VAL_TY, 1>();
    SpGEMMCSRxCSRxCSCShuffleImplBody<KIND, INDEX_CODE, VAL_CODE>()(args.out_pos,
                                                                   args.out_crd,
                                                                   args.out_vals,
                                                                   global_pos,
                                                                   global_crd,
                                                                   global_vals,
                                                                   args.global_pos.domain());
  }
};

template <VariantKind KIND>
static void spgemm_csr_csr_csc_local_tiles_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  inputs[0].remove_transform();
  inputs[3].remove_transform();
  SpGEMMCSRxCSRxCSCLocalTilesArgs args{
    outputs[0],
    outputs[1],
    outputs[2],
    inputs[0],
    inputs[1],
    inputs[2],
    inputs[3],
    inputs[4],
    inputs[5],
    context.scalars()[0].value<int64_t>(),
  };
  index_type_value_type_dispatch(
    args.A_crd.code(), args.A_vals.code(), SpGEMMCSRxCSRxCSCLocalTilesImpl<KIND>{}, args);
}

template <VariantKind KIND>
static void spgemm_csr_csr_csc_comm_compute_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& scalars = context.scalars();
  SpGEMMCSRxCSRxCSCCommComputeImpl<KIND>{}({
    context.outputs()[0],
    inputs[0],
    inputs[1],
    scalars[0].value<int32_t>(),
    scalars[1].value<int32_t>(),
  });
}

template <VariantKind KIND>
static void spgemm_csr_csr_csc_shuffle_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  SpGEMMCSRxCSRxCSCShuffleArgs args{
    outputs[0],
    outputs[1],
    outputs[2],
    inputs[0],
    inputs[1],
    inputs[2],
  };
  index_type_value_type_dispatch(
    args.out_crd.code(), args.out_vals.code(), SpGEMMCSRxCSRxCSCShuffleImpl<KIND>{}, args);
}

}  // namespace sparse

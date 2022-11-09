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
#include "sparse/array/conv/coo_to_dense.h"
#include "sparse/util/dispatch.h"

namespace sparse {

using namespace Legion;
using namespace legate;

// Since we only support this on single-threaded CPUs
// right now, don't worry about templating over the
// variant kind.
struct COOToDenseImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(COOToDenseArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto result = args.result.write_accessor<VAL_TY, 2>();
    auto rows   = args.rows.read_accessor<INDEX_TY, 1>();
    auto cols   = args.cols.read_accessor<INDEX_TY, 1>();
    auto vals   = args.vals.read_accessor<VAL_TY, 1>();
    auto dom    = args.rows.domain();

    for (coord_t pos = dom.lo()[0]; pos < dom.hi()[0] + 1; pos++) {
      auto i         = rows[pos];
      auto j         = cols[pos];
      auto val       = vals[pos];
      result[{i, j}] = val;
    }
  }
};

static void coo_to_dense_template(TaskContext& context)
{
  auto& inputs = context.inputs();
  COOToDenseArgs args{
    context.outputs()[0],
    inputs[0],
    inputs[1],
    inputs[2],
  };
  index_type_value_type_dispatch(args.rows.code(), args.vals.code(), COOToDenseImpl{}, args);
}

}  // namespace sparse

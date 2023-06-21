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

#include "sparse/array/csc/spmv.h"
#include "sparse/array/csc/spmv_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE, Type::Code VAL_CODE, typename ACC>
struct CSCSpMVColSplitImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(ACC y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 1>& x,
                  const Rect<1>& rect)
  {
    for (coord_t j = rect.lo[0]; j < rect.hi[0] + 1; j++) {
      for (size_t iA = A_pos[j].lo; iA < A_pos[j].hi + 1; iA++) {
        auto i = A_crd[iA];
        y[i] <<= A_vals[iA] * x[j];
      }
    }
  }
};

/*static*/ void CSCSpMVColSplit::cpu_variant(TaskContext& context)
{
  csc_spmv_col_split_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  CSCSpMVColSplit::register_variants();
}
}  // namespace

}  // namespace sparse

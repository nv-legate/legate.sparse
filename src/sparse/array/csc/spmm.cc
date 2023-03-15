/* Copyright 2023 NVIDIA Corporation
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

#include "sparse/array/csc/spmm.h"
#include "sparse/array/csc/spmm_template.inl"

namespace sparse {

using namespace legate;

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename ACC>
struct SpMMCSCImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(ACC A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<VAL_TY, 2>& C_vals,
                  const Rect<2>& A_rect,
                  const Rect<2>& C_rect,
                  const Rect<1>& _)
  {
    // We've already filled the output with zero, so we can just
    // accumulate into it directly.
    for (auto k = C_rect.lo[0]; k < C_rect.hi[0] + 1; k++) {
      for (size_t iB = B_pos[k].lo; iB < B_pos[k].hi + 1; iB++) {
        auto i     = B_crd[iB];
        auto B_val = B_vals[iB];
        for (auto j = A_rect.lo[1]; j < A_rect.hi[1] + 1; j++) {
          A_vals[{i, j}] <<= B_val * C_vals[{k, j}];
        }
      }
    }
  }
};

/*static*/ void SpMMCSC::cpu_variant(TaskContext& context)
{
  spmm_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { SpMMCSC::register_variants(); }
}  // namespace

}  // namespace sparse

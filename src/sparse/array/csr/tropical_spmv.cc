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

#include "sparse/array/csr/tropical_spmv.h"
#include "sparse/array/csr/tropical_spmv_template.inl"

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE>
struct CSRTropicalSpMVImplBody<VariantKind::CPU, INDEX_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;

  void operator()(const AccessorWO<INDEX_TY, 2>& y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<INDEX_TY, 2>& x,
                  const Rect<2>& rect)
  {
    INDEX_TY num_fields = rect.hi[1] - rect.lo[1] + 1;
    for (INDEX_TY i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      // Initialize the output.
      for (INDEX_TY f = 0; f < num_fields; f++) { y[{i, f}] = 0; }
      for (size_t jpos = A_pos[i].lo; jpos < A_pos[i].hi + 1; jpos++) {
        auto j         = A_crd[jpos];
        bool y_greater = true;
        for (INDEX_TY f = 0; f < num_fields; f++) {
          if (y[{i, f}] > x[{j, f}]) {
            y_greater = true;
            break;
          } else if (y[{i, f}] < x[{j, f}]) {
            y_greater = false;
            break;
          }
          // Else the fields are equal, so move onto the next field.
        }
        if (!y_greater) {
          for (INDEX_TY f = 0; f < num_fields; f++) { y[{i, f}] = x[{j, f}]; }
        }
      }
    }
  }
};

/*static*/ void CSRTropicalSpMV::cpu_variant(TaskContext& context)
{
  csr_tropical_spmv_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  CSRTropicalSpMV::register_variants();
}
}  // namespace

}  // namespace sparse

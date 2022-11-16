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

#include "sparse/array/conv/csr_to_dense.h"
#include "sparse/array/conv/csr_to_dense_template.inl"

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct CSRToDenseImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 2>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const Rect<2>& rect)
  {
    // Initialize the output array.
    for (INDEX_TY i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      for (INDEX_TY j = rect.lo[1]; j < rect.hi[1] + 1; j++) { A_vals[{i, j}] = 0.0; }
    }
    // Do the conversion.
    for (INDEX_TY i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      for (size_t jB = B_pos[i].lo; jB < B_pos[i].hi + 1; jB++) {
        INDEX_TY j     = B_crd[jB];
        A_vals[{i, j}] = B_vals[jB];
      }
    }
  }
};

/*static*/ void CSRToDense::cpu_variant(TaskContext& context)
{
  csr_to_dense_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { CSRToDense::register_variants(); }

}  // namespace

}  // namespace sparse

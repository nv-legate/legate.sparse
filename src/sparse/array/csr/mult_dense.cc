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

#include "sparse/array/csr/mult_dense.h"
#include "sparse/array/csr/mult_dense_template.inl"

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct ElemwiseMultCSRDenseArgsImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<VAL_TY, 2>& C_vals,
                  const Rect<1>& rect,
                  const Rect<1>& _)
  {
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      for (coord_t jB = B_pos[i].lo; jB < B_pos[i].hi + 1; jB++) {
        INDEX_TY j = B_crd[jB];
        A_vals[jB] = B_vals[jB] * C_vals[{i, j}];
      }
    }
  }
};

/*static*/ void ElemwiseMultCSRDense::cpu_variant(TaskContext& context)
{
  elemwise_mult_dense_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ElemwiseMultCSRDense::register_variants();
}
}  // namespace

}  // namespace sparse

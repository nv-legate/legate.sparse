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

#include "sparse/array/csr/get_diagonal.h"
#include "sparse/array/csr/get_diagonal_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct GetCSRDiagonalImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& diag,
                  const AccessorRO<Rect<1>, 1>& pos,
                  const AccessorRO<INDEX_TY, 1>& crd,
                  const AccessorRO<VAL_TY, 1>& vals,
                  const Rect<1>& rect)
  {
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      diag[i] = 0.0;
      for (size_t j_pos = pos[i].lo; j_pos < pos[i].hi + 1; j_pos++) {
        if (crd[j_pos] == i) { diag[i] = vals[j_pos]; }
      }
    }
  }
};

/*static*/ void GetCSRDiagonal::cpu_variant(TaskContext& context)
{
  get_csr_diagonal_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  GetCSRDiagonal::register_variants();
}
}  // namespace

}  // namespace sparse

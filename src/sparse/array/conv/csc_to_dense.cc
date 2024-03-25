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

#include "sparse/array/conv/csc_to_dense.h"
#include "sparse/array/conv/csc_to_dense_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct CSCToDenseImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
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
    for (INDEX_TY j = rect.lo[1]; j < rect.hi[1] + 1; j++) {
      for (size_t iB = B_pos[j].lo; iB < B_pos[j].hi + 1; iB++) {
        INDEX_TY i     = B_crd[iB];
        A_vals[{i, j}] = B_vals[iB];
      }
    }
  }
};

/*static*/ void CSCToDense::cpu_variant(TaskContext& context)
{
  csc_to_dense_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { CSCToDense::register_variants(); }

}  // namespace

}  // namespace sparse

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

#include "sparse/array/csr/spmm.h"
#include "sparse/array/csr/spmm_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct SpMMCSRImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 2>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<VAL_TY, 2>& C_vals,
                  const Rect<2>& A_rect,
                  const Rect<2>& C_rect)
  {
    // Zero out the output array.
    for (auto i = A_rect.lo[0]; i < A_rect.hi[0] + 1; i++) {
      for (auto j = C_rect.lo[1]; j < C_rect.hi[1] + 1; j++) {
        A_vals[{i, j}] = static_cast<VAL_TY>(0);
      }
    }
    for (auto i = A_rect.lo[0]; i < A_rect.hi[0] + 1; i++) {
      for (size_t kB = B_pos[i].lo; kB < B_pos[i].hi + 1; kB++) {
        auto k = B_crd[kB];
        for (auto j = C_rect.lo[1]; j < C_rect.hi[1] + 1; j++) {
          A_vals[{i, j}] += B_vals[kB] * C_vals[{k, j}];
        }
      }
    }
  }
};

template <Type::Code INDEX_CODE, Type::Code VAL_CODE, typename ACC>
struct SpMMDenseCSRImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(ACC A_vals,
                  const AccessorRO<VAL_TY, 2>& B_vals,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorRO<INDEX_TY, 1>& C_crd,
                  const AccessorRO<VAL_TY, 1>& C_vals,
                  const Rect<2>& rect,
                  const Rect<1>& _)
  {
    for (auto k = rect.lo[1]; k < rect.hi[1] + 1; k++) {
      for (size_t jB = C_pos[k].lo; jB < C_pos[k].hi + 1; jB++) {
        auto j = C_crd[jB];
        for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
          A_vals[{i, j}] <<= B_vals[{i, k}] * C_vals[jB];
        }
      }
    }
  }
};

/*static*/ void SpMMCSR::cpu_variant(TaskContext& context)
{
  spmm_template<VariantKind::CPU>(context);
}

/*static*/ void SpMMDenseCSR::cpu_variant(TaskContext& context)
{
  spmm_dense_csr_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  SpMMCSR::register_variants();
  SpMMDenseCSR::register_variants();
}
}  // namespace

}  // namespace sparse

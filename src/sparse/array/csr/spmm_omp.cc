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

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpMMCSRImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE> {
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
#pragma omp parallel for schedule(static) collapse(2)
    for (auto i = A_rect.lo[0]; i < A_rect.hi[0] + 1; i++) {
      for (auto j = C_rect.lo[1]; j < C_rect.hi[1] + 1; j++) {
        A_vals[{i, j}] = static_cast<VAL_TY>(0);
      }
    }
#pragma omp parallel for schedule(monotonic : dynamic, 128)
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

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename ACC>
struct SpMMDenseCSRImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE, ACC> {
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
// This loop ordering (i, k, j) allows for in-order accessing of all
// three tensors and avoids atomic reductions into the output at the
// cost of reading the sparse tensor C i times. This assumes that i
// is generally small.
#pragma omp parallel for schedule(static)
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      for (auto k = rect.lo[1]; k < rect.hi[1] + 1; k++) {
        for (size_t jB = C_pos[k].lo; jB < C_pos[k].hi + 1; jB++) {
          INDEX_TY j = C_crd[jB];
          A_vals[{i, j}] <<= B_vals[{i, k}] * C_vals[jB];
        }
      }
    }
  }
};

/*static*/ void SpMMCSR::omp_variant(TaskContext& context)
{
  spmm_template<VariantKind::OMP>(context);
}

/*static*/ void SpMMDenseCSR::omp_variant(TaskContext& context)
{
  spmm_dense_csr_template<VariantKind::OMP>(context);
}

}  // namespace sparse

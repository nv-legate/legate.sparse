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

#include "sparse/array/csr/mult.h"
#include "sparse/array/csr/mult_template.inl"

namespace sparse {

using namespace legate;

template <LegateTypeCode INDEX_CODE>
struct ElemwiseMultCSRCSRNNZImplBody<VariantKind::OMP, INDEX_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  void operator()(const AccessorWO<nnz_ty, 1>& nnz,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorRO<INDEX_TY, 1>& C_crd,
                  const Rect<1>& rect)
  {
#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      size_t num_nnz = 0;
      size_t jB      = B_pos[i].lo;
      size_t pB2_end = B_pos[i].hi + 1;
      size_t jC      = C_pos[i].lo;
      size_t pC2_end = C_pos[i].hi + 1;
      while (jB < pB2_end && jC < pC2_end) {
        INDEX_TY jB0 = B_crd[jB];
        INDEX_TY jC0 = C_crd[jC];
        INDEX_TY j   = std::min(jB0, jC0);
        if (jB0 == j && jC0 == j) { num_nnz++; }
        jB += (size_t)(jB0 == j);
        jC += (size_t)(jC0 == j);
      }
      nnz[i] = num_nnz;
    }
  }
};

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct ElemwiseMultCSRCSRImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;
  void operator()(const AccessorRW<Rect<1>, 1>& A_pos,
                  const AccessorWO<INDEX_TY, 1>& A_crd,
                  const AccessorWO<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorRO<INDEX_TY, 1>& C_crd,
                  const AccessorRO<VAL_TY, 1>& C_vals,
                  const Rect<1>& rect)
  {
#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      size_t nnz_pos = A_pos[i].lo;
      size_t jB      = B_pos[i].lo;
      size_t pB2_end = B_pos[i].hi + 1;
      size_t jC      = C_pos[i].lo;
      size_t pC2_end = C_pos[i].hi + 1;
      while (jB < pB2_end && jC < pC2_end) {
        INDEX_TY jB0 = B_crd[jB];
        INDEX_TY jC0 = C_crd[jC];
        INDEX_TY j   = std::min(jB0, jC0);
        if (jB0 == j && jC0 == j) {
          A_crd[nnz_pos]  = j;
          A_vals[nnz_pos] = B_vals[jB] * C_vals[jC];
          nnz_pos++;
        }
        jB += (size_t)(jB0 == j);
        jC += (size_t)(jC0 == j);
      }
    }
  }
};

void ElemwiseMultCSRCSRNNZ::omp_variant(TaskContext& context)
{
  csr_elem_mult_nnz_template<VariantKind::OMP>(context);
}

void ElemwiseMultCSRCSR::omp_variant(TaskContext& context)
{
  csr_elem_mult_template<VariantKind::OMP>(context);
}

}  // namespace sparse

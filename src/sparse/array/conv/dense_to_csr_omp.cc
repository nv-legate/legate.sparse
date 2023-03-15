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

#include "sparse/array/conv/dense_to_csr.h"
#include "sparse/array/conv/dense_to_csr_template.inl"

namespace sparse {

using namespace legate;

template <LegateTypeCode VAL_CODE>
struct DenseToCSRNNZImplBody<VariantKind::OMP, VAL_CODE> {
  using VAL_TY = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<nnz_ty, 1>& nnz,
                  const AccessorRO<VAL_TY, 2>& B_vals,
                  const Rect<2>& rect)
  {
#pragma omp parallel for schedule(static)
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      size_t row_nnz = 0;
      for (auto j = rect.lo[1]; j < rect.hi[1] + 1; j++) {
        if (B_vals[{i, j}] != static_cast<VAL_TY>(0.0)) { row_nnz++; }
      }
      nnz[i] = row_nnz;
    }
  }
};

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct DenseToCSRImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorRW<Rect<1>, 1>& A_pos,
                  const AccessorWO<INDEX_TY, 1>& A_crd,
                  const AccessorWO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 2>& B_vals,
                  const Rect<2>& rect)
  {
#pragma omp parallel for schedule(static)
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      coord_t nnz_pos = A_pos[i].lo;
      for (auto j = rect.lo[1]; j < rect.hi[1] + 1; j++) {
        if (B_vals[{i, j}] != static_cast<VAL_TY>(0.0)) {
          A_crd[nnz_pos]  = static_cast<INDEX_TY>(j);
          A_vals[nnz_pos] = B_vals[{i, j}];
          nnz_pos++;
        }
      }
    }
  }
};

/*static*/ void DenseToCSRNNZ::omp_variant(TaskContext& context)
{
  dense_to_csr_nnz_template<VariantKind::OMP>(context);
}

/*static*/ void DenseToCSR::omp_variant(TaskContext& context)
{
  dense_to_csr_template<VariantKind::OMP>(context);
}

}  // namespace sparse

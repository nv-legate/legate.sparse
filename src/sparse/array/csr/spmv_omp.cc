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

#include "sparse/array/csr/spmv.h"
#include "sparse/array/csr/spmv_template.inl"

namespace sparse {

using namespace legate;

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct CSRSpMVRowSplitImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 1>& x,
                  const Rect<1>& rect)
  {
#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      VAL_TY sum = 0.0;
      for (size_t j_pos = A_pos[i].lo; j_pos < A_pos[i].hi + 1; j_pos++) {
        auto j = A_crd[j_pos];
        sum += A_vals[j_pos] * x[j];
      }
      y[i] = sum;
    }
  }
};

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename ACC>
struct CSRSpMVColSplitImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(ACC y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 1>& x,
                  const Rect<1>& y_rect,
                  const Rect<1>& A_crd_rect,
                  const Rect<1>& x_rect)
  {
#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (coord_t i = y_rect.lo[0]; i < y_rect.hi[0] + 1; i++) {
      VAL_TY sum = 0.0;
      for (size_t j_pos = A_pos[i].lo; j_pos < A_pos[i].hi + 1; j_pos++) {
        // Because the columns have been partitioned, we take a preimage
        // back into the coordinates, densify that, and then preimage again
        // into pos. That means we may reference entries in pos that are
        // are not meant to iterate over the entire rectangle, but just
        // the coordinates covered in A_crd_rect.
        if (A_crd_rect.contains(j_pos)) {
          auto j = A_crd[j_pos];
          // We also might get coordinates that aren't within the x partition.
          if (x_rect.contains(j)) { sum += A_vals[j_pos] * x[j]; }
        }
      }
      y[i] <<= sum;
    }
  }
};

/*static*/ void CSRSpMVRowSplit::omp_variant(TaskContext& context)
{
  csr_spmv_row_split_template<VariantKind::OMP>(context);
}

/*static*/ void CSRSpMVColSplit::omp_variant(TaskContext& context)
{
  csr_spmv_col_split_template<VariantKind::OMP>(context);
}

}  // namespace sparse

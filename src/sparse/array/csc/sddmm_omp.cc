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

#include "sparse/array/csc/sddmm.h"
#include "sparse/array/csc/sddmm_template.inl"
#include "sparse/util/distal_utils.h"

#include <omp.h>

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct CSCSDDMMImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<VAL_TY, 2>& C_vals,
                  const AccessorRO<VAL_TY, 2>& D_vals,
                  const Rect<2>& rect,
                  const Rect<1>& vals_rect)
  {
    // We'll chunk up the non-zeros into pieces so that each thread
    // only needs to perform one binary search.
    auto tot_nnz     = vals_rect.volume();
    auto num_threads = omp_get_max_threads();
    auto tile_size   = (tot_nnz + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static)
    for (size_t tid = 0; tid < num_threads; tid++) {
      auto first_nnz = tid * tile_size + vals_rect.lo[0];
      size_t j_pos   = taco_binarySearchBefore(B_pos, rect.lo[1], rect.hi[1], first_nnz);
      INDEX_TY j     = static_cast<INDEX_TY>(j_pos);
      for (size_t nnz_idx = (tid * tile_size); nnz_idx < std::min((tid + 1) * tile_size, tot_nnz);
           nnz_idx++) {
        size_t i_pos = nnz_idx + vals_rect.lo[0];
        INDEX_TY i   = B_crd[i_pos];
        // Bump up our row pointer until we find this position.
        while (!B_pos[j_pos].contains(i_pos)) {
          j_pos++;
          j = j_pos;
        }
        VAL_TY sum = static_cast<VAL_TY>(0);
        for (auto k = rect.lo[0]; k < rect.hi[0] + 1; k++) {
          sum += B_vals[i_pos] * (C_vals[{i, k}] * D_vals[{k, j}]);
        }
        A_vals[i_pos] = sum;
      }
    }
  }
};

/*static*/ void CSCSDDMM::omp_variant(TaskContext& context)
{
  csc_sddmm_template<VariantKind::OMP>(context);
}

}  // namespace sparse

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

#include "sparse/array/csr/tropical_spmv.h"
#include "sparse/array/csr/tropical_spmv_template.inl"
#include "sparse/util/cuda_help.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename INDEX_TY>
__global__ void tropical_spmv_kernel(size_t rows,
                                     coord_t offset,
                                     INDEX_TY num_fields,
                                     const AccessorWO<INDEX_TY, 2> y,
                                     const AccessorRO<Rect<1>, 1> pos,
                                     const AccessorRO<INDEX_TY, 1> crd,
                                     const AccessorRO<INDEX_TY, 2> x)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  INDEX_TY i = idx + offset;
  // Initialize the output.
  for (INDEX_TY f = 0; f < num_fields; f++) { y[{i, f}] = 0; }
  for (size_t jpos = pos[i].lo; jpos < pos[i].hi + 1; jpos++) {
    auto j         = crd[jpos];
    bool y_greater = true;
    for (INDEX_TY f = 0; f < num_fields; f++) {
      if (y[{i, f}] > x[{j, f}]) {
        y_greater = true;
        break;
      } else if (y[{i, f}] < x[{j, f}]) {
        y_greater = false;
        break;
      }
      // Else the fields are equal, so move onto the next field.
    }
    if (!y_greater) {
      for (INDEX_TY f = 0; f < num_fields; f++) { y[{i, f}] = x[{j, f}]; }
    }
  }
}

template <LegateTypeCode INDEX_CODE>
struct CSRTropicalSpMVImplBody<VariantKind::GPU, INDEX_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;

  void operator()(const AccessorWO<INDEX_TY, 2>& y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<INDEX_TY, 2>& x,
                  const Rect<2>& rect)
  {
    auto stream         = get_cached_stream();
    auto rows           = rect.hi[0] - rect.lo[0] + 1;
    auto blocks         = get_num_blocks_1d(rows);
    INDEX_TY num_fields = rect.hi[1] - rect.lo[1] + 1;
    // Since we can't use cuSPARSE over this semiring, we'll implement
    // a simple row-based kernel.
    tropical_spmv_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      rows, rect.lo[0], num_fields, y, A_pos, A_crd, x);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void CSRTropicalSpMV::gpu_variant(TaskContext& context)
{
  csr_tropical_spmv_template<VariantKind::GPU>(context);
}

}  // namespace sparse

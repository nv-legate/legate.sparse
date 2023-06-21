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

#include "sparse/array/conv/dense_to_csc.h"
#include "sparse/array/conv/dense_to_csc_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code VAL_CODE>
struct DenseToCSCNNZImplBody<VariantKind::CPU, VAL_CODE> {
  using VAL_TY = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<nnz_ty, 1>& nnz,
                  const AccessorRO<VAL_TY, 2>& B_vals,
                  const Rect<2>& rect)
  {
    for (auto j = rect.lo[1]; j < rect.hi[1] + 1; j++) {
      uint64_t col_nnz = 0;
      for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
        if (B_vals[{i, j}] != static_cast<VAL_TY>(0.0)) { col_nnz++; }
      }
      nnz[j] = col_nnz;
    }
  }
};

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct DenseToCSCImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorRW<Rect<1>, 1>& A_pos,
                  const AccessorWO<INDEX_TY, 1>& A_crd,
                  const AccessorWO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 2>& B_vals,
                  const Rect<2>& rect)
  {
    for (auto j = rect.lo[1]; j < rect.hi[1] + 1; j++) {
      coord_t nnz_pos = A_pos[j].lo;
      for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
        if (B_vals[{i, j}] != static_cast<VAL_TY>(0.0)) {
          A_crd[nnz_pos]  = static_cast<INDEX_TY>(i);
          A_vals[nnz_pos] = B_vals[{i, j}];
          nnz_pos++;
        }
      }
    }
  }
};

/*static*/ void DenseToCSCNNZ::cpu_variant(TaskContext& context)
{
  dense_to_csc_nnz_template<VariantKind::CPU>(context);
}

/*static*/ void DenseToCSC::cpu_variant(TaskContext& context)
{
  dense_to_csc_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  DenseToCSCNNZ::register_variants();
  DenseToCSC::register_variants();
}

}  // namespace

}  // namespace sparse

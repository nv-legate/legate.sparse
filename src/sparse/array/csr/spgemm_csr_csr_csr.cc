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

#include "sparse/array/csr/spgemm_csr_csr_csr.h"
#include "sparse/array/csr/spgemm_csr_csr_csr_template.inl"

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE>
struct SpGEMMCSRxCSRxCSRNNZImplBody<VariantKind::CPU, INDEX_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;

  void operator()(const AccessorWO<nnz_ty, 1>& nnz,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorRO<INDEX_TY, 1>& C_crd,
                  const uint64_t A2_dim,
                  const Rect<1>& rect)
  {
    INDEX_TY initCoord = static_cast<INDEX_TY>(0);
    bool initBool      = false;
    DeferredBuffer<INDEX_TY, 1> index_list(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initCoord);
    DeferredBuffer<bool, 1> already_set(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initBool);
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      size_t index_list_size = 0;
      for (size_t kB = B_pos[i].lo; kB < B_pos[i].hi + 1; kB++) {
        auto k = B_crd[kB];
        for (size_t jC = C_pos[k].lo; jC < C_pos[k].hi + 1; jC++) {
          auto j = C_crd[jC];
          if (!already_set[j]) {
            index_list[index_list_size] = j;
            already_set[j]              = true;
            index_list_size++;
          }
        }
      }
      nnz_ty row_nnzs = 0;
      for (auto index_loc = 0; index_loc < index_list_size; index_loc++) {
        auto j         = index_list[index_loc];
        already_set[j] = false;
        row_nnzs++;
      }
      nnz[i] = row_nnzs;
    }
  }
};

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpGEMMCSRxCSRxCSRImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
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
                  const uint64_t A2_dim,
                  const Rect<1>& rect)
  {
    INDEX_TY initCoord = static_cast<INDEX_TY>(0);
    bool initBool      = false;
    VAL_TY initVal     = static_cast<VAL_TY>(0);
    DeferredBuffer<INDEX_TY, 1> index_list(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initCoord);
    DeferredBuffer<bool, 1> already_set(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initBool);
    DeferredBuffer<VAL_TY, 1> workspace(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initVal);
    for (auto i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      size_t index_list_size = 0;
      for (size_t kB = B_pos[i].lo; kB < B_pos[i].hi + 1; kB++) {
        auto k = B_crd[kB];
        for (size_t jC = C_pos[k].lo; jC < C_pos[k].hi + 1; jC++) {
          auto j = C_crd[jC];
          if (!already_set[j]) {
            index_list[index_list_size] = j;
            already_set[j]              = true;
            index_list_size++;
          }
          workspace[j] += B_vals[kB] * C_vals[jC];
        }
      }
      size_t pA2 = A_pos[i].lo;
      for (auto index_loc = 0; index_loc < index_list_size; index_loc++) {
        auto j         = index_list[index_loc];
        already_set[j] = false;
        A_crd[pA2]     = j;
        A_vals[pA2]    = workspace[j];
        pA2++;
        // Zero out the workspace once we have read the value.
        workspace[j] = 0.0;
      }
    }
  }
};

/*static*/ void SpGEMMCSRxCSRxCSRNNZ::cpu_variant(TaskContext& context)
{
  spgemm_csr_csr_csr_nnz_template<VariantKind::CPU>(context);
}

/*static*/ void SpGEMMCSRxCSRxCSR::cpu_variant(TaskContext& context)
{
  spgemm_csr_csr_csr_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  SpGEMMCSRxCSRxCSRNNZ::register_variants();
  SpGEMMCSRxCSRxCSR::register_variants();
  SpGEMMCSRxCSRxCSRGPU::register_variants();
}
}  // namespace

}  // namespace sparse

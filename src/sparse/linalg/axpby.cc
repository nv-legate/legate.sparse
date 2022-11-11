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

#include "sparse/linalg/axpby.h"
#include "sparse/linalg/axpby_template.inl"

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode VAL_CODE>
struct AXPBYImplBody<VariantKind::CPU, VAL_CODE> {
  using VAL_TY = legate_type_of<VAL_CODE>;

  void operator()(const AccessorRW<VAL_TY, 1>& y,
                  const AccessorRO<VAL_TY, 1>& x,
                  const AccessorRO<VAL_TY, 1>& alpha,
                  const AccessorRO<VAL_TY, 1>& beta,
                  const Rect<1>& rect)
  {
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      y[i] = alpha[0] * x[i] + beta[0] * y[i];
    }
  }
};

/*static*/ void AXPBY::cpu_variant(TaskContext& context)
{
  axpby_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { AXPBY::register_variants(); }
}  // namespace

}  // namespace sparse

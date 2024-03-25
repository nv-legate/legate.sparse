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

#include "sparse/integrate/runge_kutta.h"
#include "sparse/integrate/runge_kutta_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code K_CODE, Type::Code A_CODE>
struct RKCalcDyImplBody<VariantKind::CPU, K_CODE, A_CODE> {
  using K_TY = legate_type_of<K_CODE>;
  using A_TY = legate_type_of<A_CODE>;

  void operator()(const AccessorWO<K_TY, 1>& dy,
                  const AccessorRO<K_TY, 2>& K,
                  const AccessorRO<A_TY, 1>& a,
                  const int32_t s,
                  const double h,
                  const Rect<2>& rect)
  {
    // dy(i) = K(j, i) * a(j), 0 <= j < s.
    for (auto i = rect.lo[1]; i < rect.hi[1] + 1; i++) {
      K_TY acc = 0;
#pragma unroll
      for (int64_t j = 0; j < s; j++) { acc += K[{j, i}] * a[j]; }
      dy[i] = acc * h;
    }
  }
};

/*static*/ void RKCalcDy::cpu_variant(TaskContext& context)
{
  rk_calc_dy_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { RKCalcDy::register_variants(); }
}  // namespace

}  // namespace sparse

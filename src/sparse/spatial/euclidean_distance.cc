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

#include "sparse/spatial/euclidean_distance.h"
#include "sparse/spatial/euclidean_distance_template.inl"

namespace sparse {

using namespace legate;

template <LegateTypeCode VAL_CODE>
struct EuclideanCDistImplBody<VariantKind::CPU, VAL_CODE> {
  using VAL_TY = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 2>& out,
                  const AccessorRO<VAL_TY, 2>& XA,
                  const AccessorRO<VAL_TY, 2>& XB,
                  const Rect<2>& out_rect,
                  const Rect<2>& XA_rect)
  {
    for (auto i = out_rect.lo[0]; i < out_rect.hi[0] + 1; i++) {
      for (auto j = out_rect.lo[1]; j < out_rect.hi[1] + 1; j++) {
        VAL_TY diff = 0.0;
        for (auto k = XA_rect.lo[1]; k < XA_rect.hi[1] + 1; k++) {
          diff += pow((XA[{i, k}] - XB[{j, k}]), 2);
        }
        out[{i, j}] = sqrt(diff);
      }
    }
  }
};

/*static*/ void EuclideanCDist::cpu_variant(TaskContext& context)
{
  euclidean_distance_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  EuclideanCDist::register_variants();
}
}  // namespace

}  // namespace sparse

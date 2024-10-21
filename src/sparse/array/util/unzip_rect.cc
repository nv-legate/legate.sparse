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

#include "sparse/array/util/unzip_rect.h"
#include "sparse/array/util/unzip_rect_template.inl"

namespace sparse {

using namespace legate;

template <>
struct UnZipRect1ImplBody<VariantKind::CPU> {
  void operator()(const AccessorWO<int64_t, 1>& out1,
                  const AccessorWO<int64_t, 1>& out2,
                  const AccessorRO<Rect<1>, 1>& in,
                  const Rect<1>& rect)
  {
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      out1[i] = in[i].lo;
      out2[i] = in[i].hi;
    }
  }
};

/*static*/ void UnZipRect1::cpu_variant(TaskContext& context)
{
  unzip_rect_1_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { UnZipRect1::register_variants(); }
}  // namespace

}  // namespace sparse

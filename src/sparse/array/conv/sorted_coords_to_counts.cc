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

#include "sparse/array/conv/sorted_coords_to_counts.h"
#include "sparse/array/conv/sorted_coords_to_counts_template.inl"

namespace sparse {

using namespace legate;

template <Type::Code INDEX_CODE, typename ACC>
struct SortedCoordsToCountsImplBody<VariantKind::CPU, INDEX_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  void operator()(ACC out, AccessorRO<INDEX_TY, 1> in, const Domain& dom)
  {
    for (PointInDomainIterator<1> itr(dom); itr(); itr++) { out[in[*itr]] <<= 1; }
  }
};

/*static*/ void SortedCoordsToCounts::cpu_variant(TaskContext& context)
{
  sorted_coords_to_counts_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  SortedCoordsToCounts::register_variants();
}
}  // namespace

}  // namespace sparse

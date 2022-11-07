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

#include "sparse/sort/sort.h"
#include "sparse/sort/sort_template.inl"

#include <core/comm/coll.h>

namespace sparse {

using namespace Legion;
using namespace legate;

/* static */ void SortByKey::cpu_variant(legate::TaskContext& ctx)
{
  sort_by_key_template<VariantKind::CPU, decltype(thrust::host), legate::comm::coll::CollComm>(
    ctx, thrust::host, Memory::SYSTEM_MEM);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  // Sort variants have to be marked as concurrent.
  auto options = legate::VariantOptions{}.with_concurrent(true);
  SortByKey::register_variants(
    {{LEGATE_CPU_VARIANT, options}, {LEGATE_GPU_VARIANT, options}, {LEGATE_OMP_VARIANT, options}});
}
}  // namespace

}  // namespace sparse

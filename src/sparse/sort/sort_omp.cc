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
#include "sparse/sort/sort_cpu_template.inl"

#include <core/comm/coll.h>

namespace sparse {

using namespace legate;

/* static */ void SortByKey::omp_variant(legate::TaskContext& ctx)
{
  auto kind = Core::has_socket_mem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM;
  auto exec = thrust::omp::par;
  sort_by_key_template<VariantKind::OMP, decltype(exec), legate::comm::coll::CollComm>(
    ctx, exec, kind);
}

}  // namespace sparse

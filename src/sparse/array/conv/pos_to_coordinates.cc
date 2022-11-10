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

#include "sparse/array/conv/pos_to_coordinates.h"
#include "sparse/array/conv/pos_to_coordinates_template.inl"

namespace sparse {

using namespace Legion;
using namespace legate;

/*static*/ void ExpandPosToCoordinates::cpu_variant(TaskContext& context)
{
  pos_to_coordinates_template(context, thrust::host, Memory::SYSTEM_MEM);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ExpandPosToCoordinates::register_variants();
}
}  // namespace

}  // namespace sparse

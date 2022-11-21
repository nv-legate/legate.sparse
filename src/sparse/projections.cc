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

#include "projections.h"

using namespace Legion;

namespace sparse {

Legion::LogicalRegion Promote1Dto2DFunctor::project(Legion::LogicalPartition upper_bound,
                                                    const Legion::DomainPoint& input,
                                                    const Legion::Domain& launch_domain)
{
  auto color_space =
    this->runtime->get_index_partition_color_space(upper_bound.get_index_partition());
  assert(input.dim == 1 && launch_domain.dim == 1);
  assert(color_space.dim == 2);
  assert(color_space.lo()[0] == 0 && color_space.lo()[1] == 0);
  int64_t jdim = color_space.hi()[1] + 1;

  // Project our input point onto the mxn grid.
  int64_t idx = input[0];
  int64_t i   = idx / jdim;
  int64_t j   = idx % jdim;
  auto output = Point<2>{i, j};
  assert(color_space.contains(output));
  return runtime->get_logical_subregion_by_color(upper_bound, output);
}

Functor1DToRowsImplicit2D::Functor1DToRowsImplicit2D(Legion::Runtime* rt,
                                                     int32_t gx,
                                                     int32_t gy,
                                                     bool rows)
  : LegateSparseProjectionFunctor(rt), gx(gx), gy(gy), rows(rows)
{
}

Legion::LogicalRegion Functor1DToRowsImplicit2D::project(Legion::LogicalPartition upper_bound,
                                                         const Legion::DomainPoint& input,
                                                         const Legion::Domain& launch_domain)
{
  int64_t idx = input[0];
  int64_t i   = idx / gy;
  int64_t j   = idx % gy;
  // Assume we're just projecting onto the rows right now.
  if (rows) {
    return runtime->get_logical_subregion_by_color(upper_bound, i);
  } else {
    return runtime->get_logical_subregion_by_color(upper_bound, j);
  }
}

}  // namespace sparse

extern "C" {
void register_legate_sparse_1d_to_2d_functor(legion_projection_id_t proj_id,
                                             int32_t gx,
                                             int32_t gy,
                                             bool rows)
{
  auto rt = Legion::Runtime::get_runtime();
  rt->register_projection_functor(
    proj_id, new sparse::Functor1DToRowsImplicit2D(rt, gx, gy, rows), true /* silence_warnings */);
}
}

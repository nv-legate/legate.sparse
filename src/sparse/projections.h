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

#pragma once

#include "sparse/sparse_c.h"
#include "legate.h"

namespace sparse {

// LegateSparseProjectionFunctor is a base class for projection functors.
class LegateSparseProjectionFunctor : public Legion::ProjectionFunctor {
 public:
  LegateSparseProjectionFunctor(Legion::Runtime* rt) : Legion::ProjectionFunctor(rt) {}
  bool is_functional(void) const override { return true; }
  bool is_exclusive(void) const override { return true; }
  unsigned get_depth(void) const override { return 0; }
};

// Promote1Dto2DFunctor is a projection functor that up-casts an index
// into a point in a 2-D color space.
class Promote1Dto2DFunctor : public LegateSparseProjectionFunctor {
 public:
  Promote1Dto2DFunctor(Legion::Runtime* rt) : LegateSparseProjectionFunctor(rt) {}

 public:
  Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                const Legion::DomainPoint& point,
                                const Legion::Domain& launch_domain) override;
};

// Functor1DToRowsImplicit2D is a functor that is similar to Promote1Dto2DFunctor
// but works around legate internal partitioning that internally translates
// multi-dimensional tiled colorings into tilings over 1-D color spaces.
class Functor1DToRowsImplicit2D : public LegateSparseProjectionFunctor {
 public:
  Functor1DToRowsImplicit2D(Legion::Runtime* rt, int32_t gx, int32_t gy, bool rows);
  Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                const Legion::DomainPoint& point,
                                const Legion::Domain& launch_domain) override;

 private:
  int32_t gx, gy;
  bool rows = true;
};

}  // namespace sparse

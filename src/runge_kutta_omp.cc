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

#include "sparse.h"
#include "runge_kutta.h"

using namespace Legion;

namespace sparse {

void RKCalcDy::omp_variant(legate::TaskContext& ctx) {
  auto s = ctx.scalars()[0].value<int32_t>();
  auto h = ctx.scalars()[1].value<double>();
  auto& K = ctx.inputs()[0];
  auto& a = ctx.inputs()[1];
  auto& dy = ctx.outputs()[0];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (dy.transformed()) {
    dy.remove_transform();
  }

  auto K_acc = K.read_accessor<complex<double>, 2>();
  auto a_acc = a.read_accessor<double, 1>();
  auto dy_acc = dy.write_accessor<complex<double>, 1>();

  auto dom = K.domain();
  // dy(i) = K(j, i) * a(j), 0 <= j < s.
  #pragma omp parallel for schedule(static)
  for (int64_t i = dom.lo()[1]; i < dom.hi()[1] + 1; i++) {
    complex<double> acc = 0;
    #pragma unroll
    for (int64_t j = 0; j < s; j++) {
      acc += K_acc[{j, i}] * a_acc[j];
    }
    dy_acc[i] = acc * h;
  }
}

}

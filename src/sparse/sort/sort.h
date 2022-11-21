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

#include "sparse/sparse.h"
#include "sparse/sparse_c.h"
#include "legate.h"

#include <thrust/functional.h>

namespace sparse {

struct SortByKeyArgs {
  legate::TaskContext& ctx;
  const legate::Store& key1;
  const legate::Store& key2;
  const legate::Store& values;
  legate::Store& key1_out;
  legate::Store& key2_out;
  legate::Store& values_out;
};

// SortByKey sorts a set of key regions and a value region.
// Out of an input `n` regions, the first `n-1` regions are
// zipped together to be a key to sort the value region `n`.
// SortByKey operates in place.
class SortByKey : public SparseTask<SortByKey> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_SORT_BY_KEY;

  static void cpu_variant(legate::TaskContext& ctx);

#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace sparse

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

#include "sparse.h"
#include "sparse_c.h"
#include "legate.h"

namespace sparse {

class EnumerateIndependentSets : public SparseTask<EnumerateIndependentSets> {
public:
  static const int TASK_ID = LEGATE_QUANTUM_ENUMERATE_INDEP_SETS;
  static void cpu_variant(legate::TaskContext& ctx);
private:
  template<int N, typename T>
  static void cpu_variant_impl(legate::TaskContext& ctx);
};

class CreateHamiltonians : public SparseTask<CreateHamiltonians> {
public:
  static const int TASK_ID = LEGATE_QUANTUM_CREATE_HAMILTONIANS;
  static void cpu_variant(legate::TaskContext& ctx);
private:
  template<int N, typename T>
  static void cpu_variant_impl(legate::TaskContext& ctx);
};

class SetsToSizes : public SparseTask<SetsToSizes> {
public:
  static const int TASK_ID = LEGATE_QUANTUM_SETS_TO_SIZES;
  static void cpu_variant(legate::TaskContext& ctx);
private:
  template<int N, typename T>
  static void cpu_variant_impl(legate::TaskContext& ctx);
};

}

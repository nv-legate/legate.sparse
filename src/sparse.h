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

#include "legate.h"

namespace sparse {

struct Sparse {
public:
  template <typename... Args>
  static void record_variant(Args&&... args) {
    get_registrar().record_variant(std::forward<Args>(args)...);
  }
  static legate::LegateTaskRegistrar& get_registrar();
public:
  static bool has_numamem;
  static Legion::MapperID mapper_id;
};

template <typename T>
struct SparseTask : public legate::LegateTask<T> {
  using Registrar = Sparse;
};

} // namespace sparse
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

#include "core/mapping/base_mapper.h"

namespace sparse {

class LegateSparseMapper : public legate::mapping::BaseMapper {
 public:
  LegateSparseMapper(Legion::Runtime* rt,
                     Legion::Machine machine,
                     const legate::LibraryContext& context);
  virtual ~LegateSparseMapper(void) {}

 public:
  // Virtual mapping functions of BaseMapper that need to be overridden.
  virtual bool is_pure() const override { return true; }
  virtual void select_task_options(const Legion::Mapping::MapperContext ctx,
		                   const Legion::Task& task,
				   TaskOptions& options) override;
  virtual legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override;
  virtual std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override;
  virtual legate::Scalar tunable_value(legate::TunableID tunable_id) override;
};

}  // namespace sparse

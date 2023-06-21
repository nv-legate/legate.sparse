/* Copyright 2021-2022 NVIDIA Corporation
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

#include "sparse/sparse.h"
#include "projections.h"

#include "sparse/mapper/mapper.h"

#include "legate.h"

using namespace legate;

namespace sparse {

static const char* const library_name = "legate.sparse";

TaskRegistrar& Sparse::get_registrar()
{
  static TaskRegistrar registrar;
  return registrar;
}

void registration_callback()
{
  ResourceConfig config;
  // TODO (rohany): I want to use the enums here, but I'm not sure the best way
  //  to keep this in line with the Python import since there seems to be a
  //  cyclic dependency.
  // config.max_tasks = LEGATE_SPARSE_LAST_TASK;
  // config.max_projections = LEGATE_SPARSE_LAST_PROJ_FN;
  config.max_tasks = 100;
  // TODO (rohany): We're dynamically generating projections... How does cunumeric handle this?
  config.max_projections = 1000;
  auto ctx = Runtime::get_runtime()->create_library(library_name, config, std::make_unique<LegateSparseMapper>());

  Sparse::get_registrar().register_all_tasks(ctx);

  auto runtime = Legion::Runtime::get_runtime();
  auto proj_id = ctx->get_projection_id(LEGATE_SPARSE_PROJ_FN_1D_TO_2D);
  auto functor = new Promote1Dto2DFunctor(runtime);
  runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);
}

}  // namespace sparse

extern "C" {

void perform_registration(void) { Core::perform_registration<sparse::registration_callback>(); }
}

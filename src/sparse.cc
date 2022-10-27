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

#include "sparse.h"
#include "mapper.h"
#include "projections.h"

#include "legate.h"

using namespace Legion;
using namespace legate;

namespace sparse {

/* static */ bool Sparse::has_numamem   = false;
/* static */ MapperID Sparse::mapper_id = -1;
static const char* const library_name   = "legate.sparse";

LegateTaskRegistrar& Sparse::get_registrar()
{
  static LegateTaskRegistrar registrar;
  return registrar;
}

void registration_callback(Machine machine,
                           Runtime* runtime,
                           const std::set<Processor>& local_procs)
{
  ResourceConfig config;
  config.max_mappers = 1;
  // TODO (rohany): I want to use the enums here, but I'm not sure the best way
  //  to keep this in line with the Python import since there seems to be a
  //  cyclic dependency.
  // config.max_tasks = LEGATE_SPARSE_LAST_TASK;
  // config.max_projections = LEGATE_SPARSE_LAST_PROJ_FN;
  config.max_tasks = 100;
  // TODO (rohany): We're dynamically generating projections... How does cunumeric handle this?
  config.max_projections = 1000;
  LibraryContext ctx(runtime, library_name, config);

  Sparse::get_registrar().register_all_tasks(runtime, ctx);

  Sparse::mapper_id = ctx.get_mapper_id(0);
  ctx.register_mapper(new LegateSparseMapper(runtime, machine, ctx), 0);

  auto proj_id = ctx.get_projection_id(LEGATE_SPARSE_PROJ_FN_1D_TO_2D);
  auto functor = new Promote1Dto2DFunctor(runtime);
  runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);
}

}  // namespace sparse

extern "C" {

void perform_registration(void)
{
  Runtime::perform_registration_callback(sparse::registration_callback, true /* global */);
  Runtime* runtime = Runtime::get_runtime();
  Context ctx      = Runtime::get_context();
  Future fut       = runtime->select_tunable_value(
    ctx, LEGATE_SPARSE_TUNABLE_HAS_NUMAMEM, sparse::Sparse::mapper_id);
  if (fut.get_result<int32_t>() != 0) sparse::Sparse::has_numamem = true;
}
}

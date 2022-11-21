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

#include "sparse/sparse.h"
#include "sparse/sparse_c.h"
#include "sparse/cudalibs.h"

#include <stdio.h>

namespace sparse {

using namespace Legion;

CUDALibraries::CUDALibraries() : finalized_(false), cusparse_(nullptr) {}

CUDALibraries::~CUDALibraries() { finalize(); }

void CUDALibraries::finalize()
{
  if (finalized_) return;
  if (cusparse_ != nullptr) finalize_cusparse();
  finalized_ = true;
}

void CUDALibraries::finalize_cusparse()
{
  CHECK_CUSPARSE(cusparseDestroy(cusparse_));
  cusparse_ = nullptr;
}

cusparseHandle_t CUDALibraries::get_cusparse()
{
  if (this->cusparse_ == nullptr) { CHECK_CUSPARSE(cusparseCreate(&this->cusparse_)); }
  return this->cusparse_;
}

static CUDALibraries& get_cuda_libraries(Processor proc)
{
  if (proc.kind() != Processor::TOC_PROC) {
    fprintf(stderr, "Illegal request for CUDA libraries for non-GPU processor");
    LEGATE_ABORT;
  }

  static CUDALibraries cuda_libraries[LEGION_MAX_NUM_PROCS];
  const auto proc_id = proc.id & (LEGION_MAX_NUM_PROCS - 1);
  return cuda_libraries[proc_id];
}

legate::cuda::StreamView get_cached_stream()
{
  return legate::cuda::StreamPool::get_stream_pool().get_stream();
}

cusparseHandle_t get_cusparse()
{
  const auto proc = Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cusparse();
}

class LoadCUDALibsTask : public SparseTask<LoadCUDALibsTask> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_LOAD_CUDALIBS;

 public:
  static void gpu_variant(legate::TaskContext& context)
  {
    const auto proc = Processor::get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.get_cusparse();
  }
};

class UnloadCUDALibsTask : public SparseTask<UnloadCUDALibsTask> {
 public:
  static const int TASK_ID = LEGATE_SPARSE_UNLOAD_CUDALIBS;

 public:
  static void gpu_variant(legate::TaskContext& context)
  {
    const auto proc = Processor::get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.finalize();
  }
};

static void __attribute__((constructor)) register_tasks(void)
{
  LoadCUDALibsTask::register_variants();
  UnloadCUDALibsTask::register_variants();
}

}  // namespace sparse

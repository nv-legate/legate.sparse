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
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "cusparse.h"
#include <nccl.h>

#define THREADS_PER_BLOCK 128

#define CHECK_CUSPARSE(expr)                    \
  do {                                          \
    cusparseStatus_t result = (expr);           \
    checkCuSparse(result, __FILE__, __LINE__);  \
  } while (false)

#define CHECK_NCCL(expr)                    \
  do {                                      \
    ncclResult_t result = (expr);           \
    check_nccl(result, __FILE__, __LINE__); \
  } while (false)


namespace sparse {

__device__ inline size_t global_tid_1d()
{
  return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

inline size_t get_num_blocks_1d(size_t threads) {
  return (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

__host__ inline void checkCuSparse(cusparseStatus_t status, const char* file, int line) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal CUSPARSE failure with error code %d (%s) in file %s at line %d\n",
            status,
	    cusparseGetErrorString(status),
            file,
            line);
    assert(false);
  }
}

__host__ inline void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    fprintf(stderr,
            "Internal NCCL failure with error %s in file %s at line %d\n",
            ncclGetErrorString(error),
            file,
            line);
    assert(false);
  }
}


// Return a cached stream for the current GPU.
legate::cuda::StreamView get_cached_stream();
// Method to get the CUSPARSE handle associated with the current GPU.
cusparseHandle_t get_cusparse();

}

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

/* Portions of this file are additionally subject to the following license
 * and copyright.
 *  MIT License
 *
 *  Copyright (c) 2017-2019 MIT CSAIL, Adobe Systems, and other contributors.
 *
 *  Developed by:
 *
 *    The taco team
 *    http://tensor-compiler.org
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#pragma once

#include "sparse.h"
#include "cuda_help.h"
#include "distal_utils.h"

namespace sparse {

// This atomicAddWarp kernel ensures that the warp level reduction only
// happens if all threads in the warp are indeed writing to the same
// output location.
template<typename T>
__device__ inline void atomicAddWarp(T *output, size_t index, T val)
{
  size_t leader_index = __shfl_sync(__activemask(), index, 0);
  int mask = __ballot_sync(__activemask(), leader_index == index);
  if(mask == __activemask()) {
    val += __shfl_down_sync(__activemask(), val, 16);
    val += __shfl_down_sync(__activemask(), val, 8);
    val += __shfl_down_sync(__activemask(), val, 4);
    val += __shfl_down_sync(__activemask(), val, 2);
    val += __shfl_down_sync(__activemask(), val, 1);
    if(threadIdx.x % 32 == 0) {
      atomicAdd(output, val);
    }
  } else {
    atomicAdd(output, val);
  }
}

template<typename T, int DIM>
__device__ __inline__
size_t flattenPoint(T accessor, Legion::Point<DIM> point) {
  size_t base = 0;
  for (int i = 0; i < DIM; i++) {
    base += accessor.accessor.strides[i] * point[i];
  }
  return base;
}

template<typename T, typename R>
__global__ void taco_binarySearchBeforeBlock(T posArray, R* __restrict__ results, int64_t arrayStart, int64_t arrayEnd, int values_per_block, int num_blocks, int64_t offset) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx > num_blocks) {
    return;
  }
  results[idx] = taco_binarySearchBefore<T>(posArray, arrayStart, arrayEnd, idx * values_per_block + offset);
}

template<typename T>
__host__ void taco_binarySearchBeforeBlockLaunch(legate::cuda::StreamView& stream, T posArray, int64_t* __restrict__ results, int64_t arrayStart, int64_t arrayEnd, int values_per_block, int block_size, int num_blocks, int64_t offset = 0) {
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  taco_binarySearchBeforeBlock<T, int64_t><<<num_search_blocks, block_size, 0, stream>>>(posArray, results, arrayStart, arrayEnd, values_per_block, num_blocks, offset);
}

} // namespace sparse

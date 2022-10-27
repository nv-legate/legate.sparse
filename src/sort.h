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
#include <thrust/functional.h>

namespace sparse {

template <typename INDEX_TY, typename VAL_TY>
struct SortPiece {
  Legion::DeferredBuffer<INDEX_TY, 1> indices1;
  Legion::DeferredBuffer<INDEX_TY, 1> indices2;
  Legion::DeferredBuffer<VAL_TY, 1> values;
  size_t size;
};

template <typename INDEX_TY>
struct Sample {
  INDEX_TY value1;
  INDEX_TY value2;
  int32_t rank;
  size_t position;
};

template <typename T>
Legion::DeferredBuffer<T, 1> create_buffer(size_t count, Legion::Memory::Kind mem)
{
  // We have to make sure that we don't return empty buffers, as this results
  // in null pointers getting passed to the communicators, which confuses them
  // greatly.
  count = std::max(count, size_t(1));
  Legion::DeferredBuffer<T, 1> buf({0, count - 1}, mem);
  return buf;
}

template <typename INDEX_TY>
struct SampleComparator : public thrust::binary_function<Sample<INDEX_TY>, Sample<INDEX_TY>, bool> {
#if defined(__CUDACC__)
  __host__ __device__
#endif
    bool
    operator()(const Sample<INDEX_TY>& lhs, const Sample<INDEX_TY>& rhs) const
  {
    // special case for unused samples
    if (lhs.rank < 0 || rhs.rank < 0) { return rhs.rank < 0 && lhs.rank >= 0; }

    auto lhs_value = std::make_pair(lhs.value1, lhs.value2);
    auto rhs_value = std::make_pair(rhs.value1, rhs.value2);
    if (lhs_value != rhs_value) {
      return lhs_value < rhs_value;
    } else if (lhs.rank != rhs.rank) {
      return lhs.rank < rhs.rank;
    } else {
      return lhs.position < rhs.position;
    }
  }
};

// We temporarily implement our own versions of the lower_bound and upper_bound
// functions as the original implementation uses CUB. I copied these implementations
// from CUB and edited them to handle the 2-element key.
template <typename INDEX_TY>
#if defined(__CUDACC__)
__host__ __device__
#endif
  size_t
  lower_bound(const INDEX_TY* input1,
              const INDEX_TY* input2,
              size_t num_items,
              const INDEX_TY val1,
              const INDEX_TY val2)
{
  size_t retval = 0;
  while (num_items > 0) {
    size_t half = num_items >> 1;
    auto idx    = retval + half;
    auto a      = std::make_pair(input1[idx], input2[idx]);
    auto b      = std::make_pair(val1, val2);
    if (a < b) {
      retval    = retval + (half + 1);
      num_items = num_items - (half + 1);
    } else {
      num_items = half;
    }
  }

  return retval;
}

template <typename INDEX_TY>
#if defined(__CUDACC__)
__host__ __device__
#endif
  size_t
  upper_bound(const INDEX_TY* input1,
              const INDEX_TY* input2,
              size_t num_items,
              const INDEX_TY val1,
              const INDEX_TY val2)
{
  size_t retval = 0;
  while (num_items > 0) {
    size_t half = num_items >> 1;
    auto idx    = retval + half;
    auto a      = std::make_pair(input1[idx], input2[idx]);
    auto b      = std::make_pair(val1, val2);
    if (b < a) {
      num_items = half;
    } else {
      retval    = retval + (half + 1);
      num_items = num_items - (half + 1);
    }
  }

  return retval;
}

}  // namespace sparse
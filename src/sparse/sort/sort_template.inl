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

// Useful for IDEs.
#include "sparse/sort/sort.h"
#include "sparse/util/dispatch.h"

#include <thrust/binary_search.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>

namespace sparse {

using namespace Legion;
using namespace legate;

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

template <typename INDEX_TY>
void extract_samples(const INDEX_TY* data1,
                     const INDEX_TY* data2,
                     const size_t volume,
                     Sample<INDEX_TY>* samples,
                     const size_t num_local_samples,
                     const Sample<INDEX_TY> init_sample,
                     const size_t offset,
                     const size_t rank)
{
  for (size_t sample_idx = 0; sample_idx < num_local_samples; sample_idx++) {
    if (num_local_samples < volume) {
      const size_t index                    = (sample_idx + 1) * volume / num_local_samples - 1;
      samples[offset + sample_idx].value1   = data1[index];
      samples[offset + sample_idx].value2   = data2[index];
      samples[offset + sample_idx].rank     = rank;
      samples[offset + sample_idx].position = index;
    } else {
      // edge case where num_local_samples > volume
      if (sample_idx < volume) {
        samples[offset + sample_idx].value1   = data1[sample_idx];
        samples[offset + sample_idx].value2   = data2[sample_idx];
        samples[offset + sample_idx].rank     = rank;
        samples[offset + sample_idx].position = sample_idx;
      } else {
        samples[offset + sample_idx] = init_sample;
      }
    }
  }
}

template <typename INDEX_TY>
static void extract_split_positions(const INDEX_TY* data1,
                                    const INDEX_TY* data2,
                                    const size_t volume,
                                    const Sample<INDEX_TY>* samples,
                                    const size_t num_samples,
                                    size_t* split_positions,
                                    const size_t num_splitters,
                                    const size_t rank)
{
  for (size_t splitter_idx = 0; splitter_idx < num_splitters; splitter_idx++) {
    const size_t index              = (splitter_idx + 1) * num_samples / (num_splitters + 1) - 1;
    const Sample<INDEX_TY> splitter = samples[index];

    // now perform search on data to receive position *after* last element to be
    // part of the package for rank splitter_idx
    if (rank > splitter.rank) {
      // position of the last position with smaller value than splitter.value + 1
      split_positions[splitter_idx] =
        lower_bound(data1, data2, volume, splitter.value1, splitter.value2);
    } else if (rank < splitter.rank) {
      // position of the first position with value larger than splitter.value
      split_positions[splitter_idx] =
        upper_bound(data1, data2, volume, splitter.value1, splitter.value2);
    } else {
      split_positions[splitter_idx] = splitter.position + 1;
    }
  }
}

// This will be defined by sort_cpu_template.inl and sort.cu.
template <VariantKind KIND, typename INDEX_TY, typename VAL_TY, typename Policy, typename Comm>
struct SampleSorter;

template <VariantKind KIND, typename Policy, typename Comm>
struct SortByKeyImpl {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(SortByKeyArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto& ctx        = args.ctx;
    auto& key1       = args.key1;
    auto& key2       = args.key2;
    auto& values     = args.values;
    auto& key1_out   = args.key1_out;
    auto& key2_out   = args.key2_out;
    auto& values_out = args.values_out;

    auto key1_acc   = key1.read_accessor<INDEX_TY, 1>();
    auto key2_acc   = key2.read_accessor<INDEX_TY, 1>();
    auto values_acc = values.read_accessor<VAL_TY, 1>();

    auto size = key1.domain().get_volume();
    SortPiece<INDEX_TY, VAL_TY> local;
    local.indices1 = create_buffer<INDEX_TY>(size, mem);
    local.indices2 = create_buffer<INDEX_TY>(size, mem);
    local.values   = create_buffer<VAL_TY>(size, mem);
    local.size     = size;
    assert(key1.domain().get_volume() == key2.domain().get_volume() &&
           key1.domain().get_volume() == values.domain().get_volume());

    // To avoid setting off the bounds checking warnings, only
    // access these pointer locations if we have elements to
    // process. Note that we _cannot_ return early here, because
    // then there won't be a cooperating process to send and
    // receive messages in the explicit communication phase.
    if (!key1.domain().empty()) {
      // Copy the inputs into the output buffers.
      thrust::copy(exec,
                   key1_acc.ptr(key1.domain().lo()),
                   key1_acc.ptr(key1.domain().lo()) + size,
                   local.indices1.ptr(0));
      thrust::copy(exec,
                   key2_acc.ptr(key2.domain().lo()),
                   key2_acc.ptr(key2.domain().lo()) + size,
                   local.indices2.ptr(0));
      thrust::copy(exec,
                   values_acc.ptr(values.domain().lo()),
                   values_acc.ptr(values.domain().lo()) + size,
                   local.values.ptr(0));

      // Sort the local chunks of data.
      auto tuple_begin = thrust::make_tuple(local.indices1.ptr(0), local.indices2.ptr(0));
      auto tuple_end =
        thrust::make_tuple(local.indices1.ptr(0) + local.size, local.indices2.ptr(0) + local.size);
      thrust::sort_by_key(exec,
                          thrust::make_zip_iterator(tuple_begin),
                          thrust::make_zip_iterator(tuple_end),
                          local.values.ptr(0));
    }

    auto is_single = ctx.is_single_task() || (ctx.get_launch_domain().get_volume() == 1);
    // Now, we have to branch on whether or not this is actually
    // a distributed computation.
    if (is_single) {
      // In a single task, we just need to return the allocations
      // that we made.
      key1_out.return_data(local.indices1, {local.size});
      key2_out.return_data(local.indices2, {local.size});
      values_out.return_data(local.values, {local.size});
    } else {
      // Otherwise, initiate the distributed samplesort.
      auto comm   = ctx.communicators()[0].get<Comm>();
      auto result = SampleSorter<KIND, INDEX_TY, VAL_TY, Policy, Comm>{}(
        exec, local, ctx.get_task_index()[0], ctx.get_launch_domain().get_volume(), mem, &comm);
      key1_out.return_data(result.indices1, {result.size});
      key2_out.return_data(result.indices2, {result.size});
      values_out.return_data(result.values, {result.size});
    }
  }
  // The thrust execution policy for this code.
  Policy exec;
  // The kind of memory to allocate temporaries within.
  Memory::Kind mem;
};

template <VariantKind KIND, typename Policy, typename Comm>
static void sort_by_key_template(TaskContext& context, Policy exec, Memory::Kind mem)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  SortByKeyArgs args{
    context,
    inputs[0],
    inputs[1],
    inputs[2],
    outputs[0],
    outputs[1],
    outputs[2],
  };
  index_type_value_type_dispatch(
    args.key1.code(), args.values.code(), SortByKeyImpl<KIND, Policy, Comm>{exec, mem}, args);
}

}  // namespace sparse

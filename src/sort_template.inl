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
#include "sort.h"

#include <core/comm/coll.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace sparse {

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

template <typename INDEX_TY, typename VAL_TY, typename Policy>
static SortPiece<INDEX_TY, VAL_TY> samplesort(Policy exec,
                                              SortPiece<INDEX_TY, VAL_TY> local_sorted,
                                              size_t my_rank,
                                              size_t num_ranks,
                                              Legion::Memory::Kind mem,
                                              legate::comm::coll::CollComm* comm)
{
}

template <typename INDEX_TY, typename VAL_TY, typename Policy>
void SortBody(legate::TaskContext& ctx, Legion::Memory::Kind mem, Policy exec)
{
  auto& key1       = ctx.inputs()[0];
  auto& key2       = ctx.inputs()[1];
  auto& values     = ctx.inputs()[2];
  auto& key1_out   = ctx.outputs()[0];
  auto& key2_out   = ctx.outputs()[1];
  auto& values_out = ctx.outputs()[2];

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
    auto comm   = ctx.communicators()[0].get<legate::comm::coll::CollComm>();
    auto result = samplesort(
      exec, local, ctx.get_task_index()[0], ctx.get_launch_domain().get_volume(), mem, &comm);
    key1_out.return_data(result.indices1, {result.size});
    key2_out.return_data(result.indices2, {result.size});
    values_out.return_data(result.values, {result.size});
  }
}

}  // namespace sparse

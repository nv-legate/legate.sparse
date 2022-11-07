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

#include <core/comm/coll.h>
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

template <VariantKind KIND, typename INDEX_TY>
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

template <VariantKind KIND, typename INDEX_TY>
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

// By default here, we'll include the CPU/OMP implementation, and
// define the NCCL version in the sort.cu file.
template <VariantKind KIND, typename INDEX_TY, typename VAL_TY, typename Policy, typename Comm>
static SortPiece<INDEX_TY, VAL_TY> samplesort(Policy exec,
                                              SortPiece<INDEX_TY, VAL_TY> local_sorted,
                                              size_t my_rank,
                                              size_t num_ranks,
                                              Legion::Memory::Kind mem,
                                              Comm* comm)
{
  size_t volume = local_sorted.size;

  // collect local samples - for now we take num_ranks samples for every node
  // worst case this leads to 2*N/ranks elements on a single node
  size_t num_local_samples = num_ranks;

  size_t num_global_samples = num_local_samples * num_ranks;
  auto samples              = create_buffer<Sample<INDEX_TY>>(num_global_samples, mem);

  Sample<INDEX_TY> init_sample;
  {
    init_sample.rank = -1;  // init samples that are not populated
    size_t offset    = num_local_samples * my_rank;
    extract_samples<KIND>(local_sorted.indices1.ptr(0),
                          local_sorted.indices2.ptr(0),
                          volume,
                          samples.ptr(0),
                          num_local_samples,
                          init_sample,
                          offset,
                          my_rank);
  }

  // TODO (rohany): I think that we can specialize the communicator
  //  implementation choice using templates? Have a wrapper allGather
  //  that understands the type of the communicator.
  legate::comm::coll::collAllgather(samples.ptr(my_rank * num_ranks),
                                    samples.ptr(0),
                                    num_ranks * sizeof(Sample<INDEX_TY>),
                                    legate::comm::coll::CollDataType::CollInt8,
                                    *comm);

  // Sort local samples.
  thrust::stable_sort(
    exec, samples.ptr(0), samples.ptr(0) + num_global_samples, SampleComparator<INDEX_TY>());

  auto lower_bound          = thrust::lower_bound(exec,
                                         samples.ptr(0),
                                         samples.ptr(0) + num_global_samples,
                                         init_sample,
                                         SampleComparator<INDEX_TY>());
  size_t num_usable_samples = lower_bound - samples.ptr(0);

  // select splitters / positions based on samples (on device)
  const size_t num_splitters = num_ranks - 1;
  auto split_positions       = create_buffer<size_t>(num_splitters, mem);
  {
    extract_split_positions<KIND>(local_sorted.indices1.ptr(0),
                                  local_sorted.indices2.ptr(0),
                                  volume,
                                  samples.ptr(0),
                                  num_usable_samples,
                                  split_positions.ptr(0),
                                  num_splitters,
                                  my_rank);
  }

  // collect sizes2send, send to rank i: local_sort_data from positions  split_positions[i-1],
  // split_positions[i] - 1
  auto size_send = create_buffer<uint64_t>(num_ranks, mem);
  {
    size_t last_position = 0;
    for (size_t rank = 0; rank < num_ranks - 1; ++rank) {
      size_t cur_position = split_positions[rank];
      size_send[rank]     = cur_position - last_position;
      last_position       = cur_position;
    }
    size_send[num_ranks - 1] = volume - last_position;
  }

  // cleanup intermediate data structures
  samples.destroy();
  split_positions.destroy();

  // all2all exchange send/receive sizes
  auto size_recv = create_buffer<uint64_t>(num_ranks, mem);
  legate::comm::coll::collAlltoall(
    size_send.ptr(0), size_recv.ptr(0), 1, legate::comm::coll::CollDataType::CollUint64, *comm);

  // Compute the sdispls and rdispls arrays using scans over the send
  // and recieve arrays.
  std::vector<int> sendcounts(num_ranks), recvcounts(num_ranks);
  std::vector<int> sdispls(num_ranks), rdispls(num_ranks);
  uint64_t sval = 0, rval = 0;
  for (size_t r = 0; r < num_ranks; r++) {
    sdispls[r]    = sval;
    rdispls[r]    = rval;
    sendcounts[r] = size_send[r];
    recvcounts[r] = size_recv[r];
    sval += size_send[r];
    rval += size_recv[r];
  }

  auto coord1_send_buf = local_sorted.indices1;
  auto coord2_send_buf = local_sorted.indices2;
  auto vals_send_buf   = local_sorted.values;

  // allocate target buffers.
  auto coord1_recv_buf = create_buffer<INDEX_TY>(rval, mem);
  auto coord2_recv_buf = create_buffer<INDEX_TY>(rval, mem);
  auto values_recv_buf = create_buffer<VAL_TY>(rval, mem);

  // TODO (rohany): This code below assumes that INDEX_TY's is CollInt64
  //  and that VAL_TY is CollDouble.

  // All2Allv time for each buffer.
  legate::comm::coll::collAlltoallv(coord1_send_buf.ptr(0),
                                    sendcounts.data(),
                                    sdispls.data(),
                                    coord1_recv_buf.ptr(0),
                                    recvcounts.data(),
                                    rdispls.data(),
                                    legate::comm::coll::CollDataType::CollInt64,
                                    *comm);
  legate::comm::coll::collAlltoallv(coord2_send_buf.ptr(0),
                                    sendcounts.data(),
                                    sdispls.data(),
                                    coord2_recv_buf.ptr(0),
                                    recvcounts.data(),
                                    rdispls.data(),
                                    legate::comm::coll::CollDataType::CollInt64,
                                    *comm);
  legate::comm::coll::collAlltoallv(vals_send_buf.ptr(0),
                                    sendcounts.data(),
                                    sdispls.data(),
                                    values_recv_buf.ptr(0),
                                    recvcounts.data(),
                                    rdispls.data(),
                                    legate::comm::coll::CollDataType::CollDouble,
                                    *comm);

  // Clean up remaining buffers.
  size_send.destroy();
  size_recv.destroy();

  // Extract the pieces of the received data into sort pieces.
  std::vector<SortPiece<INDEX_TY, VAL_TY>> merge_buffers(num_ranks);
  for (size_t r = 0; r < num_ranks; r++) {
    auto size                 = recvcounts[r];
    merge_buffers[r].size     = size;
    merge_buffers[r].indices1 = create_buffer<INDEX_TY>(size, mem);
    merge_buffers[r].indices2 = create_buffer<INDEX_TY>(size, mem);
    merge_buffers[r].values   = create_buffer<VAL_TY>(size, mem);
    // Copy each of the corresponding pieces into the buffers.
    if (rdispls[r] < rval) {
      thrust::copy(exec,
                   coord1_recv_buf.ptr(rdispls[r]),
                   coord1_recv_buf.ptr(rdispls[r]) + size,
                   merge_buffers[r].indices1.ptr(0));
      thrust::copy(exec,
                   coord2_recv_buf.ptr(rdispls[r]),
                   coord2_recv_buf.ptr(rdispls[r]) + size,
                   merge_buffers[r].indices2.ptr(0));
      thrust::copy(exec,
                   values_recv_buf.ptr(rdispls[r]),
                   values_recv_buf.ptr(rdispls[r]) + size,
                   merge_buffers[r].values.ptr(0));
    }
  }

  // Clean up some more allocations.
  coord1_recv_buf.destroy();
  coord2_recv_buf.destroy();
  values_recv_buf.destroy();

  // Merge all of the pieces together into the result buffer.
  for (size_t stride = 1; stride < num_ranks; stride *= 2) {
    for (size_t pos = 0; pos + stride < num_ranks; pos += 2 * stride) {
      auto source1       = merge_buffers[pos];
      auto source2       = merge_buffers[pos + stride];
      auto merged_size   = source1.size + source2.size;
      auto merged_coord1 = create_buffer<INDEX_TY>(merged_size, mem);
      auto merged_coord2 = create_buffer<INDEX_TY>(merged_size, mem);
      auto merged_values = create_buffer<VAL_TY>(merged_size, mem);

      auto p_left_coord1  = source1.indices1.ptr(0);
      auto p_left_coord2  = source1.indices2.ptr(0);
      auto p_left_values  = source1.values.ptr(0);
      auto p_right_coord1 = source2.indices1.ptr(0);
      auto p_right_coord2 = source2.indices2.ptr(0);
      auto p_right_values = source2.values.ptr(0);

      auto left_zipped_begin = thrust::make_tuple(p_left_coord1, p_left_coord2);
      auto left_zipped_end =
        thrust::make_tuple(p_left_coord1 + source1.size, p_left_coord2 + source1.size);
      auto right_zipped_begin = thrust::make_tuple(p_right_coord1, p_right_coord2);
      auto right_zipped_end =
        thrust::make_tuple(p_right_coord1 + source2.size, p_right_coord2 + source2.size);
      auto merged_zipped_begin = thrust::make_tuple(merged_coord1.ptr(0), merged_coord2.ptr(0));

      thrust::merge_by_key(exec,
                           thrust::make_zip_iterator(left_zipped_begin),
                           thrust::make_zip_iterator(left_zipped_end),
                           thrust::make_zip_iterator(right_zipped_begin),
                           thrust::make_zip_iterator(right_zipped_end),
                           p_left_values,
                           p_right_values,
                           thrust::make_zip_iterator(merged_zipped_begin),
                           merged_values.ptr(0));

      // Clean up allocations that we don't need anymore.
      source1.indices1.destroy();
      source1.indices2.destroy();
      source1.values.destroy();
      source2.indices1.destroy();
      source2.indices2.destroy();
      source2.values.destroy();

      merge_buffers[pos].indices1 = merged_coord1;
      merge_buffers[pos].indices2 = merged_coord2;
      merge_buffers[pos].values   = merged_values;
      merge_buffers[pos].size     = merged_size;
    }
  }
  return merge_buffers[0];
}

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
      auto result = samplesort<KIND, INDEX_TY, VAL_TY, Policy, Comm>(
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

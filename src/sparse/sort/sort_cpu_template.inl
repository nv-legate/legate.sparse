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

#include "sparse/sort/sort.h"
#include "sparse/sort/sort_template.inl"

#include <core/comm/coll.h>

namespace sparse {

using namespace legate;

template <VariantKind KIND, typename INDEX_TY, typename VAL_TY, typename Policy, typename Comm>
struct SampleSorter {
  SortPiece<INDEX_TY, VAL_TY> operator()(Policy exec,
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
      extract_samples(local_sorted.indices1.ptr(0),
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
      extract_split_positions(local_sorted.indices1.ptr(0),
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

    // Adjust all of the send and recieve sizes by
    // the size of the actual datatypes being sent.
    std::vector<int> sendcounts_index(num_ranks), recvcounts_index(num_ranks);
    std::vector<int> sendcounts_value(num_ranks), recvcounts_value(num_ranks);
    std::vector<int> sdispls_index(num_ranks), rdispls_index(num_ranks);
    std::vector<int> sdispls_value(num_ranks), rdispls_value(num_ranks);
    for (size_t r = 0; r < num_ranks; r++) {
      sendcounts_index[r] = sendcounts[r] * sizeof(INDEX_TY);
      sendcounts_value[r] = sendcounts[r] * sizeof(VAL_TY);
      recvcounts_index[r] = recvcounts[r] * sizeof(INDEX_TY);
      recvcounts_value[r] = recvcounts[r] * sizeof(VAL_TY);
      sdispls_index[r]    = sdispls[r] * sizeof(INDEX_TY);
      sdispls_value[r]    = sdispls[r] * sizeof(VAL_TY);
      rdispls_index[r]    = rdispls[r] * sizeof(INDEX_TY);
      rdispls_value[r]    = rdispls[r] * sizeof(VAL_TY);
    }

    auto coord1_send_buf = local_sorted.indices1;
    auto coord2_send_buf = local_sorted.indices2;
    auto vals_send_buf   = local_sorted.values;

    // allocate target buffers.
    auto coord1_recv_buf = create_buffer<INDEX_TY>(rval, mem);
    auto coord2_recv_buf = create_buffer<INDEX_TY>(rval, mem);
    auto values_recv_buf = create_buffer<VAL_TY>(rval, mem);

    // All2Allv time for each buffer.
    legate::comm::coll::collAlltoallv(coord1_send_buf.ptr(0),
                                      sendcounts_index.data(),
                                      sdispls_index.data(),
                                      coord1_recv_buf.ptr(0),
                                      recvcounts_index.data(),
                                      rdispls_index.data(),
                                      legate::comm::coll::CollDataType::CollInt8,
                                      *comm);
    legate::comm::coll::collAlltoallv(coord2_send_buf.ptr(0),
                                      sendcounts_index.data(),
                                      sdispls_index.data(),
                                      coord2_recv_buf.ptr(0),
                                      recvcounts_index.data(),
                                      rdispls_index.data(),
                                      legate::comm::coll::CollDataType::CollInt8,
                                      *comm);
    legate::comm::coll::collAlltoallv(vals_send_buf.ptr(0),
                                      sendcounts_value.data(),
                                      sdispls_value.data(),
                                      values_recv_buf.ptr(0),
                                      recvcounts_value.data(),
                                      rdispls_value.data(),
                                      legate::comm::coll::CollDataType::CollInt8,
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
};

}  // namespace sparse

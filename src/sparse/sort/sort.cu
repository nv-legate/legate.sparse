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
#include "thrust_allocator.h"
#include "cuda_help.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename INDEX_TY>
__global__ static void extract_samples_kernel(const INDEX_TY* data1,
                                              const INDEX_TY* data2,
                                              const size_t volume,
                                              Sample<INDEX_TY>* samples,
                                              const size_t num_local_samples,
                                              const Sample<INDEX_TY> init_sample,
                                              const size_t offset,
                                              const size_t rank)
{
  auto sample_idx = global_tid_1d();
  if (sample_idx >= num_local_samples) return;
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

template <typename INDEX_TY, typename STREAM>
void extract_samples_gpu(const INDEX_TY* data1,
                         const INDEX_TY* data2,
                         const size_t volume,
                         Sample<INDEX_TY>* samples,
                         const size_t num_local_samples,
                         const Sample<INDEX_TY> init_sample,
                         const size_t offset,
                         const size_t rank,
                         STREAM& stream)
{
  auto blocks = get_num_blocks_1d(num_local_samples);
  if (blocks > 0) {
    extract_samples_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      data1, data2, volume, samples, num_local_samples, init_sample, offset, rank);
  }
}

template <typename INDEX_TY>
__global__ static void extract_split_positions_kernel(const INDEX_TY* data1,
                                                      const INDEX_TY* data2,
                                                      const size_t volume,
                                                      const Sample<INDEX_TY>* samples,
                                                      const size_t num_samples,
                                                      size_t* split_positions,
                                                      const size_t num_splitters,
                                                      const size_t rank)
{
  const auto splitter_idx = global_tid_1d();
  if (splitter_idx >= num_splitters) return;

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

template <typename INDEX_TY, typename STREAM>
static void extract_split_positions_gpu(const INDEX_TY* data1,
                                        const INDEX_TY* data2,
                                        const size_t volume,
                                        const Sample<INDEX_TY>* samples,
                                        const size_t num_samples,
                                        size_t* split_positions,
                                        const size_t num_splitters,
                                        const size_t rank,
                                        STREAM& stream)
{
  auto blocks = get_num_blocks_1d(num_splitters);
  if (blocks > 0) {
    extract_split_positions_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      data1, data2, volume, samples, num_samples, split_positions, num_splitters, rank);
  }
}

template <typename INDEX_TY, typename VAL_TY, typename Policy, typename Comm>
struct SampleSorter<VariantKind::GPU, INDEX_TY, VAL_TY, Policy, Comm> {
  SortPiece<INDEX_TY, VAL_TY> operator()(Policy exec,
                                         SortPiece<INDEX_TY, VAL_TY> local_sorted,
                                         size_t my_rank,
                                         size_t num_ranks,
                                         Legion::Memory::Kind mem,
                                         Comm* comm_ptr)
  {
    auto stream   = get_cached_stream();
    size_t volume = local_sorted.size;
    // To make the template dispatch work out, we take a pointer
    // to a communicator. However, the communicator we get from
    // the context is already a ncclComm_t*, so we have to do
    // one dereference here to make the types work out.
    auto comm = *comm_ptr;

    // collect local samples - for now we take num_ranks samples for every node
    // worst case this leads to 2*N/ranks elements on a single node
    size_t num_local_samples = num_ranks;

    size_t num_global_samples = num_local_samples * num_ranks;
    auto samples              = create_buffer<Sample<INDEX_TY>>(num_global_samples, mem);

    Sample<INDEX_TY> init_sample;
    {
      init_sample.rank = -1;  // init samples that are not populated
      size_t offset    = num_local_samples * my_rank;
      extract_samples_gpu(local_sorted.indices1.ptr(0),
                          local_sorted.indices2.ptr(0),
                          volume,
                          samples.ptr(0),
                          num_local_samples,
                          init_sample,
                          offset,
                          my_rank,
                          stream);
    }

    CHECK_NCCL(ncclAllGather(samples.ptr(my_rank * num_ranks),
                             samples.ptr(0),
                             num_ranks * sizeof(Sample<INDEX_TY>),
                             ncclInt8,
                             *comm,
                             stream));

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
    auto split_positions       = create_buffer<size_t>(num_splitters, Memory::Z_COPY_MEM);
    {
      extract_split_positions_gpu(local_sorted.indices1.ptr(0),
                                  local_sorted.indices2.ptr(0),
                                  volume,
                                  samples.ptr(0),
                                  num_usable_samples,
                                  split_positions.ptr(0),
                                  num_splitters,
                                  my_rank,
                                  stream);
    }

    // need to sync as we share values in between host/device
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // collect sizes2send, send to rank i: local_sort_data from positions  split_positions[i-1],
    // split_positions[i] - 1
    auto size_send = create_buffer<uint64_t>(num_ranks, Memory::Z_COPY_MEM);
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
    auto size_recv = create_buffer<uint64_t>(num_ranks, Memory::Z_COPY_MEM);
    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_ranks; r++) {
      CHECK_NCCL(ncclSend(size_send.ptr(r), 1, ncclUint64, r, *comm, stream));
      CHECK_NCCL(ncclRecv(size_recv.ptr(r), 1, ncclUint64, r, *comm, stream));
    }
    CHECK_NCCL(ncclGroupEnd());

    // need to sync as we share values in between host/device
    CHECK_CUDA(cudaStreamSynchronize(stream));

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
    std::vector<SortPiece<INDEX_TY, VAL_TY>> merge_buffers(num_ranks);
    for (size_t r = 0; r < num_ranks; r++) {
      auto size                 = recvcounts[r];
      merge_buffers[r].size     = size;
      merge_buffers[r].indices1 = create_buffer<INDEX_TY>(size, mem);
      merge_buffers[r].indices2 = create_buffer<INDEX_TY>(size, mem);
      merge_buffers[r].values   = create_buffer<VAL_TY>(size, mem);
    }

    // All2Allv time for each buffer.
    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_ranks; r++) {
      CHECK_NCCL(ncclSend(sdispls[r] >= sval ? nullptr : coord1_send_buf.ptr(sdispls[r]),
                          size_send[r] * sizeof(INDEX_TY),
                          ncclInt8,
                          r,
                          *comm,
                          stream));
      CHECK_NCCL(ncclRecv(merge_buffers[r].indices1.ptr(0),
                          size_recv[r] * sizeof(INDEX_TY),
                          ncclInt8,
                          r,
                          *comm,
                          stream));
    }
    CHECK_NCCL(ncclGroupEnd());

    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_ranks; r++) {
      CHECK_NCCL(ncclSend(sdispls[r] >= sval ? nullptr : coord2_send_buf.ptr(sdispls[r]),
                          size_send[r] * sizeof(INDEX_TY),
                          ncclInt8,
                          r,
                          *comm,
                          stream));
      CHECK_NCCL(ncclRecv(merge_buffers[r].indices2.ptr(0),
                          size_recv[r] * sizeof(INDEX_TY),
                          ncclInt8,
                          r,
                          *comm,
                          stream));
    }
    CHECK_NCCL(ncclGroupEnd());

    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_ranks; r++) {
      CHECK_NCCL(ncclSend(sdispls[r] >= sval ? nullptr : vals_send_buf.ptr(sdispls[r]),
                          size_send[r] * sizeof(VAL_TY),
                          ncclInt8,
                          r,
                          *comm,
                          stream));
      CHECK_NCCL(ncclRecv(
        merge_buffers[r].values.ptr(0), size_recv[r] * sizeof(VAL_TY), ncclInt8, r, *comm, stream));
    }
    CHECK_NCCL(ncclGroupEnd());

    // Clean up remaining buffers.
    size_send.destroy();
    size_recv.destroy();

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

/* static */ void SortByKey::gpu_variant(legate::TaskContext& ctx)
{
  auto stream = get_cached_stream();
  ThrustAllocator alloc(Memory::GPU_FB_MEM);
  auto policy = thrust::cuda::par(alloc).on(stream);
  sort_by_key_template<VariantKind::GPU, decltype(policy), ncclComm_t*>(
    ctx, policy, Memory::GPU_FB_MEM);
  CHECK_CUDA_STREAM(stream);
}

}  // namespace sparse

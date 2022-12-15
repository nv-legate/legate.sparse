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

#include "sparse/array/csr/spmv.h"
#include "sparse/array/csr/spmv_template.inl"

#include <core/comm/coll.h>

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct CSRSpMVRowSplitImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 1>& x,
                  const Rect<1>& rect)
  {
#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (coord_t i = rect.lo[0]; i < rect.hi[0] + 1; i++) {
      VAL_TY sum = 0.0;
      for (size_t j_pos = A_pos[i].lo; j_pos < A_pos[i].hi + 1; j_pos++) {
        auto j = A_crd[j_pos];
        sum += A_vals[j_pos] * x[j];
      }
      y[i] = sum;
    }
  }
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

inline void check_mpi(int error, const char* file, int line)
{
  if (error != MPI_SUCCESS) {
    fprintf(
      stderr, "Internal MPI failure with error code %d in file %s at line %d\n", error, file, line);
    assert(false);
  }
}

#define CHECK_MPI(expr)                    \
  do {                                     \
    int result = (expr);                   \
    check_mpi(result, __FILE__, __LINE__); \
  } while (false)

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename COMM>
struct CSRSpMVRowSplitExplicitCollectiveImplBody<VariantKind::OMP, INDEX_CODE, VAL_CODE, COMM> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(const AccessorWO<VAL_TY, 1>& y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 1>& x_local,
                  const Rect<1>& y_rect,
		  const Rect<1>& x_rect,
		  COMM comm,
		  int num_ranks,
		  int my_rank)
  {
    auto mpi_comm = comm->mpi_comm;
    auto mem = Sparse::has_numamem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM;
    // First do a gather to understand how much data to expect from each.
    auto counts = create_buffer<int>(num_ranks, mem);
    auto displs = create_buffer<int>(num_ranks, mem);
    int elems = x_rect.volume();
    CHECK_MPI(MPI_Gather(&elems, 1, MPI_INT32_T, counts.ptr(0), 1, MPI_INT32_T, 0, mpi_comm));

    // Use the information about counts to do an gatherv.
    int tot_size = 0;
    for (int i = 0; i < num_ranks; i++) {
      displs[i] = tot_size * sizeof(VAL_TY);
      tot_size += counts[i];
      counts[i] = counts[i] * sizeof(VAL_TY);
    }

    Legion::DeferredBuffer<VAL_TY, 1> global_x;
    if (my_rank == 0) {
      global_x = create_buffer<VAL_TY>(tot_size, mem);
      CHECK_MPI(MPI_Gatherv(x_local.ptr(x_rect.lo), x_rect.volume() * sizeof(VAL_TY), MPI_UINT8_T, global_x.ptr(0), counts.ptr(0), displs.ptr(0), MPI_UINT8_T, 0, mpi_comm));
    } else {
      CHECK_MPI(MPI_Gatherv(x_local.ptr(x_rect.lo), x_rect.volume() * sizeof(VAL_TY), MPI_UINT8_T, nullptr, nullptr, nullptr, MPI_UINT8_T, 0, mpi_comm));
    }
    CHECK_MPI(MPI_Bcast(&tot_size, 1, MPI_INT32_T, 0, mpi_comm));
    if (my_rank != 0) {
      global_x = create_buffer<VAL_TY>(tot_size, mem);
    }
    CHECK_MPI(MPI_Bcast(global_x.ptr(0), tot_size * sizeof(VAL_TY), MPI_UINT8_T, 0, mpi_comm));

#pragma omp parallel for schedule(monotonic : dynamic, 128)
    for (coord_t i = y_rect.lo[0]; i < y_rect.hi[0] + 1; i++) {
      VAL_TY sum = 0.0;
      for (size_t j_pos = A_pos[i].lo; j_pos < A_pos[i].hi + 1; j_pos++) {
        auto j = A_crd[j_pos];
        sum += A_vals[j_pos] * global_x[j];
      }
      y[i] = sum;
    }
  }
};

/*static*/ void CSRSpMVRowSplit::omp_variant(TaskContext& context)
{
  csr_spmv_row_split_template<VariantKind::OMP>(context);
}

/*static*/ void CSRSpMVRowSplitExplicitCollective::omp_variant(TaskContext& context)
{
  csr_spmv_row_split_explicit_collective_template<VariantKind::OMP, legate::comm::coll::CollComm>(context);
}

}  // namespace sparse

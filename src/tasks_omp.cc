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

#include "sparse.h"
#include "distal_utils.h"
#include "tasks.h"
#include "sort.h"
#include "sort_template.inl"

#include <omp.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

using namespace Legion;

namespace sparse {

void CSRSpMVRowSplitTropicalSemiring::omp_variant(legate::TaskContext& ctx)
{
  auto& y   = ctx.outputs()[0];
  auto& pos = ctx.inputs()[0];
  auto& crd = ctx.inputs()[1];
  auto& x   = ctx.inputs()[2];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (pos.transformed()) { pos.remove_transform(); }

  auto yacc   = y.write_accessor<coord_ty, 2>();
  auto posacc = pos.read_accessor<Rect<1>, 1>();
  auto crdacc = crd.read_accessor<coord_ty, 1>();
  auto xacc   = x.read_accessor<coord_ty, 2>();
  auto bounds = y.domain();

  auto num_fields = x.domain().hi()[1] - x.domain().lo()[1] + 1;
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (coord_ty i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
    // Initialize the output.
    for (coord_ty f = 0; f < num_fields; f++) { yacc[{i, f}] = 0; }
    for (size_t jpos = posacc[i].lo; jpos < posacc[i].hi + 1; jpos++) {
      auto j         = crdacc[jpos];
      bool y_greater = true;
      for (coord_ty f = 0; f < num_fields; f++) {
        if (yacc[{i, f}] > xacc[{j, f}]) {
          y_greater = true;
          break;
        } else if (yacc[{i, f}] < xacc[{j, f}]) {
          y_greater = false;
          break;
        }
        // Else the fields are equal, so move onto the next field.
      }
      if (!y_greater) {
        for (coord_ty f = 0; f < num_fields; f++) { yacc[{i, f}] = xacc[{j, f}]; }
      }
    }
  }
}

// SpGEMM on CSR = CSR x CSR is adapted from DISTAL/TACO generated code.
// A(i, j) = B(i, k) * C(k, j)
// Schedule:
// assemble(A, AssembleStrategy::Insert)
// precompute(j, w), w is a workspace of size A2.
// parallelize(i, NoRaces)
void SpGEMMCSRxCSRxCSRNNZ::omp_variant(legate::TaskContext& ctx)
{
  auto A2_dim = ctx.scalars()[0].value<size_t>();
  auto& nnz   = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& C_pos = ctx.inputs()[2];
  auto& C_crd = ctx.inputs()[3];

  auto nnz_acc   = nnz.write_accessor<nnz_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();

  // Allocate sparse accelerator data.
  coord_ty initValInt = 0;
  bool initValBool    = false;
  auto num_threads    = omp_get_max_threads();
  auto kind           = Sparse::has_numamem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM;
  DeferredBuffer<coord_ty, 1> index_list_all(
    kind, Rect<1>{0, (A2_dim * num_threads) - 1}, &initValInt);
  DeferredBuffer<bool, 1> already_set_all(
    kind, Rect<1>{0, (A2_dim * num_threads) - 1}, &initValBool);
  // For this computation, we assume that the rows are partitioned.
  auto bounds = B_pos.domain();
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (auto i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
    auto thread_id         = omp_get_thread_num();
    auto index_list        = index_list_all.ptr(thread_id * A2_dim);
    auto already_set       = already_set_all.ptr(thread_id * A2_dim);
    size_t index_list_size = 0;
    for (size_t kB = B_pos_acc[i].lo; kB < B_pos_acc[i].hi + 1; kB++) {
      auto k = B_crd_acc[kB];
      for (size_t jC = C_pos_acc[k].lo; jC < C_pos_acc[k].hi + 1; jC++) {
        auto j = C_crd_acc[jC];
        if (!already_set[j]) {
          index_list[index_list_size] = j;
          already_set[j]              = true;
          index_list_size++;
        }
      }
    }
    size_t row_nnzs = 0;
    for (auto index_loc = 0; index_loc < index_list_size; index_loc++) {
      auto j         = index_list[index_loc];
      already_set[j] = false;
      row_nnzs++;
    }
    nnz_acc[i] = row_nnzs;
  }
}

void SpGEMMCSRxCSRxCSR::omp_variant(legate::TaskContext& ctx)
{
  auto A2_dim  = ctx.scalars()[0].value<size_t>();
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];

  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos  = ctx.inputs()[3];
  auto& C_crd  = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  auto A_pos_acc  = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc  = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_pos_acc  = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc  = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  // Allocate sparse accelerator data.
  coord_ty initValInt  = 0;
  bool initValBool     = false;
  val_ty initValDouble = 0.0;
  auto num_threads     = omp_get_max_threads();
  auto kind            = Sparse::has_numamem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM;
  DeferredBuffer<coord_ty, 1> index_list_all(
    kind, Rect<1>{0, (A2_dim * num_threads) - 1}, &initValInt);
  DeferredBuffer<bool, 1> already_set_all(
    kind, Rect<1>{0, (A2_dim * num_threads) - 1}, &initValBool);
  DeferredBuffer<val_ty, 1> workspace_all(
    kind, Rect<1>{0, (A2_dim * num_threads) - 1}, &initValDouble);
  // For this computation, we assume that the rows are partitioned.
  auto bounds = B_pos.domain();
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (auto i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
    auto thread_id         = omp_get_thread_num();
    auto index_list        = index_list_all.ptr(thread_id * A2_dim);
    auto already_set       = already_set_all.ptr(thread_id * A2_dim);
    auto workspace         = workspace_all.ptr(thread_id * A2_dim);
    size_t index_list_size = 0;
    for (size_t kB = B_pos_acc[i].lo; kB < B_pos_acc[i].hi + 1; kB++) {
      auto k = B_crd_acc[kB];
      for (size_t jC = C_pos_acc[k].lo; jC < C_pos_acc[k].hi + 1; jC++) {
        auto j = C_crd_acc[jC];
        if (!already_set[j]) {
          index_list[index_list_size] = j;
          already_set[j]              = true;
          index_list_size++;
        }
        workspace[j] += B_vals_acc[kB] * C_vals_acc[jC];
      }
    }
    size_t pA2 = A_pos_acc[i].lo;
    for (auto index_loc = 0; index_loc < index_list_size; index_loc++) {
      auto j          = index_list[index_loc];
      already_set[j]  = false;
      A_crd_acc[pA2]  = j;
      A_vals_acc[pA2] = workspace[j];
      pA2++;
      // Zero out the workspace once we have read the value.
      workspace[j] = 0.0;
    }
  }
}

void SpGEMMCSRxCSRxCSCLocalTiles::omp_variant(legate::TaskContext& ctx)
{
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];

  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos  = ctx.inputs()[3];
  auto& C_crd  = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  // Remove the transformations on the B_pos and C_pos stores.
  B_pos.remove_transform();
  C_pos.remove_transform();

  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_pos_acc  = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc  = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  auto B_domain = B_pos.domain();
  auto C_domain = C_pos.domain();
  // Pull these out so that we don't allocate within the loop.
  auto B_lo = B_domain.lo();
  auto B_hi = B_domain.hi();
  auto C_lo = C_domain.lo();
  auto C_hi = C_domain.hi();

  // Our job right now is to perform both passes of the SpGEMM operation
  // and output instances for local CSR matrices of each result.
  auto kind = Sparse::has_numamem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM;
  DeferredBuffer<size_t, 1> nnz({B_lo[0], B_hi[0]}, kind);
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (auto i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    size_t row_nnzs = 0;
    for (auto j = C_lo[0]; j < C_hi[0] + 1; j++) {
      size_t kB_pos     = B_pos_acc[i].lo;
      size_t kB_pos_end = B_pos_acc[i].hi + 1;
      size_t kC_pos     = C_pos_acc[j].lo;
      size_t kC_pos_end = C_pos_acc[j].hi + 1;
      while (kB_pos < kB_pos_end && kC_pos < kC_pos_end) {
        auto kB = B_crd_acc[kB_pos];
        auto kC = C_crd_acc[kC_pos];
        auto k  = std::min(kB, kC);
        if (k == kB && k == kC) {
          row_nnzs++;
          break;
        }
        kB_pos += (int64_t)(kB == k);
        kC_pos += (int64_t)(kC == k);
      }
    }
    nnz[i] = row_nnzs;
  }

  // Do an in-place, inclusive scan on nnz.
  auto nnz_base = nnz.ptr(B_lo);
  thrust::inclusive_scan(thrust::omp::par, nnz_base, nnz_base + (B_hi[0] - B_lo[0] + 1), nnz_base);
  // Construct the final pos array.
  auto A_pos_acc =
    A_pos.create_output_buffer<Rect<1>, 1>(B_pos.domain().get_volume(), true /* return_buffer */);
#pragma omp parallel for schedule(static)
  for (size_t i = B_lo[0]; i < B_hi[0] + 1; i++) {
    size_t prev            = i == B_lo[0] ? 0 : nnz[i - 1];
    A_pos_acc[i - B_lo[0]] = {prev, nnz[i] - 1};
  }
  auto tot_nnz    = nnz[B_hi];
  auto A_crd_acc  = A_crd.create_output_buffer<coord_ty, 1>(tot_nnz, true /* return_buffer */);
  auto A_vals_acc = A_vals.create_output_buffer<val_ty, 1>(tot_nnz, true /* return_buffer */);

#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    // Important: We need to offset the access into the output buffers, as they
    // are not globally indexed objects.
    size_t nnz_pos = A_pos_acc[i - B_lo[0]].lo;
    for (coord_ty j = C_lo[0]; j < C_hi[0] + 1; j++) {
      double result     = 0.0;
      bool set          = false;
      size_t kB_pos     = B_pos_acc[i].lo;
      size_t kB_pos_end = B_pos_acc[i].hi + 1;
      size_t kC_pos     = C_pos_acc[j].lo;
      size_t kC_pos_end = C_pos_acc[j].hi + 1;
      while (kB_pos < kB_pos_end && kC_pos < kC_pos_end) {
        auto kB = B_crd_acc[kB_pos];
        auto kC = C_crd_acc[kC_pos];
        auto k  = std::min(kB, kC);
        if (k == kB && k == kC) {
          set = true;
          result += B_vals_acc[kB_pos] * C_vals_acc[kC_pos];
        }
        kB_pos += (int64_t)(kB == k);
        kC_pos += (int64_t)(kC == k);
      }
      // Since we're considering a row at a time and not scattering
      // across pos, we don't need to mutate pos like as done in DISTAL.
      if (set) {
        A_crd_acc[nnz_pos]  = j;
        A_vals_acc[nnz_pos] = result;
        nnz_pos++;
      }
    }
  }
}

void SpGEMMCSRxCSRxCSCCommCompute::omp_variant(legate::TaskContext& ctx)
{
  auto out         = ctx.outputs()[0].write_accessor<Rect<1>, 3>();
  auto& pos        = ctx.inputs()[0];
  auto& global_pos = ctx.inputs()[1];

  auto gx = ctx.scalars()[0].value<int32_t>();
  auto gy = ctx.scalars()[1].value<int32_t>();

  auto output_point = ctx.outputs()[0].domain().lo();
  auto oi           = output_point[0];
  auto oj           = output_point[1];

  auto global_pos_domain = global_pos.domain();

  auto i_lo        = pos.domain().lo()[0];
  auto i_hi        = pos.domain().hi()[0] + 1;
  auto i_tile_size = i_hi - i_lo;
  auto tile_size   = (i_hi - i_lo + gy - 1) / gy;
#pragma omp parallel for schedule(static)
  for (int32_t j = 0; j < gy; j++) {
    auto sub_tile_start = j * tile_size;
    auto sub_tile_end   = std::min((j + 1) * tile_size, i_tile_size);
    auto lo             = global_pos_domain.lo()[0] + sub_tile_start;
    auto hi             = global_pos_domain.lo()[0] + sub_tile_end;
    out[{oi, oj, j}]    = Rect<1>{lo, hi - 1};
  }
}

void SpGEMMCSRxCSRxCSCShuffle::omp_variant(legate::TaskContext& ctx)
{
  auto& global_pos  = ctx.inputs()[0];
  auto& global_crd  = ctx.inputs()[1];
  auto& global_vals = ctx.inputs()[2];

  auto& out_pos  = ctx.outputs()[0];
  auto& out_crd  = ctx.outputs()[1];
  auto& out_vals = ctx.outputs()[2];

  // TODO (rohany): I want a sparse instance here.
  auto global_pos_acc  = global_pos.read_accessor<Rect<1>, 1>();
  auto global_crd_acc  = global_crd.read_accessor<coord_ty, 1>();
  auto global_vals_acc = global_vals.read_accessor<val_ty, 1>();

  // I believe that there should be j rectangles here (O(sqrt(p)), so this
  // is something that we can do sequentially without paying much in perf.
  size_t total_nnzs = 0;
  std::vector<Rect<1>> rects;
  for (RectInDomainIterator<1> rect_itr(global_pos.domain()); rect_itr(); rect_itr++) {
    rects.push_back(*rect_itr);
    if (rect_itr->empty()) continue;
    auto lo = global_pos_acc[rect_itr->lo];
    auto hi = global_pos_acc[rect_itr->hi];
    total_nnzs += hi.hi[0] - lo.lo[0] + 1;
  }
  size_t total_rows = 0;
  for (auto it : rects) { total_rows = std::max(total_rows, it.volume()); }

  auto pos_acc  = out_pos.create_output_buffer<Rect<1>, 1>(total_rows, true /* return_buffer */);
  auto crd_acc  = out_crd.create_output_buffer<coord_ty, 1>(total_nnzs, true /* return_buffer */);
  auto vals_acc = out_vals.create_output_buffer<val_ty, 1>(total_nnzs, true /* return_buffer */);

  // Calculate the number of elements that each row will write.
  auto kind = Sparse::has_numamem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM;
  DeferredBuffer<size_t, 1> row_offsets({0, total_rows - 1}, kind);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < total_rows; i++) {
    size_t elems = 0;
    for (auto rect : rects) {
      auto global_pos_idx = rect.lo + i;
      if (rect.contains(global_pos_idx)) { elems += global_pos_acc[global_pos_idx].volume(); }
    }
    row_offsets[i] = elems;
  }
  // Scan over the counts to find the offsets for each row.
  thrust::exclusive_scan(
    thrust::omp::par, row_offsets.ptr(0), row_offsets.ptr(0) + total_rows, row_offsets.ptr(0));

// Write out the outputs in parallel, by row.
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (size_t i = 0; i < total_rows; i++) {
    auto offset = row_offsets[i];
    auto lo     = offset;
    for (auto rect : rects) {
      auto global_pos_idx = rect.lo + i;
      if (!rect.contains(global_pos_idx)) continue;
      for (int64_t pos = global_pos_acc[global_pos_idx].lo;
           pos < global_pos_acc[global_pos_idx].hi + 1;
           pos++) {
        crd_acc[offset]  = global_crd_acc[pos];
        vals_acc[offset] = global_vals_acc[pos];
        offset++;
      }
    }
    auto hi    = offset - 1;
    pos_acc[i] = {lo, hi};
  }
}

// SpMMCSR is adapted from DISTAL/TACO generated code where A and C are dense.
// A(i, j) = B(i, k) * C(k, j)
// Schedule:
// distribute({i, j}, {io, jo}, {ii, ji}, grid)
// communicate({A, B, C}, jo)
// reorder({io, jo, ii, k, ji})
// parallelize(ii, CPUThread, NoRaces)
void SpMMCSR::omp_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 2>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();

  // In this computation, both i and j can be partitioned.
  auto A_domain = A_vals.domain();
  auto C_domain = C_vals.domain();
  auto C_lo     = C_domain.lo();
  auto C_hi     = C_domain.hi();

// First zero out the output array.
#pragma omp parallel for schedule(static) collapse(2)
  for (int64_t i = A_domain.lo()[0]; i < A_domain.hi()[0] + 1; i++) {
    for (int64_t j = C_lo[1]; j < C_hi[1] + 1; j++) { A_vals_acc[{i, j}] = 0.0; }
  }

#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (int64_t i = A_domain.lo()[0]; i < A_domain.hi()[0] + 1; i++) {
    for (size_t kB = B_pos_acc[i].lo; kB < B_pos_acc[i].hi + 1; kB++) {
      auto k = B_crd_acc[kB];
      for (int64_t j = C_lo[1]; j < C_hi[1] + 1; j++) {
        A_vals_acc[{i, j}] += B_vals_acc[kB] * C_vals_acc[{k, j}];
      }
    }
  }
}

void SpMMDenseCSR::omp_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.reductions()[0];
  auto& B_vals = ctx.inputs()[0];
  auto& C_pos  = ctx.inputs()[1];
  auto& C_crd  = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (C_pos.transformed()) { C_pos.remove_transform(); }

  auto A_vals_acc = A_vals.reduce_accessor<SumReduction<val_ty>, true /* exclusive */, 2>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();
  auto C_pos_acc  = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc  = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  auto A_domain = A_vals.domain();
  auto C_domain = C_pos.domain();
  auto C_lo     = C_domain.lo();
  auto C_hi     = C_domain.hi();
// This loop ordering (i, k, j) allows for in-order accessing of all
// three tensors and avoids atomic reductions into the output at the
// cost of reading the sparse tensor C i times. This assumes that i
// is generally small.
#pragma omp parallel for schedule(static)
  for (coord_ty i = A_domain.lo()[0]; i < A_domain.hi()[0] + 1; i++) {
    for (coord_ty k = C_lo[0]; k < C_hi[0] + 1; k++) {
      for (size_t jB = C_pos_acc[k].lo; jB < C_pos_acc[k].hi + 1; jB++) {
        auto j = C_crd_acc[jB];
        A_vals_acc[{i, j}] <<= B_vals_acc[{i, k}] * C_vals_acc[jB];
      }
    }
  }
}

void CSRSDDMM::omp_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];
  auto& D_vals = ctx.inputs()[4];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();
  auto D_vals_acc = D_vals.read_accessor<val_ty, 2>();

  auto B_pos_domain = B_pos.domain();
  auto B_crd_domain = B_crd.domain();
  auto C_domain     = C_vals.domain();
  auto B_pos_lo     = B_pos_domain.lo();
  auto B_pos_hi     = B_pos_domain.hi();
  auto B_crd_lo     = B_crd_domain.lo();
  auto B_crd_hi     = B_crd_domain.hi();
  auto C_lo         = C_domain.lo();
  auto C_hi         = C_domain.hi();

  // Return if there are no non-zeros to process.
  if (B_crd_domain.empty()) return;

  // We'll chunk up the non-zeros into pieces so that each thread
  // only needs to perform one binary search.
  auto tot_nnz     = B_crd_domain.get_volume();
  auto num_threads = omp_get_max_threads();
  auto tile_size   = (tot_nnz + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static)
  for (size_t tid = 0; tid < num_threads; tid++) {
    auto first_nnz = tid * tile_size + B_crd_lo[0];
    size_t i_pos   = taco_binarySearchBefore(B_pos_acc, B_pos_lo[0], B_pos_hi[0], first_nnz);
    coord_ty i     = i_pos;
    for (size_t nnz_idx = (tid * tile_size); nnz_idx < std::min((tid + 1) * tile_size, tot_nnz);
         nnz_idx++) {
      size_t j_pos = nnz_idx + B_crd_lo[0];
      coord_ty j   = B_crd_acc[j_pos];
      // Bump up our row pointer until we find this position.
      while (!B_pos_acc[i_pos].contains(j_pos)) {
        i_pos++;
        i = i_pos;
      }
      val_ty sum = 0.0;
      for (coord_ty k = C_lo[1]; k < C_hi[1] + 1; k++) {
        sum += B_vals_acc[j_pos] * (C_vals_acc[{i, k}] * D_vals_acc[{k, j}]);
      }
      A_vals_acc[j_pos] = sum;
    }
  }
}

void CSCSDDMM::omp_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];
  auto& D_vals = ctx.inputs()[4];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();
  auto D_vals_acc = D_vals.read_accessor<val_ty, 2>();

  auto B_pos_domain = B_pos.domain();
  auto B_crd_domain = B_crd.domain();
  auto C_domain     = C_vals.domain();
  auto B_pos_lo     = B_pos_domain.lo();
  auto B_pos_hi     = B_pos_domain.hi();
  auto B_crd_lo     = B_crd_domain.lo();
  auto B_crd_hi     = B_crd_domain.hi();
  auto C_lo         = C_domain.lo();
  auto C_hi         = C_domain.hi();

  // Return if there are no non-zeros to process.
  if (B_crd_domain.empty()) return;

  // We'll chunk up the non-zeros into pieces so that each thread
  // only needs to perform one binary search.
  auto tot_nnz     = B_crd_domain.get_volume();
  auto num_threads = omp_get_max_threads();
  auto tile_size   = (tot_nnz + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static)
  for (size_t tid = 0; tid < num_threads; tid++) {
    auto first_nnz = tid * tile_size + B_crd_lo[0];
    size_t j_pos   = taco_binarySearchBefore(B_pos_acc, B_pos_lo[0], B_pos_hi[0], first_nnz);
    coord_ty j     = j_pos;
    for (size_t nnz_idx = (tid * tile_size); nnz_idx < std::min((tid + 1) * tile_size, tot_nnz);
         nnz_idx++) {
      size_t i_pos = nnz_idx + B_crd_lo[0];
      coord_ty i   = B_crd_acc[i_pos];
      // Bump up our row pointer until we find this position.
      while (!B_pos_acc[j_pos].contains(i_pos)) {
        j_pos++;
        j = j_pos;
      }
      val_ty sum = 0.0;
      for (coord_ty k = C_lo[1]; k < C_hi[1] + 1; k++) {
        sum += B_vals_acc[i_pos] * (C_vals_acc[{i, k}] * D_vals_acc[{k, j}]);
      }
      A_vals_acc[i_pos] = sum;
    }
  }
}

void BoundsFromPartitionedCoordinates::omp_variant(legate::TaskContext& ctx)
{
  auto& input  = ctx.inputs()[0];
  auto& output = ctx.outputs()[0];
  assert(output.is_future());

  auto input_acc  = input.read_accessor<coord_ty, 1>();
  auto output_acc = output.write_accessor<Domain, 1>();
  auto dom        = input.domain();
  if (dom.empty()) {
    output_acc[0] = {0, -1};
  } else {
    auto ptr      = input_acc.ptr(dom.lo());
    auto result   = thrust::minmax_element(thrust::omp::par, ptr, ptr + dom.get_volume());
    output_acc[0] = {*result.first, *result.second};
  }
}

void SortedCoordsToCounts::omp_variant(legate::TaskContext& ctx)
{
  auto& output  = ctx.reductions()[0];
  auto& input   = ctx.inputs()[0];
  auto max_vals = ctx.scalars()[0].value<int64_t>();
  auto dom      = input.domain();
  if (output.domain().empty() || dom.empty()) return;
  auto in  = input.read_accessor<coord_ty, 1>();
  auto out = output.reduce_accessor<SumReduction<uint64_t>, true /* exclusive */, 1>();

  // TODO (rohany): We could make a call to unique before this to avoid allocating
  //  O(rows/cols) space. For now this shouldn't be too much of a problem. Unfortunately
  //  unique counting has just recently landed, and I'm not sure when our thrust version
  //  will pick up the change: https://github.com/NVIDIA/thrust/issues/1612.
  auto kind = Sparse::has_numamem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM;
  DeferredBuffer<coord_t, 1> keys({0, max_vals - 1}, kind);
  DeferredBuffer<uint64_t, 1> counts({0, max_vals - 1}, kind);
  auto result = thrust::reduce_by_key(thrust::omp::par,
                                      in.ptr(dom.lo()),
                                      in.ptr(dom.lo()) + dom.get_volume(),
                                      thrust::make_constant_iterator(1),
                                      keys.ptr(0),
                                      counts.ptr(0));
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < (result.first - keys.ptr(0)); i++) { out[keys[i]] <<= counts[i]; }
}

void ExpandPosToCoordinates::omp_variant(legate::TaskContext& ctx)
{
  auto& pos    = ctx.inputs()[0];
  auto& result = ctx.outputs()[0];
  ExpandPosToCoordinates::expand_pos_impl(
    thrust::omp::par,
    pos.read_accessor<Rect<1>, 1>(),
    pos.domain(),
    result.write_accessor<coord_ty, 1>(),
    result.domain(),
    Sparse::has_numamem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM);
}

// ElemwiseMultCSRCSRNNZ and ElemwiseMultCSRCSR are adapted from DISTAL generated code.
// A(i, j) = B(i, j) + C(i, j)
// Schedule:
// assemble(A, AssembleStrategy::Insert)
// distribute(i, io, ii, pieces)
// parallelize(ii, CPUThread, NoRaces)
// TODO (rohany): These kernels could be templated over a functor
//  to perform different binary operations that have the intersect iteration
//  pattern (i.e. different semirings).
void ElemwiseMultCSRCSRNNZ::omp_variant(legate::TaskContext& ctx)
{
  auto& nnz   = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& C_pos = ctx.inputs()[2];
  auto& C_crd = ctx.inputs()[3];

  auto nnz_acc   = nnz.write_accessor<nnz_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();

  auto B_domain = B_pos.domain();
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    size_t num_nnz = 0;
    size_t jB      = B_pos_acc[i].lo;
    size_t pB2_end = B_pos_acc[i].hi + 1;
    size_t jC      = C_pos_acc[i].lo;
    size_t pC2_end = C_pos_acc[i].hi + 1;
    while (jB < pB2_end && jC < pC2_end) {
      size_t jB0 = B_crd_acc[jB];
      size_t jC0 = C_crd_acc[jC];
      coord_ty j = std::min(jB0, jC0);
      if (jB0 == j && jC0 == j) { num_nnz++; }
      jB += (size_t)(jB0 == j);
      jC += (size_t)(jC0 == j);
    }
    nnz_acc[i] = num_nnz;
  }
}

void ElemwiseMultCSRCSR::omp_variant(legate::TaskContext& ctx)
{
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos  = ctx.inputs()[3];
  auto& C_crd  = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  auto A_pos_acc  = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc  = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_pos_acc  = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc  = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  auto B_domain = B_pos.domain();
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    // Similarly to other codes, since we're doing the inserts for a particular
    // value of i at a time, we don't need to mutate the pos array.
    size_t nnz_pos = A_pos_acc[i].lo;
    size_t jB      = B_pos_acc[i].lo;
    size_t pB2_end = B_pos_acc[i].hi + 1;
    size_t jC      = C_pos_acc[i].lo;
    size_t pC2_end = C_pos_acc[i].hi + 1;
    while (jB < pB2_end && jC < pC2_end) {
      coord_ty jB0 = B_crd_acc[jB];
      coord_ty jC0 = C_crd_acc[jC];
      coord_ty j   = std::min(jB0, jC0);
      if (jB0 == j && jC0 == j) {
        A_crd_acc[nnz_pos]  = j;
        A_vals_acc[nnz_pos] = B_vals_acc[jB] * C_vals_acc[jC];
        nnz_pos++;
      }
      jB += (size_t)(jB0 == j);
      jC += (size_t)(jC0 == j);
    }
  }
}

void ElemwiseMultCSRDense::omp_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();

  // TODO (rohany) I'm starting with a row-based distribution here, though
  //  a non-zero based parallelization scheme even within the rows allocated
  //  to a particular processor could be used.
  auto B_domain = B_pos.domain();
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (coord_t i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    for (size_t jB = B_pos_acc[i].lo; jB < B_pos_acc[i].hi + 1; jB++) {
      coord_t j      = B_crd_acc[jB];
      A_vals_acc[jB] = B_vals_acc[jB] * C_vals_acc[{i, j}];
    }
  }
}

// CSRToDense was adapted from DISTAL generated code.
// A(i, j) = B(i, j)
// Schedule:
// distribute({i}, {io}, {ii}, pieces)
// parallelize(ii, CPUThread, NoRaces)
void CSRToDense::omp_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 2>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();

  // Initialize the output array.
  auto A_domain = A_vals.domain();
  auto A_lo     = A_domain.lo();
  auto A_hi     = A_domain.hi();
#pragma omp parallel for schedule(static) collapse(2)
  for (coord_ty i = A_lo[0]; i < A_hi[0] + 1; i++) {
    for (coord_ty j = A_lo[1]; j < A_hi[1] + 1; j++) { A_vals_acc[{i, j}] = 0.0; }
  }

  auto B_domain = B_pos.domain();
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    for (size_t jB = B_pos_acc[i].lo; jB < B_pos_acc[i].hi + 1; jB++) {
      auto j             = B_crd_acc[jB];
      A_vals_acc[{i, j}] = B_vals_acc[jB];
    }
  }
}

// CSCToDense was adapted from DISTAL generated code.
// A(i, j) = B(i, j)
// Schedule:
// distribute({j}, {jo}, {ji}, pieces)
// parallelize(ji, CPUThread, NoRaces)
void CSCToDense::omp_variant(legate::TaskContext& ctx)
{
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos  = ctx.inputs()[0];
  auto& B_crd  = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) { B_pos.remove_transform(); }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 2>();
  auto B_pos_acc  = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc  = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();

  // Initialize the output array.
  auto A_domain = A_vals.domain();
  auto A_lo     = A_domain.lo();
  auto A_hi     = A_domain.hi();
#pragma omp parallel for schedule(static) collapse(2)
  for (coord_ty i = A_lo[0]; i < A_hi[0] + 1; i++) {
    for (coord_ty j = A_lo[1]; j < A_hi[1] + 1; j++) { A_vals_acc[{i, j}] = 0.0; }
  }

  auto B_domain = B_pos.domain();
#pragma omp parallel for schedule(monotonic : dynamic, 128)
  for (coord_ty j = B_domain.lo()[0]; j < B_domain.hi()[0] + 1; j++) {
    for (size_t iB = B_pos_acc[j].lo; iB < B_pos_acc[j].hi + 1; iB++) {
      auto i             = B_crd_acc[iB];
      A_vals_acc[{i, j}] = B_vals_acc[iB];
    }
  }
}

void DenseToCSRNNZ::omp_variant(legate::TaskContext& ctx)
{
  auto& nnz    = ctx.outputs()[0];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the nnz region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (nnz.transformed()) { nnz.remove_transform(); }

  auto nnz_acc    = nnz.write_accessor<nnz_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();

  auto dom = B_vals.domain();
  auto lo  = dom.lo();
  auto hi  = dom.hi();
#pragma omp parallel for schedule(static)
  for (coord_ty i = lo[0]; i < hi[0] + 1; i++) {
    nnz_ty row_nnz = 0;
    for (coord_ty j = lo[1]; j < hi[1] + 1; j++) {
      if (B_vals_acc[{i, j}] != 0.0) { row_nnz++; }
    }
    nnz_acc[i] = row_nnz;
  }
}

void DenseToCSR::omp_variant(legate::TaskContext& ctx)
{
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (A_pos.transformed()) { A_pos.remove_transform(); }

  auto A_pos_acc  = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc  = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();

  auto dom = B_vals.domain();
  auto lo  = dom.lo();
  auto hi  = dom.hi();
#pragma omp parallel for schedule(static)
  for (coord_ty i = lo[0]; i < hi[0] + 1; i++) {
    size_t nnz_pos = A_pos_acc[i].lo;
    for (coord_ty j = lo[1]; j < hi[1] + 1; j++) {
      if (B_vals_acc[{i, j}] != 0.0) {
        A_crd_acc[nnz_pos]  = j;
        A_vals_acc[nnz_pos] = B_vals_acc[{i, j}];
        nnz_pos++;
      }
    }
  }
}

void DenseToCSCNNZ::omp_variant(legate::TaskContext& ctx)
{
  auto& nnz    = ctx.outputs()[0];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the nnz region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (nnz.transformed()) { nnz.remove_transform(); }

  auto nnz_acc    = nnz.write_accessor<nnz_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();

  auto dom = B_vals.domain();
  auto lo  = dom.lo();
  auto hi  = dom.hi();
#pragma omp parallel for schedule(static)
  for (coord_ty j = lo[1]; j < hi[1] + 1; j++) {
    uint64_t col_nnz = 0;
    for (coord_ty i = lo[0]; i < hi[0] + 1; i++) {
      if (B_vals_acc[{i, j}] != 0.0) { col_nnz++; }
    }
    nnz_acc[j] = col_nnz;
  }
}

void DenseToCSC::omp_variant(legate::TaskContext& ctx)
{
  auto& A_pos  = ctx.outputs()[0];
  auto& A_crd  = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (A_pos.transformed()) { A_pos.remove_transform(); }

  auto A_pos_acc  = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc  = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();

  auto dom = B_vals.domain();
  auto lo  = dom.lo();
  auto hi  = dom.hi();
#pragma omp parallel for schedule(static)
  for (coord_ty j = lo[1]; j < hi[1] + 1; j++) {
    int nnz_pos = A_pos_acc[j].lo;
    for (coord_ty i = lo[0]; i < hi[0] + 1; i++) {
      if (B_vals_acc[{i, j}] != 0.0) {
        A_crd_acc[nnz_pos]  = i;
        A_vals_acc[nnz_pos] = B_vals_acc[{i, j}];
        nnz_pos++;
      }
    }
  }
}

void ZipToRect1::omp_variant(legate::TaskContext& ctx)
{
  auto& output = ctx.outputs()[0];
  if (output.domain().empty()) return;
  auto output_acc = output.write_accessor<Rect<1>, 1>();
  auto lo         = ctx.inputs()[0].read_accessor<uint64_t, 1>();
  auto hi         = ctx.inputs()[1].read_accessor<uint64_t, 1>();
  auto bounds     = output.domain();
#pragma omp parallel for schedule(static)
  for (int64_t i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
    output_acc[i] = {lo[i], hi[i] - 1};
  }
}

void UnZipRect1::omp_variant(legate::TaskContext& ctx)
{
  auto bounds = ctx.inputs()[0].domain();
  if (bounds.empty()) return;
  if (ctx.outputs()[0].dim() == 1) {
    auto out1 = ctx.outputs()[0].write_accessor<int64_t, 1>();
    auto out2 = ctx.outputs()[1].write_accessor<int64_t, 1>();
    auto in   = ctx.inputs()[0].read_accessor<Rect<1>, 1>();
#pragma omp parallel for schedule(static)
    for (int64_t i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
      auto rect = in[i];
      out1[i]   = rect.lo;
      out2[i]   = rect.hi;
    }
  } else if (ctx.outputs()[0].dim() == 2) {
    auto out1 = ctx.outputs()[0].write_accessor<int64_t, 2>();
    auto out2 = ctx.outputs()[1].write_accessor<int64_t, 2>();
    auto in   = ctx.inputs()[0].read_accessor<Rect<1>, 2>();
#pragma omp parallel for schedule(static) collapse(2)
    for (int64_t i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
      for (int64_t j = bounds.lo()[1]; j < bounds.hi()[1] + 1; j++) {
        auto rect    = in[{i, j}];
        out1[{i, j}] = rect.lo;
        out2[{i, j}] = rect.hi;
      }
    }
  } else {
    assert(ctx.outputs()[0].dim() == 3);
    auto out1 = ctx.outputs()[0].write_accessor<int64_t, 3>();
    auto out2 = ctx.outputs()[1].write_accessor<int64_t, 3>();
    auto in   = ctx.inputs()[0].read_accessor<Rect<1>, 3>();
#pragma omp parallel for schedule(static) collapse(3)
    for (int64_t i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
      for (int64_t j = bounds.lo()[1]; j < bounds.hi()[1] + 1; j++) {
        for (int64_t k = bounds.lo()[2]; k < bounds.hi()[2] + 1; k++) {
          auto rect       = in[{i, j, k}];
          out1[{i, j, k}] = rect.lo;
          out2[{i, j, k}] = rect.hi;
        }
      }
    }
  }
}

void ScaleRect1::omp_variant(legate::TaskContext& ctx)
{
  auto bounds = ctx.outputs()[0].domain();
  if (bounds.empty()) return;
  auto out   = ctx.outputs()[0].read_write_accessor<Rect<1>, 1>();
  auto task  = ctx.task_;
  auto scale = task->futures[0].get_result<int64_t>();
#pragma omp parallel for schedule(static)
  for (int64_t i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
    out[i].lo = out[i].lo + scale;
    out[i].hi = out[i].hi + scale;
  }
}

void FastImageRange::omp_variant(legate::TaskContext& ctx)
{
  auto& input  = ctx.inputs()[0];
  auto& output = ctx.outputs()[0];
  if (input.transformed()) { input.remove_transform(); }
  auto in  = input.read_accessor<Rect<1>, 1>();
  auto out = output.write_accessor<Domain, 1>();
  auto dom = input.domain();
  if (dom.empty()) {
    out[0] = Rect<1>::make_empty();
  } else {
    out[0] = Rect<1>{in[dom.lo()].lo, in[dom.hi()].hi};
  }
}

void EuclideanCDist::omp_variant(legate::TaskContext& ctx)
{
  auto& out = ctx.outputs()[0];
  auto& XA  = ctx.inputs()[0];
  auto& XB  = ctx.inputs()[1];

  auto out_acc = out.write_accessor<val_ty, 2>();
  auto XA_acc  = XA.read_accessor<val_ty, 2>();
  auto XB_acc  = XB.read_accessor<val_ty, 2>();

  auto out_domain = out.domain();
  auto out_lo     = out_domain.lo();
  auto out_hi     = out_domain.hi();
  auto XA_domain  = XA.domain();
  auto XA_lo      = XA_domain.lo();
  auto XA_hi      = XA_domain.hi();
#pragma omp parallel for schedule(static) collapse(2)
  for (coord_ty i = out_lo[0]; i < out_hi[0] + 1; i++) {
    for (coord_ty j = out_lo[1]; j < out_hi[1] + 1; j++) {
      val_ty diff = 0.0;
      for (coord_ty k = XA_lo[1]; k < XA_hi[1] + 1; k++) {
        diff += pow((XA_acc[{i, k}] - XB_acc[{j, k}]), 2);
      }
      out_acc[{i, j}] = sqrt(diff);
    }
  }
}

void SortByKey::omp_variant(legate::TaskContext& ctx)
{
  auto kind = Sparse::has_numamem ? Memory::SOCKET_MEM : Memory::SYSTEM_MEM;
  auto exec = thrust::omp::par;
  SortBody<coord_ty, val_ty, decltype(exec)>(ctx, kind, exec);
}

void VecMultAdd::omp_variant(legate::TaskContext& ctx)
{
  auto& lhs        = ctx.outputs()[0];
  auto& rhs        = ctx.inputs()[0];
  auto& beta_store = ctx.inputs()[1];
  bool left        = ctx.scalars()[0].value<bool>();
  auto dom         = lhs.domain();
  auto lhs_acc     = lhs.read_write_accessor<val_ty, 1>();
  auto rhs_acc     = rhs.read_accessor<val_ty, 1>();
  auto beta        = beta_store.read_accessor<val_ty, 1>()[0];

  if (left) {
#pragma omp parallel for schedule(static)
    for (coord_ty i = dom.lo()[0]; i < dom.hi()[0] + 1; i++) {
      lhs_acc[i] = (lhs_acc[i] * beta) + rhs_acc[i];
    }
  } else {
#pragma omp parallel for schedule(static)
    for (coord_ty i = dom.lo()[0]; i < dom.hi()[0] + 1; i++) {
      lhs_acc[i] = lhs_acc[i] + (beta * rhs_acc[i]);
    }
  }
}

}  // namespace sparse

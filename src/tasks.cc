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
#include "tasks.h"

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

using namespace Legion;

namespace sparse {

void CSRSpMVRowSplitTropicalSemiring::cpu_variant(legate::TaskContext& ctx)
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

// TODO (rohany): In a real implementation, we could template this
//  implementation to use a different operator or semiring.
// SpGEMM on CSR = CSR x CSR is adapted from DISTAL/TACO generated code.
// A(i, j) = B(i, k) * C(k, j)
// Schedule:
// assemble(A, AssembleStrategy::Insert)
// precompute(j, w), w is a workspace of size A2.
void SpGEMMCSRxCSRxCSRNNZ::cpu_variant(legate::TaskContext& ctx)
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
  // TODO (rohany): Look at cunumeric for a smarter buffer allocator function.
  coord_ty initValInt = 0;
  bool initValBool    = false;
  DeferredBuffer<coord_ty, 1> index_list(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initValInt);
  DeferredBuffer<bool, 1> already_set(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initValBool);
  // For this computation, we assume that the rows are partitioned.
  auto bounds = B_pos.domain();
  for (auto i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
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
    nnz_ty row_nnzs = 0;
    for (auto index_loc = 0; index_loc < index_list_size; index_loc++) {
      auto j         = index_list[index_loc];
      already_set[j] = false;
      row_nnzs++;
    }
    nnz_acc[i] = row_nnzs;
  }
}

void SpGEMMCSRxCSRxCSR::cpu_variant(legate::TaskContext& ctx)
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
  // TODO (rohany): Look at cunumeric for a smarter buffer allocator function.
  coord_ty initValInt  = 0;
  bool initValBool     = false;
  val_ty initValDouble = 0.0;
  DeferredBuffer<coord_ty, 1> index_list(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initValInt);
  DeferredBuffer<bool, 1> already_set(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initValBool);
  DeferredBuffer<val_ty, 1> workspace(Memory::SYSTEM_MEM, Rect<1>{0, A2_dim - 1}, &initValDouble);
  // For this computation, we assume that the rows are partitioned.
  auto bounds = B_pos.domain();
  for (auto i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
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
    // TODO (rohany): I don't think that we need to mutate the pos array
    //  here since this loop iteration is already scoped to a particular
    //  row. Instead, we can just extract the position into a register.
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

// An old version of CSRxCSRxCSC SpGEMM.
// SpGEMM on CSR = CSR x CSR is adapted from DISTAL/TACO generated code.
// A(i, j) = B(i, k) * C(k, j)
// Schedule:
// assemble(A, AssembleStrategy::Insert)
// void SpGEMMCSRxCSRxCSCNNZ::cpu_variant(legate::TaskContext &ctx) {
//   auto A2_dim = ctx.scalars()[0].value<size_t>();
//   auto& nnz = ctx.outputs()[0];
//   auto& B_pos = ctx.inputs()[0];
//   auto& B_crd = ctx.inputs()[1];
//   auto& C_pos = ctx.inputs()[2];
//   auto& C_crd = ctx.inputs()[3];
//
//   auto nnz_acc = nnz.write_accessor<uint64_t, 1>();
//   auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
//   auto B_crd_acc = B_crd.read_accessor<int64_t, 1>();
//   auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
//   auto C_crd_acc = C_crd.read_accessor<int64_t, 1>();
//
//   // In this parallelization, the rows of B are partitioned.
//   // TODO (rohany): We have to do something much smarter (that likely involves manual
//   communication)
//   //  to handle partitioning of C, due to also causing parallel construction of the sparse output
//   dimension.
//   //  DISTAL can handle parallelization and distribution of dense output dimensions that have
//   sparse
//   //  children, but not distribution of sparse output dimensions.
//   auto bounds = B_pos.domain();
//   for (auto i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
//     size_t row_nnzs = 0;
//     for (auto j = 0; j < A2_dim; j++) {
//       size_t kB_pos = B_pos_acc[i].lo;
//       size_t kB_pos_end = B_pos_acc[i].hi + 1;
//       size_t kC_pos = C_pos_acc[j].lo;
//       size_t kC_pos_end = C_pos_acc[j].hi + 1;
//       while (kB_pos < kB_pos_end && kC_pos < kC_pos_end) {
//         auto kB = B_crd_acc[kB_pos];
//         auto kC = C_crd_acc[kC_pos];
//         auto k = std::min(kB, kC);
//         if (k == kB && k == kC) {
//           row_nnzs++;
//           break;
//         }
//         kB_pos += (int64_t)(kB == k);
//         kC_pos += (int64_t)(kC == k);
//       }
//     }
//     nnz_acc[i] = row_nnzs;
//   }
// }
// void SpGEMMCSRxCSRxCSC::cpu_variant(legate::TaskContext &ctx) {
//   auto A2_dim = ctx.scalars()[0].value<size_t>();
//   auto& A_pos = ctx.outputs()[0];
//   auto& A_crd = ctx.outputs()[1];
//   auto& A_vals = ctx.outputs()[2];
//
//   auto& B_pos = ctx.inputs()[0];
//   auto& B_crd = ctx.inputs()[1];
//   auto& B_vals = ctx.inputs()[2];
//   auto& C_pos = ctx.inputs()[3];
//   auto& C_crd = ctx.inputs()[4];
//   auto& C_vals = ctx.inputs()[5];
//
//   auto A_pos_acc = A_pos.read_write_accessor<Rect<1>, 1>();
//   auto A_crd_acc = A_crd.write_accessor<int64_t, 1>();
//   auto A_vals_acc = A_vals.write_accessor<double, 1>();
//   auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
//   auto B_crd_acc = B_crd.read_accessor<int64_t, 1>();
//   auto B_vals_acc = B_vals.read_accessor<double, 1>();
//   auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
//   auto C_crd_acc = C_crd.read_accessor<int64_t, 1>();
//   auto C_vals_acc = C_vals.read_accessor<double, 1>();
//
//   for (auto i = B_pos.domain().lo()[0]; i < B_pos.domain().hi()[0] + 1; i++) {
//     size_t nnz_pos = A_pos_acc[i].lo;
//     for (auto j = 0; j < A2_dim; j++) {
//       double result = 0.0;
//       bool set = false;
//       size_t kB_pos = B_pos_acc[i].lo;
//       size_t kB_pos_end = B_pos_acc[i].hi + 1;
//       size_t kC_pos = C_pos_acc[j].lo;
//       size_t kC_pos_end = C_pos_acc[j].hi + 1;
//       while (kB_pos < kB_pos_end && kC_pos < kC_pos_end) {
//         auto kB = B_crd_acc[kB_pos];
//         auto kC = C_crd_acc[kC_pos];
//         auto k = std::min(kB, kC);
//         if (k == kB && k == kC) {
//           set = true;
//           result += B_vals_acc[kB_pos] * C_vals_acc[kC_pos];
//         }
//         kB_pos += (int64_t)(kB == k);
//         kC_pos += (int64_t)(kC == k);
//       }
//       // Since we're considering a row at a time and not scattering
//       // across pos, we don't need to mutate pos like as done in DISTAL.
//       if (set) {
//         A_crd_acc[nnz_pos] = j;
//         A_vals_acc[nnz_pos] = result;
//         nnz_pos++;
//       }
//     }
//   }
// }

void SpGEMMCSRxCSRxCSCLocalTiles::cpu_variant(legate::TaskContext& ctx)
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
  auto C_lo = C_domain.lo();
  auto C_hi = C_domain.hi();

  // Our job right now is to perform both passes of the SpGEMM operation
  // and output instances for local CSR matrices of each result.
  DeferredBuffer<size_t, 1> nnz({B_pos.domain().lo()[0], B_pos.domain().hi()[0]},
                                Memory::SYSTEM_MEM);
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

  auto A_pos_acc =
    A_pos.create_output_buffer<Rect<1>, 1>(B_pos.domain().get_volume(), true /* return_buffer */);
  size_t val = 0;
  auto B_lo  = B_domain.lo()[0];
  for (size_t i = B_lo; i < B_domain.hi()[0] + 1; i++) {
    size_t newVal       = nnz[i] + val;
    A_pos_acc[i - B_lo] = {val, newVal - 1};
    val                 = newVal;
  }
  auto A_crd_acc  = A_crd.create_output_buffer<coord_ty, 1>(val, true /* return_buffer */);
  auto A_vals_acc = A_vals.create_output_buffer<val_ty, 1>(val, true /* return_buffer */);

  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    // Important: We need to offset the access into the output buffers, as they
    // are not globally indexed objects.
    size_t nnz_pos = A_pos_acc[i - B_lo].lo;
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

void SpGEMMCSRxCSRxCSCCommCompute::cpu_variant(legate::TaskContext& ctx)
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
  for (int32_t j = 0; j < gy; j++) {
    auto sub_tile_start = j * tile_size;
    auto sub_tile_end   = std::min((j + 1) * tile_size, i_tile_size);
    auto lo             = global_pos_domain.lo()[0] + sub_tile_start;
    auto hi             = global_pos_domain.lo()[0] + sub_tile_end;
    out[{oi, oj, j}]    = Rect<1>{lo, hi - 1};
  }
}

void SpGEMMCSRxCSRxCSCShuffle::cpu_variant(legate::TaskContext& ctx)
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

  size_t total_nnzs = 0;
  // TODO (rohany): I think we can assume that there will be `j` subrects.
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

  size_t offset = 0;
  for (size_t i = 0; i < total_rows; i++) {
    auto lo = offset;
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
void SpMMCSR::cpu_variant(legate::TaskContext& ctx)
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
  // Pull these out so that we don't allocate them during the loop.
  auto C_lo = C_domain.lo();
  auto C_hi = C_domain.hi();

  // First zero out the output array.
  for (coord_ty i = A_domain.lo()[0]; i < A_domain.hi()[0] + 1; i++) {
    for (coord_ty j = C_lo[1]; j < C_hi[1] + 1; j++) { A_vals_acc[{i, j}] = 0.0; }
  }

  // Next, do the computation.
  for (coord_ty i = A_domain.lo()[0]; i < A_domain.hi()[0] + 1; i++) {
    for (size_t kB = B_pos_acc[i].lo; kB < B_pos_acc[i].hi + 1; kB++) {
      auto k = B_crd_acc[kB];
      for (coord_ty j = C_lo[1]; j < C_hi[1] + 1; j++) {
        A_vals_acc[{i, j}] += B_vals_acc[kB] * C_vals_acc[{k, j}];
      }
    }
  }
}

void SpMMDenseCSR::cpu_variant(legate::TaskContext& ctx)
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
  auto A_lo     = A_domain.lo();
  auto A_hi     = A_domain.hi();
  auto C_domain = C_pos.domain();
  // TODO (rohany): We can explore different ways of scheduling
  //  this for better performance. Right now, it's not accessing
  //  either of the dense arrays along their strides.
  for (coord_ty k = C_domain.lo()[0]; k < C_domain.hi()[0] + 1; k++) {
    for (size_t jB = C_pos_acc[k].lo; jB < C_pos_acc[k].hi + 1; jB++) {
      coord_ty j = C_crd_acc[jB];
      for (coord_ty i = A_lo[0]; i < A_hi[0] + 1; i++) {
        A_vals_acc[{i, j}] <<= B_vals_acc[{i, k}] * C_vals_acc[jB];
      }
    }
  }
}

// ElemwiseMultCSRCSRNNZ and ElemwiseMultCSRCSR are adapted from DISTAL generated code.
// A(i, j) = B(i, j) + C(i, j)
// Schedule:
// assemble(A, AssembleStrategy::Insert)
// distribute(i, io, ii, pieces)
// TODO (rohany): These kernels could be templated over a functor
//  to perform different binary operations that have the intersect iteration
//  pattern (i.e. different semirings).
void ElemwiseMultCSRCSRNNZ::cpu_variant(legate::TaskContext& ctx)
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
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    size_t num_nnz = 0;
    size_t jB      = B_pos_acc[i].lo;
    size_t pB2_end = B_pos_acc[i].hi + 1;
    size_t jC      = C_pos_acc[i].lo;
    size_t pC2_end = C_pos_acc[i].hi + 1;
    while (jB < pB2_end && jC < pC2_end) {
      coord_ty jB0 = B_crd_acc[jB];
      coord_ty jC0 = C_crd_acc[jC];
      coord_ty j   = std::min(jB0, jC0);
      if (jB0 == j && jC0 == j) { num_nnz++; }
      jB += (size_t)(jB0 == j);
      jC += (size_t)(jC0 == j);
    }
    nnz_acc[i] = num_nnz;
  }
}

void ElemwiseMultCSRCSR::cpu_variant(legate::TaskContext& ctx)
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
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    // Similarly to other codes, since we're doing the inserts for a particular
    // value of i at a time, we don't need to mutate the pos array.
    size_t nnz_pos = A_pos_acc[i].lo;
    size_t jB      = B_pos_acc[i].lo;
    size_t pB2_end = B_pos_acc[i].hi + 1;
    size_t jC      = C_pos_acc[i].lo;
    size_t pC2_end = C_pos_acc[i].hi + 1;
    while (jB < pB2_end && jC < pC2_end) {
      size_t jB0 = B_crd_acc[jB];
      size_t jC0 = C_crd_acc[jC];
      coord_ty j = std::min(jB0, jC0);
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

void CSRSDDMM::cpu_variant(legate::TaskContext& ctx)
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

  auto B_domain = B_pos.domain();
  auto C_domain = C_vals.domain();
  auto C_lo     = C_domain.lo();
  auto C_hi     = C_domain.hi();

  for (coord_t i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    for (size_t jB = B_pos_acc[i].lo; jB < B_pos_acc[i].hi + 1; jB++) {
      coord_t j      = B_crd_acc[jB];
      A_vals_acc[jB] = 0.0;
      for (coord_t k = C_lo[1]; k < C_hi[1] + 1; k++) {
        A_vals_acc[jB] += B_vals_acc[jB] * (C_vals_acc[{i, k}] * D_vals_acc[{k, j}]);
      }
    }
  }
}

void CSCSDDMM::cpu_variant(legate::TaskContext& ctx)
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

  auto B_domain = B_pos.domain();
  auto C_domain = C_vals.domain();
  auto C_lo     = C_domain.lo();
  auto C_hi     = C_domain.hi();

  for (coord_t j = B_domain.lo()[0]; j < B_domain.hi()[0] + 1; j++) {
    for (size_t iB = B_pos_acc[j].lo; iB < B_pos_acc[j].hi + 1; iB++) {
      coord_t i      = B_crd_acc[iB];
      A_vals_acc[iB] = 0.0;
      for (coord_t k = C_lo[1]; k < C_hi[1] + 1; k++) {
        A_vals_acc[iB] += B_vals_acc[iB] * (C_vals_acc[{i, k}] * D_vals_acc[{k, j}]);
      }
    }
  }
}

}  // namespace sparse

namespace {  // anonymous
static void __attribute__((constructor)) register_tasks(void)
{
  sparse::CSRSpMVRowSplitTropicalSemiring::register_variants();

  sparse::SpGEMMCSRxCSRxCSRNNZ::register_variants();
  sparse::SpGEMMCSRxCSRxCSR::register_variants();
  sparse::SpGEMMCSRxCSRxCSRGPU::register_variants();

  sparse::SpGEMMCSRxCSRxCSCLocalTiles::register_variants();
  sparse::SpGEMMCSRxCSRxCSCCommCompute::register_variants();
  sparse::SpGEMMCSRxCSRxCSCShuffle::register_variants();

  sparse::SpMMCSR::register_variants();
  sparse::SpMMDenseCSR::register_variants();

  sparse::ElemwiseMultCSRCSRNNZ::register_variants();
  sparse::ElemwiseMultCSRCSR::register_variants();
  sparse::CSRSDDMM::register_variants();
  sparse::CSCSDDMM::register_variants();
}
}  // namespace

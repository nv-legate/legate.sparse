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
#include "sort_template.inl"

#include <fstream>

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

using namespace Legion;

namespace sparse {

// CSRSpMVRowSplit and its corresponding partitioning steps are generated
// from the DISTAL program:
// y(i) = A(i, j) * x(j)
// Schedule:
// distribute(i, io, ii, pieces)
// communicate({y, A, x}, io)
void CSRSpMVRowSplit::cpu_variant(legate::TaskContext &ctx) {
  // Get the pos, crd and vals regions.
  auto& y = ctx.outputs()[0];
  auto& pos = ctx.inputs()[0];
  auto& crd = ctx.inputs()[1];
  auto& vals = ctx.inputs()[2];
  auto& x = ctx.inputs()[3];

  // std::cout << "SpMV task: " << std::endl;
  // std::cout << y.domain() << std::endl;
  // std::cout << pos.domain() << std::endl;
  // std::cout << crd.domain() << std::endl;
  // std::cout << vals.domain() << std::endl;
  // std::cout << x.domain() << std::endl;

  // TODO (rohany): Template this over the value types of the arrays.
  auto yacc = y.write_accessor<val_ty, 1>();
  auto posacc = pos.read_accessor<Rect<1>, 1>();
  // TODO (rohany): Expose control over 32 or 64 bit indices.
  auto crdacc = crd.read_accessor<coord_ty, 1>();
  auto valsacc = vals.read_accessor<val_ty, 1>();
  auto xacc = x.read_accessor<val_ty, 1>();

  // TODO (rohany): We can partition the x vector here for an algorithm that
  //  doesn't replicate x, or even use a 2-D distribution of the CSR matrix.
  //  However, a 2-D distribution requires a different encoding.
  auto bounds = y.domain();
  for (size_t i = bounds.lo()[0]; i <= bounds.hi()[0]; i++) {
    // We importantly need to discard whatever data already lives in the instances.
    val_ty sum = 0.0;
    for (size_t jpos = posacc[i].lo; jpos <= posacc[i].hi; jpos++) {
      auto j = crdacc[jpos];
      sum += valsacc[jpos] * xacc[j];
    }
    yacc[i] = sum;
  }
}

void CSRSpMVRowSplitTropicalSemiring::cpu_variant(legate::TaskContext &ctx) {
  auto& y = ctx.outputs()[0];
  auto& pos = ctx.inputs()[0];
  auto& crd = ctx.inputs()[1];
  auto& x = ctx.inputs()[2];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (pos.transformed()) {
    pos.remove_transform();
  }

  auto yacc = y.write_accessor<coord_ty, 2>();
  auto posacc = pos.read_accessor<Rect<1>, 1>();
  auto crdacc = crd.read_accessor<coord_ty, 1>();
  auto xacc = x.read_accessor<coord_ty, 2>();
  auto bounds = y.domain();

  auto num_fields = x.domain().hi()[1] - x.domain().lo()[1] + 1;
  for (coord_ty i = bounds.lo()[0]; i < bounds.hi()[0] + 1; i++) {
    // Initialize the output.
    for (coord_ty f = 0; f < num_fields; f++) {
      yacc[{i, f}] = 0;
    }
    for (size_t jpos = posacc[i].lo; jpos < posacc[i].hi + 1; jpos++) {
      auto j = crdacc[jpos];
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
        for (coord_ty f = 0; f < num_fields; f++) {
          yacc[{i, f}] = xacc[{j, f}];
        }
      }
    }
  }
}

void CSCSpMVColSplit::cpu_variant(legate::TaskContext &ctx) {
  auto& y = ctx.reductions()[0];
  auto& A_pos = ctx.inputs()[0];
  auto& A_crd = ctx.inputs()[1];
  auto& A_vals = ctx.inputs()[2];
  auto& x = ctx.inputs()[3];

  auto y_acc = y.reduce_accessor<SumReduction<val_ty>, true /* exclusive */, 1>();
  auto A_pos_acc = A_pos.read_accessor<Rect<1>, 1>();
  auto A_crd_acc = A_crd.read_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.read_accessor<val_ty, 1>();
  auto x_acc = x.read_accessor<val_ty, 1>();
  auto bounds = A_pos.domain();
  for (coord_ty j = bounds.lo()[0]; j < bounds.hi()[0] + 1; j++) {
    for (size_t iA = A_pos_acc[j].lo; iA < A_pos_acc[j].hi + 1; iA++) {
      auto i = A_crd_acc[iA];
      y_acc[i] <<= A_vals_acc[iA] * x_acc[j];
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
void SpGEMMCSRxCSRxCSRNNZ::cpu_variant(legate::TaskContext &ctx) {
  auto A2_dim = ctx.scalars()[0].value<size_t>();
  auto& nnz = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& C_pos = ctx.inputs()[2];
  auto& C_crd = ctx.inputs()[3];

  auto nnz_acc = nnz.write_accessor<nnz_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();

  // Allocate sparse accelerator data.
  // TODO (rohany): Look at cunumeric for a smarter buffer allocator function.
  coord_ty initValInt = 0;
  bool initValBool = false;
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
          already_set[j] = true;
          index_list_size++;
        }
      }
    }
    nnz_ty row_nnzs = 0;
    for (auto index_loc = 0; index_loc < index_list_size; index_loc++) {
      auto j = index_list[index_loc];
      already_set[j] = false;
      row_nnzs++;
    }
    nnz_acc[i] = row_nnzs;
  }
}

void SpGEMMCSRxCSRxCSR::cpu_variant(legate::TaskContext &ctx) {
  auto A2_dim = ctx.scalars()[0].value<size_t>();
  auto& A_pos = ctx.outputs()[0];
  auto& A_crd = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];

  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos = ctx.inputs()[3];
  auto& C_crd = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  auto A_pos_acc = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  // Allocate sparse accelerator data.
  // TODO (rohany): Look at cunumeric for a smarter buffer allocator function.
  coord_ty initValInt = 0;
  bool initValBool = false;
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
          already_set[j] = true;
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
      auto j = index_list[index_loc];
      already_set[j] = false;
      A_crd_acc[pA2] = j;
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
//   // TODO (rohany): We have to do something much smarter (that likely involves manual communication)
//   //  to handle partitioning of C, due to also causing parallel construction of the sparse output dimension.
//   //  DISTAL can handle parallelization and distribution of dense output dimensions that have sparse
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

void SpGEMMCSRxCSRxCSCLocalTiles::cpu_variant(legate::TaskContext &ctx) {
  auto& A_pos = ctx.outputs()[0];
  auto& A_crd = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];

  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos = ctx.inputs()[3];
  auto& C_crd = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  // Remove the transformations on the B_pos and C_pos stores.
  B_pos.remove_transform();
  C_pos.remove_transform();

  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  auto B_domain = B_pos.domain();
  auto C_domain = C_pos.domain();
  // Pull these out so that we don't allocate within the loop.
  auto C_lo = C_domain.lo();
  auto C_hi = C_domain.hi();

  // Our job right now is to perform both passes of the SpGEMM operation
  // and output instances for local CSR matrices of each result.
  DeferredBuffer<size_t, 1> nnz({B_pos.domain().lo()[0], B_pos.domain().hi()[0]}, Memory::SYSTEM_MEM);
  for (auto i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    size_t row_nnzs = 0;
    for (auto j = C_lo[0]; j < C_hi[0] + 1; j++) {
      size_t kB_pos = B_pos_acc[i].lo;
      size_t kB_pos_end = B_pos_acc[i].hi + 1;
      size_t kC_pos = C_pos_acc[j].lo;
      size_t kC_pos_end = C_pos_acc[j].hi + 1;
      while (kB_pos < kB_pos_end && kC_pos < kC_pos_end) {
        auto kB = B_crd_acc[kB_pos];
        auto kC = C_crd_acc[kC_pos];
        auto k = std::min(kB, kC);
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

  auto A_pos_acc = A_pos.create_output_buffer<Rect<1>, 1>(B_pos.domain().get_volume(), true /* return_buffer */);
  size_t val = 0;
  auto B_lo = B_domain.lo()[0];
  for (size_t i = B_lo; i < B_domain.hi()[0] + 1; i++) {
    size_t newVal = nnz[i] + val;
    A_pos_acc[i - B_lo] = {val, newVal - 1};
    val = newVal;
  }
  auto A_crd_acc = A_crd.create_output_buffer<coord_ty, 1>(val, true /* return_buffer */);
  auto A_vals_acc = A_vals.create_output_buffer<val_ty, 1>(val, true /* return_buffer */);

  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    // Important: We need to offset the access into the output buffers, as they
    // are not globally indexed objects.
    size_t nnz_pos = A_pos_acc[i - B_lo].lo;
    for (coord_ty j = C_lo[0]; j < C_hi[0] + 1; j++) {
      double result = 0.0;
      bool set = false;
      size_t kB_pos = B_pos_acc[i].lo;
      size_t kB_pos_end = B_pos_acc[i].hi + 1;
      size_t kC_pos = C_pos_acc[j].lo;
      size_t kC_pos_end = C_pos_acc[j].hi + 1;
      while (kB_pos < kB_pos_end && kC_pos < kC_pos_end) {
        auto kB = B_crd_acc[kB_pos];
        auto kC = C_crd_acc[kC_pos];
        auto k = std::min(kB, kC);
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
        A_crd_acc[nnz_pos] = j;
        A_vals_acc[nnz_pos] = result;
        nnz_pos++;
      }
    }
  }
}

void SpGEMMCSRxCSRxCSCCommCompute::cpu_variant(legate::TaskContext &ctx) {
  auto out = ctx.outputs()[0].write_accessor<Rect<1>, 3>();
  auto& pos = ctx.inputs()[0];
  auto& global_pos = ctx.inputs()[1];

  auto gx = ctx.scalars()[0].value<int32_t>();
  auto gy = ctx.scalars()[1].value<int32_t>();

  auto output_point = ctx.outputs()[0].domain().lo();
  auto oi = output_point[0];
  auto oj = output_point[1];

  auto global_pos_domain = global_pos.domain();

  auto i_lo = pos.domain().lo()[0];
  auto i_hi = pos.domain().hi()[0] + 1;
  auto i_tile_size = i_hi - i_lo;
  auto tile_size = (i_hi - i_lo + gy - 1) / gy;
  for (int32_t j = 0; j < gy; j++) {
    auto sub_tile_start = j * tile_size;
    auto sub_tile_end = std::min((j + 1) * tile_size, i_tile_size);
    auto lo = global_pos_domain.lo()[0] + sub_tile_start;
    auto hi = global_pos_domain.lo()[0] + sub_tile_end;
    out[{oi, oj, j}] = Rect<1>{lo, hi - 1};
  }
}

void SpGEMMCSRxCSRxCSCShuffle::cpu_variant(legate::TaskContext &ctx) {
  auto& global_pos = ctx.inputs()[0];
  auto& global_crd = ctx.inputs()[1];
  auto& global_vals = ctx.inputs()[2];

  auto& out_pos = ctx.outputs()[0];
  auto& out_crd = ctx.outputs()[1];
  auto& out_vals = ctx.outputs()[2];

  // TODO (rohany): I want a sparse instance here.
  auto global_pos_acc = global_pos.read_accessor<Rect<1>, 1>();
  auto global_crd_acc = global_crd.read_accessor<coord_ty, 1>();
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
  for (auto it : rects) {
    total_rows = std::max(total_rows, it.volume());
  }

  auto pos_acc = out_pos.create_output_buffer<Rect<1>, 1>(total_rows, true /* return_buffer */);
  auto crd_acc = out_crd.create_output_buffer<coord_ty, 1>(total_nnzs, true /* return_buffer */);
  auto vals_acc = out_vals.create_output_buffer<val_ty, 1>(total_nnzs, true /* return_buffer */);

  size_t offset = 0;
  for (size_t i = 0; i < total_rows; i++) {
    auto lo = offset;
    for (auto rect : rects) {
      auto global_pos_idx = rect.lo + i;
      if (!rect.contains(global_pos_idx)) continue;
      for (int64_t pos = global_pos_acc[global_pos_idx].lo; pos < global_pos_acc[global_pos_idx].hi + 1; pos++) {
        crd_acc[offset] = global_crd_acc[pos];
        vals_acc[offset] = global_vals_acc[pos];
        offset++;
      }
    }
    auto hi = offset - 1;
    pos_acc[i] = {lo, hi};
  }
}

// SpMMCSR is adapted from DISTAL/TACO generated code where A and C are dense.
// A(i, j) = B(i, k) * C(k, j)
// Schedule:
// distribute({i, j}, {io, jo}, {ii, ji}, grid)
// communicate({A, B, C}, jo)
// reorder({io, jo, ii, k, ji})
void SpMMCSR::cpu_variant(legate::TaskContext &ctx) {
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) {
    B_pos.remove_transform();
  }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 2>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
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
    for (coord_ty j = C_lo[1]; j < C_hi[1] + 1; j++) {
      A_vals_acc[{i, j}] = 0.0;
    }
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

void SpMMDenseCSR::cpu_variant(legate::TaskContext &ctx) {
  auto& A_vals = ctx.reductions()[0];
  auto& B_vals = ctx.inputs()[0];
  auto& C_pos = ctx.inputs()[1];
  auto& C_crd = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (C_pos.transformed()) {
    C_pos.remove_transform();
  }

  auto A_vals_acc = A_vals.reduce_accessor<SumReduction<val_ty>, true /* exclusive */, 2>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty , 1>();

  auto A_domain = A_vals.domain();
  auto A_lo = A_domain.lo();
  auto A_hi = A_domain.hi();
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

// AddCSRCSRNNZ and AddCSRCSR are adapted from DISTAL generated code.
// A(i, j) = B(i, j) + C(i, j)
// Schedule:
// assemble(A, AssembleStrategy::Insert)
// distribute(i, io, ii, pieces)
// TODO (rohany): These kernels could be templated over a functor
//  to perform different binary operations that have the union iteration
//  pattern (i.e. different semirings).
void AddCSRCSRNNZ::cpu_variant(legate::TaskContext &ctx) {
  auto& nnz = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& C_pos = ctx.inputs()[2];
  auto& C_crd = ctx.inputs()[3];

  auto nnz_acc = nnz.write_accessor<nnz_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();

  auto B_domain = B_pos.domain();
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    size_t num_nnz = 0;
    size_t jB = B_pos_acc[i].lo;
    size_t pB2_end = B_pos_acc[i].hi + 1;
    size_t jC = C_pos_acc[i].lo;
    size_t pC2_end = C_pos_acc[i].hi + 1;
    while (jB < pB2_end && jC < pC2_end) {
      coord_ty jB0 = B_crd_acc[jB];
      coord_ty jC0 = C_crd_acc[jC];
      int64_t j = std::min(jB0, jC0);
      num_nnz++;
      jB += (size_t)(jB0 == j);
      jC += (size_t)(jC0 == j);
    }
    if (jB < pB2_end) {
      num_nnz += pB2_end - jB;
      jB = pB2_end;
    }
    if (jC < pC2_end) {
      num_nnz += pC2_end - jC;
      jC = pC2_end;
    }
    nnz_acc[i] = num_nnz;
  }
}

void AddCSRCSR::cpu_variant(legate::TaskContext &ctx) {
  auto& A_pos = ctx.outputs()[0];
  auto& A_crd = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos = ctx.inputs()[3];
  auto& C_crd = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  auto A_pos_acc = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  auto B_domain = B_pos.domain();
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    // Similarly to other codes, since we're doing the inserts for a particular
    // value of i at a time, we don't need to mutate the pos array.
    size_t nnz_pos = A_pos_acc[i].lo;
    size_t jB = B_pos_acc[i].lo;
    size_t pB2_end = B_pos_acc[i].hi + 1;
    size_t jC = C_pos_acc[i].lo;
    size_t pC2_end = C_pos_acc[i].hi + 1;
    while (jB < pB2_end && jC < pC2_end) {
      coord_t jB0 = B_crd_acc[jB];
      coord_t jC0 = C_crd_acc[jC];
      coord_ty j = std::min(jB0, jC0);
      if (jB0 == j && jC0 == j) {
        A_crd_acc[nnz_pos] = j;
        A_vals_acc[nnz_pos] = B_vals_acc[jB] + C_vals_acc[jC];
        nnz_pos++;
      } else if (jB0 == j) {
        A_crd_acc[nnz_pos] = j;
        A_vals_acc[nnz_pos] = B_vals_acc[jB];
        nnz_pos++;
      } else {
        A_crd_acc[nnz_pos] = j;
        A_vals_acc[nnz_pos] = C_vals_acc[jC];
        nnz_pos++;
      }
      jB += (size_t)(jB0 == j);
      jC += (size_t)(jC0 == j);
    }
    while (jB < pB2_end) {
      coord_ty j = B_crd_acc[jB];
      A_crd_acc[nnz_pos] = j;
      A_vals_acc[nnz_pos] = B_vals_acc[jB];
      nnz_pos++;
      jB++;
    }
    while (jC < pC2_end) {
      coord_ty j = C_crd_acc[jC];
      A_crd_acc[nnz_pos] = j;
      A_vals_acc[nnz_pos] = C_vals_acc[jC];
      nnz_pos++;
      jC++;
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
void ElemwiseMultCSRCSRNNZ::cpu_variant(legate::TaskContext &ctx) {
  auto& nnz = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& C_pos = ctx.inputs()[2];
  auto& C_crd = ctx.inputs()[3];

  auto nnz_acc = nnz.write_accessor<nnz_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();

  auto B_domain = B_pos.domain();
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    size_t num_nnz = 0;
    size_t jB = B_pos_acc[i].lo;
    size_t pB2_end = B_pos_acc[i].hi + 1;
    size_t jC = C_pos_acc[i].lo;
    size_t pC2_end = C_pos_acc[i].hi + 1;
    while (jB < pB2_end && jC < pC2_end) {
      coord_ty jB0 = B_crd_acc[jB];
      coord_ty jC0 = C_crd_acc[jC];
      coord_ty j = std::min(jB0, jC0);
      if (jB0 == j && jC0 == j) {
        num_nnz++;
      }
      jB += (size_t)(jB0 == j);
      jC += (size_t)(jC0 == j);
    }
    nnz_acc[i] = num_nnz;
  }
}

void ElemwiseMultCSRCSR::cpu_variant(legate::TaskContext &ctx) {
  auto& A_pos = ctx.outputs()[0];
  auto& A_crd = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_pos = ctx.inputs()[3];
  auto& C_crd = ctx.inputs()[4];
  auto& C_vals = ctx.inputs()[5];

  auto A_pos_acc = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_pos_acc = C_pos.read_accessor<Rect<1>, 1>();
  auto C_crd_acc = C_crd.read_accessor<coord_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 1>();

  auto B_domain = B_pos.domain();
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    // Similarly to other codes, since we're doing the inserts for a particular
    // value of i at a time, we don't need to mutate the pos array.
    size_t nnz_pos = A_pos_acc[i].lo;
    size_t jB = B_pos_acc[i].lo;
    size_t pB2_end = B_pos_acc[i].hi + 1;
    size_t jC = C_pos_acc[i].lo;
    size_t pC2_end = C_pos_acc[i].hi + 1;
    while (jB < pB2_end && jC < pC2_end) {
      size_t jB0 = B_crd_acc[jB];
      size_t jC0 = C_crd_acc[jC];
      coord_ty j = std::min(jB0, jC0);
      if (jB0 == j && jC0 == j) {
        A_crd_acc[nnz_pos] = j;
        A_vals_acc[nnz_pos] = B_vals_acc[jB] * C_vals_acc[jC];
        nnz_pos++;
      }
      jB += (size_t)(jB0 == j);
      jC += (size_t)(jC0 == j);
    }
  }
}

void ElemwiseMultCSRDense::cpu_variant(legate::TaskContext &ctx) {
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) {
    B_pos.remove_transform();
  }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();

  auto B_domain = B_pos.domain();
  for (coord_t i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    for (size_t jB = B_pos_acc[i].lo; jB < B_pos_acc[i].hi + 1; jB++) {
      coord_t j = B_crd_acc[jB];
      A_vals_acc[jB] = B_vals_acc[jB] * C_vals_acc[{i, j}];
    }
  }
}

void CSRSDDMM::cpu_variant(legate::TaskContext &ctx) {
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];
  auto& D_vals = ctx.inputs()[4];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) {
    B_pos.remove_transform();
  }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();
  auto D_vals_acc = D_vals.read_accessor<val_ty, 2>();

  auto B_domain = B_pos.domain();
  auto C_domain = C_vals.domain();
  auto C_lo = C_domain.lo();
  auto C_hi = C_domain.hi();

  for (coord_t i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    for (size_t jB = B_pos_acc[i].lo; jB < B_pos_acc[i].hi + 1; jB++) {
      coord_t j = B_crd_acc[jB];
      A_vals_acc[jB] = 0.0;
      for (coord_t k = C_lo[1]; k < C_hi[1] + 1; k++) {
        A_vals_acc[jB] += B_vals_acc[jB] * (C_vals_acc[{i, k}] * D_vals_acc[{k, j}]);
      }
    }
  }
}

void CSCSDDMM::cpu_variant(legate::TaskContext &ctx) {
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];
  auto& C_vals = ctx.inputs()[3];
  auto& D_vals = ctx.inputs()[4];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) {
    B_pos.remove_transform();
  }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();
  auto C_vals_acc = C_vals.read_accessor<val_ty, 2>();
  auto D_vals_acc = D_vals.read_accessor<val_ty, 2>();

  auto B_domain = B_pos.domain();
  auto C_domain = C_vals.domain();
  auto C_lo = C_domain.lo();
  auto C_hi = C_domain.hi();

  for (coord_t j = B_domain.lo()[0]; j < B_domain.hi()[0] + 1; j++) {
    for (size_t iB = B_pos_acc[j].lo; iB < B_pos_acc[j].hi + 1; iB++) {
      coord_t i = B_crd_acc[iB];
      A_vals_acc[iB] = 0.0;
      for (coord_t k = C_lo[1]; k < C_hi[1] + 1; k++) {
        A_vals_acc[iB] += B_vals_acc[iB] * (C_vals_acc[{i, k}] * D_vals_acc[{k, j}]);
      }
    }
  }
}

void BoundsFromPartitionedCoordinates::cpu_variant(legate::TaskContext &ctx) {
  auto& input = ctx.inputs()[0];
  auto& output = ctx.outputs()[0];
  assert(output.is_future());

  auto input_acc = input.read_accessor<coord_ty, 1>();
  auto output_acc = output.write_accessor<Domain, 1>();
  auto dom = input.domain();
  if (dom.empty()) {
    output_acc[0] = {0, -1};
  } else {
    auto ptr = input_acc.ptr(dom.lo());
    auto result = thrust::minmax_element(thrust::host, ptr, ptr + dom.get_volume());
    output_acc[0] = {*result.first, *result.second};
  }
}

void SortedCoordsToCounts::cpu_variant(legate::TaskContext &ctx) {
  auto& output = ctx.reductions()[0];
  auto& input = ctx.inputs()[0];
  auto dom = input.domain();
  if (output.domain().empty()) return;
  auto in = input.read_accessor<coord_ty, 1>();
  auto out = output.reduce_accessor<SumReduction<uint64_t>, true /* exclusive */, 1>();
  for (PointInDomainIterator<1> itr(dom); itr(); itr++) {
    out[in[*itr]] <<= 1;
  }
}

void ExpandPosToCoordinates::cpu_variant(legate::TaskContext &ctx) {
  auto& pos = ctx.inputs()[0];
  auto& result = ctx.outputs()[0];
  ExpandPosToCoordinates::expand_pos_impl(
    thrust::host,
    pos.read_accessor<Rect<1>, 1>(),
    pos.domain(),
    result.write_accessor<coord_ty, 1>(),
    result.domain(),
    Memory::SYSTEM_MEM
  );
}

// TODO (rohany): This kernel could be templated to support both CSR and CSC.
// CSRToDense was adapted from DISTAL generated code.
// A(i, j) = B(i, j)
// Schedule:
// distribute({i}, {io}, {ii}, pieces)
void CSRToDense::cpu_variant(legate::TaskContext &ctx) {
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) {
    B_pos.remove_transform();
  }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 2>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();

  // Initialize the output array.
  auto A_domain = A_vals.domain();
  auto A_lo = A_domain.lo();
  auto A_hi = A_domain.hi();
  for (coord_ty i = A_lo[0]; i < A_hi[0] + 1; i++) {
    for (coord_ty j = A_lo[1]; j < A_hi[1] + 1; j++) {
      A_vals_acc[{i, j}] = 0.0;
    }
  }

  auto B_domain = B_pos.domain();
  for (coord_ty i = B_domain.lo()[0]; i < B_domain.hi()[0] + 1; i++) {
    for (size_t jB = B_pos_acc[i].lo; jB < B_pos_acc[i].hi + 1; jB++) {
      auto j = B_crd_acc[jB];
      A_vals_acc[{i, j}] = B_vals_acc[jB];
    }
  }
}

// CSCToDense was adapted from DISTAL generated code.
// A(i, j) = B(i, j)
// Schedule:
// distribute({i}, {io}, {ii}, pieces)
void CSCToDense::cpu_variant(legate::TaskContext &ctx) {
  auto& A_vals = ctx.outputs()[0];
  auto& B_pos = ctx.inputs()[0];
  auto& B_crd = ctx.inputs()[1];
  auto& B_vals = ctx.inputs()[2];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (B_pos.transformed()) {
    B_pos.remove_transform();
  }

  auto A_vals_acc = A_vals.write_accessor<val_ty, 2>();
  auto B_pos_acc = B_pos.read_accessor<Rect<1>, 1>();
  auto B_crd_acc = B_crd.read_accessor<coord_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 1>();

  // Initialize the output array.
  auto A_domain = A_vals.domain();
  auto A_lo = A_domain.lo();
  auto A_hi = A_domain.hi();
  for (coord_ty i = A_lo[0]; i < A_hi[0] + 1; i++) {
    for (coord_ty j = A_lo[1]; j < A_hi[1] + 1; j++) {
      A_vals_acc[{i, j}] = 0.0;
    }
  }

  auto B_domain = B_pos.domain();
  for (coord_ty j = B_domain.lo()[0]; j < B_domain.hi()[0] + 1; j++) {
    for (size_t iB = B_pos_acc[j].lo; iB < B_pos_acc[j].hi + 1; iB++) {
      auto i = B_crd_acc[iB];
      A_vals_acc[{i, j}] = B_vals_acc[iB];
    }
  }
}

void COOToDense::cpu_variant(legate::TaskContext& ctx) {
  auto& result_store = ctx.outputs()[0];
  auto& rows_store = ctx.inputs()[0];
  auto& cols_store = ctx.inputs()[1];
  auto& vals_store = ctx.inputs()[2];
  assert(ctx.is_single_task());

  auto result = result_store.write_accessor<val_ty, 2>();
  auto rows = rows_store.read_accessor<coord_ty, 1>();
  auto cols = cols_store.read_accessor<coord_ty, 1>();
  auto vals = vals_store.read_accessor<val_ty, 1>();
  auto dom = rows_store.domain();
  for (coord_ty pos = dom.lo()[0]; pos < dom.hi()[0] + 1; pos++) {
    auto i = rows[pos];
    auto j = cols[pos];
    auto val = vals[pos];
    result[{i, j}] = val;
  }
}

void DenseToCSRNNZ::cpu_variant(legate::TaskContext &ctx) {
  auto& nnz = ctx.outputs()[0];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the nnz region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (nnz.transformed()) {
    nnz.remove_transform();
  }

  auto nnz_acc = nnz.write_accessor<nnz_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();

  auto dom = B_vals.domain();
  auto lo = dom.lo();
  auto hi = dom.hi();
  for (coord_ty i = lo[0]; i < hi[0] + 1; i++) {
    uint64_t row_nnz = 0;
    for (coord_ty j = lo[1]; j < hi[1] + 1; j++) {
      if (B_vals_acc[{i, j}] != 0.0) {
        row_nnz++;
      }
    }
    nnz_acc[i] = row_nnz;
  }
}

void DenseToCSR::cpu_variant(legate::TaskContext &ctx) {
  auto& A_pos = ctx.outputs()[0];
  auto& A_crd = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (A_pos.transformed()) {
    A_pos.remove_transform();
  }

  auto A_pos_acc = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();

  auto dom = B_vals.domain();
  auto lo = dom.lo();
  auto hi = dom.hi();
  for (coord_ty i = lo[0]; i < hi[0] + 1; i++) {
    int nnz_pos = A_pos_acc[i].lo;
    for (coord_ty j = lo[1]; j < hi[1] + 1; j++) {
      if (B_vals_acc[{i, j}] != 0.0) {
        A_crd_acc[nnz_pos] = j;
        A_vals_acc[nnz_pos] = B_vals_acc[{i, j}];
        nnz_pos++;
      }
    }
  }
}

void DenseToCSCNNZ::cpu_variant(legate::TaskContext &ctx) {
  auto& nnz = ctx.outputs()[0];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the nnz region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (nnz.transformed()) {
    nnz.remove_transform();
  }

  auto nnz_acc = nnz.write_accessor<nnz_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();

  auto dom = B_vals.domain();
  auto lo = dom.lo();
  auto hi = dom.hi();
  for (coord_ty j = lo[1]; j < hi[1] + 1; j++) {
    uint64_t col_nnz = 0;
    for (coord_ty i = lo[0]; i < hi[0] + 1; i++) {
      if (B_vals_acc[{i, j}] != 0.0) {
        col_nnz++;
      }
    }
    nnz_acc[j] = col_nnz;
  }
}

void DenseToCSC::cpu_variant(legate::TaskContext &ctx) {
  auto& A_pos = ctx.outputs()[0];
  auto& A_crd = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_vals = ctx.inputs()[0];

  // We have to promote the pos region for the auto-parallelizer to kick in,
  // so remove the transformation before proceeding.
  if (A_pos.transformed()) {
    A_pos.remove_transform();
  }

  auto A_pos_acc = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_vals_acc = B_vals.read_accessor<val_ty, 2>();

  auto dom = B_vals.domain();
  auto lo = dom.lo();
  auto hi = dom.hi();
  for (coord_ty j = lo[1]; j < hi[1] + 1; j++) {
    int nnz_pos = A_pos_acc[j].lo;
    for (coord_ty i = lo[0]; i < hi[0] + 1; i++) {
      if (B_vals_acc[{i, j}] != 0.0) {
        A_crd_acc[nnz_pos] = i;
        A_vals_acc[nnz_pos] = B_vals_acc[{i, j}];
        nnz_pos++;
      }
    }
  }
}

void DIAToCSRNNZ::cpu_variant(legate::TaskContext &ctx) {
  assert(ctx.is_single_task());
  auto& nnz = ctx.outputs()[0];
  auto& B_offsets = ctx.inputs()[0];
  auto& B_data = ctx.inputs()[1];

  auto nnz_acc = nnz.write_accessor<nnz_ty, 1>();
  auto B_offsets_acc = B_offsets.read_accessor<coord_ty, 1>();
  auto B_data_acc = B_data.read_accessor<val_ty, 2>();

  auto m = ctx.scalars()[0].value<int64_t>();
  auto n = ctx.scalars()[1].value<int64_t>();

  // Initialize the nnz array.
  for (PointInDomainIterator<1> itr(nnz.domain()); itr(); itr++) {
    nnz_acc[*itr] = 0;
  }

  auto data_domain = B_data.domain();
  // Figure out the number of non-zeros per row.
  for (PointInDomainIterator<1> offsets_itr(B_offsets.domain()); offsets_itr(); offsets_itr++) {
    auto offset = B_offsets_acc[*offsets_itr];
    int64_t col = 0;
    int64_t row = 0;
    int64_t start = data_domain.lo()[1];
    int64_t end = data_domain.hi()[1] + 1;
    if (offset > 0) {
      col += offset;
      start += offset;
    } else {
      row += (-offset);
      end = m + offset;
    }
    for (int64_t k = start; k < end; k++) {
      if (row >= m || col >= n) break;
      auto elem = B_data_acc[{*offsets_itr, k}];
      if (elem != 0.0) {
        nnz_acc[row]++;
      }
      col++;
      row++;
    }
  }
}

void DIAToCSR::cpu_variant(legate::TaskContext &ctx) {
  assert(ctx.is_single_task());
  auto& A_pos = ctx.outputs()[0];
  auto& A_crd = ctx.outputs()[1];
  auto& A_vals = ctx.outputs()[2];
  auto& B_offsets = ctx.inputs()[0];
  auto& B_data = ctx.inputs()[1];

  auto A_pos_acc = A_pos.read_write_accessor<Rect<1>, 1>();
  auto A_crd_acc = A_crd.write_accessor<coord_ty, 1>();
  auto A_vals_acc = A_vals.write_accessor<val_ty, 1>();
  auto B_offsets_acc = B_offsets.read_accessor<int64_t, 1>();
  auto B_data_acc = B_data.read_accessor<val_ty, 2>();

  int m = ctx.scalars()[0].value<int64_t>();
  int n = ctx.scalars()[1].value<int64_t>();

  auto data_domain = B_data.domain();
  for (PointInDomainIterator<1> offsets_itr(B_offsets.domain()); offsets_itr(); offsets_itr++) {
    auto offset = B_offsets_acc[*offsets_itr];
    int64_t col = 0;
    int64_t row = 0;
    int64_t start = data_domain.lo()[1];
    int64_t end = data_domain.hi()[1] + 1;
    if (offset > 0) {
      col += offset;
      start += offset;
    } else {
      row += (-offset);
      end = m + offset;
    }
    for (int64_t k = start; k < end; k++) {
      if (row >= m || col >= n) break;
      auto elem = B_data_acc[{*offsets_itr, k}];
      if (elem != 0.0) {
        auto j_pos = A_pos_acc[row].lo;
        A_pos_acc[row].lo = A_pos_acc[row].lo + 1;
        A_crd_acc[j_pos] = col;
        A_vals_acc[j_pos] = elem;
      }
      col++;
      row++;
    }
  }

  // TODO (rohany): Note that we do this differently than in DISTAL, where we
  //  always allocate a temporary ghost region, so we can iterate in standard order.
  // Finalize the changes to the pos array.
  for (int64_t it = A_pos.domain().hi()[0]; it >= 0; it--) {
    if (it == 0) {
      A_pos_acc[0].lo = 0;
    } else {
      A_pos_acc[it].lo = A_pos_acc[it - 1].lo;
    }
  }

}

void ZipToRect1::cpu_variant(legate::TaskContext &ctx) {
  auto& output = ctx.outputs()[0];
  if (output.domain().empty()) return;
  auto output_acc = output.write_accessor<Rect<1>, 1>();
  auto lo = ctx.inputs()[0].read_accessor<uint64_t, 1>();
  auto hi = ctx.inputs()[1].read_accessor<uint64_t, 1>();

  for (PointInDomainIterator<1> itr(output.domain()); itr(); itr++) {
    output_acc[*itr] = {lo[*itr], hi[*itr] - 1};
  }
}

void UnZipRect1::cpu_variant(legate::TaskContext &ctx) {
  if (ctx.outputs()[0].domain().empty()) return;
  if (ctx.outputs()[0].dim() == 1) {
    auto out1 = ctx.outputs()[0].write_accessor<int64_t, 1>();
    auto out2 = ctx.outputs()[1].write_accessor<int64_t, 1>();
    auto in = ctx.inputs()[0].read_accessor<Rect<1>, 1>();
    for (PointInDomainIterator<1> itr(ctx.inputs()[0].domain()); itr(); itr++) {
      auto rect = in[*itr];
      out1[*itr] = rect.lo;
      out2[*itr] = rect.hi;
    }
  } else if (ctx.outputs()[0].dim() == 2) {
    auto out1 = ctx.outputs()[0].write_accessor<int64_t, 2>();
    auto out2 = ctx.outputs()[1].write_accessor<int64_t, 2>();
    auto in = ctx.inputs()[0].read_accessor<Rect<1>, 2>();
    for (PointInDomainIterator<2> itr(ctx.inputs()[0].domain()); itr(); itr++) {
      auto rect = in[*itr];
      out1[*itr] = rect.lo;
      out2[*itr] = rect.hi;
    }
  } else {
    assert(ctx.outputs()[0].dim() == 3);
    auto out1 = ctx.outputs()[0].write_accessor<int64_t, 3>();
    auto out2 = ctx.outputs()[1].write_accessor<int64_t, 3>();
    auto in = ctx.inputs()[0].read_accessor<Rect<1>, 3>();
    for (PointInDomainIterator<3> itr(ctx.inputs()[0].domain()); itr(); itr++) {
      auto rect = in[*itr];
      out1[*itr] = rect.lo;
      out2[*itr] = rect.hi;
    }
  }
}

void ScaleRect1::cpu_variant(legate::TaskContext &ctx) {
  if (ctx.outputs()[0].domain().empty()) return;
  auto out = ctx.outputs()[0].read_write_accessor<Rect<1>, 1>();
  auto task = ctx.task_;
  auto scale = task->futures[0].get_result<int64_t>();
  for (PointInDomainIterator<1> itr(ctx.outputs()[0].domain()); itr(); itr++) {
    out[*itr].lo = out[*itr].lo + scale;
    out[*itr].hi = out[*itr].hi + scale;
  }
}

template<typename T>
void UpcastFutureToRegion::cpu_variant_impl(legate::TaskContext &ctx) {
  auto& in_fut = ctx.inputs()[0];
  const T* src;
  T* dst;
  switch (in_fut.dim()) {
    case 0: {
      // Futures can be 0-dimensional. legate doesn't appear to complain
      // if we make a 1-D accessor of a 0-D "store".
      dst = ctx.outputs()[0].write_accessor<T, 1>().ptr(0);
      src = ctx.inputs()[0].read_accessor<T, 1>().ptr(0);
      break;
    }
    case 1: {
      dst = ctx.outputs()[0].write_accessor<T, 1>().ptr(0);
      src = ctx.inputs()[0].read_accessor<T, 1>().ptr(0);
      break;
    }
    case 2: {
      dst = ctx.outputs()[0].write_accessor<T, 2>().ptr({0, 0});
      src = ctx.inputs()[0].read_accessor<T, 2>().ptr({0, 0});
      break;
    }
    case 3: {
      dst = ctx.outputs()[0].write_accessor<T, 3>().ptr({0, 0, 0});
      src = ctx.inputs()[0].read_accessor<T, 3>().ptr({0, 0, 0});
      break;
    }
  }
  memcpy(dst, src, sizeof(T));
}

void UpcastFutureToRegion::cpu_variant(legate::TaskContext &ctx) {
  assert(ctx.is_single_task());
  auto future_size = ctx.scalars()[0].value<size_t>();
  switch (future_size) {
    case 1: {
      UpcastFutureToRegion::cpu_variant_impl<uint8_t>(ctx);
      break;
    }
    case 2: {
      UpcastFutureToRegion::cpu_variant_impl<uint16_t>(ctx);
      break;
    }
    case 4: {
      UpcastFutureToRegion::cpu_variant_impl<uint32_t>(ctx);
      break;
    }
    case 8: {
      UpcastFutureToRegion::cpu_variant_impl<uint64_t>(ctx);
      break;
    }
    default: {
      assert(false);
    }
  }
}

void FastImageRange::cpu_variant(legate::TaskContext &ctx) {
  auto& input = ctx.inputs()[0];
  auto& output = ctx.outputs()[0];
  if (input.transformed()) {
    input.remove_transform();
  }
  auto in = input.read_accessor<Rect<1>, 1>();
  auto out = output.write_accessor<Domain, 1>();
  auto dom = input.domain();
  if (dom.empty()) {
    out[0] = Rect<1>::make_empty();
  } else {
    out[0] = Rect<1>{in[dom.lo()].lo, in[dom.hi()].hi};
  }
}

// Much of this code was adapted from the matrix-market file IO module
// within DISTAL.
void ReadMTXToCOO::cpu_variant(legate::TaskContext &ctx) {
  assert(ctx.is_single_task());
  // Regardless of how inputs are added, scalar future return values are at the front.
  auto& m_store = ctx.outputs()[0];
  auto& n_store = ctx.outputs()[1];
  auto& nnz_store = ctx.outputs()[2];
  auto& rows = ctx.outputs()[3];
  auto& cols = ctx.outputs()[4];
  auto& vals = ctx.outputs()[5];
  auto filename = ctx.scalars()[0].value<std::string>();
  std::fstream file;
  file.open(filename, std::fstream::in);

  // Parse the header. The header is structured as follows:
  //  %%MatrixMarket type format field symmetry
  std::string line;
  std::getline(file, line);
  std::stringstream lineStream(line);
  std::string head, type, formats, field, symmetry;
  lineStream >> head >> type >> formats >> field >> symmetry;
  assert(head == "%%MatrixMarket" && "Unknown header of MatrixMarket");
  assert(type == "matrix" && "must have type matrix");
  assert(formats == "coordinate" && "must be coordinate");
  enum ValueKind {
    REAL,
    PATTERN,
    INTEGER,
  };
  ValueKind valueKind;
  if (field == "real") {
    valueKind = REAL;
  } else if (field == "pattern") {
    valueKind = PATTERN;
  } else if (field == "integer") {
    valueKind = INTEGER;
  } else {
    assert(false && "unknown field");
  }
  bool symmetric = false;
  if (symmetry == "symmetric") {
    symmetric = true;
  } else if (symmetry == "general") { /* Do nothing. */ }
  else {
    assert(false && "unknown symmetry");
  }

  // Skip comments at the top of the file.
  std::string token;
  do {
    std::stringstream lineStream(line);
    lineStream >> token;
    if (token[0] != '%') {
      break;
    }
  } while (std::getline(file, line));

  char* linePtr = (char*)line.data();
  coord_ty m, n;
  size_t lines;
  {
    std::vector<coord_ty> dimensions;
    while (size_t dimension = strtoull(linePtr, &linePtr, 10)) {
      dimensions.push_back(static_cast<coord_ty>(dimension));
    }
    m = dimensions[0];
    n = dimensions[1];
    lines = dimensions[2];
  }

  size_t bufSize = lines;
  if (symmetric) {
    bufSize *= 2;
  }

  auto row_acc = rows.create_output_buffer<coord_ty, 1>(bufSize, true /* return_data */);
  auto col_acc = cols.create_output_buffer<coord_ty, 1>(bufSize, true /* return_data */);
  auto vals_acc = vals.create_output_buffer<val_ty, 1>(bufSize, true /* return_data */);

  size_t idx = 0;
  while (std::getline(file, line)) {
    char* linePtr = (char*)line.data();
    coord_ty coordX = strtoll(linePtr, &linePtr, 10);
    coord_ty coordY = strtoll(linePtr, &linePtr, 10);
    // MTX coordinates 1 indexed rather than 0 indexed.
    row_acc[idx] = coordX - 1;
    col_acc[idx] = coordY - 1;
    double val;
    if (valueKind == PATTERN) {
      val = 1.0;
    } else if (valueKind == INTEGER) {
      val = strtol(linePtr, &linePtr, 10);
    } else {
      val = strtod(linePtr, &linePtr);
    }
    vals_acc[idx] = val;
    idx++;
    if (symmetric && coordX != coordY) {
      row_acc[idx] = coordY - 1;
      col_acc[idx] = coordX - 1;
      vals_acc[idx] = val;
      idx++;
    }
  }

  file.close();
  m_store.write_accessor<int64_t, 1>()[0] = int64_t(m);
  n_store.write_accessor<int64_t, 1>()[0] = int64_t(n);
  nnz_store.write_accessor<uint64_t, 1>()[0] = uint64_t(idx);
}

void EuclideanCDist::cpu_variant(legate::TaskContext &ctx) {
  auto& out = ctx.outputs()[0];
  auto& XA = ctx.inputs()[0];
  auto& XB = ctx.inputs()[1];

  auto out_acc = out.write_accessor<val_ty, 2>();
  auto XA_acc = XA.read_accessor<val_ty, 2>();
  auto XB_acc = XB.read_accessor<val_ty, 2>();

  auto out_domain = out.domain();
  auto out_lo = out_domain.lo();
  auto out_hi = out_domain.hi();
  auto XA_domain = XA.domain();
  auto XA_lo = XA_domain.lo();
  auto XA_hi = XA_domain.hi();
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

void GetCSRDiagonal::cpu_variant(legate::TaskContext &ctx) {
  auto& diag = ctx.outputs()[0];
  auto& pos = ctx.inputs()[0];
  auto& crd = ctx.inputs()[1];
  auto& vals = ctx.inputs()[2];

  auto diag_acc = diag.write_accessor<val_ty, 1>();
  auto pos_acc = pos.read_accessor<Rect<1>, 1>();
  auto crd_acc = crd.read_accessor<coord_ty, 1>();
  auto vals_acc = vals.read_accessor<val_ty, 1>();

  auto dom = pos.domain();
  for (coord_ty i = dom.lo()[0]; i < dom.hi()[0] + 1; i++) {
    diag_acc[i] = 0.0;
    for (size_t j_pos = pos_acc[i].lo; j_pos < pos_acc[i].hi + 1; j_pos++) {
      if (crd_acc[j_pos] == i) {
        diag_acc[i] = vals_acc[j_pos];
      }
    }
  }
}

void SortByKey::cpu_variant(legate::TaskContext &ctx) {
  SortBody<coord_ty, val_ty, decltype(thrust::host)>(ctx, Memory::SYSTEM_MEM, thrust::host);
}

void VecMultAdd::cpu_variant(legate::TaskContext& ctx) {
  auto& lhs = ctx.outputs()[0];
  auto& rhs = ctx.inputs()[0];
  auto& beta_store = ctx.inputs()[1];
  bool left = ctx.scalars()[0].value<bool>();
  auto dom = lhs.domain();
  auto lhs_acc = lhs.read_write_accessor<val_ty, 1>();
  auto rhs_acc = rhs.read_accessor<val_ty, 1>();
  auto beta = beta_store.read_accessor<val_ty, 1>()[0];
  
  if (left) {
    for (coord_ty i = dom.lo()[0]; i < dom.hi()[0] + 1; i++) {
      lhs_acc[i] = (lhs_acc[i] * beta) + rhs_acc[i];
    }
  } else {
    for (coord_ty i = dom.lo()[0]; i < dom.hi()[0] + 1; i++) {
      lhs_acc[i] = lhs_acc[i] + (beta * rhs_acc[i]);
    }
  }
}

} // namespace sparse

namespace { // anonymous
static void __attribute__((constructor)) register_tasks(void) {
  sparse::CSRSpMVRowSplit::register_variants();
  sparse::CSRSpMVRowSplitTropicalSemiring::register_variants();
  sparse::CSCSpMVColSplit::register_variants();

  sparse::SpGEMMCSRxCSRxCSRNNZ::register_variants();
  sparse::SpGEMMCSRxCSRxCSR::register_variants();
  sparse::SpGEMMCSRxCSRxCSRGPU::register_variants();

  sparse::SpGEMMCSRxCSRxCSCLocalTiles::register_variants();
  sparse::SpGEMMCSRxCSRxCSCCommCompute::register_variants();
  sparse::SpGEMMCSRxCSRxCSCShuffle::register_variants();

  sparse::SpMMCSR::register_variants();
  sparse::SpMMDenseCSR::register_variants();

  sparse::AddCSRCSRNNZ::register_variants();
  sparse::AddCSRCSR::register_variants();
  sparse::ElemwiseMultCSRCSRNNZ::register_variants();
  sparse::ElemwiseMultCSRCSR::register_variants();
  sparse::ElemwiseMultCSRDense::register_variants();
  sparse::CSRSDDMM::register_variants();
  sparse::CSCSDDMM::register_variants();

  sparse::CSRToDense::register_variants();
  sparse::CSCToDense::register_variants();
  sparse::COOToDense::register_variants();
  sparse::DenseToCSRNNZ::register_variants();
  sparse::DenseToCSR::register_variants();
  sparse::DenseToCSCNNZ::register_variants();
  sparse::DenseToCSC::register_variants();
  sparse::DIAToCSRNNZ::register_variants();
  sparse::DIAToCSR::register_variants();
  sparse::BoundsFromPartitionedCoordinates::register_variants();
  sparse::SortedCoordsToCounts::register_variants();
  sparse::ExpandPosToCoordinates::register_variants();

  sparse::ZipToRect1::register_variants();
  sparse::UnZipRect1::register_variants();
  sparse::ScaleRect1::register_variants();
  sparse::UpcastFutureToRegion::register_variants();
  sparse::FastImageRange::register_variants();

  sparse::ReadMTXToCOO::register_variants();

  sparse::EuclideanCDist::register_variants();
  sparse::SortByKey::register_variants();

  sparse::GetCSRDiagonal::register_variants();
  sparse::VecMultAdd::register_variants();
}
}

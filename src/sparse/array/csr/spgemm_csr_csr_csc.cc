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

#include "sparse/array/csr/spgemm_csr_csr_csc.h"
#include "sparse/array/csr/spgemm_csr_csr_csc_template.inl"

namespace sparse {

using namespace Legion;
using namespace legate;

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpGEMMCSRxCSRxCSCLocalTilesImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;
  void operator()(Store& A_pos_store,
                  Store& A_crd_store,
                  Store& A_vals_store,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorRO<INDEX_TY, 1>& C_crd,
                  const AccessorRO<VAL_TY, 1>& C_vals,
                  const Rect<1>& B_rect,
                  const Rect<1>& C_rect)
  {
    // Our job right now is to perform both passes of the SpGEMM operation
    // and output instances for local CSR matrices of each result.
    DeferredBuffer<size_t, 1> nnz({B_rect.lo[0], B_rect.hi[0]}, Memory::SYSTEM_MEM);
    for (auto i = B_rect.lo[0]; i < B_rect.hi[0] + 1; i++) {
      size_t row_nnzs = 0;
      for (auto j = C_rect.lo[0]; j < C_rect.hi[0] + 1; j++) {
        size_t kB_pos     = B_pos[i].lo;
        size_t kB_pos_end = B_pos[i].hi + 1;
        size_t kC_pos     = C_pos[j].lo;
        size_t kC_pos_end = C_pos[j].hi + 1;
        while (kB_pos < kB_pos_end && kC_pos < kC_pos_end) {
          auto kB = B_crd[kB_pos];
          auto kC = C_crd[kC_pos];
          auto k  = std::min(kB, kC);
          if (k == kB && k == kC) {
            row_nnzs++;
            break;
          }
          kB_pos += (size_t)(kB == k);
          kC_pos += (size_t)(kC == k);
        }
      }
      nnz[i] = row_nnzs;
    }

    auto A_pos =
      A_pos_store.create_output_buffer<Rect<1>, 1>(B_rect.volume(), true /* return_buffer */);
    size_t val = 0;
    for (auto i = B_rect.lo[0]; i < B_rect.hi[0] + 1; i++) {
      size_t newVal           = nnz[i] + val;
      A_pos[i - B_rect.lo[0]] = {val, newVal - 1};
      val                     = newVal;
    }
    auto A_crd  = A_crd_store.create_output_buffer<INDEX_TY, 1>(val, true /* return_buffer */);
    auto A_vals = A_vals_store.create_output_buffer<VAL_TY, 1>(val, true /* return_buffer */);

    for (auto i = B_rect.lo[0]; i < B_rect.hi[0] + 1; i++) {
      // Important: We need to offset the access into the output buffers, as they
      // are not globally indexed objects.
      size_t nnz_pos = A_pos[i - B_rect.lo[0]].lo;
      for (auto j = C_rect.lo[0]; j < C_rect.hi[0] + 1; j++) {
        VAL_TY result     = static_cast<VAL_TY>(0);
        bool set          = false;
        size_t kB_pos     = B_pos[i].lo;
        size_t kB_pos_end = B_pos[i].hi + 1;
        size_t kC_pos     = C_pos[j].lo;
        size_t kC_pos_end = C_pos[j].hi + 1;
        while (kB_pos < kB_pos_end && kC_pos < kC_pos_end) {
          auto kB = B_crd[kB_pos];
          auto kC = C_crd[kC_pos];
          auto k  = std::min(kB, kC);
          if (k == kB && k == kC) {
            set = true;
            result += B_vals[kB_pos] * C_vals[kC_pos];
          }
          kB_pos += (size_t)(kB == k);
          kC_pos += (size_t)(kC == k);
        }
        // Since we're considering a row at a time and not scattering
        // across pos, we don't need to mutate pos like as done in DISTAL.
        if (set) {
          A_crd[nnz_pos]  = j;
          A_vals[nnz_pos] = result;
          nnz_pos++;
        }
      }
    }
  }
};

template <>
struct SpGEMMCSRxCSRxCSCCommComputeImplBody<VariantKind::CPU> {
  void operator()(const AccessorWO<Rect<1>, 3>& out,
                  const Store& pos,
                  const Store& global_pos,
                  const int32_t gx,
                  const int32_t gy,
                  const Point<3>& output_point)
  {
    auto oi                = output_point[0];
    auto oj                = output_point[1];
    auto global_pos_domain = global_pos.domain();
    auto i_lo              = pos.domain().lo()[0];
    auto i_hi              = pos.domain().hi()[0] + 1;
    auto i_tile_size       = i_hi - i_lo;
    auto tile_size         = (i_hi - i_lo + gy - 1) / gy;
    for (int32_t j = 0; j < gy; j++) {
      auto sub_tile_start = j * tile_size;
      auto sub_tile_end   = std::min((j + 1) * tile_size, i_tile_size);
      auto lo             = global_pos_domain.lo()[0] + sub_tile_start;
      auto hi             = global_pos_domain.lo()[0] + sub_tile_end;
      out[{oi, oj, j}]    = Rect<1>{lo, hi - 1};
    }
  }
};

template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
struct SpGEMMCSRxCSRxCSCShuffleImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(Store& out_pos,
                  Store& out_crd,
                  Store& out_vals,
                  const AccessorRO<Rect<1>, 1>& global_pos,
                  const AccessorRO<INDEX_TY, 1>& global_crd,
                  const AccessorRO<VAL_TY, 1>& global_vals,
                  const Domain& global_pos_domain)
  {
    size_t total_nnzs = 0;
    std::vector<Rect<1>> rects;
    for (RectInDomainIterator<1> rect_itr(global_pos_domain); rect_itr(); rect_itr++) {
      rects.push_back(*rect_itr);
      if (rect_itr->empty()) continue;
      auto lo = global_pos[rect_itr->lo];
      auto hi = global_pos[rect_itr->hi];
      total_nnzs += hi.hi[0] - lo.lo[0] + 1;
    }
    size_t total_rows = 0;
    for (auto it : rects) { total_rows = std::max(total_rows, it.volume()); }

    auto pos_acc  = out_pos.create_output_buffer<Rect<1>, 1>(total_rows, true /* return_buffer */);
    auto crd_acc  = out_crd.create_output_buffer<INDEX_TY, 1>(total_nnzs, true /* return_buffer */);
    auto vals_acc = out_vals.create_output_buffer<VAL_TY, 1>(total_nnzs, true /* return_buffer */);

    size_t offset = 0;
    for (size_t i = 0; i < total_rows; i++) {
      auto lo = offset;
      for (auto rect : rects) {
        auto global_pos_idx = rect.lo + i;
        if (!rect.contains(global_pos_idx)) continue;
        for (int64_t pos = global_pos[global_pos_idx].lo; pos < global_pos[global_pos_idx].hi + 1;
             pos++) {
          crd_acc[offset]  = global_crd[pos];
          vals_acc[offset] = global_vals[pos];
          offset++;
        }
      }
      auto hi    = offset - 1;
      pos_acc[i] = {lo, hi};
    }
  }
};

/*static*/ void SpGEMMCSRxCSRxCSCLocalTiles::cpu_variant(TaskContext& context)
{
  spgemm_csr_csr_csc_local_tiles_template<VariantKind::CPU>(context);
}

/*static*/ void SpGEMMCSRxCSRxCSCCommCompute::cpu_variant(TaskContext& context)
{
  spgemm_csr_csr_csc_comm_compute_template<VariantKind::CPU>(context);
}

/*static*/ void SpGEMMCSRxCSRxCSCShuffle::cpu_variant(TaskContext& context)
{
  spgemm_csr_csr_csc_shuffle_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  SpGEMMCSRxCSRxCSCLocalTiles::register_variants();
  SpGEMMCSRxCSRxCSCCommCompute::register_variants();
  SpGEMMCSRxCSRxCSCShuffle::register_variants();
}
}  // namespace

}  // namespace sparse

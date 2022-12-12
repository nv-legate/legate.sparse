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
#include "sparse/util/cusparse_utils.h"
#include "sparse/util/dispatch.h"

namespace sparse {

template <>
struct CSRSpMVRowSplitImpl<VariantKind::GPU> {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(CSRSpMVRowSplitArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto& y      = args.y;
    auto& A_pos  = args.A_pos;
    auto& A_crd  = args.A_crd;
    auto& A_vals = args.A_vals;
    auto& x      = args.x;

    // Break out early if the iteration space partition is empty.
    if (y.domain().empty() || A_crd.domain().empty()) return;

    // Get context sensitive objects.
    auto handle = get_cusparse();
    auto stream = get_cached_stream();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    // Construct the CUSPARSE objects from individual regions.
    auto cusparse_y = makeCuSparseDenseVec<VAL_TY>(y);
    // In order to play nicely with cuSPARSE and weak-scale (on distribution
    // friendly inputs), we have to do some trickery. The first happens when
    // we launch our tasks: we take the image of the selected coordinates
    // onto the x vector, resulting in a sparse partition of x. Next, we map
    // x densely, so that we get a dense vector fitted to the size of the
    // coordinates, where communication is done just for the selected pieces.
    // Now, we need to pass a dense vector of the correct size into cuSPARSE's
    // SpMV. We can abuse the fact that a proper SpMV implementation should only
    // read the components of x corresponding to encoded columns in the input matrix.
    // Note that we don't have to do any of this for the CPU/OMP implementations
    // since those codes use the accessor types directly. If we switched to calling
    // an external library we would need to do something similar to this.
    auto x_domain = x.domain();
    // We set the number of columns to the upper bound of the domain of x. This
    // shrinks the number of columns to the largest column index in the selected
    // partition of the input matrix.
    auto cols = x_domain.hi()[0] + 1;
    // Next, we grab a pointer to the start of the x instance. Since x is densely
    // encoded, we can shift this pointer based on the lower bound of the input
    // domain to get a "fake" pointer to the start of a densely encoded vector
    // of length cols. We can bank on cuSPARSE's implementation to not read any
    // of the memory locations before the lower bound of the x_vals_domain due
    // to the properties of the resulting image partition.
    auto x_raw_ptr = x.read_accessor<VAL_TY, 1>().ptr(x_domain.lo());
    auto x_ptr     = x_raw_ptr - size_t(x_domain.lo()[0]);
    cusparseDnVecDescr_t cusparse_x;
    CHECK_CUSPARSE(cusparseCreateDnVec(
      &cusparse_x, cols, const_cast<VAL_TY*>(x_ptr), cusparseDataType<VAL_TY>()));
    auto cusparse_A = makeCuSparseCSR<INDEX_TY, VAL_TY>(A_pos, A_crd, A_vals, cols);

    // Make the CUSPARSE calls.
    VAL_TY alpha   = 1.0;
    VAL_TY beta    = 0.0;
    size_t bufSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           cusparse_A,
                                           cusparse_x,
                                           &beta,
                                           cusparse_y,
                                           cusparseDataType<VAL_TY>(),
#if (CUSPARSE_VER_MAJOR < 11 || CUSPARSE_VER_MINOR < 2)
                                           CUSPARSE_MV_ALG_DEFAULT,
#else
                                           CUSPARSE_SPMV_ALG_DEFAULT,
#endif
                                           &bufSize));
    // Allocate a buffer if we need to.
    void* workspacePtr = nullptr;
    if (bufSize > 0) {
      DeferredBuffer<char, 1> buf({0, bufSize - 1}, Memory::GPU_FB_MEM);
      workspacePtr = buf.ptr(0);
    }
    // Finally do the SpMV.
    CHECK_CUSPARSE(cusparseSpMV(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                cusparse_A,
                                cusparse_x,
                                &beta,
                                cusparse_y,
                                cusparseDataType<VAL_TY>(),
#if (CUSPARSE_VER_MAJOR < 11 || CUSPARSE_VER_MINOR < 2)
                                CUSPARSE_MV_ALG_DEFAULT,
#else
                                CUSPARSE_SPMV_ALG_DEFAULT,
#endif
                                workspacePtr));
    // Destroy the created objects.
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_y));
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_x));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_A));
    CHECK_CUDA_STREAM(stream);
  }
};

template <typename INDEX_TY, typename VAL_TY>
__global__ void spmv_col_split_kernel(const AccessorRD<SumReduction<VAL_TY>, true /* exclusive */, 1> y,
                                      const AccessorRO<Rect<1>, 1> A_pos,
                                      const AccessorRO<INDEX_TY, 1> A_crd,
                                      const AccessorRO<VAL_TY, 1> A_vals,
                                      const AccessorRO<VAL_TY, 1> x,
                                      const Rect<1> y_rect,
                                      const Rect<1> A_crd_rect,
                                      const Rect<1> x_rect)
{
  auto idx = global_tid_1d();
  if (idx >= y_rect.volume()) return;
  auto i     = idx + y_rect.lo[0];
  VAL_TY sum = 0.0;
  for (size_t j_pos = A_pos[i].lo; j_pos < A_pos[i].hi + 1; j_pos++) {
    // Because the columns have been partitioned, we take a preimage
    // back into the coordinates, densify that, and then preimage again
    // into pos. That means we may reference entries in pos that are
    // are not meant to iterate over the entire rectangle, but just
    // the coordinates covered in A_crd_rect.
    if (A_crd_rect.contains(j_pos)) {
      auto j = A_crd[j_pos];
      // We also might get coordinates that aren't within the x partition.
      if (x_rect.contains(j)) { sum += A_vals[j_pos] * x[j]; }
    }
  }
  y[i] <<= sum;
}

// Trying to use cuSPARSE in this column split case is definitely tricky.
// To make things a bit simpler for us, start with a simple CUDA kernel
// for now, and we can move to a cuSPARSE kernel if necessary.
template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE, typename ACC>
struct CSRSpMVColSplitImplBody<VariantKind::GPU, INDEX_CODE, VAL_CODE, ACC> {
  using INDEX_TY = legate_type_of<INDEX_CODE>;
  using VAL_TY   = legate_type_of<VAL_CODE>;

  void operator()(ACC y,
                  const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<VAL_TY, 1>& x,
                  const Rect<1>& y_rect,
                  const Rect<1>& A_crd_rect,
                  const Rect<1>& x_rect)
  {
    auto stream = get_cached_stream();
    auto elems  = y_rect.volume();
    auto blocks = get_num_blocks_1d(elems);
    spmv_col_split_kernel<INDEX_TY, VAL_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      y, A_pos, A_crd, A_vals, x, y_rect, A_crd_rect, x_rect);
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void CSRSpMVRowSplit::gpu_variant(TaskContext& context)
{
  csr_spmv_row_split_template<VariantKind::GPU>(context);
}

/*static*/ void CSRSpMVColSplit::gpu_variant(TaskContext& context)
{
  csr_spmv_col_split_template<VariantKind::GPU>(context);
}

}  // namespace sparse

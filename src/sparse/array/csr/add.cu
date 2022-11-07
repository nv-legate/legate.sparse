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

#include "sparse/array/csr/add.h"
#include "sparse/array/csr/add_template.inl"
#include "sparse/util/cusparse_utils.h"

namespace sparse {

template <>
struct AddCSRCSRNNZImpl<VariantKind::GPU> {
  template <LegateTypeCode INDEX_CODE>
  void operator()(AddCSRCSRNNZArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;

    auto nnz   = args.nnz.write_accessor<nnz_ty, 1>();
    auto B_pos = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto C_pos = args.C_pos.read_accessor<Rect<1>, 1>();
    auto C_crd = args.C_crd.read_accessor<INDEX_TY, 1>();
    auto cols  = args.cols;

    auto B_pos_domain = args.B_pos.domain();
    auto B_crd_domain = args.B_crd.domain();
    auto C_pos_domain = args.C_pos.domain();
    auto C_crd_domain = args.C_crd.domain();

    // Break out early if the iteration space partition is empty.
    if (B_pos_domain.empty() || C_pos_domain.empty()) { return; }

    // Get context sensitive objects.
    auto handle = get_cusparse();
    auto stream = get_cached_stream();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    auto rows = B_pos_domain.get_volume();

    // Cast the pos arrays into CSR indptr arrays.
    DeferredBuffer<int32_t, 1> B_indptr({0, rows}, Memory::GPU_FB_MEM);
    DeferredBuffer<int32_t, 1> C_indptr({0, rows}, Memory::GPU_FB_MEM);
    {
      auto blocks = get_num_blocks_1d(rows);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, B_pos.ptr(B_pos_domain.lo()), B_indptr.ptr(0));
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, C_pos.ptr(C_pos_domain.lo()), C_indptr.ptr(0));
    }

    // If our index types are already 32 bit integers, then we don't need
    // to do any casting. Otherwise, we need to create some temporaries.
    const int32_t* B_crd_int = nullptr;
    const int32_t* C_crd_int = nullptr;
    if constexpr (INDEX_CODE == LegateTypeCode::INT32_LT) {
      B_crd_int = B_crd.ptr(B_crd_domain.lo());
      C_crd_int = C_crd.ptr(C_crd_domain.lo());
    } else {
      DeferredBuffer<int32_t, 1> B_crd_int_buf({0, B_crd_domain.get_volume() - 1},
                                               Memory::GPU_FB_MEM);
      DeferredBuffer<int32_t, 1> C_crd_int_buf({0, C_crd_domain.get_volume() - 1},
                                               Memory::GPU_FB_MEM);
      {
        auto elems  = B_crd_domain.get_volume();
        auto blocks = get_num_blocks_1d(elems);
        cast<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          elems, B_crd_int_buf.ptr(0), B_crd.ptr(B_crd_domain.lo()));
      }
      {
        auto elems  = C_crd_domain.get_volume();
        auto blocks = get_num_blocks_1d(elems);
        cast<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          elems, C_crd_int_buf.ptr(0), C_crd.ptr(C_crd_domain.lo()));
      }
      B_crd_int = B_crd_int_buf.ptr(0);
      C_crd_int = C_crd_int_buf.ptr(0);
    }

    // We can now start to make calls to cuSPARSE.
    cusparseMatDescr_t A, B, C;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&A));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(A, index_base));
    CHECK_CUSPARSE(cusparseSetMatType(A, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&B));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(B, index_base));
    CHECK_CUSPARSE(cusparseSetMatType(B, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&C));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(C, index_base));
    CHECK_CUSPARSE(cusparseSetMatType(C, CUSPARSE_MATRIX_TYPE_GENERAL));

    // Figure out the necessary buffer size.
    double alpha   = 1.0;
    size_t bufSize = 0;
    DeferredBuffer<int32_t, 1> nnz_int({0, rows}, Memory::GPU_FB_MEM);
    // Since this is just the NNZ calculation, we don't actually
    // need to dispatch on the value type.
    CHECK_CUSPARSE(cusparseDcsrgeam2_bufferSizeExt(handle,
                                                   rows,
                                                   cols,
                                                   &alpha,
                                                   B,
                                                   B_crd_domain.get_volume(),
                                                   nullptr,
                                                   B_indptr.ptr(0),
                                                   B_crd_int,
                                                   &alpha,
                                                   C,
                                                   C_crd_domain.get_volume(),
                                                   nullptr,
                                                   C_indptr.ptr(0),
                                                   C_crd_int,
                                                   A,
                                                   nullptr,
                                                   nnz_int.ptr(0),
                                                   nullptr,
                                                   &bufSize));
    // Allocate a buffer if we need to.
    void* workspacePtr = nullptr;
    if (bufSize > 0) {
      DeferredBuffer<char, 1> buf({0, bufSize - 1}, Memory::GPU_FB_MEM);
      workspacePtr = buf.ptr(0);
    }
    int32_t tot_nnz = 0;
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle,
                                        rows,
                                        cols,
                                        B,
                                        B_crd_domain.get_volume(),
                                        B_indptr.ptr(0),
                                        B_crd_int,
                                        C,
                                        C_crd_domain.get_volume(),
                                        C_indptr.ptr(0),
                                        C_crd_int,
                                        A,
                                        nnz_int.ptr(0),
                                        &tot_nnz,
                                        workspacePtr));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(A));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(B));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(C));
    // Finally, turn the computed indptr array into the nnz array.
    {
      auto blocks = get_num_blocks_1d(rows);
      localIndptrToNnz<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, nnz.ptr(args.nnz.domain().lo()), nnz_int.ptr(0));
    }
    CHECK_CUDA_STREAM(stream);
  }
};

// The old cuSPARSE isn't happy with the complex types that
// we use, so add a utility to help us with this casting.
template <LegateTypeCode CODE>
struct OldCuSparseTypeOf {
  using type = legate_type_of<CODE>;
};

template <>
struct OldCuSparseTypeOf<LegateTypeCode::COMPLEX64_LT> {
  using type = cuComplex;
};

template <>
struct OldCuSparseTypeOf<LegateTypeCode::COMPLEX128_LT> {
  using type = cuDoubleComplex;
};

template <LegateTypeCode CODE>
using old_cusparse_type_of = typename OldCuSparseTypeOf<CODE>::type;

template <>
struct AddCSRCSRImpl<VariantKind::GPU> {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(AddCSRCSRArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto A_pos  = args.A_pos.read_write_accessor<Rect<1>, 1>();
    auto A_crd  = args.A_crd.write_accessor<INDEX_TY, 1>();
    auto A_vals = args.A_vals.write_accessor<VAL_TY, 1>();
    auto B_pos  = args.B_pos.read_accessor<Rect<1>, 1>();
    auto B_crd  = args.B_crd.read_accessor<INDEX_TY, 1>();
    auto B_vals = args.B_vals.read_accessor<VAL_TY, 1>();
    auto C_pos  = args.C_pos.read_accessor<Rect<1>, 1>();
    auto C_crd  = args.C_crd.read_accessor<INDEX_TY, 1>();
    auto C_vals = args.C_vals.read_accessor<VAL_TY, 1>();
    auto cols   = args.cols;

    // We have to do the same thing as in the nnz call to cast
    // our inputs into inputs that look like what cusparse can take.
    auto A_pos_domain  = args.A_pos.domain();
    auto A_crd_domain  = args.A_crd.domain();
    auto A_vals_domain = args.A_vals.domain();
    auto B_pos_domain  = args.B_pos.domain();
    auto B_crd_domain  = args.B_crd.domain();
    auto B_vals_domain = args.B_vals.domain();
    auto C_pos_domain  = args.C_pos.domain();
    auto C_crd_domain  = args.C_crd.domain();
    auto C_vals_domain = args.C_vals.domain();

    // Break out early if the iteration space partition is empty.
    if (B_pos_domain.empty() || C_pos_domain.empty()) { return; }

    // Get context sensitive objects.
    auto handle = get_cusparse();
    auto stream = get_cached_stream();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));
    auto rows = B_pos_domain.get_volume();

    // Convert all of the pos arrays into CSR indptr arrays.
    DeferredBuffer<int32_t, 1> A_indptr({0, rows}, Memory::GPU_FB_MEM);
    DeferredBuffer<int32_t, 1> B_indptr({0, rows}, Memory::GPU_FB_MEM);
    DeferredBuffer<int32_t, 1> C_indptr({0, rows}, Memory::GPU_FB_MEM);
    {
      auto blocks = get_num_blocks_1d(rows);
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, A_pos.ptr(A_pos_domain.lo()), A_indptr.ptr(0));
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, B_pos.ptr(B_pos_domain.lo()), B_indptr.ptr(0));
      convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, C_pos.ptr(C_pos_domain.lo()), C_indptr.ptr(0));
    }

    // Similiarly as the NNZ task, we don't need to copy any data
    // if we were actually given 32 bit integer coordinates.
    int32_t* A_crd_int       = nullptr;
    const int32_t* B_crd_int = nullptr;
    const int32_t* C_crd_int = nullptr;
    if constexpr (INDEX_CODE == LegateTypeCode::INT32_LT) {
      A_crd_int = A_crd.ptr(A_crd_domain.lo());
      B_crd_int = B_crd.ptr(B_crd_domain.lo());
      C_crd_int = C_crd.ptr(C_crd_domain.lo());
    } else {
      DeferredBuffer<int32_t, 1> A_crd_int_buf({0, A_crd_domain.get_volume() - 1},
                                               Memory::GPU_FB_MEM);
      DeferredBuffer<int32_t, 1> B_crd_int_buf({0, B_crd_domain.get_volume() - 1},
                                               Memory::GPU_FB_MEM);
      DeferredBuffer<int32_t, 1> C_crd_int_buf({0, C_crd_domain.get_volume() - 1},
                                               Memory::GPU_FB_MEM);
      {
        auto elems  = B_crd_domain.get_volume();
        auto blocks = get_num_blocks_1d(elems);
        cast<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          elems, B_crd_int_buf.ptr(0), B_crd.ptr(B_crd_domain.lo()));
      }
      {
        auto elems  = C_crd_domain.get_volume();
        auto blocks = get_num_blocks_1d(elems);
        cast<int32_t, INDEX_TY><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          elems, C_crd_int_buf.ptr(0), C_crd.ptr(C_crd_domain.lo()));
      }
      A_crd_int = A_crd_int_buf.ptr(0);
      B_crd_int = B_crd_int_buf.ptr(0);
      C_crd_int = C_crd_int_buf.ptr(0);
    }

    // We can now start to make calls to cuSPARSE.
    cusparseMatDescr_t A, B, C;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&A));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(A, index_base));
    CHECK_CUSPARSE(cusparseSetMatType(A, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&B));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(B, index_base));
    CHECK_CUSPARSE(cusparseSetMatType(B, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&C));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(C, index_base));
    CHECK_CUSPARSE(cusparseSetMatType(C, CUSPARSE_MATRIX_TYPE_GENERAL));

    auto funcs = []() -> auto
    {
      if constexpr (VAL_CODE == LegateTypeCode::FLOAT_LT) {
        return std::make_pair(cusparseScsrgeam2_bufferSizeExt, cusparseScsrgeam2);
      } else if constexpr (VAL_CODE == LegateTypeCode::DOUBLE_LT) {
        return std::make_pair(cusparseDcsrgeam2_bufferSizeExt, cusparseDcsrgeam2);
      } else if constexpr (VAL_CODE == LegateTypeCode::COMPLEX64_LT) {
        return std::make_pair(cusparseCcsrgeam2_bufferSizeExt, cusparseCcsrgeam2);
      } else {
        return std::make_pair(cusparseZcsrgeam2_bufferSizeExt, cusparseZcsrgeam2);
      }
    }
    ();

    using cusparse_val_ty = old_cusparse_type_of<VAL_CODE>;

    // Figure out the necessary buffer size.
    const VAL_TY alpha = 1.0;
    size_t bufSize     = 0;
    CHECK_CUSPARSE(
      funcs.first(handle,
                  rows,
                  cols,
                  reinterpret_cast<const cusparse_val_ty*>(&alpha),
                  B,
                  B_crd_domain.get_volume(),
                  reinterpret_cast<const cusparse_val_ty*>(B_vals.ptr(B_vals_domain.lo())),
                  B_indptr.ptr(0),
                  B_crd_int,
                  reinterpret_cast<const cusparse_val_ty*>(&alpha),
                  C,
                  C_crd_domain.get_volume(),
                  reinterpret_cast<const cusparse_val_ty*>(C_vals.ptr(C_vals_domain.lo())),
                  C_indptr.ptr(0),
                  C_crd_int,
                  A,
                  reinterpret_cast<const cusparse_val_ty*>(A_vals.ptr(A_vals_domain.lo())),
                  A_indptr.ptr(0),
                  A_crd_int,
                  &bufSize));
    // Allocate a buffer if we need to.
    void* workspacePtr = nullptr;
    if (bufSize > 0) {
      DeferredBuffer<char, 1> buf({0, bufSize - 1}, Memory::GPU_FB_MEM);
      workspacePtr = buf.ptr(0);
    }
    CHECK_CUSPARSE(
      funcs.second(handle,
                   rows,
                   cols,
                   reinterpret_cast<const cusparse_val_ty*>(&alpha),
                   B,
                   B_crd_domain.get_volume(),
                   reinterpret_cast<const cusparse_val_ty*>(B_vals.ptr(B_vals_domain.lo())),
                   B_indptr.ptr(0),
                   B_crd_int,
                   reinterpret_cast<const cusparse_val_ty*>(&alpha),
                   C,
                   C_crd_domain.get_volume(),
                   reinterpret_cast<const cusparse_val_ty*>(C_vals.ptr(C_vals_domain.lo())),
                   C_indptr.ptr(0),
                   C_crd_int,
                   A,
                   reinterpret_cast<cusparse_val_ty*>(A_vals.ptr(A_vals_domain.lo())),
                   A_indptr.ptr(0),
                   A_crd_int,
                   workspacePtr));
    // Clean up the created cuSPARSE objects.
    CHECK_CUSPARSE(cusparseDestroyMatDescr(A));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(B));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(C));
    // Now, we need to clean up the results and package them into the global
    // data structures. Note that the global pos array has already been computed
    // and will not be changed by the call to cusparse*csrgeam2, so we only
    // need to cast the coordinates back to the desired type.
    if constexpr (INDEX_CODE != LegateTypeCode::INT32_LT) {
      auto elems  = A_crd_domain.get_volume();
      auto blocks = get_num_blocks_1d(elems);
      cast<INDEX_TY, int32_t>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(elems, A_crd.ptr(A_crd_domain.lo()), A_crd_int);
    }
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void AddCSRCSRNNZ::gpu_variant(TaskContext& context)
{
  add_csr_csr_nnz_template<VariantKind::GPU>(context);
}

/*static*/ void AddCSRCSR::gpu_variant(TaskContext& context)
{
  add_csr_csr_template<VariantKind::GPU>(context);
}

}  // namespace sparse

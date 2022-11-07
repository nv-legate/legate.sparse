#=============================================================================
# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build legate sparse shared libraries" ON)
option(legate_sparse_EXCLUDE_LEGATE_CORE_FROM_ALL "Exclude legate.core targets from Legate Sparse's 'all' target" OFF)

##############################################################################
# - Project definition -------------------------------------------------------

# TODO (rohany): Do we need something like this for Legate Sparse?
# Write the version header
# rapids_cmake_write_version_file(include/cunumeric/version_config.hpp)

# Needed to integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - Build Type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ############################################################################
# * conda environment --------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

##############################################################################
# - Dependencies -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

find_package(OpenMP)

option(Legion_USE_CUDA "Use CUDA" ON)
option(Legion_USE_OpenMP "Use OpenMP" ${OpenMP_FOUND})
option(Legion_BOUNDS_CHECKS "Build cuNumeric with bounds checks (expensive)" OFF)

###
# If we find legate.core already configured on the system, it will report
# whether it was compiled with bounds checking (Legion_BOUNDS_CHECKS),
# CUDA (Legion_USE_CUDA), and OpenMP (Legion_USE_OpenMP).
#
# We use the same variables as legate.core because we want to enable/disable
# each of these features based on how legate.core was configured (it doesn't
# make sense to build cuNumeric's CUDA bindings if legate.core wasn't built
# with CUDA support).
###
include(cmake/thirdparty/get_legate_core.cmake)

if(Legion_USE_CUDA)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cuda_arch_helpers.cmake)
  # Needs to run before `rapids_cuda_init_architectures`
  set_cuda_arch_from_names()
  # Needs to run before `enable_language(CUDA)`
  rapids_cuda_init_architectures(legate_sparse)
  enable_language(CUDA)
  # Since Legate Sparseonly enables CUDA optionally we need to manually include
  # the file that rapids_cuda_init_architectures relies on `project` calling
  if(CMAKE_PROJECT_legate_sparse_INCLUDE)
    include("${CMAKE_PROJECT_legate_sparse_INCLUDE}")
  endif()

  # Must come after enable_language(CUDA)
  # Use `-isystem <path>` instead of `-isystem=<path>`
  # because the former works with clangd intellisense
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET legate-sparse-exports
    INSTALL_EXPORT_SET legate-sparse-exports
  )

  include(cmake/thirdparty/get_nccl.cmake)
endif()

##############################################################################
# - Legate Sparse ------------------------------------------------------------

set(legate_sparse_SOURCES "")
set(legate_sparse_CXX_DEFS "")
set(legate_sparse_CUDA_DEFS "")
set(legate_sparse_CXX_OPTIONS "")
set(legate_sparse_CUDA_OPTIONS "")

include(cmake/Modules/set_cpu_arch_flags.cmake)
set_cpu_arch_flags(legate_sparse_CXX_OPTIONS)

list(APPEND legate_sparse_SOURCES
  src/sparse/array/csc/spmv.cc
  src/sparse/array/csr/add.cc
  src/sparse/array/csr/get_diagonal.cc
  src/sparse/array/csr/spmv.cc
  src/sparse/mapper/mapper.cc

  src/projections.cc
  src/quantum.cc
  src/runge_kutta.cc
  src/tasks.cc
)

if(Legion_USE_OpenMP)
  list(APPEND legate_sparse_SOURCES
    src/sparse/array/csc/spmv_omp.cc
    src/sparse/array/csr/add_omp.cc
    src/sparse/array/csr/get_diagonal_omp.cc
    src/sparse/array/csr/spmv_omp.cc

    src/quantum_omp.cc
    src/runge_kutta_omp.cc
    src/tasks_omp.cc
  )
endif()

if(Legion_USE_CUDA)
  list(APPEND legate_sparse_SOURCES
    src/sparse/array/csc/spmv.cu
    src/sparse/array/csr/get_diagonal.cu
    src/sparse/array/csr/add.cu
    src/sparse/array/csr/spmv.cu

    src/cudalibs.cu
    src/sort.cu
    src/tasks.cu
  )
endif()

list(APPEND legate_sparse_SOURCES
  # This must always be the last file!
  # It guarantees we do our registration callback
  # only after all task variants are recorded
  src/sparse.cc
)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  list(APPEND legate_sparse_CXX_DEFS DEBUG_LEGATE_SPARSE)
  list(APPEND legate_sparse_CUDA_DEFS DEBUG_LEGATE_SPARSE)
endif()

if(Legion_BOUNDS_CHECKS)
  list(APPEND legate_sparse_CXX_DEFS BOUNDS_CHECKS)
  list(APPEND legate_sparse_CUDA_DEFS BOUNDS_CHECKS)
endif()

list(APPEND legate_sparse_CUDA_OPTIONS -Xfatbin=-compress-all)
list(APPEND legate_sparse_CUDA_OPTIONS --expt-extended-lambda)
list(APPEND legate_sparse_CUDA_OPTIONS --expt-relaxed-constexpr)
list(APPEND legate_sparse_CXX_OPTIONS -Wno-deprecated-declarations)
list(APPEND legate_sparse_CUDA_OPTIONS -Wno-deprecated-declarations)

add_library(legate_sparse ${legate_sparse_SOURCES})
add_library(legate_sparse::legate_sparse ALIAS legate_sparse)

set_target_properties(legate_sparse
           PROPERTIES BUILD_RPATH                         "\$ORIGIN"
                      INSTALL_RPATH                       "\$ORIGIN"
                      CXX_STANDARD                        17
                      CXX_STANDARD_REQUIRED               ON
                      POSITION_INDEPENDENT_CODE           ON
                      INTERFACE_POSITION_INDEPENDENT_CODE ON
                      CUDA_STANDARD                       17
                      CUDA_STANDARD_REQUIRED              ON
                      LIBRARY_OUTPUT_DIRECTORY            lib)

target_link_libraries(legate_sparse
   PUBLIC legate::core
          $<TARGET_NAME_IF_EXISTS:NCCL::NCCL>
  PRIVATE
          # Add Conda library and include paths
          $<TARGET_NAME_IF_EXISTS:conda_env>
          $<TARGET_NAME_IF_EXISTS:CUDA::cusparse>
          $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>)

# Change THRUST_DEVICE_SYSTEM for `.cpp` files
if(Legion_USE_OpenMP)
  list(APPEND legate_sparse_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND legate_sparse_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
elseif(NOT Legion_USE_CUDA)
  list(APPEND legate_sparse_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND legate_sparse_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()

target_compile_options(legate_sparse
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${legate_sparse_CXX_OPTIONS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${legate_sparse_CUDA_OPTIONS}>")

target_compile_definitions(legate_sparse
  PUBLIC  "$<$<COMPILE_LANGUAGE:CXX>:${legate_sparse_CXX_DEFS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${legate_sparse_CUDA_DEFS}>")

target_include_directories(legate_sparse
  PRIVATE
    $<BUILD_INTERFACE:${legate_sparse_SOURCE_DIR}/src>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

if(Legion_USE_CUDA)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=])
  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(legate_sparse PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
endif()

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_sparse
        DESTINATION ${lib_dir}
        EXPORT legate-sparse-exports)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide targets for Legate Sparse, an aspiring drop-in replacement for Scipy.Sparse at scale.

Imported Targets:
  - legate_sparse::legate_sparse

]=])

string(JOIN "\n" code_string
  "set(Legion_USE_CUDA ${Legion_USE_CUDA})"
  "set(Legion_USE_OpenMP ${Legion_USE_OpenMP})"
  "set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS})"
)

if(DEFINED Legion_USE_Python)
  string(APPEND code_string "\nset(Legion_USE_Python ${Legion_USE_Python})")
endif()

if(DEFINED Legion_NETWORKS)
  string(APPEND code_string "\nset(Legion_NETWORKS ${Legion_NETWORKS})")
endif()

rapids_export(
  INSTALL legate_sparse
  EXPORT_SET legate-sparse-exports
  GLOBAL_TARGETS legate_sparse
  NAMESPACE legate_sparse::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD legate_sparse
  EXPORT_SET legate-sparse-exports
  GLOBAL_TARGETS legate_sparse
  NAMESPACE legate-sparse::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

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

option(FIND_LEGATE_SPARSE_CPP "Search for existing Legate Sparse C++ installations before defaulting to local files"
       OFF)

##############################################################################
# - Dependencies -------------------------------------------------------------

# If the user requested it we attempt to find legate sparse.
if(FIND_LEGATE_SPARSE_CPP)
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${legate_sparse_version} legate_sparse parsed_ver)
  rapids_find_package(legate_sparse ${parsed_ver} EXACT CONFIG
                      GLOBAL_TARGETS     legate_sparse::legate_sparse # TODO (rohany): Not sure if this target is correct...
                      BUILD_EXPORT_SET   legate-sparse-python-exports
                      INSTALL_EXPORT_SET legate-sparse-python-exports)
else()
  set(legate_sparse_FOUND OFF)
endif()

if(NOT legate_sparse_found_FOUND)
  set(SKBUILD OFF)
  set(Legion_USE_Python ON)
  set(Legion_BUILD_BINDINGS ON)
  add_subdirectory(. "${CMAKE_CURRENT_SOURCE_DIR}/build")
  set(SKBUILD ON)
endif()

add_custom_target("generate_install_info_py" ALL
  COMMAND ${CMAKE_COMMAND}
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_install_info_py.cmake"
  COMMENT "Generate install_info.py"
  VERBATIM
)

execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -E -DLEGATE_USE_PYTHON_CFFI
    -P "${CMAKE_CURRENT_SOURCE_DIR}/src/sparse/sparse_c.h"
  ECHO_ERROR_VARIABLE
  OUTPUT_VARIABLE header
  COMMAND_ERROR_IS_FATAL ANY
)

set(libpath "")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/sparse/install_info.py.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/sparse/install_info.py"
@ONLY)

add_library(legate_sparse_python INTERFACE)
add_library(legate_sparse::legate_sparse_python ALIAS legate_sparse_python)
target_link_libraries(legate_sparse_python INTERFACE legate::core)

##############################################################################
# - install targets ----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_sparse_python
        DESTINATION ${lib_dir}
        EXPORT legate-sparse-python-exports)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide Python targets for Legate Sparse, an aspiring drop-in replacement for Scipy.Sparse at scale.

Imported Targets:
  - legate_sparse::legate_sparse_python

]=])

set(code_string "")

rapids_export(
  INSTALL legate_sparse_python
  EXPORT_SET legate-sparse-python-exports
  GLOBAL_TARGETS legate_sparse_python
  NAMESPACE legate_sparse::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD legate_sparse_python
  EXPORT_SET legate-sparse-python-exports
  GLOBAL_TARGETS legate_sparse_python
  NAMESPACE legate_sparse::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

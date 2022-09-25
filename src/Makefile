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

ifndef LEGATE_DIR
$(error LEGATE_DIR variable is not defined, aborting build)
endif

include $(LEGATE_DIR)/share/legate/config.mk

LIBNAME = liblegatesparse

CC_FLAGS ?=
CC_FLAGS += -I. -I$(TBLIS_PATH)/include -I$(THRUST_PATH)
CC_FLAGS += -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

LD_FLAGS ?=
LD_FLAGS +=
LD_FLAGS +=
ifeq ($(strip $(USE_CUDA)),1)
LD_FLAGS += -lcusparse
LD_FLAGS +=
LD_FLAGS += -L$(NCCL_PATH)/lib -lnccl -Wl,-rpath,$(NCCL_PATH)/lib
endif
NVCC_FLAGS ?=
NVCC_FLAGS += -I. -I$(NCCL_PATH)/include

ifeq ($(strip $(DEBUG)),1)
CC_FLAGS += -DDEBUG_LEGATE_SPARSE
NVCC_FLAGS += -DDEBUG_LEGATE_SPARSE
endif

CHECK_BOUNDS ?= 0
ifeq ($(strip $(CHECK_BOUNDS)),1)
CC_FLAGS += -DBOUNDS_CHECKS
endif

GEN_CPU_SRC = tasks.cc mapper.cc projections.cc quantum.cc
GEN_GPU_SRC = tasks.cu cudalibs.cu sort.cu

ifeq ($(strip $(USE_OPENMP)),1)
GEN_CPU_SRC += tasks_omp.cc
endif

# This must always be the last file! It guarantees we do our registration
# callback only after all task variants are recorded.
GEN_CPU_SRC += sparse.cc

include $(LEGATE_DIR)/share/legate/legate.mk

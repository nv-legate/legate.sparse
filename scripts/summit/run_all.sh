#!/bin/bash

# Print usage if requested.
if [[ $# -ge 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    echo "Usage: $(basename "${BASH_SOURCE[0]}")"
    echo "Arguments read from the environment:"
    echo "  OUT_DIR : Directory to dump output to"
    echo "  DOT : Run the dot microbenchmark (set to 1 to enable)"
    echo "  LEGATE_CPU : Run Legate with CPUs"
    echo "  LEGATE_GPU : Run Legate with GPUs"
    echo "  PETSC_CPU : Run PETSc with CPUs"
    echo "  PETSC_GPU : Run PETSc with GPUs"
    echo "  SCIPY : Run SciPy"
    echo "  CUPY : Run CuPy"
    exit
fi

if [[ ! -n $OUT_DIR ]]; then
    echo "The OUT_DIR variable must be set."
    exit 1
fi

# Initialize the output directory, if it doesn't exist already.
mkdir -p $OUT_DIR

# Location of the PETSc benchmarks on Summit.
export PETSC_DIR=$SCRATCH_DIR/petsc-pde-benchmark
# By default, we'll run each experiment for 12 iterations and take the
# average of the middle 10 executions.
export EXP_ITERS=${EXP_ITERS:-12}

# Default the number of requested sockets and GPUs.
export EXP_SOCKETS=${EXP_SOCKETS:-"1 2 4 8 16 32 64"}
export EXP_GPUS=${EXP_GPUS:-"1 3 6 12 24 48 96 192"}

# Configuration to run the DOT microbenchmark.
if [[ -n $DOT ]]; then
    if [[ -n $LEGATE_CPU ]]; then
        CPU_SOCKETS="$EXP_SOCKETS" ./scripts/summit/run_legate_dot_microbenchmark.sh 2>&1 | tee $OUT_DIR/legate_cpu_dot.out
    fi
    if [[ -n $LEGATE_GPU ]]; then
        GPUS="$EXP_GPUS" ./scripts/summit/run_legate_dot_microbenchmark.sh 2>&1 | tee $OUT_DIR/legate_gpu_dot.out
    fi
    if [[ -n $SCIPY ]]; then
        SCIPY_SOCKETS="1 2" ./scripts/summit/run_scipy_dot_microbenchmark.sh 2>&1 | tee $OUT_DIR/scipy_cpu_dot.out
    fi
    if [[ -n $CUPY ]]; then
        CUPY=1 ./scripts/summit/run_scipy_dot_microbenchmark.sh 2>&1 | tee $OUT_DIR/cupy_gpu_dot.out
    fi
    if [[ -n $PETSC_CPU ]]; then
        CPU_SOCKETS="$EXP_SOCKETS" ./scripts/summit/run_petsc_dot_microbenchmark.sh 2>&1 | tee $OUT_DIR/petsc_cpu_dot.out
    fi
    if [[ -n $PETSC_GPU ]]; then
        GPUS="$EXP_GPUS" ./scripts/summit/run_petsc_dot_microbenchmark.sh 2>&1 | tee $OUT_DIR/petsc_gpu_dot.out
    fi
fi

# TODO (rohany): Add PDE, GMG, Quantum and SparseML.
#!/bin/bash

if [[ ! -n $QUANTUM_DIR ]]; then
    echo "The QUANTUM_DIR variable must be set."
    exit 1
fi

# Activate the correct conda env, just in case.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cupy

EXP_ITERS=${EXP_ITERS:-1}
ITERS=25
COMMON_ARGS="-iters $ITERS -l 9"

# For this experiment, we use the same scaling between CPU
# sockets and GPUs, since we don't have precise control
# over problem sizes.
weak_scale_frac() {
    python3 scripts/summit/quantum_fractions.py $1
}

# Run the benchmark for SciPy.
if [[ -n $SCIPY_SOCKETS ]]; then
    for sockets in $SCIPY_SOCKETS ; do
        cmd="jsrun -n 1 -c $((20 * $sockets)) -b rs python3 $QUANTUM_DIR/demo_integration.py -package scipy $COMMON_ARGS -frac $(weak_scale_frac $sockets) -load-ham $SCRATCH_DIR/rydberg-hams-$(weak_scale_frac $sockets).npz $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "CPU SOCKETS = $sockets:"
            echo $cmd
            eval $cmd
        done
    done
fi

# Run the benchmark for CuPy.
if [[ -n $CUPY_GPUS ]]; then
    export CUPY_CACHE_DIR=/tmp/
    cmd="jsrun -n 1 -c ALL_CPUS -g 1 -b rs python $QUANTUM_DIR/demo_integration.py -package cupy $COMMON_ARGS -frac $(weak_scale_frac 1) -load-ham $SCRATCH_DIR/rydberg-hams-$(weak_scale_frac 1).npz  $ARGS"
    for iter in $(seq 1 $EXP_ITERS); do
        echo "GPUS = 1:"
        echo $cmd
        eval $cmd
    done
fi

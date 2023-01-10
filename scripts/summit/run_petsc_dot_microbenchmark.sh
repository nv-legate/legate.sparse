#!/bin/bash

if [[ ! -n $PETSC_DIR ]]; then
    echo "The PETSC_DIR variable must be set."
    exit 1
fi

EXP_ITERS=${EXP_ITERS:-1}
ITERS=100
SIZE=10000000
COMMON_ARGS="-i $ITERS -nnz-per-row 11"

weak_scale_cpu() {
    # We start scaling for CPUs at the problem size used for 3 GPUs to
    # create comparable plots on Summit.
    python3 -c "print($SIZE * $1 * 3)"
}
weak_scale_gpu() {
    python3 -c "print($SIZE * $1)"
}

# CPU runs.
if [[ -n $CPU_SOCKETS ]]; then
    for sockets in $CPU_SOCKETS; do
        cmd="jsrun -n $(($sockets * 20)) -c 1 -b rs $PETSC_DIR/dot_microbenchmark -n $(weak_scale_cpu $sockets) $COMMON_ARGS $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "CPU SOCKETS = $sockets:"
            echo $cmd
            eval $cmd
        done
    done
fi

# GPU runs.
if [[ -n $GPUS ]]; then
    GPU_ARGS="$COMMON_ARGS -vec_type cuda -mat_type aijcusparse -gpu"
    for gpus in $GPUS; do
        cmd="jsrun -n $gpus -g 1 -c 4 -b rs --smpiargs=\"-gpu\" $PETSC_DIR/dot_microbenchmark $GPU_ARGS -n $(weak_scale_gpu $gpus) $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "GPUS = $gpus:"
            echo $cmd
            eval $cmd
        done
    done
fi
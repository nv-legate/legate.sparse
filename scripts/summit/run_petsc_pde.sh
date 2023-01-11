#!/bin/bash

if [[ ! -n $PETSC_DIR ]]; then
    echo "The PETSC_DIR variable must be set."
    exit 1
fi

ITERS=300
SIZE=6000
EXP_ITERS=${EXP_ITERS:-1}

COMMON_ARGS="-ksp_max_it $ITERS -ksp_type cg -pc_type none -ksp_norm_type unpreconditioned -ksp_atol 1e-10 -ksp_rtol 1e-10"

weak_scale_cpu() {
    # We start scaling for CPUs at the problem size used for 3 GPUs to
    # create comparable plots on Summit.
    python3 -c "import math; print(int($SIZE * math.sqrt($1 * 3)))"
}
weak_scale_gpu() {
    python3 -c "import math; print(int($SIZE * math.sqrt($1)))"
}

# CPU runs.
if [[ -n $CPU_SOCKETS ]]; then
    echo "CPU BENCHMARKS:"
    for sockets in $CPU_SOCKETS; do
        cmd="jsrun -n $(($sockets * 20)) -c 1 -b rs $PETSC_DIR/main -nx $(weak_scale_cpu $sockets) -ny $(weak_scale_cpu $sockets) $COMMON_ARGS $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "CPU SOCKETS = $sockets:"
            echo $cmd
            eval $cmd
        done
    done
fi

# GPU runs.
if [[ -n $GPUS ]]; then
    GPU_ARGS="$COMMON_ARGS -vec_type cuda -mat_type aijcusparse"
    echo "GPU BENCHMARKS:"
    for gpus in $GPUS; do
        cmd="jsrun -n $gpus -g 1 -c 4 -b rs --smpiargs=\"-gpu\" $PETSC_DIR/main $GPU_ARGS -nx $(weak_scale_gpu $gpus) -ny $(weak_scale_gpu $gpus) $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "GPUS = $gpus:"
            echo $cmd
            eval $cmd
        done
    done
fi
#!/bin/bash

if [[ ! -n $PETSC_DIR ]]; then
    echo "The PETSC_DIR variable must be set."
    exit 1
fi

ITERS=300
SIZE=6000

COMMON_ARGS="-ksp_max_it $ITERS -ksp_type cg -pc_type none -ksp_norm_type unpreconditioned -ksp_atol 1e-10 -ksp_rtol 1e-10"

# TODO (rohany): Maybe deduplicate this?
weak_scale() {
    python3 -c "import math; print(int($SIZE * math.sqrt($1)))"
}

# CPU runs.
if [[ -n $CPU_SOCKETS ]]; then
    echo "CPU BENCHMARKS:"
    for sockets in $CPU_SOCKETS; do
        cmd="jsrun -n $(($sockets * 20)) -c 1 -b rs $PETSC_DIR/main -nx $(weak_scale $sockets) -ny $(weak_scale $sockets) $COMMON_ARGS $ARGS"
        echo "CPU SOCKETS = $CPU_SOCKETS:"
        echo $cmd
        eval $cmd
    done
fi

# GPU runs.
if [[ -n $GPUS ]]; then
    GPU_ARGS="$COMMON_ARGS -vec_type cuda -mat_type aijcusparse"
    echo "GPU BENCHMARKS:"
    for gpus in $GPUS; do
        cmd="jsrun -n $gpus -g 1 -c 4 -b rs --smpiargs=\"-gpu\" $PETSC_DIR/main $GPU_ARGS -nx $(weak_scale $gpus) -ny $(weak_scale $gpus) $ARGS"
        echo "GPUS = $GPUS:"
        echo $cmd
        eval $cmd
    done
fi
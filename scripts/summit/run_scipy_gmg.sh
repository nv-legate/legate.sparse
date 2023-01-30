#!/bin/bash
export PYTHONUNBUFFERED=1

# Activate the correct conda env, just in case.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cupy

EXP_ITERS=${EXP_ITERS:-1}
ITERS=250
SIZE=4500
COMMON_ARGS="-m $ITERS"

weak_scale_cpu() {
    # We start scaling for CPUs at the problem size used for 3 GPUs to
    # create comparable plots on Summit.
    python3 -c "import math; print(int($SIZE * math.sqrt($1 * 3)))"
}

# Run the benchmark for SciPy.
if [[ -n $SCIPY_SOCKETS ]]; then
    for sockets in $SCIPY_SOCKETS ; do
        cmd="jsrun -n 1 -c $((20 * $sockets)) -b rs python3 examples/gmg.py -package scipy $COMMON_ARGS -n $(weak_scale_cpu $sockets) $ARGS"
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
    cmd="jsrun -n 1 -c ALL_CPUS -g 1 -b rs python examples/gmg.py -package cupy $COMMON_ARGS -n $SIZE $ARGS"
    for iter in $(seq 1 $EXP_ITERS); do
        echo "GPUS = 1:"
        echo $cmd
        eval $cmd
    done
fi

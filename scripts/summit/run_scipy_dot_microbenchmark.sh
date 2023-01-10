#!/bin/bash
export PYTHONUNBUFFERED=1

# Activate the correct conda env, just in case.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cupy

ITERS=100
SIZE=10000000
COMMON_ARGS="-i $ITERS"

weak_scale() {
    # We start scaling for CPUs at the problem size used for 3 GPUs to
    # create comparable plots on Summit.
    python3 -c "print($SIZE * $1 * 3)"
}

# Run the benchmark for SciPy.
if [[ -n $SCIPY_SOCKETS ]]; then
    for sockets in $SCIPY_SOCKETS ; do
        cmd="jsrun -n 1 -c $((20 * $sockets)) -b rs python3 examples/dot_microbenchmark.py -package scipy $COMMON_ARGS -n $(weak_scale $sockets) $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "CPU SOCKETS = $sockets:"
            echo $cmd
            eval $cmd
        done
    done
fi

# Run the benchmark for CuPy.
if [[ -n $CUPY ]]; then
    export CUPY_CACHE_DIR=/tmp/
    cmd="jsrun -n 1 -c ALL_CPUS -g 1 -b rs python examples/dot_microbenchmark.py -package cupy $COMMON_ARGS -n $SIZE $ARGS"
    for iter in $(seq 1 $EXP_ITERS); do
        echo "GPUS = 1:"
        echo $cmd
        eval $cmd
    done
fi
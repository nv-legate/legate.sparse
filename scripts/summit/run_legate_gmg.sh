#!/bin/bash

# Activate the correct conda env, just in case.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate legate

# Environment configuration.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
export LEGATE_TEST=1
export PYTHONUNBUFFERED=1
export LEGATE_FIELD_REUSE_FREQ=1
EXP_ITERS=${EXP_ITERS:-1}

ITERS=200
SIZE=4500
COMMON_ARGS="--launcher jsrun -m $ITERS"

weak_scale_cpu() {
    # We start scaling for CPUs at the problem size used for 3 GPUs to
    # create comparable plots on Summit.
    python3 -c "import math; print(int($SIZE * math.sqrt($1 * 3)))"
}
weak_scale_gpu() {
    python3 -c "import math; print(int($SIZE * math.sqrt($1)))"
}

# CPU runs.

nodes() {
    python3 -c "print(($1 + 1) // 2)"
}

if [[ -n $CPU_SOCKETS ]]; then
    SYS_MEM=200000
    ranks() {
        python3 -c "print(min($1, 2))"
    }
    BIND_ARGS1="--cpu-bind 0-83 --mem-bind 0 --nic-bind mlx5_0,mlx5_1"
    BIND_ARGS2="--cpu-bind 0-83/88-171 --mem-bind 0/8 --nic-bind mlx5_0,mlx5_1/mlx5_2,mlx5_3"
    bind_args() {
        python3 -c "print('$BIND_ARGS2' if $1 > 1 else '$BIND_ARGS1')"
    }
    # For some reason, we seem less likely to hit a Realm bug if we use less OMP threads.
    OMPTHREADS=14
    UTILITY=2
    for sockets in $CPU_SOCKETS; do
        cmd="legate examples/gmg.py --nodes $(nodes $sockets) --ranks-per-node $(ranks $sockets) -n $(weak_scale_cpu $sockets) --omps 1 --ompthreads $OMPTHREADS --cpus 1 --sysmem $SYS_MEM --utility $UTILITY --eager-alloc-percentage 30 $(bind_args $sockets) $COMMON_ARGS $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "CPU SOCKETS = $sockets:"
            echo $cmd
            eval $cmd
        done
    done
fi

# GPU runs.

GPU_MEM=13500
nodes() {
    python3 -c "print(($1 + 5) // 6)"
}

SYS_MEM=220000
BIND_ARGS1="--cpu-bind 0-83 --mem-bind 0 --gpu-bind 0,1,2 --nic-bind mlx5_0,mlx5_1"
BIND_ARGS2="--cpu-bind 0-83/88-171 --mem-bind 0/8 --gpu-bind 0,1,2/3,4,5 --nic-bind mlx5_0,mlx5_1/mlx5_2,mlx5_3"
ranks() {
    python3 -c "print(2 if $1 > 3 else 1)"
}
gpus_per_node() {
    python3 -c "print(3 if $1 >= 3 else $1)"
}
bind_args() {
    python3 -c "print('$BIND_ARGS2' if $1 > 3 else '$BIND_ARGS1')"
}
# We use 3 OMP procs per rank so that the equivalence set heuristics don't get confused
# when we switch from OMPs to GPUs after the initial build phase.
GPU_ARGS="-cunumeric:preload-cudalibs $COMMON_ARGS --fbmem $GPU_MEM --cpus 1 --sysmem $SYS_MEM --utility 4 --launcher-extra=\"--smpiargs='-disable_gpu_hooks'\" --omps 3 --ompthreads 3 -lg:eager_alloc_percentage 10 -lg:eager_alloc_percentage_override system_mem=25"

if [[ -n $GPUS ]]; then
    for gpus in $GPUS; do
        cmd="legate examples/gmg.py $GPU_ARGS --nodes $(nodes $gpus) --ranks-per-node $(ranks $gpus) --gpus $(gpus_per_node $gpus) $(bind_args $gpus) -n $(weak_scale_gpu $gpus) $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "GPUS = $gpus:"
            echo $cmd
            eval $cmd
        done
    done
fi

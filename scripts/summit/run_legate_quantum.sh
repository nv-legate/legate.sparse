#!/bin/bash

if [[ ! -n $QUANTUM_DIR ]]; then
    echo "The QUANTUM_DIR variable must be set."
    exit 1
fi

# Activate the correct conda env, just in case.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate legate

# Environment configuration.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
export LEGATE_TEST=1
export LEGATE_FIELD_REUSE_FREQ=1
export PYTHONUNBUFFERED=1
EXP_ITERS=${EXP_ITERS:-1}

ITERS=25
COMMON_ARGS="--launcher jsrun -iters $ITERS -l 9"

# For this experiment, we use the same scaling between CPU
# sockets and GPUs, since we don't have precise control
# over problem sizes.
weak_scale_frac() {
    python3 scripts/summit/quantum_fractions.py $1
}

# CPU runs.
nodes() {
    python3 -c "print(($1 + 1) // 2)"
}

if [[ -n $CPU_SOCKETS ]]; then
    SYS_MEM=150000
    ranks() {
        python3 -c "print(min($1, 2))"
    }
    BIND_ARGS1="--cpu-bind 0-83 --mem-bind 0 --nic-bind mlx5_0,mlx5_1"
    BIND_ARGS2="--cpu-bind 0-83/88-171 --mem-bind 0/8 --nic-bind mlx5_0,mlx5_1/mlx5_2,mlx5_3"
    bind_args() {
        python3 -c "print('$BIND_ARGS2' if $1 > 1 else '$BIND_ARGS1')"
    }
    OMPTHREADS=16
    UTILITY=2
    for sockets in $CPU_SOCKETS; do
        cmd="legate $QUANTUM_DIR/demo_integration.py --nodes $(nodes $sockets) --ranks-per-node $(ranks $sockets) -frac $(weak_scale_frac $sockets) --omps 1 --ompthreads $OMPTHREADS --cpus 1 --sysmem $SYS_MEM --utility $UTILITY $(bind_args $sockets) $COMMON_ARGS $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "CPU SOCKETS = $sockets:"
            echo $cmd
            eval $cmd
        done
    done
fi

# GPU runs.
GPU_MEM=14000
nodes() {
    # We're only using 4 GPUs per node.
    python3 -c "print(($1 + 3) // 4)"
}

SYS_MEM=150000
BIND_ARGS1="--cpu-bind 0-83 --mem-bind 0 --gpu-bind 0,1,2 --nic-bind mlx5_0,mlx5_1"
BIND_ARGS2="--cpu-bind 0-83/88-171 --mem-bind 0/8 --gpu-bind 0,1,2/3,4,5 --nic-bind mlx5_0,mlx5_1/mlx5_2,mlx5_3"
ranks() {
    python3 -c "print(2 if $1 > 2 else 1)"
}
gpus_per_node() {
    # Importantly, we're only going to use at most 2 GPUs per rank.
    python3 -c "print(2 if $1 >= 2 else $1)"
}
bind_args() {
    python3 -c "print('$BIND_ARGS2' if $1 > 2 else '$BIND_ARGS1')"
}
GPU_ARGS="-cunumeric:preload-cudalibs $COMMON_ARGS --fbmem $GPU_MEM --cpus 1 --sysmem $SYS_MEM -lg:eager_alloc_percentage 5 -lg:eager_alloc_percentage_override system_mem=50 --utility 2 --launcher-extra=\"--smpiargs='-disable_gpu_hooks'\" --omps 1 --ompthreads 8"

if [[ -n $GPUS ]]; then
    for gpus in $GPUS; do
        cmd="legate $QUANTUM_DIR/demo_integration.py $GPU_ARGS --nodes $(nodes $gpus) --ranks-per-node $(ranks $gpus) --gpus $(gpus_per_node $gpus) $(bind_args $gpus) -frac $(weak_scale_frac $gpus) $ARGS"
        for iter in $(seq 1 $EXP_ITERS); do
            echo "GPUS = $gpus:"
            echo $cmd
            eval $cmd
        done
    done
fi

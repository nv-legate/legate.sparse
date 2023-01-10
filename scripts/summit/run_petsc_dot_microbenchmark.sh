#!/bin/bash

if [[ ! -n $PETSC_DIR ]]; then
	echo "The PETSC_DIR variable must be set."
	exit 1
fi

ITERS=200
SIZE=10000000
COMMON_ARGS="-i $ITERS -nnz-per-row 11"

weak_scale() {
    python3 -c "print($SIZE * $1)"
}

# CPU runs.
if [[ -n $CPU_SOCKETS ]]; then
	for sockets in $CPU_SOCKETS; do
		cmd="jsrun -n $(($sockets * 20)) -c 1 -b rs $PETSC_DIR/dot_microbenchmark -n $(weak_scale $sockets) $COMMON_ARGS $ARGS"
		echo $cmd
		eval $cmd
	done
fi

# GPU runs.
if [[ -n $GPUS ]]; then
	GPU_ARGS="$COMMON_ARGS -vec_type cuda -mat_type aijcusparse -gpu"
	for gpus in $GPUS; do
		cmd="jsrun -n $gpus -g 1 -c 4 -b rs --smpiargs=\"-gpu\" $PETSC_DIR/dot_microbenchmark $GPU_ARGS -n $(weak_scale $gpus) $ARGS"
		echo $cmd
		eval $cmd
	done
fi
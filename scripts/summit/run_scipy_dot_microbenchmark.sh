#!/bin/bash
export PYTHONUNBUFFERED=1

ITERS=100
SIZE=10000000
COMMON_ARGS="-i $ITERS"

weak_scale() {
    python3 -c "print($SIZE * $1)"
}

# Run the benchmark for SciPy.
if [[ -n $SCIPY_SOCKETS ]]; then
	for sockets in $SCIPY_SOCKETS ; do
		cmd="jsrun -n 1 -c $((20 * $sockets)) -b rs python3 examples/dot_microbenchmark.py -package scipy $COMMON_ARGS -n $(weak_scale $sockets) $ARGS"
		echo "CPU SOCKETS = $sockets:"
		echo $cmd
		eval $cmd
	done
fi

# Run the benchmark for CuPy.
if [[ -n $CUPY ]]; then
	export CUPY_CACHE_DIR=/tmp/
	cmd="jsrun -n 1 -c ALL_CPUS -g 1 -b rs python examples/dot_microbenchmark.py -package cupy $COMMON_ARGS -n $SIZE $ARGS"
	echo $cmd
	eval $cmd
fi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=100)
parser.add_argument("-i", type=int, default=25)
parser.add_argument("-op", choices=["spmv", "spmm"], default="spmv")
parser.add_argument("-k", type=int, default=32)
parser.add_argument(
    "-package", choices=["legate", "cupy", "scipy"], default="legate"
)
args, _ = parser.parse_known_args()
n = args.n
iters = args.i
op = args.op

if args.package == "legate":
    import cunumeric as np
    from legate.timing import time

    import sparse

    use_legate = True
else:
    from time import perf_counter_ns

    def time():
        return perf_counter_ns() / 1000.0

    if args.package == "cupy":
        import cupy as np
        import cupyx.scipy.sparse as sparse
    elif args.package == "scipy":
        import numpy as np
        import scipy.sparse as sparse
    use_legate = False

# Create a banded diagonal matrix with 5 diagonals.
A = sparse.diags(
    [1, 1, 1, 1, 1],
    [-2, -1, 0, 1, 2],
    shape=(n, n),
    format="csr",
    dtype=np.float64,
)
if op == "spmv":
    x = np.ones((n,))
    y = np.zeros((n,))
else:
    x = np.ones((n, args.k))
    y = np.zeros((n, args.k))


def f():
    if use_legate:
        global y
        A.dot(x, out=y)
    else:
        y = A.dot(x)


# Run one to warm up the system.
f()

start = time()
for i in range(iters):
    f()
end = time()
total_s = (end - start) / 1000.0 / 1000.0

print(f"Iterations / sec: {iters / total_s}")

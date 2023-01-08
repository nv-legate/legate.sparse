import argparse

from benchmark import parse_common_args

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=100)
parser.add_argument("-i", type=int, default=25)
parser.add_argument("-op", choices=["spmv", "spmm"], default="spmv")
parser.add_argument("-k", type=int, default=32)
args, _ = parser.parse_known_args()
n = args.n
iters = args.i
op = args.op
_, timer, np, sparse, _, use_legate = parse_common_args()

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

timer.start()
for i in range(iters):
    f()
total = timer.stop() / 1000.0

print(f"Iterations / sec: {iters / total}")

import argparse

from benchmark import parse_common_args

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=100)
parser.add_argument("-i", type=int, default=25)
parser.add_argument("-nnz-per-row", type=int, default=11)
args, _ = parser.parse_known_args()
n = args.n
iters = args.i
nnz_per_row = args.nnz_per_row
_, timer, np, sparse, _, use_legate = parse_common_args()

# Create a banded diagonal matrix with nnz_per_row diagonals.
A = sparse.diags(
    [1] * nnz_per_row,
    [x - (nnz_per_row // 2) for x in range(nnz_per_row)],
    shape=(n, n),
    format="csr",
    dtype=np.float64,
)
B = A.copy()


def f():
    C = A @ B
    return C


# Run one to warm up the system.
f()

timer.start()
for i in range(iters):
    f()
total = timer.stop() / 1000.0

print(f"Iterations / sec: {iters / total}")

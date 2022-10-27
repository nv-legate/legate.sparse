# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import cunumeric as np
    from legate.timing import time

    import sparse as sparse
    import sparse.linalg as linalg

    use_legate = True
except (RuntimeError, ImportError):
    from time import perf_counter_ns

    import numpy as np
    import scipy
    import scipy.sparse as sparse
    import scipy.sparse.linalg as linalg

    def time():
        return perf_counter_ns() / 1000.0

    def spmv(A, x, out):
        for i in range(A.shape[0]):
            begin, end = A.indptr[i], A.indptr[i + 1]
            indices = A.indices[begin:end]
            out[i] = max(x[indices].tolist())

    scipy.sparse.csr_matrix.tropical_spmv = spmv
    use_legate = False

import argparse


def stencil_grid(S, grid, dtype=None, format=None):
    N_v = int(np.prod(grid))  # number of vertices in the mesh
    N_s = int((S != 0).sum())  # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = np.cumprod([1] + list(reversed(grid)))[:-1]
    indices = tuple(i.copy() for i in S.nonzero())
    for i, s in zip(indices, S.shape):
        i -= s // 2

    for stride, coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = np.repeat(S[S != 0], N_v).reshape((N_s, N_v))

    indices = np.vstack(indices).T

    # zero boundary connections
    for index, diag in zip(indices, data):
        diag = diag.reshape(grid)
        for n, i in enumerate(index):
            if i > 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(0, i)
                s = tuple(s)
                diag[s] = 0
            elif i < 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(i, None)
                s = tuple(s)
                diag[s] = 0

    # remove diagonals that lie outside matrix
    mask = abs(diags) < N_v
    if not mask.all():
        diags = diags[mask]
        data = data[mask]

    # sum duplicate diagonals
    if len(np.unique(diags)) != len(diags):
        new_diags = np.unique(diags)
        new_data = np.zeros((len(new_diags), data.shape[1]), dtype=data.dtype)

        for dia, dat in zip(diags, data):
            n = np.searchsorted(new_diags, dia)
            new_data[n, :] += dat

        diags = new_diags
        data = new_data

    return sparse.dia_matrix((data, diags), shape=(N_v, N_v)).asformat(format)


def poisson2D(N):
    M = 2
    stencil = np.zeros((3,) * M, dtype=float)
    for i in range(M):
        stencil[(1,) * i + (0,) + (1,) * (M - i - 1)] = -1
        stencil[(1,) * i + (2,) + (1,) * (M - i - 1)] = -1
    stencil[(1,) * M] = 2 * M
    return stencil_grid(stencil, (N, N))


def diffusion2D(N, epsilon=1.0, theta=0.0):
    eps = float(epsilon)  # for brevity
    theta = float(theta)

    C = np.cos(theta)
    S = np.sin(theta)
    CS = C * S
    CC = C**2
    SS = S**2

    a = (-1 * eps - 1) * CC + (-1 * eps - 1) * SS + (3 * eps - 3) * CS
    b = (2 * eps - 4) * CC + (-4 * eps + 2) * SS
    c = (-1 * eps - 1) * CC + (-1 * eps - 1) * SS + (-3 * eps + 3) * CS
    d = (-4 * eps + 2) * CC + (2 * eps - 4) * SS
    e = (8 * eps + 8) * CC + (8 * eps + 8) * SS

    stencil = np.array([[a, b, c], [d, e, d], [c, b, a]]) / 6.0
    return stencil_grid(stencil, (N, N))


def strength(A, theta=0.0):
    if theta == 0:
        return A

    B = abs(A.copy())
    D = B.diagonal()
    B.data *= B.data >= (theta * np.sqrt(D[B.row] * D[B.col]))
    B.eliminate_zeros()

    max_val = B.max(axis=0).data
    B.data /= max_val[B.col]
    return B


def fit_candidates(A, B):
    # TODO (rohany): This COO conversion might be unnecessary.
    Q = A.copy().tocoo()
    Q.data = B.ravel() ** 2

    # WAR since sparse sum return an invalid 'matrix' type
    R = np.sqrt(np.array(Q.sum(axis=0)))

    Q.data /= R.ravel()[Q.col]
    return Q, R


def estimate_spectral_radius(A, maxiter=15):
    x = np.random.rand(A.shape[0])

    for _ in range(maxiter):
        x /= np.linalg.norm(x)
        y = A @ x
        x, y = y, x

    return np.dot(x, y) / np.linalg.norm(y)


def smooth_prolongator(A, T, k=1, omega=4.0 / 3.0, D=None):
    if D is None:
        D = A.diagonal()
    D_inv = 1.0 / D
    A_coo = A.tocoo()
    D_inv_S = A.copy()
    # This is another way of computing the following advanced
    # indexing operation by exploting the properties of the COO matrix.
    # D_inv_S.data *= D_inv[A_coo.row]
    reps = np.bincount(A_coo.row, minlength=D_inv.size)
    assert reps.size == D_inv.size
    assert reps.sum() == D_inv_S.nnz
    D_inv_S.data *= np.repeat(D_inv, reps)

    spectral_radius = estimate_spectral_radius(D_inv_S)
    # TODO (rohany): We don't have the __rmul__ dispatch...
    # D_inv_S = (omega / spectral_radius) * D_inv_S
    D_inv_S = D_inv_S * (omega / spectral_radius)

    # TODO (rohany): This implicit conversion should happen in the CSR matmul
    # operator.
    P = T.tocsr()
    for _ in range(k):
        P = P - (D_inv_S @ P)

    return P, spectral_radius


def maximal_independent_set(C, k=1, invalid=None):
    assert C.shape[0] == C.shape[1]
    N = C.shape[0]

    random_values = np.random.randint(0, np.iinfo(np.int64).max, size=N)

    x = np.array(
        np.vstack([np.ones_like(random_values), random_values, np.arange(N)]).T
    )
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    active_nodes = N
    if invalid is not None:
        x[invalid, 0] = -1
        active_nodes -= invalid.sum()

    C = C.tocsr()
    while True:
        C.tropical_spmv(x, out=z)

        for _ in range(1, k):
            y, z = z, y
            C.tropical_spmv(y, out=z)

        mis_node = np.where((x[:, 0] == 1) & (z[:, 2] == np.arange(N)))[0]
        x[mis_node, 0] = 2

        non_mis_node = np.where((x[:, 0] == 1) & (z[:, 0] == 2))[0]
        x[non_mis_node, 0] = 0

        active_nodes -= len(mis_node) + len(non_mis_node)
        if active_nodes == 0:
            break

        assert (active_nodes > 0) and (active_nodes < N)

    return np.where(x[:, 0] == 2)[0]


def coloring(C):
    N = C.shape[0]

    color = 0
    colors = -np.ones(N, dtype=np.int32)

    num_invalid = 0
    invalid = np.zeros(N, dtype=bool)
    while num_invalid < N:
        mis = maximal_independent_set(C, invalid=invalid)

        colors[mis] = color
        color += 1

        invalid[mis] = True
        num_invalid += len(mis)

    return colors


def mis_aggregate(C):
    C = C.tocsr()
    mis = maximal_independent_set(C, 2)

    N_fine, N_coarse = C.shape[0], mis.size

    # TODO (rohany): Importantly, this can't be a uint32 right now...
    x = np.zeros((N_fine, 2), dtype=np.int64)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    x[mis, 0] = 2
    x[mis, 1] = np.arange(N_coarse)
    C.tropical_spmv(x, out=y)

    y[:, 0] += x[:, 0]
    C.tropical_spmv(y, out=z)

    # TODO (rohany): This data is currently _ok_ as a 32-bit integer because
    #  we cast incoming data to the expected types in the COO constructor.
    data = np.ones(N_fine, dtype=np.uint32)
    row = np.arange(N_fine)
    col = z[:, 1]

    return sparse.coo_matrix((data, (row, col)), shape=(N_fine, N_coarse)), mis


class Level:
    def __init__(self, R=None, A=None, P=None, D=None, B=None, rho_DinvA=None):
        self.R = R
        self.A = A
        self.P = P
        self.D = D
        self.B = B
        self.rho_DinvA = rho_DinvA
        # Cache several workspace allocations on the level.
        self._dense_A = None
        self._residual_alloc = None
        self._coarse_b_alloc = None
        self._coarse_x_alloc = None
        self._coarse_P_alloc = None
        self._presmoother_workspace = None
        self._postsmoother_workspace = None

    @property
    def dense_A(self):
        if self._dense_A is None:
            self._dense_A = self.A.todense()
        return self._dense_A

    @property
    def residual_alloc(self):
        if self._residual_alloc is None:
            self._residual_alloc = np.zeros(self.A.shape[0])
        return self._residual_alloc

    @property
    def coarse_b_alloc(self):
        if self._coarse_b_alloc is None:
            self._coarse_b_alloc = np.zeros(self.R.shape[0])
        return self._coarse_b_alloc

    @property
    def coarse_x_alloc(self):
        if self._coarse_x_alloc is None:
            self._coarse_x_alloc = np.zeros(self.R.shape[0])
        return self._coarse_x_alloc

    @property
    def coarse_P_alloc(self):
        if self._coarse_P_alloc is None:
            self._coarse_P_alloc = np.zeros(self.P.shape[0])
        return self._coarse_P_alloc

    def presmoother(self, x, b, omega=4.0 / 3.0):
        if self._presmoother_workspace is None:
            self._presmoother_workspace = np.zeros(self.D.shape[0])
        workspace = self._presmoother_workspace
        # x[:] = (omega / rho_DinvA) * b / D
        np.divide(omega, self.rho_DinvA, out=workspace)
        np.multiply(workspace, b, out=workspace)
        np.divide(workspace, self.D, out=x)

    def postsmoother(self, x, b, omega=4.0 / 3.0):
        if self._postsmoother_workspace is None:
            self._postsmoother_workspace = np.zeros(self.D.shape[0])
        workspace = self._postsmoother_workspace
        # y = A @ x
        # x += (omega / rho_DinvA) * (b - y) / D
        self.A.dot(x, out=workspace)
        np.subtract(b, workspace, out=workspace)
        np.divide(workspace, self.rho_DinvA, out=workspace)
        np.multiply(omega, workspace, out=workspace)
        np.divide(workspace, self.D, out=workspace)
        x += workspace


def build_hierarchy(A, B, theta=0, max_coarse=10):
    assert B.shape[1] == 1

    levels = [Level(R=None, A=A, P=None, D=None, B=B, rho_DinvA=None)]

    lvl = 0

    # Adding in some type assertions for performance checking.
    if use_legate:
        assert isinstance(A, sparse.csr_array)

    while levels[-1].A.shape[0] > max_coarse:
        A = levels[-1].A
        B = levels[-1].B
        D = A.diagonal()

        # filter connections based on symmetric strength measure
        # not strictly necessary
        C = strength(A, theta=theta)

        # build coarse grid structure based on MIS(2) aggregation
        AggOp, roots = mis_aggregate(C)

        # build coarse grid values based on near-nullspace vectors (B)
        T, B_coarse = fit_candidates(AggOp, B)

        # smooth tentative prolongator to improve convergence
        # not strictly necessary
        P, rho_DinvA = smooth_prolongator(A, T, k=1, D=D)

        # Keep all our matrices in CSR.
        R = P.T.tocsr()

        levels[-1] = Level(R, A, P, D, B, rho_DinvA)

        # Form coarse grid.
        A_coarse = R @ A @ P

        if use_legate:
            assert isinstance(A_coarse, sparse.csr_array)

        levels.append(Level(None, A_coarse, None, None, B_coarse, None))

        lvl += 1

    return levels


def cycle(levels, lvl, x, b):
    A = levels[lvl].A
    levels[lvl].presmoother(x, b)

    # Use workspace allocations for these operations per level.
    # residual = b - A @ x
    # coarse_b = levels[lvl].R @ residual
    # coarse_x = np.zeros_like(coarse_b)
    residual = levels[lvl].residual_alloc
    A.dot(x, out=residual)
    np.subtract(b, residual, out=residual)
    coarse_b = levels[lvl].coarse_b_alloc
    levels[lvl].R.dot(residual, out=coarse_b)
    coarse_x = levels[lvl].coarse_x_alloc

    if lvl == len(levels) - 2:
        np.linalg.solve(levels[-1].dense_A, coarse_b, out=coarse_x)
    else:
        cycle(levels, lvl + 1, coarse_x, coarse_b)

    # x += levels[lvl].P @ coarse_x
    levels[lvl].P.dot(coarse_x, out=levels[lvl].coarse_P_alloc)
    x += levels[lvl].coarse_P_alloc
    levels[lvl].postsmoother(x, b)


def test(A, levels=None, plot=False):
    N = A.shape[0]
    x0 = np.zeros(N)
    b = np.ones(N)

    # To avoid penalizing benchmarking runs, don't record the residuals
    # unless we are plotting data. If we should end up recording all of
    # the residuals, add workspaces to avoid allocating on each iteration.
    residuals = []
    if plot:

        def callback(x):
            r = b - A @ x
            normr = np.linalg.norm(r)
            residuals.append(normr)

    else:
        callback = None

    if levels is not None:
        # Handle in matvec when we're supposed to write our outputs into
        # a given input array.
        def matvec(b, out=None):
            if out is None:
                out = np.zeros_like(b)
            else:
                out.fill(0.0)
            cycle(levels, 0, out, b)
            return out

        M = linalg.LinearOperator(A.shape, matvec=matvec)
        conv_test = 5
    else:
        M = None
        conv_test = 25
    _, iters = linalg.cg(
        A, b=b, x0=x0, M=M, callback=callback, conv_test_iters=conv_test
    )

    return residuals, iters


def operator_complexity(levels):
    return sum(level.A.nnz for level in levels) / float(levels[0].A.nnz)


def grid_complexity(levels):
    return sum(level.A.shape[0] for level in levels) / float(
        levels[0].A.shape[0]
    )


def print_diagnostics(levels):
    """Print basic statistics about the multigrid hierarchy."""
    output = "MultilevelSolver\n"
    output += f"Number of Levels:     {len(levels)}\n"
    output += f"Operator Complexity: {operator_complexity(levels):6.3f}\n"
    output += f"Grid Complexity:     {grid_complexity(levels):6.3f}\n"

    total_nnz = sum(level.A.nnz for level in levels)

    #          123456712345678901 123456789012 123456789
    #               0       10000        49600 [52.88%]
    output += "  level   unknowns     nonzeros\n"
    for n, level in enumerate(levels):
        A = level.A
        ratio = 100 * A.nnz / total_nnz
        output += f"{n:>6} {A.shape[1]:>11} {A.nnz:>12} [{ratio:2.2f}%]\n"

    print(output)


if __name__ == "__main__":
    np.random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("-nodes", type=int, default=64)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-r", "--reference-solve", action="store_true")
    parser.add_argument("-pyamg-init", action="store_true")
    args, _ = parser.parse_known_args()

    num_nodes = args.nodes
    if args.pyamg_init:
        import pyamg

        sten = pyamg.gallery.diffusion_stencil_2d(epsilon=0.1, type="FD")
        A = sparse.csr_array(
            pyamg.gallery.stencil_grid(sten, (args.nodes, args.nodes)).tocsr()
        )
    else:
        # A = poisson2D(num_nodes).tocsr()
        A = diffusion2D(num_nodes, epsilon=0.1, theta=np.pi / 4).tocsr()

    start_build = time()
    B = np.ones((A.shape[0], 1))
    levels = build_hierarchy(A, B)
    end_build = time()
    print_diagnostics(levels)

    start_amg_solve = time()
    amg_residuals, iters = test(A, levels, plot=args.output is not None)
    stop_amg_solve = time()
    build_time = (end_build - start_build) / 1000.0
    solve_time = (stop_amg_solve - start_amg_solve) / 1000.0
    print(f"AMG build time             : {build_time:.3f} ms.")
    print(f"Preconditioned solve time  : {solve_time:.3f} ms.")
    print(
        f"Solver throughput          : {float(iters) / (solve_time / 1000)} "
        "iters/sec."
    )
    print(f"AMG+CG total time          : {build_time + solve_time:.3f} ms.")

    if args.reference_solve or args.output is not None:
        start_normal_solve = time()
        cg_residuals, iters = test(A, plot=args.output is not None)
        end_normal_solve = time()
        print(
            "Normal solve execution time: "
            f"{(end_normal_solve - start_normal_solve) / 1000.0:.3f} ms"
        )

    if args.output is not None:
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        plt.semilogy(amg_residuals, "ob-", label="AMG+CG")
        plt.semilogy(cg_residuals, "og-", label="CG")
        plt.legend()
        plt.savefig(args.output)

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

import argparse
import math
from collections import namedtuple
from time import time_ns as time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg


def reference_M(A, B=None):
    import pyamg

    smoother = ("jacobi", {"omega": 4.0 / 3.0, "iterations": 1})
    ml = pyamg.aggregation.smoothed_aggregation_solver(
        A,
        B=B,
        keep=True,
        improve_candidates=None,
        presmoother=smoother,
        postsmoother=smoother,
    )
    return ml


def reference_solve(ml, A, b, x0):
    M = ml.aspreconditioner()

    residuals = []

    def callback(x):
        r = b - A @ x
        normr = np.linalg.norm(r)
        residuals.append(normr)

    sp.sparse.linalg.cg(A, b=b, x0=x0, M=M, callback=callback)

    return residuals


def trimesh(vertices, indices, labels=False):
    from matplotlib import collections

    vertices, indices = np.asarray(vertices), np.asarray(indices)

    triangles = vertices[indices.ravel(), :].reshape(
        (indices.shape[0], indices.shape[1], 2)
    )
    col = collections.PolyCollection(
        triangles, lw=1, edgecolor="black", facecolor="gray", alpha=0.5
    )

    sub = plt.gca()
    sub.add_collection(col, autolim=True)
    plt.axis("off")
    sub.autoscale_view()


def draw_graph(mesh, P):
    N = int(math.sqrt(mesh.shape[0]))
    grid = np.meshgrid(range(N), range(N))
    V = np.vstack(list(map(np.ravel, grid))).T
    E = np.vstack((mesh.row, mesh.col)).T

    c = ["red" if p == 0 else "green" for p in P]

    plt.figure()
    sub = plt.gca()
    trimesh(V, E, False)
    sub.scatter(V[:, 0], V[:, 1], marker="o", s=400, c=c)

    for i in range(V.shape[0]):
        sub.annotate(str(i), (V[i, 0], V[i, 1]), ha="center", va="center")

    plt.show()


def csr_allclose(a, b, rtol=1e-8, atol=1e-8):
    c = np.abs(np.abs(a - b) - rtol * np.abs(b))
    return c.max() <= atol


def stencil_grid(S, grid, dtype=None, format=None):
    N_v = np.prod(grid)  # number of vertices in the mesh
    N_s = (S != 0).sum()  # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = np.cumprod([1] + list(reversed(grid)))[:-1]
    indices = tuple(i.copy() for i in S.nonzero())
    for i, s in zip(indices, S.shape):
        i -= s // 2

    for stride, coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = S[S != 0].repeat(N_v).reshape(N_s, N_v)

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

    return scipy.sparse.dia_matrix((data, diags), shape=(N_v, N_v)).asformat(
        format
    )


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

    B = abs(A.copy().tocoo())
    D = B.diagonal()
    B.data *= B.data >= (theta * np.sqrt(D[B.row] * D[B.col]))
    B.eliminate_zeros()

    max_val = B.max(axis=0).data
    B.data /= max_val[B.col]
    return B


def fit_candidates(A, B):
    Q = A.copy().tocoo()
    Q.data = B.ravel() ** 2

    # WAR since sparse sum return an invalid 'matrix' type
    R = np.sqrt(np.array(Q.T.sum(axis=1)))
    # R = np.sqrt(Q.T.sum(axis=1))

    Q.data /= R.ravel()[Q.col]
    return Q, R


def estimate_spectral_radius(A, maxiter=15):
    x = np.random.rand(A.shape[0])

    for _ in range(maxiter):
        x /= np.linalg.norm(x)
        y = A @ x
        x, y = y, x

    return np.dot(x, y) / np.linalg.norm(y)


def smooth_prolongator(A, T, k=1, omega=4.0 / 3.0):
    D_inv = 1.0 / A.diagonal()
    D_inv_S = A.copy().tocoo()
    D_inv_S.data *= D_inv[D_inv_S.row]

    spectral_radius = estimate_spectral_radius(D_inv_S)
    D_inv_S = (omega / spectral_radius) * D_inv_S

    P = T
    for _ in range(k):
        P = P - (D_inv_S @ P)

    return P, spectral_radius


def mis_spmv(A, x, y):
    for i in range(A.shape[0]):
        begin, end = A.indptr[i], A.indptr[i + 1]
        indices = A.indices[begin:end]

        # collect elements from x based on the structure of A
        # convert the set to list and compare tuples based on pythonic ordering
        # https://stackoverflow.com/questions/5292303/how-does-tuple-comparison-work-in-python
        # tolist() is the most time consuming operation in entire program
        y[i] = max(x[indices].tolist())


def maximal_independent_set(C, k=1):
    assert C.shape[0] == C.shape[1]
    N = C.shape[0]

    # random_values = np.random.randint(0, np.iinfo(np.uint32).max, size=N)
    random_values = np.random.randint(0, np.iinfo(np.int64).max, size=N)

    x = np.vstack([np.ones_like(random_values), random_values, np.arange(N)]).T
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    C = C.tocsr()
    while True:
        mis_spmv(C, x, z)

        for _ in range(1, k):
            y, z = z, y
            mis_spmv(C, y, z)

        mis_node = np.where((x[:, 0] == 1) & (z[:, 2] == np.arange(N)))[0]
        x[mis_node, 0] = 2

        non_mis_node = np.where((x[:, 0] == 1) & (z[:, 0] == 2))[0]
        x[non_mis_node, 0] = 0

        active_nodes = (x[:, 0] == 1).sum()

        if active_nodes == 0:
            break

    return np.where(x[:, 0] == 2)[0]


def mis_aggregate(C):
    C = C.tocsr()
    mis = maximal_independent_set(C, 2)

    N_fine, N_coarse = C.shape[0], mis.size

    x = np.zeros((N_fine, 2), dtype=np.uint32)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    x[mis, 0] = 2
    x[mis, 1] = np.arange(N_coarse)
    mis_spmv(C, x, y)

    y[:, 0] += x[:, 0]
    mis_spmv(C, y, z)

    data = np.ones(N_fine, dtype=np.uint32)
    row = np.arange(N_fine)
    col = z[:, 1]

    return (
        sp.sparse.coo_matrix((data, (row, col)), shape=(N_fine, N_coarse)),
        mis,
    )


def build_hierarchy(A, B, theta=0, max_coarse=10):
    assert B.shape[1] == 1

    Level = namedtuple("Level", ["R", "A", "P", "D", "B", "rho_DinvA"])
    levels = [Level(R=None, A=A, P=None, D=None, B=B, rho_DinvA=None)]

    lvl = 0

    while levels[-1].A.shape[0] > max_coarse:
        A = levels[-1].A
        B = levels[-1].B

        # filter connections based on symmetric strength measure
        # not strictly necessary
        C = strength(A, theta=theta)

        # build coarse grid structure based on MIS(2) aggregation
        AggOp, roots = mis_aggregate(C)

        # build coarse grid values based on near-nullspace vectors (B)
        T, B_coarse = fit_candidates(AggOp, B)

        # smooth tentative prolongator to improve convergence
        # not strictly necessary
        P, rho_DinvA = smooth_prolongator(A, T, k=1)

        R = P.T

        levels[-1] = Level(R, A, P, A.diagonal(), B, rho_DinvA)

        # form coarse grid
        A_coarse = R @ A @ P

        levels.append(Level(None, A_coarse, None, None, B_coarse, None))

        lvl += 1

    return levels


def presmoother(A, x, b, D, rho_DinvA, omega=4.0 / 3.0):
    x[:] = (omega / rho_DinvA) * b / D


def postsmoother(A, x, b, D, rho_DinvA, omega=4.0 / 3.0):
    y = A @ x
    x += (omega / rho_DinvA) * (b - y) / D


def cycle(levels, lvl, x, b):
    A = levels[lvl].A
    D = levels[lvl].D
    rho_DinvA = levels[lvl].rho_DinvA

    presmoother(A, x, b, D, rho_DinvA)

    residual = b - A @ x
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)

    if lvl == len(levels) - 2:
        coarse_x = np.linalg.solve(levels[-1].A.todense(), coarse_b)
    else:
        cycle(levels, lvl + 1, coarse_x, coarse_b)

    x += levels[lvl].P @ coarse_x

    postsmoother(A, x, b, D, rho_DinvA)


def plot_mis(A):
    mis = maximal_independent_set(A)
    P = np.zeros(A.shape[0])
    P[mis] = 1
    draw_graph(A.tocoo(), P)


def test(A, levels=None):
    N = A.shape[0]
    x0 = np.zeros(N)
    b = np.ones(N)

    residuals = []

    def callback(x):
        r = b - A @ x
        normr = np.linalg.norm(r)
        residuals.append(normr)

    if levels is not None:

        def matvec(b):
            x = np.zeros_like(b)
            cycle(levels, 0, x, b)
            return x

        M = sp.sparse.linalg.LinearOperator(A.shape, matvec=matvec)
    else:
        M = None

    sp.sparse.linalg.cg(A, b=b, x0=x0, M=M, callback=callback, atol=1e-08)

    return residuals


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
        A = sp.sparse.csr_array(
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
    residuals = test(A, levels)
    stop_amg_solve = time()
    plt.semilogy(residuals, "ob-", label="AMG+CG")
    build_time = (end_build - start_build) / 1.0e6
    solve_time = (stop_amg_solve - start_amg_solve) / 1.0e6
    print(f"AMG build time: {build_time} ms.")
    print(f"Preconditioned solve time: {solve_time} ms.")
    print(
        f"Solver throughput: {float(len(residuals)) / (solve_time / 1000)} "
        "iters/sec."
    )
    print(f"AMG+CG total time: {build_time + solve_time} ms.")

    if args.reference_solve or args.output is not None:
        start_normal_solve = time()
        residuals = test(A)
        end_normal_solve = time()
        plt.semilogy(residuals, "og-", label="CG")
        print(
            "Normal solve execution time: "
            f"{(end_normal_solve - start_normal_solve) / 1.0e6} ms"
        )

    if args.output is not None:
        plt.legend()
        plt.show()

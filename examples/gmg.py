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

import numpy
from benchmark import get_phase_procs, parse_common_args


def stencil_grid(S, grid, dtype=None, format=None):
    N_v = int(numpy.prod(grid))  # number of vertices in the mesh
    N_s = int((S != 0).sum())  # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = numpy.cumprod([1] + list(reversed(grid)))[:-1]
    indices = tuple(i.copy() for i in S.nonzero())
    for i, s in zip(indices, S.shape):
        i -= s // 2

    for stride, coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = np.repeat(S[S != 0], N_v).reshape((N_s, N_v))

    indices = np.vstack(indices).T

    # zero boundary connections
    for idx in range(indices.shape[0]):
        # We do this instead of
        #  for index, diag in zip(indices, data):
        # to avoid unnecessary materialization into numpy arrays.
        index = indices[idx, :]
        diag = data[idx, :]
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
    diag_size = N * N - 1
    first = np.full((N - 1), -1.0)
    chunks = np.concatenate([np.zeros(1), first])
    diag_a = np.concatenate(
        [first, np.tile(chunks, (diag_size - (N - 1)) // N)]
    )
    diag_g = -1.0 * np.ones(N * (N - 1))
    diag_c = 4.0 * np.ones(N * N)

    # We construct a sequence of main diagonal elements,
    diagonals = [diag_g, diag_a, diag_c, diag_a, diag_g]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-N, -1, 0, 1, N]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to
    # .toarray() is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
    # d2mat = diags(diagonals, offsets, dtype=np.float64).tocsr()
    # TODO (rohany): We want to have this conversion occur in parallel so that
    #  we can effectively weak scale. Unfortunately, I can't figure out how to
    #  adapt the scipy.sparse DIA->CSC method to work for DIA->CSR conversions.
    #  I made an attempt at using the transpose of the DIA matrix -> CSC -> CSR
    #  via a final transpose, but it turns out the direct implementation of
    #  transpose on DIA matrices uses alot of memory and is slow due to the use
    #  of indirection copies. Since we know that this matrix is symmetric, we
    #  directly use the DIA->CSC conversion, and then take the transpose to get
    #  a CSR matrix back.
    return sparse.diags(diagonals, offsets, dtype=np.float64).tocsc().T


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


def max_eigenvalue(A, iters=15):
    # Compute eigenvector associated with maximum eigenvalue via power
    # iteration.  This is the same as Steven's imp for estimating spectral
    # radius.
    x1 = np.random.rand(A.shape[1]).reshape(-1, 1)
    for _ in range(iters):
        x1 = A @ x1
        x1 /= np.linalg.norm(x1)
    # Compute and return max eigenvalue via Raleigh quotient.
    # This is np.dot(A @ x1, x1) / np.dot(x1, x1)
    # but since x1 is a unit vector, we can assume denominator is 1.
    return np.dot(x1.T, A @ x1).item()


class GMG(object):
    """
    Geometric Multigrid solver for the 2D Poisson problem.

    - Source on correctness of restriction / prolongation operators: [1]
    - Sources on V-cycle algorithm: [1, 2, 3, 4]
    - Source on preconditioned conjugate gradient and Gauss-Seidel smoothing: [4]

    [1] https://www.researchgate.net/publication/220690328_A_Multigrid_Tutorial_2nd_Edition
    [2] https://github.com/pyamg/pyamg
    [3] http://www.cs.columbia.edu/cg/pdfs/28_GPUSim.pdf
    [4] https://netlib.org/utk/people/JackDongarra/PAPERS/HPCG-benchmark.pdf
    """  # noqa: E501

    def __init__(self, A, shape, levels, smoother, gridop, machine):
        self.A = A
        self.shape = shape
        self.N = numpy.product(self.shape)
        self.levels = levels
        self.restriction_op = {
            "injection": injection_operator,
            "linear": linear_operator,
        }[gridop]
        self.smoother = {"symgs": SYMGS, "jacobi": WeightedJacobi}[smoother]()
        self.operators = self.compute_operators(A)
        self.temp = None
        self.machine = machine
        self.proc_kind = machine.preferred_kind

    def compute_operators(self, A):
        operators = []
        dim = self.N
        self.smoother.init_level_params(A, 0)
        for level in range(self.levels):
            R, dim = self.compute_restriction_level(dim)
            P = R.T.tocsr()
            # assert sparse.issparse(P)
            A = R @ A @ P
            # assert sparse.issparse(A)
            self.smoother.init_level_params(A, level + 1)
            operators.append((R, A, P))
        return operators

    def cycle(self, r):
        # Kick off the cycle with the top-level machine.
        with self.machine:
            return self._cycle(self.A, r, 0, self.machine)

    def _cycle(self, A, r, level, machine):
        if level == self.levels - 1:
            return self.smoother.coarse(A, r, None, level=level)
        x = None
        # Do one pre-smoothing iteration.
        R, coarse_A, P = self.operators[level]
        x = self.smoother.pre(A, r, x, level=level)
        # Compute the residual.
        fine_r = r - A.dot(x)

        # Restrict the residual.
        if use_legate:
            coarse_r = R.dot(fine_r, spmv_domain_part=True)
        else:
            coarse_r = R.dot(fine_r)

        # Compute coarse solution using a subset of the machine.
        ratio = (fine_r.shape[0] // coarse_r.shape[0]) // 2
        num_procs = max(machine.count(self.proc_kind) // ratio, 1)
        with machine[:num_procs]:
            coarse_x = self._cycle(
                coarse_A, coarse_r, level + 1, machine[:num_procs]
            )

        fine_x = P @ coarse_x
        x_corrected = x + fine_x
        # Do one post-smoothing iteration.
        return self.smoother.post(A, r, x_corrected, level=level)

    def compute_restriction_level(self, fine_dim):
        return self.restriction_op(fine_dim)

    def linear_operator(self):
        return linalg.LinearOperator(
            self.A.shape, dtype=float, matvec=lambda r: self.cycle(r)
        )


class SYMGS(object):
    def init_level_params(self, A, level):
        pass

    def __call__(self, A, r, x, level):
        if x is None:
            x = np.zeros_like(r)
        symgs_c(A.indptr, A.indices, A.data, x, r)  # noqa: F821
        return x

    pre = post = coarse = __call__


class WeightedJacobi(object):
    def __init__(self, omega=4.0 / 3.0):
        # Basically, similar solution to PyAMG.
        self.level_params = []
        self._init_omega = omega

    def init_level_params(self, A, level):
        D_inv = 1.0 / A.diagonal()
        # We need to create a new sparse matrix with just this modified
        # diagonal of A. sparse.eye doesn't have this nob, but we can take
        # the output of sparse.eye and mess with it to get the matrix
        # that we want.
        D_inv_mat = sparse.eye(
            A.shape[0], n=A.shape[1], dtype=A.dtype, format="csr"
        )
        D_inv_mat.data = 1.0 / D_inv
        spectral_radius = max_eigenvalue(A @ D_inv_mat)
        omega = self._init_omega / spectral_radius
        self.level_params.append((omega, D_inv))
        assert len(self.level_params) - 1 == level

    def __call__(self, A, r, x, level):
        omega, D_inv = self.level_params[level]
        return (1 - omega) * x + omega * (r - A @ x + x / D_inv) * D_inv

    def pre(self, A, r, x, level):
        if x is not None:
            raise Exception("Expected x is None.")
        omega, D_inv = self.level_params[level]
        return omega * r * D_inv

    def post(self, A, r, x, level):
        omega, D_inv = self.level_params[level]
        return x + omega * (r - A @ x) * D_inv

    def coarse(self, A, r, x, level):
        return self.pre(A, r, x, level)
        # return sparse.linalg.spsolve(A, r)


def injection_operator(fine_dim):
    fine_shape = (int(np.sqrt(fine_dim)),) * 2
    coarse_shape = fine_shape[0] // 2, fine_shape[1] // 2
    coarse_dim = numpy.product(coarse_shape)
    Rp = np.arange(coarse_dim + 1)
    Rx = np.ones((coarse_dim,), dtype=np.float64)
    ij = np.arange(coarse_dim, dtype=np.int64)
    i = ij % coarse_shape[1]
    j = ij // coarse_shape[1]
    Rj = 2 * i + 2 * j * coarse_shape[1]
    R = sparse.csr_matrix(
        (Rx, Rj, Rp), shape=(coarse_dim, fine_dim), dtype=np.float64
    )
    return R, coarse_dim


def linear_operator(fine_dim):
    import numpy

    fine_shape = (int(np.sqrt(fine_dim)),) * 2
    coarse_shape = fine_shape[0] // 2, fine_shape[1] // 2
    coarse_dim = np.product(coarse_shape)
    # Construct CSR directly.
    Rp = numpy.empty(coarse_dim + 1, dtype=np.int64)
    # Get an upper bound on the total number of non-zeroes, and construct Rj
    # and Rx based on this bound.  Computing this value exactly is tedious and
    # the extra allocation can be truncated at the end.  We won't need more
    # than 9*coarse_dim rows.
    nnz = 9 * coarse_dim
    Rj = numpy.empty((nnz,), dtype=np.int64)
    Rx = numpy.empty((nnz,), dtype=np.float64)
    p = 0

    def flatten(i, j):
        return i * fine_shape[1] + j

    for ij in range(coarse_dim):
        Rp[ij] = p
        # For linear interpolation,
        # we have 9 points over which to average in the 2d case.
        # The coefficient matrix will act as a stencil operator.
        i, j = (ij // coarse_shape[1]), (ij % coarse_shape[1])
        # Corners.
        # r[2*i-1, 2*j-1] = 1/16
        # r[2*i-1, 2*j+1] = 1/16
        # r[2*i+1, 2*j-1] = 1/16
        # r[2*i+1, 2*j+1] = 1/16
        # Edges.
        # r[2*i, 2*j+1] = 2/16
        # r[2*i, 2*j-1] = 2/16
        # r[2*i-1, 2*j] = 2/16
        # r[2*i+1, 2*j] = 2/16
        # Center.
        # r[2 * i, 2 * j] = 4/16
        # Ensure indices are constructed in order.
        # Assumes row-major ordering.
        if 0 <= 2 * i - 1:
            if 0 <= 2 * j - 1:
                # top-left
                Rj[p], Rx[p] = flatten(2 * i - 1, 2 * j - 1), 1 / 16
                p += 1
            # top-middle
            Rj[p], Rx[p] = flatten(2 * i - 1, 2 * j), 2 / 16
            p += 1
            if 2 * j + 1 < fine_dim:
                # top-right
                Rj[p], Rx[p] = flatten(2 * i - 1, 2 * j + 1), 1 / 16
                p += 1
        if 0 <= 2 * j - 1:
            # middle-left
            Rj[p], Rx[p] = flatten(2 * i, 2 * j - 1), 2 / 16
            p += 1
        # middle-middle
        Rj[p], Rx[p] = flatten(2 * i, 2 * j), 4 / 16
        p += 1
        if 2 * j + 1 < fine_dim:
            # middle-right
            Rj[p], Rx[p] = flatten(2 * i, 2 * j + 1), 2 / 16
            p += 1
        if 2 * i + 1 < fine_dim:
            if 0 <= 2 * j - 1:
                # bottom-left
                Rj[p], Rx[p] = flatten(2 * i + 1, 2 * j - 1), 1 / 16
                p += 1
            # bottom-middle
            Rj[p], Rx[p] = flatten(2 * i + 1, 2 * j), 2 / 16
            p += 1
            if 2 * j + 1 < fine_dim:
                # bottom-right
                Rj[p], Rx[p] = flatten(2 * i + 1, 2 * j + 1), 1 / 16
                p += 1

    Rp[coarse_dim] = p
    Rx, Rj, Rp = np.array(Rx[:p]), np.array(Rj[:p]), np.array(Rp)
    R = sparse.csr_matrix((Rx[:p], Rj[:p], Rp), shape=(coarse_dim, fine_dim))
    return R, coarse_dim


def required_driver_memory(N):
    NN = N * N
    fine_shape = (int(np.sqrt(NN)),) * 2
    coarse_shape = fine_shape[0] // 2, fine_shape[1] // 2
    coarse_dim = numpy.product(coarse_shape)
    nnz = 9 * coarse_dim
    elements = nnz + coarse_dim + 1
    bytes = elements * 8
    mb = bytes / 10**6
    print("Max required driver memory for N=%d is %fMB" % (N, mb))


def execute(N, data, smoother, gridop, levels, maxiter, tol, verbose, timer):
    build, solve = get_phase_procs(use_legate)
    timer.start()
    with build:
        if data == "poisson":
            A = poisson2D(N).tocsr()
            b = np.random.rand(N**2)
        elif data == "diffusion":
            A = diffusion2D(N).tocsr()
            b = np.random.rand(N**2)
        else:
            raise NotImplementedError(data)
        print(f"Data creation time: {timer.stop()} ms")

        assert (
            smoother == "jacobi"
        ), "Only Jacobi smoother is currently supported."

        if verbose:

            def callback(x):
                print(f"Residual: {np.linalg.norm(b-A.matvec(x))}")

        else:
            callback = None

        required_driver_memory(N)
        timer.start()
        mg_solver = GMG(
            A=A,
            shape=(N, N),
            levels=levels,
            smoother=smoother,
            gridop=gridop,
            machine=solve,
        )
        M = mg_solver.linear_operator()
        print(f"GMG init time: {timer.stop()} ms")

    with solve:
        # Warm up the runtime.
        float(
            np.linalg.norm(
                A.dot(
                    np.zeros(
                        A.shape[1],
                    )
                )
            )
        )
        float(
            np.linalg.norm(
                M.matvec(
                    np.zeros(
                        M.shape[1],
                    )
                )
            )
        )
        # Make another call to random here as well.
        float(np.linalg.norm(np.random.rand(b.shape[0])))
        timer.start()
        x, iters = linalg.cg(
            A, b, tol=tol, maxiter=maxiter, M=M, callback=callback
        )
        total = timer.stop()
        if tol <= np.linalg.norm(x):
            print("Converged in %d iterations" % iters)
        else:
            print("Failed to converge in %d iterations" % iters)
        print(f"Solve Time: {total} ms")
        print(f"Iterations / sec: {iters / (total / 1000.0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=16,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        choices=["poisson", "diffusion"],
        type=str,
        default="poisson",
        help="The problem instance to solve.",
    )
    parser.add_argument(
        "-s",
        "--smoother",
        dest="smoother",
        choices=["jacobi", "symgs"],
        type=str,
        default="jacobi",
        help="Smoother to use.",
    )
    parser.add_argument(
        "-g",
        "--gridop",
        dest="gridop",
        choices=["linear", "injection"],
        type=str,
        default="injection",
        help="Intergrid transfer operator to use.",
    )
    parser.add_argument(
        "-l",
        "--levels",
        dest="levels",
        type=int,
        default=2,
        help="Number of multigrid levels.",
    )
    parser.add_argument(
        "-m",
        "--maxiter",
        type=int,
        default=None,
        dest="maxiter",
        help="bound the maximum number of iterations",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="print verbose output",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        dest="tol",
        help="convergence check threshold",
    )

    args, _ = parser.parse_known_args()
    _, timer, np, sparse, linalg, use_legate = parse_common_args()
    execute(**vars(args), timer=timer)

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

import cunumeric as np
import scipy as sp

import sparse as sparse

sparse.csr_array.asfptype = lambda x: x
sparse.csr_array.format = "csr"

sp.sparse._arrays.csr_array = sparse.csr_array
sp.sparse.dia_matrix = sparse.dia_matrix
sp.sparse.isspmatrix = lambda x: True


def symmetric_strength_of_connection(A, theta=0.0):
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
    Q = A.copy().tocoo()
    Q.data = B.ravel() ** 2

    R = np.sqrt(np.array(Q.T.sum(axis=1)))

    Q.data /= R.ravel()[Q.col]
    return Q, R


def estimate_spectral_radius(A, maxiter=15):
    x = np.random.rand(A.shape[0])

    for _ in range(maxiter):
        x /= np.linalg.norm(x)
        y = A @ x
        x, y = y, x

    return np.dot(x, y) / np.linalg.norm(y)


def jacobi_prolongation_smoother(S, T, C, B, omega=4.0 / 3.0, degree=1):
    D_inv = 1.0 / S.diagonal()
    S_coo = S.tocoo()
    D_inv_S = S.copy()
    D_inv_S.data *= D_inv[S_coo.row]

    spectral_radius = estimate_spectral_radius(D_inv_S)
    S.rho_D_inv = spectral_radius
    D_inv_S = D_inv_S * (omega / spectral_radius)

    P = T.tocsr()
    for _ in range(degree):
        P = P - (D_inv_S @ P)

    return P


def maximal_independent_set(C, k=1, invalid=None):
    assert C.shape[0] == C.shape[1]
    N = C.shape[0]

    random_values = np.random.randint(0, np.iinfo(np.int64).max, size=N)

    x = np.vstack([np.ones_like(random_values), random_values, np.arange(N)]).T
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


def PMIS(C):
    C = C.tocsr()
    mis = maximal_independent_set(C, 2)

    N_fine, N_coarse = C.shape[0], mis.size

    x = np.zeros((N_fine, 2), dtype=np.int64)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    x[mis, 0] = 2
    x[mis, 1] = np.arange(N_coarse)
    C.tropical_spmv(x, out=y)

    y[:, 0] += x[:, 0]
    C.tropical_spmv(y, out=z)

    data = np.ones(N_fine, dtype=np.uint32)
    row = np.arange(N_fine)
    col = z[:, 1]

    return sparse.coo_matrix((data, (row, col)), shape=(N_fine, N_coarse)), mis


def jacobi(A, x, b, iterations=1, omega=1.0):
    D = A.diagonal()
    rho_DinvA = A.rho_D_inv
    y = A @ x
    x += (omega / rho_DinvA) * (b - y) / D


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


def patch(pyamg):
    from .patcher import patch_all_symbol_imports

    patchers = patch_all_symbol_imports(
        pyamg.strength.symmetric_strength_of_connection, skip_substring="test"
    )
    for patcher in patchers:
        mock_request = patcher.start()
        mock_request.side_effect = symmetric_strength_of_connection

    patchers = patch_all_symbol_imports(
        pyamg.aggregation.standard_aggregation, skip_substring="test"
    )
    for patcher in patchers:
        mock_request = patcher.start()
        mock_request.side_effect = PMIS

    patchers = patch_all_symbol_imports(
        pyamg.aggregation.fit_candidates, skip_substring="test"
    )
    for patcher in patchers:
        mock_request = patcher.start()
        mock_request.side_effect = fit_candidates

    patchers = patch_all_symbol_imports(
        pyamg.aggregation.jacobi_prolongation_smoother, skip_substring="test"
    )
    for patcher in patchers:
        mock_request = patcher.start()
        mock_request.side_effect = jacobi_prolongation_smoother

    patchers = patch_all_symbol_imports(
        pyamg.relaxation.relaxation.jacobi, skip_substring="test"
    )
    for patcher in patchers:
        mock_request = patcher.start()
        mock_request.side_effect = jacobi

    patchers = patch_all_symbol_imports(
        pyamg.gallery.stencil_grid, skip_substring="test"
    )
    for patcher in patchers:
        mock_request = patcher.start()
        mock_request.side_effect = stencil_grid

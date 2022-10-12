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

# This PDE solving application is derived from
# https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html.

import argparse
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-nx", type=int, default=101)
parser.add_argument("-ny", type=int, default=101)
parser.add_argument("-plot", action="store_true")
parser.add_argument("-plot_filename", default=None, type=str)
parser.add_argument("-throughput", action="store_true")
parser.add_argument("-max_iter", type=int, default=None)
parser.add_argument("-package", type=str, default="legate", choices=["legate", "cupy", "scipy"])
args, _ = parser.parse_known_args()
if args.throughput and args.max_iter is None:
    print("Must provide -max_iter when using -throughput.")
    sys.exit(1)

# Handle imports depending on what package we're supposed to use.
if args.package == "legate":
    from sparse import diags, linalg
    import cunumeric as np
    from legate.timing import time
else:
    from time import perf_counter_ns
    def time():
        return perf_counter_ns() / 1000.0
    if args.package == "cupy":
        import cupy as np
        from cupyx.scipy.sparse import diags, linalg
    elif args.package == "scipy":
        from scipy.sparse import diags, linalg
        import numpy as np


# Grid parameters.
nx = args.nx              # number of points in the x direction
ny = args.ny              # number of points in the y direction
xmin, xmax = 0.0, 1.0     # limits in the x direction
ymin, ymax = -0.5, 0.5    # limits in the y direction
lx = xmax - xmin          # domain length in the x direction
ly = ymax - ymin          # domain length in the y direction
dx = lx / (nx-1)          # grid spacing in the x direction
dy = ly / (ny-1)          # grid spacing in the y direction

# Create the gridline locations and the mesh grid;
# see notebook 02_02_Runge_Kutta for more details
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
# We pass the argument `indexing='ij'` to np.meshgrid
# as x and y should be associated respectively with the
# rows and columns of X, Y.
X, Y = np.meshgrid(x, y, indexing='ij')

# Compute the rhs. Note that we non-dimensionalize the coordinates
# x and y with the size of the domain in their respective dire-
# ctions.
b = (np.sin(np.pi*X)*np.cos(np.pi*Y)
     + np.sin(5.0*np.pi*X)*np.cos(5.0*np.pi*Y))

# b is currently a 2D array. We need to convert it to a column-major
# ordered 1D array. This is done with the flatten numpy function.
# We use the parameter 'F' to specify that we want want column-major
# ordering. The letter 'F' is used because this is the natural
# ordering of the popular Fortran language. For row-major
# ordering you can pass 'C' as paremeter, which is the natural
# ordering for the C language.
# More info
# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
bflat = b[1:-1, 1:-1].flatten('F')

# Allocate array for the (full) solution, including boundary values
p = np.empty((nx, ny))

def d2_mat_dirichlet_2d(nx, ny, dx, dy):
    """
    Constructs the matrix for the centered second-order accurate
    second-order derivative for Dirichlet boundary conditions in 2D

    Parameters
    ----------
    nx : integer
        number of grid points in the x direction
    ny : integer
        number of grid points in the y direction
    dx : float
        grid spacing in the x direction
    dy : float
        grid spacing in the y direction

    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order deri-
        vative with Dirichlet boundary conditions
    """
    a = 1.0 / dx**2
    g = 1.0 / dy**2
    c = -2.0*a - 2.0*g

    # The below is a slightly inefficient (but full cunumeric) implementation
    # of the following python code to construct the input diagonal. We can't
    # use this code right now because cunumeric doesn't support strided slicing.
    # diag_a = a * numpy.ones((nx-2)*(ny-2)-1)
    # diag_a[nx-3::nx-2] = 0.0
    diag_size = (nx - 2) * (ny - 2) - 1
    first = np.full((nx - 3), a)
    chunks = np.concatenate([np.zeros(1), first])
    diag_a = np.concatenate([first, np.tile(chunks, (diag_size - (nx - 3)) // (nx - 2))])
    diag_g = g * np.ones((nx-2)*(ny-3))
    diag_c = c * np.ones((nx-2)*(ny-2))

    # We construct a sequence of main diagonal elements,
    diagonals = [diag_g, diag_a, diag_c, diag_a, diag_g]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-(nx-2), -1, 0, 1, nx-2]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to .toarray()
    # is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
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
    d2mat = diags(diagonals, offsets, dtype=np.float64).tocsc().T

    # Return the final array
    return d2mat


def p_exact_2d(X, Y):
    """Computes the exact solution of the Poisson equation in the domain
    [0, 1]x[-0.5, 0.5] with rhs:
    b = (np.sin(np.pi * X) * np.cos(np.pi * Y) +
    np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y))

    Parameters
    ----------
    X : numpy.ndarray
        array of x coordinates for all grid points
    Y : numpy.ndarray
        array of y coordinates for all grid points

    Returns
    -------
    sol : numpy.ndarray
        exact solution of the Poisson equation
    """

    sol = (-1.0/(2.0*np.pi**2)*np.sin(np.pi*X)*np.cos(np.pi*Y)
           - 1.0/(50.0*np.pi**2)*np.sin(5.0*np.pi*X)*np.cos(5.0*np.pi*Y))

    return sol


A = d2_mat_dirichlet_2d(nx, ny, dx, dy)
# Warm up the runtime and legate by performing an SpMV on A
# before timing. This makes sure that any deppart operations
# using A are completed before timing.
_ = A.dot(np.zeros((A.shape[1],)))
start = time()
# If we're testing throughput, run only the prescribed number of iterations.
if args.throughput:
    p_sol, iters = linalg.cg(A, bflat, tol=1e-10, maxiter=args.max_iter)
else:
    p_sol, iters = linalg.cg(A, bflat, tol=1e-10)
    assert(np.allclose((A @ p_sol), bflat))
stop = time()
total = (stop - start) / 1000.0
if args.throughput:
    print(f"Iterations / sec: {args.max_iter / (total / 1000.0)}")
    sys.exit(0)
else:
    print(f"Total time: {total} ms")
pvec = np.reshape(p_sol, (nx-2, ny-2), order='F')

# Construct the full solution and apply boundary conditions
p[1:-1, 1:-1] = pvec
p[0, :] = 0
p[-1, :] = 0
p[:, 0] = 0
p[:, -1] = 0

p_e = p_exact_2d(X, Y)

print(f"Iterative method error: {np.sqrt(np.sum((p - p_e) ** 2))}")

if args.plot:
    assert(args.plot_filename is not None)
    plt.switch_backend("Agg")
    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(16,5))
    # We shall now use the
    # matplotlib.pyplot.contourf function.
    # As X and Y, we pass the mesh data.
    #
    # For more info
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contourf.html
    #
    ax_1.contourf(X, Y, p_e, 20)
    ax_2.contourf(X, Y, p, 20)

    # plot along the line y=0:
    jc = int(ly/(2*dy))
    ax_3.plot(x, p_e[:,jc], '*', color='red', markevery=2, label=r'$p_e$')
    ax_3.plot(x, p[:,jc], label=r'$p$')

    # add some labels and titles
    ax_1.set_xlabel(r'$x$')
    ax_1.set_ylabel(r'$y$')
    ax_1.set_title('Exact solution')

    ax_2.set_xlabel(r'$x$')
    ax_2.set_ylabel(r'$y$')
    ax_2.set_title('Numerical solution')

    ax_3.set_xlabel(r'$x$')
    ax_3.set_ylabel(r'$p$')
    ax_3.set_title(r'$p(x,0)$')

    ax_3.legend()
    plt.savefig(args.plot_filename)

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
    import pyamg
    from pyamg_to_legate.wrapper import patch

    patch(pyamg)
    from legate.timing import time
except (RuntimeError, ImportError):
    from time import perf_counter_ns

    import numpy as np
    import pyamg

    def time():
        return perf_counter_ns() / 1000.0


from scipy.sparse.linalg import cg

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-nodes", type=int, default=64)
    args, _ = parser.parse_known_args()

    sten = pyamg.gallery.diffusion_stencil_2d(epsilon=0.1, type="FD")
    A = pyamg.gallery.stencil_grid(
        sten, (args.num_nodes, args.num_nodes)
    ).tocsr()

    smoother = ("jacobi", {"omega": 4.0 / 3.0, "iterations": 1})
    B = np.ones((A.shape[0], 1))

    class coarse_solver(object):
        def __call__(self, A, b):
            return np.linalg.solve(A.todense(), b)

        def __repr__(self):
            return "numpy.linalg.solve"

    start_build = time()
    ml = pyamg.aggregation.smoothed_aggregation_solver(
        A,
        B=B,
        improve_candidates=None,
        presmoother=smoother,
        postsmoother=smoother,
        symmetry="symmetric",
        coarse_solver=coarse_solver(),
    )
    end_build = time()
    build_time = (end_build - start_build) / 1000.0
    print(ml)

    b = np.ones((A.shape[0]))
    M = ml.aspreconditioner(cycle="V")
    iters = 0

    def callback(_):
        global iters
        iters += 1

    start_amg_solve = time()
    x, info = cg(A, b, tol=1e-8, maxiter=30, M=M, callback=callback)
    stop_amg_solve = time()
    solve_time = (stop_amg_solve - start_amg_solve) / 1000.0
    print(f"AMG build time             : {build_time:.3f} ms.")
    print(f"Preconditioned solve time  : {solve_time:.3f} ms.")
    print(
        f"Solver Throughput          : {float(iters) / (solve_time / 1000.0)}"
        f" iters/sec."
    )
    print(f"AMG+CG total time          : {build_time + solve_time:.3f} ms.")

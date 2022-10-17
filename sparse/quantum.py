# This file contains utility functions for constructing the input of
# the rydberg simulation in parallel. It should in principle be a separate
# legate library just for the simulation, but setting up new legate libraries
# is very annoying. Instead, we'll just stick it in here and reuse alot of
# the infrastructure that we already have set up.

import cunumeric as np
import networkx as nx
import pyarrow as pa
from legate.core import Rect, ReductionOp
from legate.core.launcher import Broadcast, TaskLauncher
from legate.core.partition import DomainPartition, Tiling
from legate.core.shape import Shape

from .array import (
    CompressedBase,
    coord_ty,
    csr_array,
    factor_int,
    get_store_from_cunumeric_array,
    nnz_ty,
    store_to_cunumeric_array,
)
from .config import SparseOpCode, domain_ty, types
from .runtime import ctx, runtime


class LegateHamiltonianDriver:
    def __init__(
        self,
        energies: tuple = (1,),
        graph: nx.Graph = None,
        dtype=np.complex64,
    ):
        """
        Default is that the first element in transition is the higher energy s.
        """
        self.energies = energies

        # This method creates a sparse matrix representing interactions between
        # different hamiltonian groups. For each possible excitation state, it
        # adds an edge in the graph between the state and all states that are 1
        # excitation behind that state. The original version of this code
        # directly constructed the sparse matrix as a COO matrix and converted
        # it to CSR. The problem we ran into was the deferred memory usage,
        # especially around the sort.  To alleviate this problem, we take
        # advantage of several properties:
        # * The matrix is symmetric.
        # * Each excitation size adds coordinates that are logically "in a
        #   different zone" than the previous excitation size.
        # Thus, we only need to sort each group of coordinates for a particular
        # excitation size, rather than all of them at once. To make this
        # possible, we rely on the symmetricity -- the symmetricity allows us
        # to consider the lower and upper halves of the matrix independently,
        # and then add them together to get the final matrix. Without
        # symmetricity, we would not be able to apply the group-wise sorting
        # approach.

        self.ip = [1]
        # Maintain the upper and lower halves of the matrix.
        rows_lower_group, cols_lower_group = [], []
        rows_upper_group, cols_upper_group = [], []
        sets, nbrs = None, None
        # Maintain ID offsets for the previous and current sets.
        prev_offset, offset = 0, 1
        for k in range(1, graph.number_of_nodes()):
            # Find independence sets of size k.
            prev_sets = sets
            sets, nbrs = enumerate_independent_sets(
                graph, k, prevk_sets=sets, prevk_queues=nbrs
            )
            # Remember the number of independence sets that we found.
            self.ip.append(sets.shape[0])
            # Use the independence sets of size k and k - 1
            # to build the hamiltonian coordinates. Create unbound
            # stores for the upper and lower halves of the matrix.
            rows_lower = ctx.create_store(coord_ty, ndim=1)
            cols_lower = ctx.create_store(coord_ty, ndim=1)
            rows_upper = ctx.create_store(coord_ty, ndim=1)
            cols_upper = ctx.create_store(coord_ty, ndim=1)

            # Here we have to play some tricks with partitioning to make
            # sure that we don't blow out the memory of the machine by
            # requiring any region to be fully materialized in one particular
            # memory. Since the dependency pattern of the hamiltonian creation
            # is that the entire previous set of hamiltonians is needed to
            # compute the next piece, we do a 2-D replication strategy, similar
            # to the 3-D matrix-multiply algorithms, where data is somewhat
            # replicated but still partitioned across one dimension of the
            # processor grid. In particular, we create a grid where the x
            # dimension partitions the current sets and the y-dimension
            # partitions the prior sets, and then we launch over the full
            # grid, where each processor takes the corresponding pieces
            # of its x and y dimensions. Notably, we don't do this replication
            # algorithm for k = 1, and let the auto-parallelizer kick in for
            # that simple case.
            num_procs = runtime.num_procs
            xdim, ydim = factor_int(num_procs)
            # Always try to make the x-dimension of the grid larger than the y.
            xdim, ydim = (xdim, ydim) if xdim > ydim else (ydim, xdim)
            grid = Shape((xdim, ydim))
            sets_tiling = Shape(((sets.shape[0] + grid[0] - 1) // grid[0], 1))
            prev_sets_tiling = None
            if prev_sets is not None:
                prev_sets_tiling = Shape(
                    ((prev_sets.shape[0] + grid[1] - 1) // grid[1], 1)
                )

            rows_proj_fn = runtime.get_1d_to_2d_functor_id(
                grid[0], grid[1], True
            )  # True is for the rows.
            cols_proj_fn = runtime.get_1d_to_2d_functor_id(
                grid[0], grid[1], False
            )  # False is for the cols.
            flattened_grid = Shape((num_procs,))

            # Compute the lower.
            if k == 1:
                task = ctx.create_task(SparseOpCode.CREATE_HAMILTONIANS)
                task.add_scalar_arg(graph.number_of_nodes(), types.int32)
                task.add_scalar_arg(k, types.int32)
                task.add_scalar_arg(offset, types.uint64)
                task.add_scalar_arg(True, bool)  # Lower.
                task.add_input(sets)
                task.add_output(rows_lower)
                task.add_output(cols_lower)
            else:
                # Since we're using output regions, we need to launch
                # over a 1 dimensional grid and use a projection functor
                # to reconstruct the grid.
                task = ctx.create_manual_task(
                    SparseOpCode.CREATE_HAMILTONIANS,
                    launch_domain=Rect(hi=flattened_grid),
                )
                task.add_scalar_arg(graph.number_of_nodes(), types.int32)
                task.add_scalar_arg(k, types.int32)
                task.add_scalar_arg(offset, types.uint64)
                task.add_scalar_arg(True, bool)  # Lower.
                # To partition a 1-D object along 2 dimensions, we need
                # to promote it into a two dimensional object as well.
                task.add_input(
                    sets.promote(1).partition(Tiling(sets_tiling, grid)),
                    proj=rows_proj_fn,
                )
                task.add_output(rows_lower)
                task.add_output(cols_lower)
                task.add_input(
                    prev_sets.promote(1).partition(
                        Tiling(prev_sets_tiling, grid)
                    ),
                    proj=cols_proj_fn,
                )
                task.add_scalar_arg(prev_offset, types.uint64)
            task.execute()

            # Compute the upper.
            if k == 1:
                task = ctx.create_task(SparseOpCode.CREATE_HAMILTONIANS)
                task.add_scalar_arg(graph.number_of_nodes(), types.int32)
                task.add_scalar_arg(k, types.int32)
                task.add_scalar_arg(offset, types.uint64)
                task.add_scalar_arg(False, bool)  # Upper.
                task.add_input(sets)
                task.add_output(rows_upper)
                task.add_output(cols_upper)
            else:
                task = ctx.create_manual_task(
                    SparseOpCode.CREATE_HAMILTONIANS,
                    launch_domain=Rect(hi=flattened_grid),
                )
                task.add_scalar_arg(graph.number_of_nodes(), types.int32)
                task.add_scalar_arg(k, types.int32)
                task.add_scalar_arg(offset, types.uint64)
                task.add_scalar_arg(False, bool)  # Upper.
                task.add_input(
                    sets.promote(1).partition(Tiling(sets_tiling, grid)),
                    proj=rows_proj_fn,
                )
                task.add_output(rows_upper)
                task.add_output(cols_upper)
                task.add_input(
                    prev_sets.promote(1).partition(
                        Tiling(prev_sets_tiling, grid)
                    ),
                    proj=cols_proj_fn,
                )
                task.add_scalar_arg(prev_offset, types.uint64)
            task.execute()

            prev_offset = offset
            offset += int(sets.shape[0])

            # As with enumerate_independent_sets, we don't want Legate
            # to be using the resulting Weighted partition for anything.
            reset_output_store_partition(rows_lower)
            reset_output_store_partition(cols_lower)
            reset_output_store_partition(rows_upper)
            reset_output_store_partition(cols_upper)

            # Next, sort each chunk of coordinates for the upper and lower
            # halves.  We reset the partitions in between each call so that we
            # evenly distribute future operations across all memories. See the
            # comment below about why np.flip is applied. We do it here to
            # avoid accumulating too much extra space. Unfortunately, np.flip
            # is not implemented with distributed support, so we can't use it.
            # We get around this by just making all of the coordinates negative
            # before sorting, then un-doing the operation after the sort to get
            # the reverse order.

            rows_lower = get_store_from_cunumeric_array(
                -1 * store_to_cunumeric_array(rows_lower)
            )
            cols_lower = get_store_from_cunumeric_array(
                -1 * store_to_cunumeric_array(cols_lower)
            )
            rows_lower, cols_lower = sort_by_key(rows_lower, cols_lower)
            reset_output_store_partition(rows_lower)
            reset_output_store_partition(cols_lower)
            rows_lower = get_store_from_cunumeric_array(
                -1 * store_to_cunumeric_array(rows_lower)
            )
            cols_lower = get_store_from_cunumeric_array(
                -1 * store_to_cunumeric_array(cols_lower)
            )
            rows_lower_group.append(store_to_cunumeric_array(rows_lower))
            cols_lower_group.append(store_to_cunumeric_array(cols_lower))

            rows_upper = get_store_from_cunumeric_array(
                -1 * store_to_cunumeric_array(rows_upper)
            )
            cols_upper = get_store_from_cunumeric_array(
                -1 * store_to_cunumeric_array(cols_upper)
            )
            rows_upper, cols_upper = sort_by_key(rows_upper, cols_upper)
            reset_output_store_partition(rows_upper)
            reset_output_store_partition(cols_upper)
            rows_upper = get_store_from_cunumeric_array(
                -1 * store_to_cunumeric_array(rows_upper)
            )
            cols_upper = get_store_from_cunumeric_array(
                -1 * store_to_cunumeric_array(cols_upper)
            )
            rows_upper_group.append(store_to_cunumeric_array(rows_upper))
            cols_upper_group.append(store_to_cunumeric_array(cols_upper))

            # If there is no frontier to explore, then we're done.
            if np.sum(sets_to_sizes(nbrs, graph)) == 0:
                break

        self.nstates = int(np.sum(self.ip))

        # After we've constructed all of the coordinates, we back-offset them
        # by coord' = (self.nstates - 1) - coord. As a result, everything is
        # backwards, since we started with low coordinates at the front of each
        # coordinate group, and sorted each in ascending order. We undo this
        # operation by concatenating each group of coordinates in reverse, and
        # reversing each array of coordinates.

        rows_upper = (self.nstates - 1) - np.concatenate(
            list(reversed(rows_upper_group))
        )
        cols_upper = (self.nstates - 1) - np.concatenate(
            list(reversed(cols_upper_group))
        )
        vals_upper = np.ones(rows_upper.shape[0], dtype=np.float64)

        rows_lower = (self.nstates - 1) - np.concatenate(
            list(reversed(rows_lower_group))
        )
        cols_lower = (self.nstates - 1) - np.concatenate(
            list(reversed(cols_lower_group))
        )
        vals_lower = np.ones(rows_lower.shape[0], dtype=np.float64)

        # Next, directly create CSR matrices, instead of going through the
        # constructor that implicitly performs a global sort.
        upper = raw_create_csr(
            rows_upper, cols_upper, vals_upper, (self.nstates, self.nstates)
        )
        lower = raw_create_csr(
            rows_lower, cols_lower, vals_lower, (self.nstates, self.nstates)
        )

        # Finally, the resulting matrix is the sum of the upper and lower
        # halves.
        self._hamiltonian = upper + lower

    @property
    def hamiltonian(self):
        if self.energies[0] == 1:
            return self._hamiltonian
        return self.energies[0] * self._hamiltonian


class LegateHamiltonianMIS(object):
    def __init__(
        self, graph: nx.Graph, poly=None, energies=(1, 1), dtype=np.complex64
    ):
        """
        Generate a vector corresponding to the diagonal of the MIS Hamiltonian.
        """
        if energies == (1, 1):
            energies = (1,)
        self.graph = graph
        self.n = self.graph.number_of_nodes()
        self.energies = energies
        self.optimization = "max"
        self._is_diagonal = True
        self.nstates = int(np.sum(poly))
        self.dtype = dtype

        # This code does the same thing as the below code in the special case
        # of equations that we are considering. In particular, the below code
        # constructs an array of consecutive integers where each integer should
        # appear the number of times corresponding to the independence
        # polynomial. We use arange and repeat to achieve that.
        levels = np.arange(len(poly))
        C = np.array(np.flip(np.repeat(levels, poly)).astype(self.dtype))
        enum_states = np.arange(self.nstates)
        self._hamiltonian = csr_array(
            (C, (enum_states, enum_states)), shape=(self.nstates, self.nstates)
        )

        # These are your independent sets of the original graphs, ordered by
        # node and size
        #
        # node_weights = []
        # for i in range(self.graph.number_of_nodes()):
        #     if hasattr(self.graph.nodes[i], 'weight'):
        #         node_weights.append(self.graph.nodes[i]['weight'])
        #     else:
        #         node_weights.append(1)
        # import numpy
        # node_weights = numpy.asarray(node_weights)
        # sets = enumerate_independent_sets(self.graph)
        # # Generate a list of integers corresponding to the independent sets
        # # in binary
        # # All ones
        # k = self.nstates - 2
        # self.mis_size = len(poly) - 1
        # C = numpy.zeros(self.nstates, dtype=self.dtype)
        # C[-1] = 0
        # for i in sets:
        #     C[k] = numpy.sum([node_weights[j] for j in i])
        #     k -= 1
        # print(C)
        # C = np.array(C)
        # assert(np.array_equal(result, C))
        # enum_states = np.arange(self.nstates)
        # self._hamiltonian = sparse.csr_matrix(
        #   (C, (enum_states, enum_states)), shape=(self.nstates, self.nstates)
        # )

    @property
    def hamiltonian(self):
        if self.energies[0] == 1:
            return self._hamiltonian
        return self.energies[0] * self._hamiltonian

    @property
    def _diagonal_hamiltonian(self):
        return self.hamiltonian.data.reshape(-1, 1)

    @property
    def optimum(self):
        # This needs to be recomputed because the optimum depends on the
        # energies
        return np.max(self._diagonal_hamiltonian.real)

    @property
    def minimum_energy(self):
        # This needs to be recomputed because it depends on the energies
        return np.min(self._diagonal_hamiltonian.real)

    def cost_function(self, state):
        # Returns <s|C|s>
        return np.real(
            np.matmul(np.conj(state).T, self._diagonal_hamiltonian * state)
        )

    def optimum_overlap(self, state):
        # Returns \sum_i <s|opt_i><opt_i|s>
        optimum_indices = np.argwhere(
            self._diagonal_hamiltonian == self.optimum
        ).T[0]
        # Construct an operator that is zero everywhere except at the optimum
        optimum = np.zeros(self._diagonal_hamiltonian.shape)
        optimum[optimum_indices] = 1
        return np.real(np.matmul(np.conj(state).T, optimum * state))

    def approximation_ratio(self, state):
        # Returns <s|C|s>/optimum
        return self.cost_function(state) / self.optimum


def reset_output_store_partition(store):
    if store.shape[0] < runtime.num_procs:
        store.reset_key_partition()
    else:
        num_procs = runtime.num_procs
        tile_size = (store.shape[0] + num_procs - 1) // num_procs
        part = store.partition_by_tiling(tile_size).partition
        store.set_key_partition(part)


registered_set_types = {}
set_bit_ty = types.uint8
set_type_id = 1000


def get_set_ty(n):
    if n in registered_set_types:
        return registered_set_types[n]
    bits = set_bit_ty.bit_width
    num_elems = (n + bits - 1) // bits
    fields = [(f"f{i}", set_bit_ty) for i in range(num_elems)]
    ty = pa.struct(fields)
    ctx.type_system.add_type(
        ty,
        num_elems * set_bit_ty.bit_width // 8,
        set_type_id + len(registered_set_types),
    )
    registered_set_types[n] = ty
    return ty


def sets_to_sizes(sets, graph):
    result = ctx.create_store(types.uint64, shape=sets.shape)
    task = ctx.create_task(SparseOpCode.SETS_TO_SIZES)
    task.add_input(sets)
    task.add_output(result)
    task.add_alignment(sets, result)
    task.add_scalar_arg(graph.number_of_nodes(), types.int32)
    task.execute()
    return store_to_cunumeric_array(result)


def independence_polynomial(graph: nx.Graph):
    ip = [1]
    sets, nbrs = None, None
    for k in range(1, graph.number_of_nodes()):
        sets, nbrs = enumerate_independent_sets(
            graph, k, prevk_sets=sets, prevk_queues=nbrs
        )
        ip.append(sets.shape[0])
        if np.all(sets_to_sizes(nbrs, graph) == 0):
            break
    return ip


# sort_by_key is a method extracted from array.py that does the COO sort by
# key operation for the input sets of rows and columns.
def sort_by_key(rows, cols):
    vals_arr = np.ones((rows.shape[0]), dtype=np.float64)
    vals = get_store_from_cunumeric_array(vals_arr)

    rows_res = ctx.create_store(rows.type, ndim=1)
    cols_res = ctx.create_store(cols.type, ndim=1)
    vals_res = ctx.create_store(vals.type, ndim=1)
    task = ctx.create_task(SparseOpCode.SORT_BY_KEY)
    # Add all of the unbounded outputs.
    task.add_output(rows_res)
    task.add_output(cols_res)
    task.add_output(vals_res)
    # Add all of the inputs.
    task.add_input(rows)
    task.add_input(cols)
    task.add_input(vals)
    # The inputs need to be aligned.
    task.add_alignment(rows, cols)
    task.add_alignment(cols, vals)
    task.add_cpu_communicator()
    task.execute()
    return rows_res, cols_res


# This code is extracted from array.py and is the final step of CSR
# matrix construction after coordinates have been sorted.
def raw_create_csr(rows, cols, vals, shape):
    rows_store = get_store_from_cunumeric_array(rows)
    cols_store = get_store_from_cunumeric_array(cols)
    vals_store = get_store_from_cunumeric_array(vals)
    # Explicitly partition the rows into equal components to get the number of
    # non-zeros per row. We'll then partition up the non-zeros array according
    # to the per-partition ranges given by the min and max of each partition.
    num_procs = runtime.num_procs
    # TODO (rohany): If I try to partition this on really small inputs (like
    # size 0 or 1 stores) across multiple processors, I see some sparse
    # non-deterministic failures. I haven't root caused these, and I'm running
    # out of time to figure them out. It seems just not partitioning the input
    # on these really small matrices side-steps the underlying issue.
    if rows_store.shape[0] <= num_procs:
        num_procs = 1
    row_tiling = (rows_store.shape[0] + num_procs - 1) // num_procs
    rows_part = rows_store.partition(
        Tiling(Shape(row_tiling), Shape(num_procs))
    )
    # In order to bypass legate.core's current inability to handle representing
    # stores as FutureMaps, we drop below the ManualTask API to launch as task
    # ourselves and use the returned future map directly instead of letting the
    # core try and extract data from it.
    launcher = TaskLauncher(
        ctx,
        SparseOpCode.BOUNDS_FROM_PARTITIONED_COORDINATES,
        error_on_interference=False,
        tag=ctx.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG,
    )
    launcher.add_input(
        rows_store, rows_part.get_requirement(1, 0), tag=1
    )  # LEGATE_CORE_KEY_STORE_TAG
    bounds_store = ctx.create_store(
        domain_ty, shape=(1,), optimize_scalar=True
    )
    launcher.add_output(bounds_store, Broadcast(None, 0), tag=0)
    result = launcher.execute(Rect(hi=(num_procs,)))

    q_nnz = get_store_from_cunumeric_array(np.zeros((shape[0],), dtype=nnz_ty))
    task = ctx.create_manual_task(
        SparseOpCode.SORTED_COORDS_TO_COUNTS,
        launch_domain=Rect(hi=(num_procs,)),
    )
    task.add_input(rows_part)
    task.add_reduction(
        q_nnz.partition(
            DomainPartition(q_nnz.shape, Shape(num_procs), result)
        ),
        ReductionOp.ADD,
    )
    task.add_scalar_arg(shape[0], types.int64)
    task.execute()
    # TODO (rohany): On small inputs, it appears that I get a non-deterministic
    # failure, which appears either as a segfault or an incorrect output. This
    # appears to show up only an OpenMP processors, and appears when running
    # the full test suite with 4 opemps and 2 openmp threads. My notes from
    # debugging this are as follows:
    #  * The result of the sort appears to be correct.
    #  * We start with a valid COO matrix.
    #  * Adding print statements make the bug much harder to reproduce.
    #  * In particular, the bug is harder to reproduce when q_nnz is printed
    #    out before the `self.nnz_to_pos(q_nnz)` line here.
    #  * If q_nnz is printed out after the `self.nnz_to_pos(q_nnz)` line, then
    #    the computation looks correct but an incorrect pos array is generated.
    pos, _ = CompressedBase.nnz_to_pos_cls(q_nnz)
    return csr_array(
        (vals_store, cols_store, pos), shape=shape, dtype=np.float64
    )


def enumerate_independent_sets(
    graph: nx.Graph, k: int, prevk_sets=None, prevk_queues=None
):
    assert k > 0
    n = graph.number_of_nodes()
    comp = nx.complement(graph)
    comp_adj_mat = np.array(nx.to_numpy_array(comp))
    comp_adj_mat_store = get_store_from_cunumeric_array(comp_adj_mat)
    output_sets = ctx.create_store(get_set_ty(n), ndim=1)
    output_nbrs = ctx.create_store(get_set_ty(n), ndim=1)

    if k == 1:
        # If k == 1, we don't need a prior level.
        task = ctx.create_task(SparseOpCode.ENUMERATE_INDEPENDENT_SETS)
        task.add_input(comp_adj_mat_store)
        task.add_output(output_sets)
        task.add_output(output_nbrs)
        task.add_broadcast(comp_adj_mat_store)
        task.add_scalar_arg(graph.number_of_nodes(), types.int32)
        task.add_scalar_arg(k, types.int32)
        task.execute()
    else:
        task = ctx.create_task(SparseOpCode.ENUMERATE_INDEPENDENT_SETS)
        task.add_input(comp_adj_mat_store)
        task.add_input(prevk_sets)
        task.add_input(prevk_queues)
        task.add_output(output_sets)
        task.add_output(output_nbrs)
        task.add_broadcast(comp_adj_mat_store)
        task.add_alignment(prevk_sets, prevk_queues)
        task.add_scalar_arg(graph.number_of_nodes(), types.int32)
        task.add_scalar_arg(k, types.int32)
        task.execute()

    # We don't want to rely on the Weighted partition that we get
    # back on each store to run future operations. Reset it
    # so that the the solver will use an equal partition instead.
    reset_output_store_partition(output_sets)
    reset_output_store_partition(output_nbrs)

    return output_sets, output_nbrs

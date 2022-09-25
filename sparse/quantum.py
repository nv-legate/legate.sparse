# This file contains utility functions for constructing the input of
# the rydberg simulation in parallel. It should in principle be a separate
# legate library just for the simulation, but setting up new legate libraries
# is very annoying. Instead, we'll just stick it in here and reuse alot of
# the infrastructure that we already have set up.

import cunumeric as np
import networkx as nx

from .array import get_store_from_cunumeric_array, store_to_cunumeric_array, coord_ty, csr_array
from .config import SparseOpCode, types
from .runtime import ctx, runtime


class LegateHamiltonianDriver:
    def __init__(self, energies: tuple = (1,), graph: nx.Graph = None, dtype=np.complex64):
        """
        Default is that the first element in transition is the higher energy s.
        """
        self.energies = energies

        self.ip = [1]
        rows_group, cols_group = [], []
        sets, nbrs = None, None
        # TODO (rohany): Comment this...
        prev_offset, offset = 0, 1
        for k in range(1, graph.number_of_nodes()):
            # Find independence sets of size k.
            prev_sets = sets
            sets, nbrs = enumerate_independent_sets(graph, k, prevk_sets=sets, prevk_queues=nbrs)
            # Remember the number of independence sets that we found.
            self.ip.append(sets.shape[0])
            # Use the independence sets of size k and k - 1
            # to build the hamiltonian coordinates.
            rows = ctx.create_store(coord_ty, ndim=1)
            cols = ctx.create_store(coord_ty, ndim=1)
            task = ctx.create_task(SparseOpCode.CREATE_HAMILTONIANS)
            task.add_scalar_arg(k, types.int32)
            task.add_scalar_arg(graph.number_of_nodes(), types.int32)
            task.add_scalar_arg(offset, types.uint64)
            task.add_input(sets)
            task.add_output(rows)
            task.add_output(cols)
            if k > 1:
                # TODO (rohany): See if we can get around broadcasting this.
                task.add_input(prev_sets)
                task.add_broadcast(prev_sets)
                task.add_scalar_arg(prev_offset, types.uint64)
            task.execute()

            prev_offset = offset
            offset += int(sets.shape[0])

            reset_output_store_partition(rows)
            reset_output_store_partition(cols)

            rows_group.append(store_to_cunumeric_array(rows))
            cols_group.append(store_to_cunumeric_array(cols))

            # If there is no frontier to explore, then we're done.
            if np.sum(sets_to_sizes(nbrs)) == 0:
                break
            print("Looping...", k)

        self.nstates = np.sum(self.ip)
        # TODO (rohany): Back-offset the values.
        rows = (self.nstates - 1) - np.concatenate(rows_group)
        cols = (self.nstates - 1) - np.concatenate(cols_group)
        # reset_output_store_partition(get_store_from_cunumeric_array(rows))
        # reset_output_store_partition(get_store_from_cunumeric_array(cols))
        vals = np.ones(rows.shape[0], dtype=dtype)
        self._hamiltonian = csr_array((vals, (rows, cols)), shape=(self.nstates, self.nstates))

    @property
    def hamiltonian(self):
        if self.energies[0] == 1:
            return self._hamiltonian
        return self.energies[0] * self._hamiltonian


class LegateHamiltonianMIS(object):
    def __init__(self, graph: nx.Graph, poly=None, energies=(1, 1), dtype=np.complex64):
        """
        Generate a vector corresponding to the diagonal of the MIS Hamiltonian.
        """
        if energies == (1, 1):
            energies = (1,)
        self.graph = graph
        self.n = self.graph.number_of_nodes()
        self.energies = energies
        self.optimization = 'max'
        self._is_diagonal = True
        self.nstates = int(np.sum(poly))
        self.dtype = dtype

        # This code does the same thing as the below code in the special
        # case of equations that we are considering. In particular, the
        # below code constructs an array of consecutive integers where
        # each integer should appear the number of times corresponding to
        # the independence polynomial. We use arange and repeat to achieve that.
        levels = np.arange(len(poly))
        C = np.array(np.flip(np.repeat(levels, poly)).astype(self.dtype))
        enum_states = np.arange(self.nstates)
        self._hamiltonian = csr_array((C, (enum_states, enum_states)), shape=(self.nstates, self.nstates))

        # These are your independent sets of the original graphs, ordered by node and size
        # node_weights = []
        # for i in range(self.graph.number_of_nodes()):
        #     if hasattr(self.graph.nodes[i], 'weight'):
        #         node_weights.append(self.graph.nodes[i]['weight'])
        #     else:
        #         node_weights.append(1)
        # import numpy
        # node_weights = numpy.asarray(node_weights)
        # sets = enumerate_independent_sets(self.graph)
        # # Generate a list of integers corresponding to the independent sets in binary
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
        # self._hamiltonian = sparse.csr_matrix((C, (enum_states, enum_states)), shape=(self.nstates, self.nstates))

    @property
    def hamiltonian(self):
        if self.energies[0] == 1:
            return self._hamiltonian
        return self.energies[0] * self._hamiltonian

    @property
    def _diagonal_hamiltonian(self):
        return self.hamiltonian.data.reshape(-1,1)

    @property
    def optimum(self):
        # This needs to be recomputed because the optimum depends on the energies
        return np.max(self._diagonal_hamiltonian.real)

    @property
    def minimum_energy(self):
        # This needs to be recomputed because it depends on the energies
        return np.min(self._diagonal_hamiltonian.real)

    def cost_function(self, state):
        # Returns <s|C|s>
        return np.real(np.matmul(np.conj(state).T, self._diagonal_hamiltonian * state))

    def optimum_overlap(self, state):
        # Returns \sum_i <s|opt_i><opt_i|s>
        optimum_indices = np.argwhere(self._diagonal_hamiltonian == self.optimum).T[0]
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


set_ty_np = np.uint64
set_ty = types.uint64

def print_set(store):
    task = ctx.create_task(SparseOpCode.EXPAND_SET)
    task.add_input(store)
    task.add_broadcast(store)
    task.execute()


def sets_to_sizes(sets):
    result = ctx.create_store(types.uint64, shape=sets.shape)
    task = ctx.create_task(SparseOpCode.SETS_TO_SIZES)
    task.add_input(sets)
    task.add_output(result)
    task.add_alignment(sets, result)
    task.execute()
    return store_to_cunumeric_array(result)



def independence_polynomial(graph: nx.Graph):
    ip = [1]
    sets, nbrs = None, None
    for k in range(1, graph.number_of_nodes()):
        sets, nbrs = enumerate_independent_sets(graph, k, prevk_sets=sets, prevk_queues=nbrs)
        ip.append(sets.shape[0])
        if np.all(sets_to_sizes(nbrs) == 0):
            break
    return ip


# TODO (rohany): Make the previous inputs a store.
def enumerate_independent_sets(graph: nx.Graph, k: int, prevk_sets=None, prevk_queues=None):
    # TODO (rohany): Shouldn't care too much about this case.
    if k == 0:
        return []
    # TODO (rohany): We'll start now with just int64, can go to the
    #  variable length int after I can get this to work.
    assert graph.number_of_nodes() <= 64

    comp = nx.complement(graph)
    comp_adj_mat = np.array(nx.to_numpy_array(comp))
    comp_adj_mat_store = get_store_from_cunumeric_array(comp_adj_mat)
    output_sets = ctx.create_store(set_ty, ndim=1)
    output_nbrs = ctx.create_store(set_ty, ndim=1)

    if k == 1:
        # If k == 1, we don't need a prior level.
        task = ctx.create_task(SparseOpCode.ENUMERATE_INDEPENDENT_SETS)
        task.add_input(comp_adj_mat_store)
        task.add_output(output_sets)
        task.add_output(output_nbrs)
        task.add_broadcast(comp_adj_mat_store)
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
        task.add_scalar_arg(k, types.int32)
        task.execute()

    # So that prints make sense...
    # print("SETS:")
    # print_set(output_sets)
    # print("NEIGHBORS:")
    # print_set(output_nbrs)

    # TODO (rohany): Comment this... The basic idea here is that
    #  we don't want to use the weighted partition we get here
    #  to perform future ops on, we want to reset it to be an
    #  equal partition.
    reset_output_store_partition(output_sets)
    reset_output_store_partition(output_nbrs)

    return output_sets, output_nbrs

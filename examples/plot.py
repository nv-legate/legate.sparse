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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def trimesh(vertices, indices, labels=False):
    from matplotlib import collections

    vertices, indices = np.asarray(vertices), np.asarray(indices)

    triangles = vertices[indices.ravel(),:].reshape((indices.shape[0], indices.shape[1], 2))
    col = collections.PolyCollection(triangles, lw=1, edgecolor='black', facecolor='gray', alpha=0.5)

    sub = plt.gca()
    sub.add_collection(col,autolim=True)
    plt.axis('off')
    sub.autoscale_view()

def draw_graph(mesh, P):
    N = int(math.sqrt(mesh.shape[0]))
    grid = np.meshgrid(range(N),range(N))
    V = np.vstack(list(map(np.ravel, grid))).T
    E = np.vstack((mesh.row, mesh.col)).T

    c = ['red' if p == 0 else 'green' for p in P]

    plt.figure()
    sub = plt.gca()
    trimesh(V, E, False)
    sub.scatter(V[:, 0], V[:, 1], marker='o', s=400, c=c)

    for i in range(V.shape[0]):
        sub.annotate(str(i), (V[i, 0], V[i, 1]), ha='center', va='center')

    plt.show()

def plot_mis(A):
    mis = maximal_independent_set(A)
    P = np.zeros(A.shape[0])
    P[mis] = 1
    draw_graph(A.tocoo(), P)

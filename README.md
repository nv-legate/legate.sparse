NOTE: Legate Sparse is currently in an experimental state and not production quality
====================================================================================

# Legate Sparse

Legate Sparse is a [Legate](https://github.com/nv-legate/legate.core) library
that aims to provide a distributed and accelerated drop-in replacement for the
[scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) library
on top of the [Legion](https://legion.stanford.edu) runtime. Legate Sparse
interoperates with [cuNumeric](https://github.com/nv-legate/cunumeric) to
enable writing programs that operate on distributed dense and sparse arrays.
For some examples, take a look at the `examples/` directory. We have implemented
a [PDE Solver](examples/pde.py), as well as [Geometric](examples/gmg.py) 
and [Algebraic](examples/amg.py) multi-grid solvers using Legate Sparse. More complex
and interesting applications are on the way -- stay tuned!

# Building

To use Legate Sparse, you must build
[legate.core](https://github.com/magnatelee/legate.core) and
[cuNumeric](https://github.com/nv-legate/cunumeric) from source.  In
particular, the following
[branch](https://github.com/magnatelee/legate.core/tree/legate-sparse-branch-23.09/) of legate.core
and the following
[branch](https://github.com/nv-legate/cunumeric/tree/branch-23.09) of cuNumeric
must be used as many necessary changes have not yet made their way into the
main branches of Legate and cuNumeric.

## Instructions

First, clone [quickstart](https://github.com/rohany/quickstart) and 
checkout the `legate-sparse` branch. This repository contains several scripts to cover
common machines and use cases. Then clone the following fork of [legate.core](https://github.com/magnatelee/legate.core/),
and the official repository of [cuNumeric](https://github.com/nv-legate/cunumeric). For legate.core,
check out the `legate-sparse-branch-23.09`, and the `branch-23.09` branch of cuNumeric. Then, clone Legate Sparse. 
We recommend the following directory organization:
```
legate/
  quickstart/
  legate.core/
  cunumeric/
  legate.sparse/
```

Second, set up a `conda` environment:
```
quickstart/setup_conda.sh
```
Running `./quickstart/setup_conda.sh --help` will display different options that allow you to customize
the created `conda` installation and environment.

Third, install `legate.core`:
```
cd legate.core/
git clone https://gitlab.com/StanfordLegion/legion/
cd legion && git checkout control_replication && cd ../
LEGION_DIR=legion ../quickstart/build.sh
```

Fourth, install `cunumeric`:
```
cd cunumeric/
../quickstart/build.sh
```

Finally, install `legate.sparse`:
```
cd legate.sparse
../quickstart/build.sh
```

The `quickstart/build.sh` script will attempt to auto-detect the machine you are
running on, if it is a common machine that the Legate or Legion developers frequently
use. Otherwise, it will ask for additional information to be specified, such as the
GPU architecture or network interconnect.

# Usage

To write programs using Legate Sparse, import the `sparse` module, which
contains methods and types found in `scipy.sparse`.
```[python]
import sparse.io as io
mat = io.mmread("testdata/test.mtx").tocsr()
print((mat + mat).todense())
"""
[[4. 0. 0. 6. 0.]
 [0. 0. 0. 0. 8.]
 [0. 0. 0. 0. 0.]
 [6. 0. 0. 0. 0.]
 [0. 8. 0. 0. 0.]]
"""
```

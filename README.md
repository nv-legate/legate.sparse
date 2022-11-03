NOTE: Legate Sparse is currently in an experimental state and not production quality
====================================================================================

# Legate Sparse

Legate Sparse is a [Legate](https://github.com/nv-legate/legate.core) library
that aims to provide a distributed and accelerated drop-in replacement for the
[scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) library
on top of the [Legion](https://legion.stanford.edu) runtime. Legate Sparse
interoperates with [cuNumeric](https://github.com/nv-legate/cunumeric) to
enable writing programs that operate on distributed dense and sparse arrays.

# Usage

To use Legate Sparse, you must build
[Legate](https://github.com/nv-legate/legate.core) and
[cuNumeric](https://github.com/nv-legate/cunumeric) from source.  In
particular, the following
[branch](https://github.com/rohany/legate.core/tree/legate-sparse) of Legate
and the following
[branch](https://github.com/rohany/cunumeric/tree/legate-sparse) of cuNumeric
must be used as many necessary changes have not yet made their way into the
main branches of Legate and cuNumeric.

After these Legate and cuNumeric have been installed, run
```
./install.py
```
to install Legate Sparse. `install.py --help` will list more installation
options for Legate Sparse.

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

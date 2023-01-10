import argparse

from typing_extensions import Protocol


class Timer(Protocol):
    def start(self):
        ...

    def stop(self):
        """
        Blocks execution until everything before it has completed. Returns the
        duration since the last call to start(), in milliseconds.
        """
        ...


class LegateTimer(Timer):
    def __init__(self):
        self._start_future = None

    def start(self):
        from legate.timing import time

        self._start_future = time()

    def stop(self):
        from legate.timing import time

        end_future = time()
        return (end_future - self._start_future) / 1000.0


class CuPyTimer(Timer):
    def __init__(self):
        self._start_event = None

    def start(self):
        from cupy import cuda

        self._start_event = cuda.Event()
        self._start_event.record()

    def stop(self):
        from cupy import cuda

        end_event = cuda.Event()
        end_event.record()
        end_event.synchronize()
        return cuda.get_elapsed_time(self._start_event, end_event)


class NumPyTimer(Timer):
    def __init__(self):
        self._start_time = None

    def start(self):
        from time import perf_counter_ns

        self._start_time = perf_counter_ns() / 1000.0

    def stop(self):
        from time import perf_counter_ns

        end_time = perf_counter_ns() / 1000.0
        return (end_time - self._start_time) / 1000.0


def parse_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-package",
        type=str,
        default="legate",
        choices=["legate", "cupy", "scipy"],
    )
    args, _ = parser.parse_known_args()
    if args.package == "legate":
        timer = LegateTimer()
        import cunumeric as np

        import sparse
        import sparse.linalg as linalg

        use_legate = True
    elif args.package == "cupy":
        timer = CuPyTimer()
        import cupy as np
        import cupyx.scipy.sparse as sparse
        from cupyx.scipy.sparse import linalg

        use_legate = False
    else:
        timer = NumPyTimer()
        import numpy as np
        import scipy.sparse as sparse
        from scipy.sparse import linalg

        use_legate = False
    return args.package, timer, np, sparse, linalg, use_legate

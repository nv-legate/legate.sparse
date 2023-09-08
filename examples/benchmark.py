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


# DummyScope is a class that is a no-op context
# manager so that we can run both CuPy and SciPy
# programs with resource scoping.
class DummyScope:
    def __init__(self):
        ...

    def __enter__(self):
        ...

    def __exit__(self, _, __, ___):
        ...

    def __getitem__(self, item):
        return self

    def count(self, _):
        return 1

    @property
    def preferred_kind(self):
        return None


def get_phase_procs(use_legate: bool):
    if use_legate:
        from legate.core import get_machine
        from legate.core.machine import ProcessorKind

        all_devices = get_machine()
        num_gpus = all_devices.count(ProcessorKind.GPU)
        num_omps = all_devices.count(ProcessorKind.OMP)

        # Prefer CPUs for the "build" phase of applications.
        if num_omps > 0:
            build_procs = all_devices.only(ProcessorKind.OMP)
        else:
            build_procs = all_devices.only(ProcessorKind.CPU)

        # Prefer GPUs for the "solve" phase of applications.
        if num_gpus > 0:
            solve_procs = all_devices.only(ProcessorKind.GPU)
        elif num_omps > 0:
            solve_procs = all_devices.only(ProcessorKind.OMP)
        else:
            solve_procs = all_devices.only(ProcessorKind.CPU)
        return build_procs, solve_procs
    else:
        return DummyScope(), DummyScope()


def parse_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
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

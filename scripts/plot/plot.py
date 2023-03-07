#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from parse import parse

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# seaborn.set_theme()
# seaborn.set_style('ticks')

# Constants for machine description and labels.
GPUS_PER_SOCKET = 3
SYSTEM_KEY = "System"
PROCS_KEY = "Sockets/GPUs"
THROUGHPUT = "Throughput"
LEGATE_CPU = "Legate-CPU"
LEGATE_CPU_NO_INSTANCES = "Legate-CPU-Map-Sep"
LEGATE_GPU = "Legate-GPU"
LEGATE_GPU_NO_INSTANCES = "Legate-GPU-Map-Sep"
PETSC_CPU = "PETSc-CPU"
PETSC_GPU = "PETSc-GPU"
SCIPY = "SciPy"
CUPY = "CuPy"
CUPY_LABEL = "CuPy (1 GPU)"
DOT = "DOT"
PDE = "PDE"
GMG = "GMG"
QUANTUM = "QUANTUM"
SPARSEML = "SPARSEML"
MARKER_SIZE = 10


def is_system_gpu(system):
    return system in [
        LEGATE_GPU,
        CUPY,
        PETSC_GPU,
        LEGATE_GPU_NO_INSTANCES,
        CUPY_LABEL,
    ]


rawPalette = seaborn.color_palette()
palette = {
    LEGATE_CPU: rawPalette[0],
    LEGATE_GPU: rawPalette[1],
    PETSC_CPU: rawPalette[2],
    PETSC_GPU: rawPalette[3],
    SCIPY: rawPalette[4],
    CUPY: rawPalette[5],
    CUPY_LABEL: rawPalette[5],
    LEGATE_CPU_NO_INSTANCES: rawPalette[6],
    LEGATE_GPU_NO_INSTANCES: rawPalette[7],
}
markers = {
    LEGATE_CPU: "^",
    LEGATE_GPU: "*",
    PETSC_CPU: "P",
    PETSC_GPU: "X",
    SCIPY: "D",
    CUPY: "s",
    CUPY_LABEL: "s",
    LEGATE_CPU_NO_INSTANCES: "d",
    LEGATE_GPU_NO_INSTANCES: "p",
}


def parse_execution_logs(filename):
    data = defaultdict(lambda: [])
    with open(filename, "r") as f:
        procs = None
        for line in f.readlines():
            line = line.strip()
            if "CPU SOCKETS = " in line or "GPUS = " in line:
                # Parse the number of processors.
                if "CPU SOCKETS" in line:
                    procs = int(parse("CPU SOCKETS = {:d}:", line)[0])
                else:
                    procs = int(parse("GPUS = {:d}:", line)[0])
            if "iterations / sec:" in line.lower():
                # Parse the iterations / second.
                assert procs is not None
                iterpersec = float(
                    parse("iterations / sec: {:f}", line.lower())[0]
                )
                data[procs].append(iterpersec)
    return data


def clean_raw_data(proc_to_iter_list):
    # Here we drop the fastest and slowest times, and then average the
    # remaining runs to get the result.
    result = []
    for procs, iters in proc_to_iter_list.items():
        iters.remove(min(iters))
        iters.remove(max(iters))
        avg = numpy.mean(iters)
        result.append((procs, avg))

    # Return the resulting sorted list.
    return sorted(result)


def project_system_name(proc_iter_pairs, system):
    if system == CUPY:
        system = CUPY_LABEL
    return [(system, p, t) for p, t in proc_iter_pairs]


def raw_cpu_data_to_df(sys_proc_iter_tuples):
    # Convert each CPU socket to the number of GPUs.
    data = [
        (system, f"{procs}/{procs * GPUS_PER_SOCKET}", tp)
        for system, procs, tp in sys_proc_iter_tuples
    ]
    return pandas.DataFrame(data, columns=[SYSTEM_KEY, PROCS_KEY, THROUGHPUT])


def raw_gpu_data_to_df(sys_proc_iter_tuples):
    # Convert each GPU count to the number of CPU sockets.
    data = [
        (system, f"{max(1, procs // GPUS_PER_SOCKET)}/{procs}", tp)
        for system, procs, tp in sys_proc_iter_tuples
    ]
    return pandas.DataFrame(data, columns=[SYSTEM_KEY, PROCS_KEY, THROUGHPUT])


def direct_procs_to_df(sys_proc_iter_tuples):
    data = [
        (system, str(procs), tp) for system, procs, tp in sys_proc_iter_tuples
    ]
    return pandas.DataFrame(data, columns=[SYSTEM_KEY, PROCS_KEY, THROUGHPUT])


def parse_standard_cpu_weakscaling_data(filename, system):
    return raw_cpu_data_to_df(
        project_system_name(
            clean_raw_data(parse_execution_logs(filename)), system
        )
    )


def parse_standard_gpu_weakscaling_data(filename, system):
    return raw_gpu_data_to_df(
        project_system_name(
            clean_raw_data(parse_execution_logs(filename)), system
        )
    )


def standard_weak_scaling_plot(sys_to_file, title, outfile=None, procs=None):
    systems_to_data = []
    for system, filename in sys_to_file:
        if is_system_gpu(system):
            systems_to_data.append(
                (system, parse_standard_gpu_weakscaling_data(filename, system))
            )
        else:
            systems_to_data.append(
                (system, parse_standard_cpu_weakscaling_data(filename, system))
            )
    # Re-order the data so that the GPU points are first
    # (this makes the 1/1 point) for GPUs come first.
    systems_to_data = list(
        filter(lambda x: is_system_gpu(x[0]), systems_to_data)
    ) + list(filter(lambda x: not is_system_gpu(x[0]), systems_to_data))
    data = pandas.concat([data for _, data in systems_to_data])
    if procs is not None:
        data = data[data.apply(lambda x: x[PROCS_KEY] in procs, axis=1)]
    ax = seaborn.lineplot(
        data,
        x=PROCS_KEY,
        y=THROUGHPUT,
        hue=SYSTEM_KEY,
        style=SYSTEM_KEY,
        palette=palette,
        markers=markers,
        markersize=MARKER_SIZE,
    )
    ax.set_yscale("log", base=10)
    ax.set_title(title)
    ax.set_ylabel("Throughput (iterations / second)")
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    ax.legend(ncol=2, title="System")
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
        plt.clf()


def summit_quantum_weak_scaling_plot(sys_to_file, title, outfile=None):
    frames = []
    for system, filename in sys_to_file:
        frames.append(
            direct_procs_to_df(
                project_system_name(
                    clean_raw_data(parse_execution_logs(filename)), system
                )
            )
        )
    data = pandas.concat(frames)
    ax = seaborn.lineplot(
        data,
        x=PROCS_KEY,
        y=THROUGHPUT,
        hue=SYSTEM_KEY,
        style=SYSTEM_KEY,
        palette=palette,
        markers=markers,
        markersize=MARKER_SIZE,
    )
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Sockets or GPUs")
    ax.set_ylabel("Throughput (iterations / second)")
    ax.set_title(title)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
        plt.clf()


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str)
parser.add_argument("result_dir", type=str)
parser.add_argument("-machine", type=str, default="summit")
args = parser.parse_args()
data_dir = args.data_dir
result_dir = args.result_dir
machine = args.machine


def get_system_data_path_pair(system, bench):
    cleaned_system = system.replace("-", "_").lower()
    suffix = ""
    if is_system_gpu(system) and "gpu" not in system.lower():
        suffix = "gpu"
    elif not is_system_gpu(system) and "cpu" not in system.lower():
        suffix = "cpu"
    path = os.path.join(
        data_dir,
        machine,
        cleaned_system
        + (("_" + suffix) if suffix != "" else "")
        + "_"
        + bench.lower()
        + ".out",
    )
    return (system, path)


# Generate the SpMV Dot Microbenchmark Plot.
standard_weak_scaling_plot(
    [
        get_system_data_path_pair(LEGATE_CPU, DOT),
        get_system_data_path_pair(LEGATE_GPU, DOT),
        get_system_data_path_pair(SCIPY, DOT),
        get_system_data_path_pair(CUPY, DOT),
        get_system_data_path_pair(PETSC_CPU, DOT),
        get_system_data_path_pair(PETSC_GPU, DOT),
    ],
    "SpMV Microbenchmark",
    os.path.join(result_dir, "spmv-microbenchmark.pdf"),
)

# PDE.
standard_weak_scaling_plot(
    [
        get_system_data_path_pair(LEGATE_CPU, PDE),
        get_system_data_path_pair(LEGATE_GPU, PDE),
        get_system_data_path_pair(SCIPY, PDE),
        get_system_data_path_pair(CUPY, PDE),
        get_system_data_path_pair(PETSC_CPU, PDE),
        get_system_data_path_pair(PETSC_GPU, PDE),
    ],
    "Conjugate Gradient Solver",
    os.path.join(result_dir, "pde.pdf"),
)

# GMG.
standard_weak_scaling_plot(
    [
        get_system_data_path_pair(LEGATE_CPU, GMG),
        get_system_data_path_pair(LEGATE_GPU, GMG),
        get_system_data_path_pair(SCIPY, GMG),
        get_system_data_path_pair(CUPY, GMG),
    ],
    "Geometric Multi-Grid Solver",
    os.path.join(result_dir, "gmg.pdf"),
)

# GMG Non Library Aware Mapping.
standard_weak_scaling_plot(
    [
        get_system_data_path_pair(LEGATE_CPU, GMG),
        get_system_data_path_pair(LEGATE_GPU, GMG),
        get_system_data_path_pair(LEGATE_GPU_NO_INSTANCES, GMG),
        get_system_data_path_pair(LEGATE_CPU_NO_INSTANCES, GMG),
    ],
    "Mapping Optimization Affect on Geometric Multi-Grid Solver",
    os.path.join(result_dir, "gmg-instances.pdf"),
    procs=["1/1", "1/3", "2/6", "4/12"],
)

# Quantum.
summit_quantum_weak_scaling_plot(
    [
        get_system_data_path_pair(LEGATE_CPU, QUANTUM),
        get_system_data_path_pair(LEGATE_GPU, QUANTUM),
        get_system_data_path_pair(SCIPY, QUANTUM),
        get_system_data_path_pair(CUPY, QUANTUM),
    ],
    "Quantum Simulation",
    os.path.join(result_dir, "quantum.pdf"),
)

#!/usr/bin/env python3

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

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys

import setuptools

# Flush output on newlines
sys.stdout.reconfigure(line_buffering=True)

os_name = platform.system()

if os_name == "Linux":
    dylib_ext = ".so"
elif os_name == "Darwin":
    dylib_ext = ".dylib"
else:
    raise Exception("install.py script does not work on %s" % os_name)


class BooleanFlag(argparse.Action):
    def __init__(
            self,
            option_strings,
            dest,
            default,
            required=False,
            help="",
            metavar=None,
    ):
        assert all(not opt.startswith("--no") for opt in option_strings)

        def flatten(list):
            return [item for sublist in list for item in sublist]

        option_strings = flatten(
            [
                [opt, "--no-" + opt[2:], "--no" + opt[2:]]
                if opt.startswith("--")
                else [opt]
                for opt in option_strings
            ]
        )
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            const=None,
            default=default,
            type=bool,
            choices=None,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, not option_string.startswith("--no"))


def execute_command(args, verbose, cwd=None, shell=False):
    if verbose:
        print("EXECUTING: ", args)
    subprocess.check_call(args, cwd=cwd, shell=shell)


def load_json_config(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except IOError:
        return None


def dump_json_config(filename, value):
    with open(filename, "w") as f:
        return json.dump(value, f)


def find_compile_flag(flag, makefile):
    with open(makefile, "r") as f:
        for line in f:
            toks = line.split()
            if len(toks) == 3 and toks[0] == flag:
                return toks[2] == "1"
    assert False, f"Compile flag '{flag}' not found"


def build_legate_sparse(
        sparse_dir,
        install_dir,
        nccl_dir,
        thrust_dir,
        debug,
        debug_release,
        check_bounds,
        clean_first,
        python_only,
        thread_count,
        verbose,
        unknown,
):
    src_dir = os.path.join(sparse_dir, "src")

    # Build the source files.
    if not python_only:
        make_flags = [
            "LEGATE_DIR=%s" % install_dir,
            "NCCL_PATH=%s" % nccl_dir,
            "THRUST_PATH=%s" % thrust_dir,
            "DEBUG=%s" % (1 if debug else 0),
            "DEBUG_RELEASE=%s" % (1 if debug_release else 0),
            "CHECK_BOUNDS=%s" % (1 if check_bounds else 0),
            "PREFIX=%s" % install_dir,
        ]
        if clean_first:
            execute_command(
                ["make"] + make_flags + ["clean"],
                cwd=src_dir,
                verbose=verbose,
            )
        execute_command(
            ["make"] + make_flags + ["-j", str(thread_count), "install"],
            cwd=src_dir,
            verbose=verbose,
        )

    try:
        shutil.rmtree(os.path.join(sparse_dir, "build"))
    except FileNotFoundError:
        pass

    cmd = [
        sys.executable,
        "setup.py",
        "install",
        "--recurse",
        "--prefix",
        install_dir,
    ]
    # Work around breaking change in setuptools 60
    if int(setuptools.__version__.split(".")[0]) >= 60:
        cmd += ["--single-version-externally-managed", "--root=/"]
    if unknown is not None:
        if "--prefix" in unknown:
            raise Exception(
                "cuNumeric cannot be installed in a different location than "
                "Legate Core, please remove the --prefix argument"
            )
        cmd += unknown
    execute_command(cmd, cwd=sparse_dir, verbose=verbose)


def install_legate_sparse(
    legate_dir,
    thrust_dir,
    debug,
    debug_release,
    check_bounds,
    clean_first,
    python_only,
    thread_count,
    verbose,
    unknown,
):
    print("Verbose build is ", "on" if verbose else "off")
    if verbose:
        print("Options are:\n")
        print("legate_dir: ", legate_dir, "\n")
        print("debug: ", debug, "\n")
        print("debug_release: ", debug_release, "\n")
        print("check_bounds: ", check_bounds, "\n")
        print("clean_first: ", clean_first, "\n")
        print("python_only: ", python_only, "\n")
        print("thread_count: ", thread_count, "\n")
        print("verbose: ", verbose, "\n")
        print("unknown: ", unknown, "\n")

    sparse_dir = os.path.dirname(os.path.realpath(__file__))

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    # Check to see if we installed Legate Core
    legate_config = os.path.join(sparse_dir, ".legate.core.json")
    if legate_dir is None:
        legate_dir = load_json_config(legate_config)
    if legate_dir is None or not os.path.exists(legate_dir):
        raise Exception(
            "You need to provide a Legate Core installation using "
            "the '--with-core' flag"
        )
    legate_dir = os.path.realpath(legate_dir)
    dump_json_config(legate_config, legate_dir)

    # Find list of already-installed libraries.
    libs_path = os.path.join(legate_dir, "share", ".legate-libs.json")
    try:
        with open(libs_path, "r") as f:
            libs_config = json.load(f)
    except (FileNotFoundError, IOError, json.JSONDecodeError):
        libs_config = {}

    # Match the core's setting regarding CUDA support.
    makefile_path = os.path.join(legate_dir, "share", "legate", "config.mk")
    nccl_dir = None
    cuda = find_compile_flag("USE_CUDA", makefile_path)
    if cuda:
        if "nccl" not in libs_config:
            raise Exception(
                "Failed to find NCCL path in the Legate installation. "
                "Make sure you installed Legate core correctly. "
                "If the problem persists, please open a GitHub issue for it. "
            )
        nccl_dir = libs_config["nccl"]

    # Find Thrust installation.
    thrust_global_config = os.path.join(
        legate_dir, "share", "legate", ".thrust.json"
    )
    if thrust_dir is None:
        thrust_dir = load_json_config(thrust_global_config)
    thrust_local_config = os.path.join(sparse_dir, ".thrust.json")
    if thrust_dir is None:
        thrust_dir = load_json_config(thrust_local_config)
    if thrust_dir is None:
        raise Exception(
            "Could not find Thrust installation, use '--with-thrust' to "
            "specify a location."
        )
    thrust_dir = os.path.realpath(thrust_dir)
    dump_json_config(thrust_local_config, thrust_dir)

    build_legate_sparse(
        sparse_dir,
        legate_dir,
        nccl_dir,
	thrust_dir,
        debug,
        debug_release,
        check_bounds,
        clean_first,
        python_only,
        thread_count,
        verbose,
        unknown,
    )


def driver():
    parser = argparse.ArgumentParser(description="Install Legate Sparse.")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG", "0") == "1",
        help="Build cuNumeric with no optimizations.",
    )
    parser.add_argument(
        "--debug-release",
        dest="debug_release",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG_RELEASE", "0") == "1",
        help="Build cuNumeric with optimizations, but include debugging "
             "symbols.",
    )
    parser.add_argument(
        "--check-bounds",
        dest="check_bounds",
        action="store_true",
        required=False,
        default=False,
        help="Build cuNumeric with bounds checks.",
    )
    parser.add_argument(
        "--with-core",
        dest="legate_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("LEGATE_DIR"),
        help="Path to Legate Core installation directory.",
    )
    parser.add_argument(
        "--with-thrust",
        dest="thrust_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("THRUST_PATH"),
        help="Path to Thrust installation directory.",
    )
    parser.add_argument(
        "--clean",
        dest="clean_first",
        action=BooleanFlag,
        default=True,
        help="Clean before build.",
    )
    parser.add_argument(
        "--python-only",
        dest="python_only",
        action="store_true",
        required=False,
        default=False,
        help="Reinstall only the Python package.",
    )
    parser.add_argument(
        "-j",
        dest="thread_count",
        nargs="?",
        type=int,
        required=False,
        help="Number of threads used to compile.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose build output.",
    )
    args, unknown = parser.parse_known_args()

    install_legate_sparse(unknown=unknown, **vars(args))


if __name__ == "__main__":
    driver()

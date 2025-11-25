"""PLUMED Python bindings with bundled library."""

import os
import sys
from pathlib import Path

_pkg_dir = Path(__file__).parent
_lib_dir = _pkg_dir / "_lib"

# Find bundled PLUMED kernel library
if sys.platform == "darwin":
    _kernel_path = _lib_dir / "lib" / "libplumedKernel.dylib"
    _pycv_plugin = _lib_dir / "lib" / "PythonCVInterface.dylib"
elif sys.platform.startswith("linux"):
    _kernel_path = _lib_dir / "lib" / "libplumedKernel.so"
    _pycv_plugin = _lib_dir / "lib" / "PythonCVInterface.so"
else:
    _kernel_path = None
    _pycv_plugin = None

# Set PLUMED_KERNEL environment variable
if _kernel_path and _kernel_path.exists():
    os.environ.setdefault("PLUMED_KERNEL", str(_kernel_path))
    BUNDLED_KERNEL_PATH = _kernel_path
else:
    BUNDLED_KERNEL_PATH = None

# pycv plugin path
BUNDLED_PYCV_PATH = _pycv_plugin if _pycv_plugin and _pycv_plugin.exists() else None

# Find bundled PLUMED executable
_plumed_bin = _lib_dir / "bin" / "plumed"
BUNDLED_PLUMED_BIN = _plumed_bin if _plumed_bin.exists() else None

# Import Cython bindings
try:
    from . import _plumed_core

    for name in dir(_plumed_core):
        if not name.startswith("_"):
            globals()[name] = getattr(_plumed_core, name)
except ImportError as e:
    import warnings

    warnings.warn(f"PLUMED Python bindings not available: {e}")

__all__ = [
    "BUNDLED_KERNEL_PATH",
    "BUNDLED_PLUMED_BIN",
    "BUNDLED_PYCV_PATH",
    "get_pycv_path",
    "cli",
]


def cli() -> None:
    """Run the bundled PLUMED command-line tool."""
    import subprocess

    if BUNDLED_PLUMED_BIN is None or not BUNDLED_PLUMED_BIN.exists():
        print("Error: Bundled PLUMED executable not found.", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    # PLUMED_ROOT must point to lib/plumed/ where scripts/ and patches/ are
    env["PLUMED_ROOT"] = str(_lib_dir / "lib" / "plumed")

    # Add library directory to library path so plumed can find libplumedKernel.so
    lib_path = str(_lib_dir / "lib")
    if sys.platform == "darwin":
        env["DYLD_LIBRARY_PATH"] = lib_path + ":" + env.get("DYLD_LIBRARY_PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run([str(BUNDLED_PLUMED_BIN)] + sys.argv[1:], env=env)
    sys.exit(result.returncode)


def get_pycv_path() -> str:
    """Get path to pycv plugin for PLUMED LOAD command.

    Returns
    -------
    str
        Absolute path to PythonCVInterface shared library.

    Raises
    ------
    RuntimeError
        If pycv plugin is not available.

    Example
    -------
    >>> import plumed
    >>> pycv_path = plumed.get_pycv_path()
    >>> plumed_input = [f"LOAD FILE={pycv_path}", ...]

    See Also
    --------
    https://www.plumed.org/doc-master/user-doc/html/_p_y_c_v_i_n_t_e_r_f_a_c_e.html
    """
    if BUNDLED_PYCV_PATH is None or not BUNDLED_PYCV_PATH.exists():
        raise RuntimeError("pycv plugin not found.")
    return str(BUNDLED_PYCV_PATH)

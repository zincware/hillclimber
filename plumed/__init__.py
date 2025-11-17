"""PLUMED Python bindings with bundled library.

This package provides Python bindings to PLUMED with the PLUMED library
bundled directly. No separate PLUMED installation is required.
"""

import os
import sys
from pathlib import Path

# Get the bundled library directory
_lib_dir = Path(__file__).parent

# Determine library extension based on platform
if sys.platform == "darwin":
    _lib_pattern = "libplumedKernel*.dylib"
elif sys.platform.startswith("linux"):
    _lib_pattern = "libplumedKernel.so*"
else:
    _lib_pattern = None

# Find the bundled PLUMED kernel library
_kernel_path = None
if _lib_pattern:
    lib_files = list(_lib_dir.glob(_lib_pattern))
    if lib_files:
        # Use the first match (should only be one)
        _kernel_path = lib_files[0]

# Set PLUMED_KERNEL environment variable to point to bundled library
# This is used by the plumed Python module to locate the library
if _kernel_path is not None and _kernel_path.exists():
    os.environ.setdefault("PLUMED_KERNEL", str(_kernel_path))
    BUNDLED_KERNEL_PATH = _kernel_path
else:
    BUNDLED_KERNEL_PATH = None

# Find bundled PLUMED executable
_plumed_bin = _lib_dir / "bin" / "plumed"
if _plumed_bin.exists():
    BUNDLED_PLUMED_BIN = _plumed_bin
else:
    BUNDLED_PLUMED_BIN = None

# Import the compiled extension module and re-export its contents
# The extension is built as _plumed_core.cpython-*.so in this directory
try:
    # Import the _plumed_core extension and re-export all its public members
    from . import _plumed_core

    # Re-export all public symbols from the extension
    for name in dir(_plumed_core):
        if not name.startswith("_"):
            globals()[name] = getattr(_plumed_core, name)

    _plumed_extension = _plumed_core

except ImportError as e:
    # If bindings fail to load, still provide the path information
    _plumed_extension = None
    import warnings
    warnings.warn(f"PLUMED Python bindings not available: {e}")

__all__ = ["BUNDLED_KERNEL_PATH", "BUNDLED_PLUMED_BIN"]

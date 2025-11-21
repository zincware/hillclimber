"""Custom Hatchling build hook to compile and bundle PLUMED library.

This hook builds the PLUMED C++ library from the git submodule and bundles it
with the Python wheel. This enables hillclimber to work without requiring users
to separately install the PLUMED library.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PlumedBuildHook(BuildHookInterface):
    """Build hook that compiles PLUMED and bundles it with the wheel."""

    PLUGIN_NAME = "plumed"

    def initialize(self, version: str, build_data: dict) -> None:
        """Build PLUMED library and prepare it for bundling.

        Parameters
        ----------
        version : str
            The version of the package being built.
        build_data : dict
            Build metadata that can be modified to control the build process.
        """
        print("=" * 70)
        print("PLUMED Build Hook: Starting PLUMED compilation")
        print("=" * 70)

        # Define directories
        plumed_src = Path(self.root) / "external" / "plumed2"
        build_dir = Path(self.root) / "build" / "plumed"
        install_dir = Path(self.root) / "build" / "plumed-install"
        pkg_lib_dir = Path(self.root) / "src" / "plumed"

        # Verify submodule exists
        if not plumed_src.exists() or not (plumed_src / "configure").exists():
            raise RuntimeError(
                f"PLUMED source not found at {plumed_src}. "
                "Please run 'git submodule update --init --recursive'"
            )

        # Create build directories
        build_dir.mkdir(parents=True, exist_ok=True)
        install_dir.mkdir(parents=True, exist_ok=True)
        pkg_lib_dir.mkdir(parents=True, exist_ok=True)

        # Build PLUMED
        self._configure_plumed(plumed_src, build_dir, install_dir)
        self._build_plumed(plumed_src)
        self._install_plumed(plumed_src)

        # Copy library to package
        self._bundle_library(install_dir, pkg_lib_dir)

        # Build Python bindings with Cython
        self._build_python_bindings(install_dir, pkg_lib_dir, build_data)

        # Include plumed package in wheel (maps src/plumed -> plumed in wheel)
        build_data.setdefault("force_include", {})[str(pkg_lib_dir)] = "plumed"

        # Mark wheel as platform-specific (contains compiled code)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        print("=" * 70)
        print("PLUMED Build Hook: Completed successfully")
        print("=" * 70)

    def _configure_plumed(
        self, src_dir: Path, build_dir: Path, install_dir: Path
    ) -> None:
        """Configure PLUMED with appropriate flags.

        Parameters
        ----------
        src_dir : Path
            PLUMED source directory.
        build_dir : Path
            Build directory (not used - configure must run in source dir).
        install_dir : Path
            Installation directory.
        """
        print("\n[1/4] Configuring PLUMED...")

        configure_cmd = [
            "./configure",
            f"--prefix={install_dir}",
            "--disable-dependency-tracking",
            "--enable-modules=all",  # Build all modules
            "--enable-shared",  # Build shared library
            "--disable-static",  # Don't need static library
        ]

        # Platform-specific configuration
        if sys.platform == "darwin":
            # macOS: Use @loader_path for relative RPATH
            configure_cmd.extend(
                [
                    "--disable-ld-r",
                    "LDFLAGS=-Wl,-rpath,@loader_path",
                ]
            )
        elif sys.platform.startswith("linux"):
            # Linux: Use $ORIGIN for relative RPATH
            # Disable -Werror to avoid build failures from warnings in PLUMED source
            configure_cmd.extend([
                "LDFLAGS=-Wl,-rpath,$ORIGIN",
                "CXXFLAGS=-Wno-error",
            ])

        print(f"Configure command: {' '.join(configure_cmd)}")
        print(f"Running in: {src_dir}")

        result = subprocess.run(
            configure_cmd,
            cwd=src_dir,  # Must run in source directory
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"PLUMED configure failed with code {result.returncode}")

        print("Configuration complete.")

    def _build_plumed(self, src_dir: Path) -> None:
        """Build PLUMED library.

        Parameters
        ----------
        src_dir : Path
            PLUMED source directory containing configured build.
        """
        print("\n[2/4] Building PLUMED...")

        # Determine number of parallel jobs
        try:
            import multiprocessing

            njobs = multiprocessing.cpu_count()
        except (ImportError, NotImplementedError):
            njobs = 2

        make_cmd = ["make", f"-j{njobs}"]
        print(f"Build command: {' '.join(make_cmd)}")
        print(f"Running in: {src_dir}")

        result = subprocess.run(
            make_cmd,
            cwd=src_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:])  # Last 2000 chars
            print("STDERR:", result.stderr[-2000:])
            raise RuntimeError(f"PLUMED build failed with code {result.returncode}")

        print("Build complete.")

    def _install_plumed(self, src_dir: Path) -> None:
        """Install PLUMED to installation directory.

        Parameters
        ----------
        src_dir : Path
            PLUMED source directory.
        """
        print("\n[3/4] Installing PLUMED...")

        result = subprocess.run(
            ["make", "install"],
            cwd=src_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"PLUMED install failed with code {result.returncode}")

        print("Installation complete.")

    def _bundle_library(self, install_dir: Path, pkg_lib_dir: Path) -> None:
        """Copy PLUMED library and executables to package directory.

        Parameters
        ----------
        install_dir : Path
            PLUMED installation directory.
        pkg_lib_dir : Path
            Package library directory where library will be bundled.
        """
        print("\n[4/4] Bundling PLUMED library and executables...")

        # Determine library extension
        if sys.platform == "darwin":
            lib_pattern = "libplumedKernel*.dylib"
        elif sys.platform.startswith("linux"):
            lib_pattern = "libplumedKernel*.so*"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        # Find and copy library files
        # Create lib subdirectory to match PLUMED's expected structure
        pkg_lib_subdir = pkg_lib_dir / "lib"
        pkg_lib_subdir.mkdir(exist_ok=True)

        lib_dir = install_dir / "lib"
        lib_files = list(lib_dir.glob(lib_pattern))

        if not lib_files:
            raise RuntimeError(f"No PLUMED library found matching {lib_pattern} in {lib_dir}")

        # In CI, auditwheel/delocate will handle library bundling to avoid duplicates
        is_ci = os.environ.get("CIBUILDWHEEL") == "1"

        for lib_file in lib_files:
            # Copy to lib subdirectory for executables and Cython compilation
            dest = pkg_lib_subdir / lib_file.name
            print(f"  Copying {lib_file.name} -> {dest}")
            shutil.copy2(lib_file, dest)
            dest.chmod(0o755)

            # Skip root copy in CI - auditwheel/delocate will bundle libs properly
            if not is_ci:
                dest_root = pkg_lib_dir / lib_file.name
                shutil.copy2(lib_file, dest_root)
                dest_root.chmod(0o755)

        # Copy PLUMED executables (plumed, sum_hills, etc.)
        bin_dir = install_dir / "bin"
        pkg_bin_dir = pkg_lib_dir / "bin"
        pkg_bin_dir.mkdir(exist_ok=True)

        if bin_dir.exists():
            plumed_exe = bin_dir / "plumed"
            if plumed_exe.exists():
                dest = pkg_bin_dir / "plumed"
                print(f"  Copying plumed executable -> {dest}")
                shutil.copy2(plumed_exe, dest)
                dest.chmod(0o755)
            else:
                print(f"  Warning: plumed executable not found at {plumed_exe}")
        else:
            print(f"  Warning: bin directory not found at {bin_dir}")

        # Copy Plumed.h header for Python bindings
        include_dir = install_dir / "include"
        plumed_h = include_dir / "plumed" / "wrapper" / "Plumed.h"
        if plumed_h.exists():
            dest_include = pkg_lib_dir / "include"
            dest_include.mkdir(exist_ok=True)
            shutil.copy2(plumed_h, dest_include / "Plumed.h")
            print(f"  Copied Plumed.h for Python bindings")

        # Copy lib/plumed data directory (module configs, etc.)
        plumed_data_dir = lib_dir / "plumed"
        if plumed_data_dir.exists():
            dest_data = pkg_lib_subdir / "plumed"
            if dest_data.exists():
                shutil.rmtree(dest_data)
            shutil.copytree(plumed_data_dir, dest_data)
            print(f"  Copied PLUMED data directory to {dest_data}")

        # Copy scripts/ and patches/ directories needed by PLUMED CLI
        for subdir in ["scripts", "patches"]:
            src_subdir = install_dir / "lib" / "plumed" / subdir
            if not src_subdir.exists():
                src_subdir = install_dir / subdir
            if src_subdir.exists():
                dest_subdir = pkg_lib_dir / subdir
                if dest_subdir.exists():
                    shutil.rmtree(dest_subdir)
                shutil.copytree(src_subdir, dest_subdir)
                print(f"  Copied {subdir}/ directory to {dest_subdir}")

        print("Library and executable bundling complete.")

    def _build_python_bindings(
        self, install_dir: Path, pkg_lib_dir: Path, build_data: dict
    ) -> None:
        """Build PLUMED Python bindings using Cython.

        Parameters
        ----------
        install_dir : Path
            PLUMED installation directory.
        pkg_lib_dir : Path
            Package library directory containing bundled library.
        build_data : dict
            Build metadata to update with Python extension information.
        """
        print("\n[5/5] Building Python bindings with Cython...")

        # Copy Python binding sources from PLUMED
        python_src = Path(self.root) / "external" / "plumed2" / "python"

        # Check if we have the necessary files
        pyx_file = python_src / "plumed.pyx"
        pxd_file = python_src / "cplumed.pxd"

        if not pyx_file.exists() or not pxd_file.exists():
            print("  Warning: Cython source files not found, skipping Python bindings")
            return

        # Copy binding sources to a temporary build location
        temp_build = Path(self.root) / "build" / "plumed_bindings"
        temp_build.mkdir(parents=True, exist_ok=True)

        # Copy and rename to _plumed_core to avoid name conflict with package
        shutil.copy2(pyx_file, temp_build / "_plumed_core.pyx")
        shutil.copy2(pxd_file, temp_build / "cplumed.pxd")

        # Compile the Cython extension
        try:
            from Cython.Build import cythonize
            from setuptools import Distribution, Extension
            from setuptools.command.build_ext import build_ext
        except ImportError as e:
            print(f"  Warning: Could not import Cython/setuptools: {e}")
            print("  Skipping Python bindings compilation")
            return

        # Create extension module
        # Build it as "plumed._plumed_core" to avoid top-level conflict
        include_dir = pkg_lib_dir / "include"
        lib_dir = pkg_lib_dir / "lib"

        extensions = [
            Extension(
                "plumed._plumed_core",  # Build as plumed._plumed_core submodule
                [str(temp_build / "_plumed_core.pyx")],  # Use renamed file
                include_dirs=[str(include_dir)],
                library_dirs=[str(lib_dir)],
                libraries=["plumedKernel"],
                runtime_library_dirs=[str(lib_dir)] if sys.platform.startswith("linux") else [],
                extra_compile_args=[
                    "-D__PLUMED_HAS_DLOPEN",
                    "-D__PLUMED_WRAPPER_LINK_RUNTIME=1",
                    "-D__PLUMED_WRAPPER_IMPLEMENTATION=1",
                    "-D__PLUMED_WRAPPER_EXTERN=0",
                ],
            )
        ]

        # Cythonize
        print("  Cythonizing _plumed_core.pyx...")
        ext_modules = cythonize(extensions, language_level=3, compiler_directives={"embedsignature": True})

        # Build the extension
        print("  Compiling extension...")
        dist = Distribution({"ext_modules": ext_modules})
        cmd = build_ext(dist)
        cmd.build_lib = str(pkg_lib_dir)
        cmd.build_temp = str(temp_build / "temp")
        cmd.inplace = False
        cmd.ensure_finalized()
        cmd.run()

        # Move extension from plumed/plumed/ to plumed/
        # (setuptools creates plumed/plumed/ for "plumed._plumed_core")
        nested_ext_dir = pkg_lib_dir / "plumed"
        if nested_ext_dir.exists():
            import glob
            ext_pattern = str(nested_ext_dir / "_plumed_core*.so")
            ext_files = glob.glob(ext_pattern)
            for ext_file in ext_files:
                dest = pkg_lib_dir / Path(ext_file).name
                shutil.move(ext_file, dest)
                print(f"  Moved {Path(ext_file).name} to {dest}")

                # Fix library paths on macOS using install_name_tool
                if sys.platform == "darwin":
                    self._fix_macos_library_paths(dest, pkg_lib_dir)

            # Remove the nested directory if empty
            try:
                nested_ext_dir.rmdir()
            except OSError:
                pass

        print("  Python bindings compiled successfully!")

    def _fix_macos_library_paths(self, ext_path: Path, pkg_lib_dir: Path) -> None:
        """Fix library paths in macOS extension to use @loader_path.

        Parameters
        ----------
        ext_path : Path
            Path to the compiled extension (.so file).
        pkg_lib_dir : Path
            Package library directory containing the bundled library.
        """
        print(f"  Fixing macOS library paths in {ext_path.name}...")

        # Get current library dependencies
        result = subprocess.run(
            ["otool", "-L", str(ext_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  Warning: otool failed: {result.stderr}")
            return

        # Find and fix libplumedKernel paths
        for line in result.stdout.splitlines():
            line = line.strip()
            if "libplumedKernel" in line:
                # Extract the path (first part before the parenthesis)
                old_path = line.split()[0]
                # New path relative to the extension location
                new_path = "@loader_path/lib/libplumedKernel.dylib"

                change_result = subprocess.run(
                    ["install_name_tool", "-change", old_path, new_path, str(ext_path)],
                    capture_output=True,
                    text=True,
                )

                if change_result.returncode == 0:
                    print(f"  Changed {old_path} -> {new_path}")
                else:
                    print(f"  Warning: install_name_tool failed: {change_result.stderr}")

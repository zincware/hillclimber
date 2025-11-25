"""Custom Hatchling build hook to compile and bundle PLUMED library.

This hook builds the PLUMED C++ library from the git submodule and installs it
directly into src/plumed/_lib. This enables hillclimber to work without requiring
users to separately install the PLUMED library.
"""

import glob
import multiprocessing
import shutil
import subprocess
import sys
from pathlib import Path

from Cython.Build import cythonize
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


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
        install_dir = Path(self.root) / "src" / "plumed" / "_lib"

        # Verify submodule exists
        if not plumed_src.exists() or not (plumed_src / "configure").exists():
            raise RuntimeError(
                f"PLUMED source not found at {plumed_src}. "
                "Please run 'git submodule update --init --recursive'"
            )

        # Clean previous install
        if install_dir.exists():
            shutil.rmtree(install_dir)
        install_dir.mkdir(parents=True, exist_ok=True)

        # Build PLUMED directly into src/plumed/_lib
        self._configure_plumed(plumed_src, install_dir)
        self._build_plumed(plumed_src)
        self._install_plumed(plumed_src)

        # Build Python bindings with Cython
        pkg_dir = Path(self.root) / "src" / "plumed"
        self._build_python_bindings(install_dir, pkg_dir)

        # Include plumed package in wheel (maps src/plumed -> plumed in wheel)
        build_data.setdefault("force_include", {})[str(pkg_dir)] = "plumed"

        # Mark wheel as platform-specific (contains compiled code)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        print("=" * 70)
        print("PLUMED Build Hook: Completed successfully")
        print("=" * 70)

    def _configure_plumed(self, src_dir: Path, install_dir: Path) -> None:
        """Configure PLUMED with appropriate flags.

        Parameters
        ----------
        src_dir : Path
            PLUMED source directory.
        install_dir : Path
            Installation directory (src/plumed/_lib).
        """
        print("\n[1/4] Configuring PLUMED...")

        configure_cmd = [
            "./configure",
            f"--prefix={install_dir}",
            "--disable-dependency-tracking",
            "--enable-modules=all",
            "--enable-shared",
            "--disable-static",
        ]

        # Platform-specific configuration
        if sys.platform == "darwin":
            configure_cmd.extend(
                [
                    "--disable-ld-r",
                    "LDFLAGS=-Wl,-rpath,@loader_path",
                ]
            )
        elif sys.platform.startswith("linux"):
            configure_cmd.extend(
                [
                    "LDFLAGS=-Wl,-rpath,$ORIGIN",
                    "CXXFLAGS=-Wno-error",
                ]
            )

        print(f"Configure command: {' '.join(configure_cmd)}")
        print(f"Running in: {src_dir}")

        subprocess.check_call(
            configure_cmd,
            cwd=src_dir,
        )

    def _build_plumed(self, src_dir: Path) -> None:
        """Build PLUMED library.

        Parameters
        ----------
        src_dir : Path
            PLUMED source directory containing configured build.
        """
        print("\n[2/4] Building PLUMED...")

        njobs = multiprocessing.cpu_count()

        make_cmd = ["make", f"-j{njobs}"]
        print(f"Build command: {' '.join(make_cmd)}")

        subprocess.check_call(
            make_cmd,
            cwd=src_dir,
        )

    def _install_plumed(self, src_dir: Path) -> None:
        """Install PLUMED to src/plumed/_lib.

        Parameters
        ----------
        src_dir : Path
            PLUMED source directory.
        """
        print("\n[3/4] Installing PLUMED...")

        subprocess.check_call(
            ["make", "install"],
            cwd=src_dir,
        )

        print("Installation complete.")

    def _build_python_bindings(self, install_dir: Path, pkg_dir: Path) -> None:
        """Build PLUMED Python bindings using Cython.

        Parameters
        ----------
        install_dir : Path
            PLUMED installation directory (_lib).
        pkg_dir : Path
            Package directory (src/plumed).
        """
        print("\n[4/4] Building Python bindings with Cython...")

        python_src = Path(self.root) / "external" / "plumed2" / "python"
        pyx_file = python_src / "plumed.pyx"
        pxd_file = python_src / "cplumed.pxd"

        if not pyx_file.exists() or not pxd_file.exists():
            print("  Warning: Cython source files not found, skipping Python bindings")
            return

        # Copy binding sources to temp build location
        temp_build = Path(self.root) / "build" / "plumed_bindings"
        temp_build.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pyx_file, temp_build / "_plumed_core.pyx")
        shutil.copy2(pxd_file, temp_build / "cplumed.pxd")

        # Use headers and library from _lib
        include_dir = install_dir / "include" / "plumed" / "wrapper"
        lib_dir = install_dir / "lib"

        extensions = [
            Extension(
                "plumed._plumed_core",
                [str(temp_build / "_plumed_core.pyx")],
                include_dirs=[str(include_dir)],
                library_dirs=[str(lib_dir)],
                libraries=["plumedKernel"],
                runtime_library_dirs=[str(lib_dir)]
                if sys.platform.startswith("linux")
                else [],
                extra_compile_args=[
                    "-D__PLUMED_HAS_DLOPEN",
                    "-D__PLUMED_WRAPPER_LINK_RUNTIME=1",
                    "-D__PLUMED_WRAPPER_IMPLEMENTATION=1",
                    "-D__PLUMED_WRAPPER_EXTERN=0",
                ],
            )
        ]

        print("  Cythonizing _plumed_core.pyx...")
        ext_modules = cythonize(
            extensions, language_level=3, compiler_directives={"embedsignature": True}
        )

        print("  Compiling extension...")
        dist = Distribution({"ext_modules": ext_modules})
        cmd = build_ext(dist)
        cmd.build_lib = str(pkg_dir)
        cmd.build_temp = str(temp_build / "temp")
        cmd.inplace = False
        cmd.ensure_finalized()
        cmd.run()

        # Move extension from nested plumed/plumed/ to plumed/
        nested_ext_dir = pkg_dir / "plumed"
        if nested_ext_dir.exists():
            ext_pattern = str(nested_ext_dir / "_plumed_core*.so")
            for ext_file in glob.glob(ext_pattern):
                dest = pkg_dir / Path(ext_file).name
                shutil.move(ext_file, dest)
                print(f"  Moved {Path(ext_file).name} to {dest}")

                # Fix library paths on macOS
                if sys.platform == "darwin":
                    self._fix_macos_library_paths(dest)

            try:
                nested_ext_dir.rmdir()
            except OSError:
                pass

        print("  Python bindings compiled successfully!")

    def _fix_macos_library_paths(self, ext_path: Path) -> None:
        """Fix library paths in macOS extension to use @loader_path."""
        print(f"  Fixing macOS library paths in {ext_path.name}...")

        result = subprocess.run(
            ["otool", "-L", str(ext_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return

        for line in result.stdout.splitlines():
            line = line.strip()
            if "libplumedKernel" in line:
                old_path = line.split()[0]
                new_path = "@loader_path/_lib/lib/libplumedKernel.dylib"

                subprocess.run(
                    ["install_name_tool", "-change", old_path, new_path, str(ext_path)],
                    capture_output=True,
                )
                print(f"  Changed {old_path} -> {new_path}")

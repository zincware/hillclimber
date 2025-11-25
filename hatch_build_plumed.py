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
        src_dir = Path(self.root) / "src"

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

        # Build pycv plugin (PythonCVInterface and plumedCommunications)
        self._build_pycv(install_dir, src_dir)

        # Include plumed package in wheel (maps src/plumed -> plumed in wheel)
        build_data.setdefault("force_include", {})[str(pkg_dir)] = "plumed"

        # Include plumedCommunications.*.so as top-level module
        for so_file in glob.glob(str(src_dir / "plumedCommunications*.so")):
            build_data["force_include"][so_file] = Path(so_file).name

        # Include plumedCommunications.pyi type stub for IDE autocompletion
        pyi_file = src_dir / "plumedCommunications.pyi"
        if pyi_file.exists():
            build_data["force_include"][str(pyi_file)] = "plumedCommunications.pyi"

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
            "--disable-python",  # We build our own Python bindings with Cython
        ]

        # Platform-specific configuration
        if sys.platform == "darwin":
            configure_cmd.extend(
                [
                    "--disable-ld-r",
                    "LDFLAGS=-Wl,-rpath,@loader_path/../lib",
                ]
            )
        elif sys.platform.startswith("linux"):
            configure_cmd.extend(
                [
                    "LDFLAGS=-Wl,-rpath,$ORIGIN/../lib",
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

        make_cmd = ["make", "lib", f"-j{njobs}"]
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
                runtime_library_dirs=["$ORIGIN/_lib/lib"]
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

    def _build_pycv(self, install_dir: Path, src_dir: Path) -> None:
        """Build pycv plugin and plumedCommunications module.

        Parameters
        ----------
        install_dir : Path
            PLUMED installation directory (_lib).
        src_dir : Path
            Source directory (src/) where plumedCommunications.so will be installed.
        """
        print("\n[5/5] Building pycv plugin...")

        pycv_src = Path(self.root) / "external" / "plumed2" / "plugins" / "pycv"
        if not pycv_src.exists():
            print("  Warning: pycv source not found, skipping pycv build")
            return

        build_dir = Path(self.root) / "build" / "pycv_wheel"

        # Clean previous build
        if build_dir.exists():
            shutil.rmtree(build_dir)

        # Copy source to avoid modifying submodule
        shutil.copytree(pycv_src / "src", build_dir / "src")

        # Patch ActionWithPython.cpp to remove EnsureGlobalDLOpen
        # This static initializer causes issues when loaded from Python
        # because it tries to dladdr on Py_Initialize which may not be resolved yet
        self._patch_pycv_source(build_dir)

        # Create patched CMakeLists.txt
        self._create_pycv_cmake(build_dir, install_dir, src_dir)

        cmake_build_dir = build_dir / "build"
        cmake_build_dir.mkdir(exist_ok=True)

        # Get pybind11 cmake dir (from build dependency)
        import cmake
        import pybind11

        pybind11_cmake = pybind11.get_cmake_dir()
        cmake_bin = Path(cmake.CMAKE_BIN_DIR) / "cmake"

        # Platform-specific settings
        if sys.platform == "darwin":
            lib_suffix = ".dylib"
            rpath = "@loader_path/plumed/_lib/lib"
        else:
            lib_suffix = ".so"
            rpath = "$ORIGIN/plumed/_lib/lib"

        kernel_lib = install_dir / "lib" / f"libplumedKernel{lib_suffix}"

        cmake_args = [
            str(cmake_bin),
            "-S",
            str(build_dir),
            "-B",
            str(cmake_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_cmake}",
            f"-DPlumed_INCLUDEDIR={install_dir / 'include'}",
            f"-DPLUMED_KERNEL_LIB={kernel_lib}",
            f"-DPYCV_LIB_DESTINATION={install_dir / 'lib'}",
            f"-DPYCV_MODULE_DESTINATION={src_dir}",
            f"-DCMAKE_INSTALL_RPATH={rpath}",
            "-DCMAKE_MACOSX_RPATH=ON",
        ]

        print(f"  CMake configure: {' '.join(cmake_args)}")
        subprocess.check_call(cmake_args)

        print("  Building pycv...")
        njobs = multiprocessing.cpu_count()
        subprocess.check_call(
            [str(cmake_bin), "--build", str(cmake_build_dir), "-j", str(njobs)]
        )

        print("  Installing pycv...")
        subprocess.check_call([str(cmake_bin), "--install", str(cmake_build_dir)])

        # Fix library paths on macOS
        if sys.platform == "darwin":
            self._fix_pycv_library_paths(src_dir, install_dir)

        print("  pycv plugin built successfully!")

    def _patch_pycv_source(self, build_dir: Path) -> None:
        """Apply patches to pycv source for hillclimber compatibility.

        These patches are required because hillclimber loads the pycv plugin
        INTO an already-running Python interpreter (via ASE's PLUMED calculator),
        rather than starting a new embedded interpreter.

        Patches Applied
        ---------------
        1. PythonCVInterface.h: Change dataContainer from py::dict to unique_ptr
           - Reason: py::dict{} default constructor calls PyDict_New() which
             requires the GIL, but C++ member initialization happens BEFORE
             the constructor body can acquire the GIL.
           - Fix: Defer initialization to constructor body after GIL acquisition.

        2. PythonCVInterface.cpp: Initialize dataContainer + import module
           - Reason: Must initialize _dataContainerPtr after acquiring GIL.
           - Also imports plumedCommunications to register pybind11 types.

        3. PlumedPythonEmbeddedModule.cpp: Use accessor for dataContainer
           - Reason: dataContainer is now accessed via unique_ptr, need accessor.

        4. PythonCVInterface.cpp & PythonFunction.cpp: Import plumedCommunications
           - Reason: When PLUMED loads PythonCVInterface.dylib via LOAD FILE=,
             and the C++ code calls pyCalculate(this), pybind11 needs to convert
             the C++ pointer to a Python object. The type binding is registered
             in plumedCommunications.so, not in PythonCVInterface.dylib.
           - Fix: Import plumedCommunications before any type conversions to
             ensure the type bindings are registered in pybind11's registry.
           - Without this: "TypeError: Unregistered type : PLMD::pycv::..."

        Parameters
        ----------
        build_dir : Path
            Build directory containing copied pycv source.
        """
        src_dir = build_dir / "src"

        # Note: We keep EnsureGlobalDLOpen(&Py_Initialize) as it's needed to make
        # Python symbols globally visible when PLUMED loads PythonCVInterface.
        # This is required on macOS for proper symbol resolution.

        # Patch 1: PythonCVInterface.h - Defer dataContainer initialization
        pycv_h = src_dir / "PythonCVInterface.h"
        if pycv_h.exists():
            content = pycv_h.read_text()
            patched = content.replace(
                "::pybind11::dict dataContainer {};",
                "std::unique_ptr<::pybind11::dict> _dataContainerPtr;\n"
                "  ::pybind11::dict& dataContainer_ref() { return *_dataContainerPtr; }",
            )
            pycv_h.write_text(patched)
            print("  Patched PythonCVInterface.h")

        # Patch 2: PythonCVInterface.cpp - Init dataContainer + import module
        pycv_cpp = src_dir / "PythonCVInterface.cpp"
        if pycv_cpp.exists():
            content = pycv_cpp.read_text()
            patched = content.replace(
                "    py::gil_scoped_acquire gil;\n    //Loading the python module",
                "    py::gil_scoped_acquire gil;\n"
                "    _dataContainerPtr = std::make_unique<py::dict>();\n"
                '    py::module::import("plumedCommunications");\n'
                "    //Loading the python module",
            )
            pycv_cpp.write_text(patched)
            print("  Patched PythonCVInterface.cpp")

        # Patch 3: PlumedPythonEmbeddedModule.cpp - Use dataContainer accessor
        embed_cpp = src_dir / "PlumedPythonEmbeddedModule.cpp"
        if embed_cpp.exists():
            content = embed_cpp.read_text()
            patched = content.replace(
                '.def_readwrite("data",&PLMD::pycv::PythonCVInterface::dataContainer,',
                '.def_property("data",\n'
                "    [](PLMD::pycv::PythonCVInterface* self) -> py::dict& {\n"
                "      return self->dataContainer_ref();\n"
                "    },\n"
                "    [](PLMD::pycv::PythonCVInterface* self, py::dict& val) {\n"
                "      *(self->_dataContainerPtr) = val;\n"
                "    },",
            )
            embed_cpp.write_text(patched)
            print("  Patched PlumedPythonEmbeddedModule.cpp")

        # Patch 4: PythonFunction.cpp - Import plumedCommunications
        pyfn_cpp = src_dir / "PythonFunction.cpp"
        if pyfn_cpp.exists():
            content = pyfn_cpp.read_text()
            patched = content.replace(
                "    py::gil_scoped_acquire gil;\n    //Loading the python module",
                "    py::gil_scoped_acquire gil;\n"
                '    py::module::import("plumedCommunications");\n'
                "    //Loading the python module",
            )
            pyfn_cpp.write_text(patched)
            print("  Patched PythonFunction.cpp")

    def _create_pycv_cmake(
        self, build_dir: Path, install_dir: Path, src_dir: Path
    ) -> None:
        """Create CMakeLists.txt for pycv build.

        Parameters
        ----------
        build_dir : Path
            Build directory for pycv.
        install_dir : Path
            PLUMED installation directory.
        src_dir : Path
            Source directory for module installation.
        """
        cmake_content = """cmake_minimum_required(VERSION 3.15...3.27)
project(pycv_hillclimber VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Use provided PLUMED paths instead of find_package
set(Plumed_INCLUDEDIR "${Plumed_INCLUDEDIR}" CACHE PATH "PLUMED include dir")
set(PLUMED_KERNEL_LIB "${PLUMED_KERNEL_LIB}" CACHE FILEPATH "PLUMED kernel library")
set(PYCV_LIB_DESTINATION "lib" CACHE PATH "Install destination for PythonCVInterface")
set(PYCV_MODULE_DESTINATION "." CACHE PATH "Install destination for plumedCommunications")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fno-gnu-unique USE_NO_GNU_UNIQUE)
if(USE_NO_GNU_UNIQUE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-gnu-unique")
endif()

# PythonCVInterface shared library (PLUMED plugin)
# IMPORTANT: Do NOT link against pybind11::embed or libpython!
# This library will be loaded INTO an already-running Python interpreter.
# Python symbols will be resolved at load time from the running interpreter.
add_library(PythonCVInterface SHARED
    src/ActionWithPython.cpp
    src/PythonCVInterface.cpp
    src/PythonFunction.cpp)

target_include_directories(PythonCVInterface PUBLIC
    src
    ${Plumed_INCLUDEDIR}
    ${Plumed_INCLUDEDIR}/plumed
    ${Python_INCLUDE_DIRS})

# Use pybind11::module target for proper extension module setup
# This sets up necessary compile definitions without linking Python libraries
# The Python symbols will be resolved from the loading Python process
target_link_libraries(PythonCVInterface PRIVATE pybind11::module)
target_link_libraries(PythonCVInterface PUBLIC ${PLUMED_KERNEL_LIB})
set_target_properties(PythonCVInterface PROPERTIES PREFIX "")

# PythonCVInterface lives in same dir as libplumedKernel
# Use platform-appropriate RPATH ($ORIGIN on Linux, @loader_path on macOS)
if(UNIX AND NOT APPLE)
  set_target_properties(PythonCVInterface PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# On macOS, allow undefined symbols (they'll be resolved from Python at load time)
if(APPLE)
  target_link_options(PythonCVInterface PRIVATE "-undefined" "dynamic_lookup")
endif()

# On Linux, allow undefined symbols as well
if(UNIX AND NOT APPLE)
  target_link_options(PythonCVInterface PRIVATE "-Wl,--allow-shlib-undefined")
endif()

install(TARGETS PythonCVInterface DESTINATION ${PYCV_LIB_DESTINATION})

# plumedCommunications Python module
pybind11_add_module(plumedCommunications src/PlumedPythonEmbeddedModule.cpp)
target_link_libraries(plumedCommunications PRIVATE pybind11::headers)
target_link_libraries(plumedCommunications PUBLIC PythonCVInterface)

# plumedCommunications is at root, needs to find libs in plumed/_lib/lib/
# Use platform-appropriate RPATH (passed via CMAKE_INSTALL_RPATH)
if(UNIX AND NOT APPLE)
  set_target_properties(plumedCommunications PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
    BUILD_WITH_INSTALL_RPATH TRUE)
endif()

install(TARGETS plumedCommunications DESTINATION ${PYCV_MODULE_DESTINATION})
"""
        cmake_file = build_dir / "CMakeLists.txt"
        cmake_file.write_text(cmake_content)

    def _fix_pycv_library_paths(self, src_dir: Path, install_dir: Path) -> None:
        """Fix library paths for pycv on macOS.

        Parameters
        ----------
        src_dir : Path
            Source directory containing plumedCommunications.so.
        install_dir : Path
            PLUMED installation directory containing PythonCVInterface.dylib.
        """
        # Fix PythonCVInterface.dylib install name and library references
        pycv_lib = install_dir / "lib" / "PythonCVInterface.dylib"
        if pycv_lib.exists():
            print(f"  Fixing install name for {pycv_lib.name}...")
            subprocess.run(
                [
                    "install_name_tool",
                    "-id",
                    "@loader_path/PythonCVInterface.dylib",
                    str(pycv_lib),
                ],
                capture_output=True,
            )

            # Fix libplumedKernel reference in PythonCVInterface.dylib
            result = subprocess.run(
                ["otool", "-L", str(pycv_lib)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if "libplumedKernel" in line and not line.startswith("@"):
                        old_path = line.split()[0]
                        new_path = "@loader_path/libplumedKernel.dylib"
                        subprocess.run(
                            [
                                "install_name_tool",
                                "-change",
                                old_path,
                                new_path,
                                str(pycv_lib),
                            ],
                            capture_output=True,
                        )
                        print(f"    Changed {old_path} -> {new_path}")

        # Fix plumedCommunications.so library references
        for so_file in glob.glob(str(src_dir / "plumedCommunications*.so")):
            so_path = Path(so_file)
            print(f"  Fixing library paths in {so_path.name}...")

            result = subprocess.run(
                ["otool", "-L", str(so_path)],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                continue

            for line in result.stdout.splitlines():
                line = line.strip()
                # Fix PythonCVInterface reference
                if "PythonCVInterface" in line and not line.startswith("@"):
                    old_path = line.split()[0]
                    new_path = "@loader_path/plumed/_lib/lib/PythonCVInterface.dylib"
                    subprocess.run(
                        [
                            "install_name_tool",
                            "-change",
                            old_path,
                            new_path,
                            str(so_path),
                        ],
                        capture_output=True,
                    )
                    print(f"    Changed {old_path} -> {new_path}")

                # Fix libplumedKernel reference
                if "libplumedKernel" in line and not line.startswith("@"):
                    old_path = line.split()[0]
                    new_path = "@loader_path/plumed/_lib/lib/libplumedKernel.dylib"
                    subprocess.run(
                        [
                            "install_name_tool",
                            "-change",
                            old_path,
                            new_path,
                            str(so_path),
                        ],
                        capture_output=True,
                    )
                    print(f"    Changed {old_path} -> {new_path}")

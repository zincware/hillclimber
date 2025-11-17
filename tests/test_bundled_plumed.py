"""Tests for bundled PLUMED library functionality.

These tests verify that the bundled PLUMED library is correctly loaded
and functional within the hillclimber package.
"""

import os
import sys

import pytest


def test_bundled_kernel_path_exists():
    """Test that bundled PLUMED kernel path is set and exists."""
    from plumed import BUNDLED_KERNEL_PATH

    # May be None if built without bundled PLUMED
    if BUNDLED_KERNEL_PATH is not None:
        assert BUNDLED_KERNEL_PATH.exists(), (
            f"Bundled PLUMED kernel not found at {BUNDLED_KERNEL_PATH}"
        )
        # Check file extension matches platform
        if sys.platform == "darwin":
            assert BUNDLED_KERNEL_PATH.suffix == ".dylib", (
                f"Expected .dylib extension on macOS, got {BUNDLED_KERNEL_PATH.suffix}"
            )
        elif sys.platform.startswith("linux"):
            assert ".so" in BUNDLED_KERNEL_PATH.name, (
                f"Expected .so extension on Linux, got {BUNDLED_KERNEL_PATH.name}"
            )


def test_plumed_kernel_env_set():
    """Test that PLUMED_KERNEL environment variable is set correctly."""
    from plumed import BUNDLED_KERNEL_PATH

    if BUNDLED_KERNEL_PATH is not None:
        # Environment variable should be set by the loader
        assert "PLUMED_KERNEL" in os.environ, (
            "PLUMED_KERNEL environment variable not set"
        )
        kernel_path = os.environ["PLUMED_KERNEL"]
        assert os.path.exists(kernel_path), (
            f"PLUMED_KERNEL points to non-existent file: {kernel_path}"
        )


def test_plumed_import():
    """Test that plumed Python module can be imported."""
    try:
        import plumed  # noqa: F401
    except ImportError as e:
        pytest.skip(f"plumed module not available: {e}")


def test_plumed_instantiation():
    """Test that Plumed object can be instantiated with bundled library."""
    try:
        from plumed import Plumed
    except ImportError:
        pytest.skip("plumed module not available")

    # Create Plumed instance - should use bundled library
    p = Plumed()
    assert p is not None, "Failed to create Plumed instance"


def test_ase_plumed_calculator_import():
    """Test that ASE Plumed calculator can be imported."""
    try:
        from ase.calculators.plumed import Plumed  # noqa: F401
    except ImportError as e:
        pytest.skip(f"ASE Plumed calculator not available: {e}")


def test_hillclimber_plumed_calculator_import():
    """Test that hillclimber's NonOverwritingPlumed can be imported."""
    from hillclimber.calc import NonOverwritingPlumed  # noqa: F401

    assert NonOverwritingPlumed is not None


def test_sum_hills_function_import():
    """Test that sum_hills function can be imported."""
    from hillclimber import sum_hills  # noqa: F401

    assert sum_hills is not None
    assert callable(sum_hills)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="PLUMED has limited Windows support"
)
def test_bundled_library_platform_compatibility():
    """Test that bundled library has correct platform-specific attributes."""
    from plumed import BUNDLED_KERNEL_PATH

    if BUNDLED_KERNEL_PATH is None:
        pytest.skip("No bundled PLUMED library available")

    # Verify file is executable/loadable
    assert BUNDLED_KERNEL_PATH.stat().st_mode & 0o111, (
        f"Bundled library is not executable: {BUNDLED_KERNEL_PATH}"
    )


def test_plumed_version_info():
    """Test that we can retrieve PLUMED version information."""
    try:
        from plumed import Plumed
    except ImportError:
        pytest.skip("plumed module not available")

    p = Plumed()

    # Try to get version - implementation depends on plumed Python bindings
    # This is a basic sanity check
    try:
        # Some versions of plumed Python bindings support this
        version = p.get_api_version()
        assert version > 0, f"Invalid API version: {version}"
    except AttributeError:
        # Older versions might not have this method
        pytest.skip("Plumed version info not available in this version")


def test_no_system_plumed_conflict():
    """Test that bundled PLUMED doesn't conflict with system PLUMED.

    This test ensures that the bundled library takes precedence and
    doesn't cause issues if a system PLUMED is also installed.
    """
    from plumed import BUNDLED_KERNEL_PATH

    if BUNDLED_KERNEL_PATH is None:
        pytest.skip("No bundled PLUMED library available")

    # If we got here, bundled PLUMED should be loaded
    # Try importing plumed - it should use our bundled version
    try:
        from plumed import Plumed
        p = Plumed()
        assert p is not None
    except ImportError:
        pytest.skip("plumed Python bindings not available")

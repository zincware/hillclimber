"""Tests for pycv plugin integration.

These tests verify that:
1. The pycv plugin is correctly built and bundled
2. plumedCommunications module is importable
3. pycv works with ASE MD simulations
4. Type stubs match the runtime module
"""

import os
import sys

import pytest


class TestPycvPath:
    """Tests for pycv plugin path."""

    def test_get_pycv_path_returns_string(self):
        """Test that get_pycv_path returns a string."""
        import plumed

        pycv_path = plumed.get_pycv_path()
        assert isinstance(pycv_path, str)

    def test_pycv_path_exists(self):
        """Test that pycv plugin file exists."""
        import plumed

        pycv_path = plumed.get_pycv_path()
        assert os.path.exists(pycv_path)

    def test_pycv_path_is_shared_library(self):
        """Test that pycv path points to shared library."""
        import plumed

        pycv_path = plumed.get_pycv_path()
        assert pycv_path.endswith((".dylib", ".so"))


class TestPlumedCommunicationsImport:
    """Tests for plumedCommunications module import."""

    def test_import_top_level(self):
        """Test top-level plumedCommunications import."""
        import plumedCommunications as PLMD

        assert PLMD is not None

    def test_defaults_submodule(self):
        """Test defaults submodule is available."""
        import plumedCommunications as PLMD

        assert hasattr(PLMD, "defaults")
        assert hasattr(PLMD.defaults, "COMPONENT")
        assert hasattr(PLMD.defaults, "COMPONENT_NODEV")

    def test_component_defaults_structure(self):
        """Test COMPONENT defaults have correct structure."""
        import plumedCommunications as PLMD

        component = PLMD.defaults.COMPONENT
        component_nodev = PLMD.defaults.COMPONENT_NODEV

        # COMPONENT should have derivatives enabled
        assert component["derivative"] is True
        assert component["period"] is None

        # COMPONENT_NODEV should have derivatives disabled
        assert component_nodev["derivative"] is False
        assert component_nodev["period"] is None

    def test_python_cv_interface_class(self):
        """Test PythonCVInterface class is available."""
        import plumedCommunications as PLMD

        assert hasattr(PLMD, "PythonCVInterface")

    def test_python_function_class(self):
        """Test PythonFunction class is available."""
        import plumedCommunications as PLMD

        assert hasattr(PLMD, "PythonFunction")

    def test_python_cv_interface_has_docstrings(self):
        """Test that PythonCVInterface methods have docstrings for help()."""
        import plumedCommunications as PLMD

        # These docstrings come from pybind11 definitions
        assert PLMD.PythonCVInterface.getStep.__doc__ is not None
        assert PLMD.PythonCVInterface.getPositions.__doc__ is not None

    def test_stub_file_exists(self):
        """Test that the .pyi stub file exists for IDE autocompletion."""
        from pathlib import Path

        stub_path = Path(__file__).parent.parent / "src" / "plumedCommunications.pyi"
        assert stub_path.exists(), f"Stub file not found at {stub_path}"


class TestPycvIntegration:
    """Integration tests for pycv with ASE."""

    @pytest.fixture
    def pycv_path(self):
        """Get pycv plugin path."""
        import plumed

        return plumed.get_pycv_path()

    def test_pycv_pyfunction_with_ase(self, pycv_path, tmp_path):
        """Integration test: PYFUNCTION action with ASE simulation."""
        import ase
        import ase.units
        from ase.calculators.lj import LennardJones
        from ase.calculators.plumed import Plumed
        from ase.md.langevin import Langevin

        # Create test function module
        fn_code = """
import plumedCommunications as PLMD
import numpy as np

plumedInit = {"Value": PLMD.defaults.COMPONENT_NODEV}

def plumedCalculate(action: PLMD.PythonFunction):
    args = action.arguments()
    return float(np.sum(args))
"""
        fn_file = tmp_path / "test_fn.py"
        fn_file.write_text(fn_code)

        sys.path.insert(0, str(tmp_path))

        # Create Argon dimer
        system = ase.Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])

        # LJ calculator
        lj_calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=10.0, smooth=True)

        # PLUMED input with PYFUNCTION
        colvar_file = tmp_path / "colvar_fn.out"

        plumed_input = [
            f"LOAD FILE={pycv_path}",
            "d: DISTANCE ATOMS=1,2",
            "fn: PYFUNCTION ARG=d IMPORT=test_fn",
            f"PRINT FILE={colvar_file} ARG=fn STRIDE=1",
        ]

        calc = Plumed(
            calc=lj_calc,
            input=plumed_input,
            timestep=2.0 * ase.units.fs,
            atoms=system,
            kT=120.0 * ase.units.kB,
        )

        system.calc = calc

        # Run short MD
        dyn = Langevin(
            atoms=system,
            timestep=2.0 * ase.units.fs,
            temperature_K=120.0,
            friction=0.01,
        )

        dyn.run(steps=5)

        assert colvar_file.exists()

        # Verify output has data (at least initial step)
        colvar_content = colvar_file.read_text()
        lines = [
            line
            for line in colvar_content.strip().split("\n")
            if line and not line.startswith("#")
        ]
        assert len(lines) >= 1

    def test_pycv_interface_with_ase(self, pycv_path, tmp_path):
        """Integration test: PYCVINTERFACE action with ASE simulation."""
        import ase
        import ase.units
        from ase.calculators.lj import LennardJones
        from ase.calculators.plumed import Plumed
        from ase.md.langevin import Langevin

        # Create test CV module that computes distance
        cv_code = """
import plumedCommunications as PLMD
import numpy as np

plumedInit = {"Value": PLMD.defaults.COMPONENT}

def plumedCalculate(action: PLMD.PythonCVInterface):
    x = action.getPositions()
    diff = x[1] - x[0]
    dist = float(np.sqrt(np.sum(diff**2)))

    # Gradient: d(dist)/d(r_i)
    grad = np.zeros((2, 3))
    grad[0] = -diff / dist
    grad[1] = diff / dist

    # Box gradient (zeros)
    box_grad = np.zeros((3, 3))

    return dist, grad, box_grad
"""
        cv_file = tmp_path / "test_cv.py"
        cv_file.write_text(cv_code)

        sys.path.insert(0, str(tmp_path))

        # Create Argon dimer
        system = ase.Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])

        # LJ calculator
        lj_calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=10.0, smooth=True)

        # PLUMED input with PYCVINTERFACE
        colvar_file = tmp_path / "colvar_cv.out"

        plumed_input = [
            f"LOAD FILE={pycv_path}",
            "cv: PYCVINTERFACE ATOMS=1,2 IMPORT=test_cv",
            f"PRINT FILE={colvar_file} ARG=cv STRIDE=1",
        ]

        calc = Plumed(
            calc=lj_calc,
            input=plumed_input,
            timestep=2.0 * ase.units.fs,
            atoms=system,
            kT=120.0 * ase.units.kB,
        )

        system.calc = calc

        # Run short MD
        dyn = Langevin(
            atoms=system,
            timestep=2.0 * ase.units.fs,
            temperature_K=120.0,
            friction=0.01,
        )

        dyn.run(steps=5)

        assert colvar_file.exists()

        # Verify output has data
        colvar_content = colvar_file.read_text()
        lines = [
            line
            for line in colvar_content.strip().split("\n")
            if line and not line.startswith("#")
        ]
        assert len(lines) >= 1

        # Verify first CV value is close to initial distance
        # Note: PLUMED uses nm internally, ASE uses Angstrom
        # 3.8 Angstrom = 0.38 nm
        first_line = lines[0].split()
        cv_value = float(first_line[1])
        assert 0.3 < cv_value < 0.5, f"Expected distance ~0.38 nm, got {cv_value}"

    def test_pycvinterface_md_simulation(self, pycv_path, tmp_path):
        """Integration test: Run actual MD simulation with PYCVINTERFACE.

        This test verifies that PYCVINTERFACE works correctly during
        a multi-step MD simulation with Langevin dynamics.
        """
        import ase
        import ase.units
        from ase.calculators.lj import LennardJones
        from ase.calculators.plumed import Plumed
        from ase.md.langevin import Langevin

        # Create test CV module with distance calculation
        cv_code = """
import plumedCommunications as PLMD
import numpy as np

plumedInit = {"Value": PLMD.defaults.COMPONENT}

def plumedCalculate(action: PLMD.PythonCVInterface):
    x = action.getPositions()
    diff = x[1] - x[0]
    dist = float(np.sqrt(np.sum(diff**2)))

    # Analytical gradient
    grad = np.zeros((2, 3))
    grad[0] = -diff / dist
    grad[1] = diff / dist

    # Box gradient (zeros for non-periodic)
    box_grad = np.zeros((3, 3))

    return dist, grad, box_grad
"""
        cv_file = tmp_path / "md_cv.py"
        cv_file.write_text(cv_code)

        sys.path.insert(0, str(tmp_path))

        # Create Argon dimer with known initial distance
        initial_distance = 3.8
        system = ase.Atoms(
            "Ar2", positions=[[0.0, 0.0, 0.0], [initial_distance, 0.0, 0.0]]
        )

        # LJ calculator for Argon
        lj_calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=10.0, smooth=True)

        # PLUMED input
        colvar_file = tmp_path / "md_colvar.out"
        plumed_input = [
            f"LOAD FILE={pycv_path}",
            "cv: PYCVINTERFACE ATOMS=1,2 IMPORT=md_cv",
            f"PRINT FILE={colvar_file} ARG=cv STRIDE=1",
        ]

        calc = Plumed(
            calc=lj_calc,
            input=plumed_input,
            timestep=2.0 * ase.units.fs,
            atoms=system,
            kT=120.0 * ase.units.kB,
        )

        system.calc = calc

        # Run MD simulation
        n_steps = 20
        dyn = Langevin(
            atoms=system,
            timestep=2.0 * ase.units.fs,
            temperature_K=120.0,
            friction=0.01,
        )
        dyn.run(steps=n_steps)

        # Verify output file exists and has correct number of entries
        assert colvar_file.exists()
        colvar_content = colvar_file.read_text()
        lines = [
            line
            for line in colvar_content.strip().split("\n")
            if line and not line.startswith("#")
        ]

        # Should have at least some entries from the simulation
        assert len(lines) >= 1

        # Parse all CV values and verify they're physically reasonable
        # Note: PLUMED uses nm internally, ASE uses Angstrom
        cv_values = [float(line.split()[1]) for line in lines]

        # All distances should be positive and reasonable (0.1-1.0 nm = 1-10 Angstrom)
        for cv in cv_values:
            assert 0.1 < cv < 1.0, f"Unreasonable distance: {cv} nm"

        # First value should be close to initial distance (3.8 A = 0.38 nm)
        initial_distance_nm = initial_distance / 10.0  # Convert A to nm
        assert abs(cv_values[0] - initial_distance_nm) < 0.01

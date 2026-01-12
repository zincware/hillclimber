"""Tests for PyCV (Python Collective Variables) implementation.

These tests verify that:
1. PyCV to_plumed generates correct PYCVINTERFACE commands
2. Adapter scripts are correctly generated
3. PyCV integrates properly with MetaDynamicsModel
4. Full output matches expected format (per CLAUDE.md requirements)
"""

import numpy as np
import pytest
from ase import Atoms

import hillclimber as hc
from hillclimber.pycv import PyCV


# --- Simple Test PyCV Subclass ---


class SimpleDistanceCV(PyCV):
    """A simple distance CV for testing - returns distance and zero gradients."""

    def compute(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Compute distance between first two atoms with zero gradients."""
        positions = atoms.get_positions()
        diff = positions[1] - positions[0]
        dist = float(np.sqrt(np.sum(diff**2)))

        # Return zero gradients for simplicity
        grad = np.zeros((len(atoms), 3))

        return dist, grad


# --- Unit Tests ---


class TestPyCVBasic:
    """Basic PyCV functionality tests."""

    def test_pycv_with_atoms_none_selects_all_atoms(self):
        """Test PyCV with atoms=None selects all atoms."""
        cv = SimpleDistanceCV(atoms=None, prefix="d_all")

        atoms = Atoms("Ar3", positions=[[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]])

        labels, commands = cv.to_plumed(atoms)

        expected = [
            "d_all: PYCVINTERFACE ATOMS=1,2,3 IMPORT=_pycv_d_all",
        ]

        assert labels == ["d_all"]
        assert commands == expected

    def test_pycv_with_index_list(self):
        """Test PyCV with direct list of atom indices."""
        cv = SimpleDistanceCV(atoms=[0, 1], prefix="d_simple")

        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.8, 0, 0]])

        labels, commands = cv.to_plumed(atoms)

        expected = [
            "d_simple: PYCVINTERFACE ATOMS=1,2 IMPORT=_pycv_d_simple",
        ]

        assert labels == ["d_simple"]
        assert commands == expected

    def test_pycv_with_index_selector(self):
        """Test PyCV with IndexSelector."""
        cv = SimpleDistanceCV(
            atoms=hc.IndexSelector(indices=[[0, 1]]),
            prefix="d_idx",
        )

        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.8, 0, 0]])

        labels, commands = cv.to_plumed(atoms)

        expected = [
            "d_idx: PYCVINTERFACE ATOMS=1,2 IMPORT=_pycv_d_idx",
        ]

        assert labels == ["d_idx"]
        assert commands == expected

    def test_pycv_empty_selection_raises(self):
        """Test that empty atom selection raises ValueError."""
        cv = SimpleDistanceCV(atoms=[], prefix="d_empty")

        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.8, 0, 0]])

        with pytest.raises(ValueError, match="Empty atom selection"):
            cv.to_plumed(atoms)

    def test_pycv_get_img_returns_image(self):
        """Test that get_img returns a PIL Image."""
        cv = SimpleDistanceCV(atoms=[0, 1], prefix="d_simple")
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.8, 0, 0]])

        img = cv.get_img(atoms)

        from PIL import Image

        assert isinstance(img, Image.Image)


class TestPyCVInitArgs:
    """Tests for get_init_args serialization."""

    def test_init_args_with_none(self):
        """Test get_init_args with atoms=None."""
        cv = SimpleDistanceCV(atoms=None, prefix="d_all")

        init_args = cv.get_init_args()

        assert init_args == "atoms=None, prefix='d_all'"

    def test_init_args_with_list(self):
        """Test get_init_args with list of indices."""
        cv = SimpleDistanceCV(atoms=[0, 1, 2], prefix="d_test")

        init_args = cv.get_init_args()

        assert init_args == "atoms=[0, 1, 2], prefix='d_test'"

    def test_init_args_with_index_selector(self):
        """Test get_init_args with IndexSelector."""
        cv = SimpleDistanceCV(
            atoms=hc.IndexSelector(indices=[[0, 1]]),
            prefix="d_idx",
        )

        init_args = cv.get_init_args()

        assert init_args == "atoms=IndexSelector(indices=[[0, 1]]), prefix='d_idx'"

    def test_init_args_with_smarts_selector(self):
        """Test get_init_args with SMARTSSelector (has multiple fields)."""
        cv = SimpleDistanceCV(
            atoms=hc.SMARTSSelector(pattern="[OH]", hydrogens="include"),
            prefix="d_smarts",
        )

        init_args = cv.get_init_args()

        assert (
            init_args
            == "atoms=SMARTSSelector(pattern='[OH]', hydrogens='include'), prefix='d_smarts'"
        )


class TestPyCVAdapterScript:
    """Tests for adapter script generation."""

    def test_adapter_script_content(self, tmp_path):
        """Test that adapter script has correct structure."""
        cv = SimpleDistanceCV(atoms=[0, 1], prefix="d_test")
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.8, 0, 0]])

        script_path = cv.write_adapter_script(
            directory=tmp_path,
            atoms=atoms,
            cv_class_module="tests.test_pycv_cv",
            cv_class_name="SimpleDistanceCV",
            cv_init_args="atoms=[0, 1], prefix='d_test'",
        )

        assert script_path.exists()
        assert script_path.name == "_pycv_d_test.py"

        content = script_path.read_text()

        # Check key elements
        assert "plumedCommunications" in content
        assert "_SYMBOLS = ['Ar', 'Ar']" in content
        assert "from tests.test_pycv_cv import SimpleDistanceCV" in content
        assert "plumedInit" in content
        assert "plumedCalculate" in content
        assert "Atoms(symbols=_SYMBOLS" in content

    def test_adapter_script_symbols_preserved(self, tmp_path):
        """Test that atomic symbols are correctly preserved in adapter script."""
        cv = SimpleDistanceCV(atoms=[0, 1, 2], prefix="d_multi")
        atoms = Atoms("CHO", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])

        script_path = cv.write_adapter_script(
            directory=tmp_path,
            atoms=atoms,
            cv_class_module="test_module",
            cv_class_name="SimpleDistanceCV",
            cv_init_args="atoms=[0, 1, 2], prefix='d_multi'",
        )

        content = script_path.read_text()
        assert "_SYMBOLS = ['C', 'H', 'O']" in content


class TestPyCVWithMetaDynamics:
    """Integration tests with MetaDynamicsModel."""

    def test_pycv_metad_to_plumed(self):
        """Test full output format from MetaDynamicsModel with PyCV."""
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.8, 0, 0]])
        cv = SimpleDistanceCV(atoms=[0, 1], prefix="pycv")

        bias = hc.MetadBias(
            cv=cv,
            sigma=0.2,
            grid_min=0.0,
            grid_max=10.0,
            grid_bin=200,
        )

        config = hc.MetaDynamicsConfig(
            height=1.0,
            pace=500,
            biasfactor=10.0,
            temp=300.0,
        )

        model = hc.MetaDynamicsModel(
            config=config,
            data=[atoms],
            bias_cvs=[bias],
            model=None,  # type: ignore
        )

        result = model.to_plumed(atoms)

        # LOAD FILE path is dynamic (contains pycv plugin path), so check structure
        assert len(result) == 4
        assert result[0] == "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877"
        assert result[1].startswith("LOAD FILE=") and "PythonCVInterface" in result[1]
        assert result[2] == "pycv: PYCVINTERFACE ATOMS=1,2 IMPORT=_pycv_pycv"
        assert result[3] == (
            "metad: METAD ARG=pycv HEIGHT=1.0 PACE=500 TEMP=300.0 FILE=HILLS "
            "BIASFACTOR=10.0 SIGMA=0.2 GRID_MIN=0.0 GRID_MAX=10.0 GRID_BIN=200"
        )


class TestPyCVCompute:
    """Tests for the compute method behavior."""

    def test_compute_returns_distance_and_gradients(self):
        """Test that compute returns correct distance and gradients."""
        cv = SimpleDistanceCV(atoms=[0, 1], prefix="d")
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.8, 0, 0]])

        value, grad = cv.compute(atoms)

        # Distance should be 3.8
        assert abs(value - 3.8) < 1e-10

        # Gradients should be zeros (our simple test CV)
        assert np.allclose(grad, 0)


class TestPyCVIntegration:
    """Integration tests that run PyCV with actual PLUMED execution."""

    def test_pycv_full_md_simulation(self, tmp_path):
        """Integration test: PyCV with MetaDynamicsModel in actual MD simulation."""
        import ase
        import ase.units
        from ase.calculators.lj import LennardJones
        from ase.md.langevin import Langevin

        # Create Argon dimer
        atoms = ase.Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])

        cv = SimpleDistanceCV(atoms=[0, 1], prefix="pycv_dist")

        bias = hc.MetadBias(cv=cv, sigma=0.1, grid_min=0.0, grid_max=10.0, grid_bin=100)

        config = hc.MetaDynamicsConfig(
            height=0.01,
            pace=5,
            temp=120.0,
        )

        class MockModel:
            def get_calculator(self, **kwargs):
                return LennardJones(sigma=3.4, epsilon=0.0104, rc=10.0, smooth=True)

        model = hc.MetaDynamicsModel(
            config=config,
            data=[atoms],
            bias_cvs=[bias],
            model=MockModel(),  # type: ignore
        )

        # get_calculator automatically adds directory to sys.path for PyCV imports
        calc = model.get_calculator(directory=tmp_path)

        # Verify files were created
        assert (tmp_path / "_pycv_pycv_dist.py").exists()
        assert (tmp_path / "plumed.dat").exists()

        # Run MD simulation
        atoms.calc = calc

        dyn = Langevin(
            atoms=atoms,
            timestep=2.0 * ase.units.fs,
            temperature_K=120.0,
            friction=0.01,
        )

        dyn.run(steps=10)

        # Verify HILLS file was created
        assert (tmp_path / "HILLS").exists()

    def test_pycv_with_selector_adapter_script_generation(
        self, tmp_path, small_ethanol_water
    ):
        """Test that PyCV with AtomSelector generates correct adapter scripts."""
        atoms = small_ethanol_water.copy()

        # Use SMARTSSelector
        oxygen_selector = hc.SMARTSSelector(pattern="[O]")
        cv = SimpleDistanceCV(atoms=oxygen_selector, prefix="pycv_oxygen")

        # Write adapter script
        script_path = cv.write_adapter_script(
            directory=tmp_path,
            atoms=atoms,
            cv_class_module="tests.test_pycv_cv",
            cv_class_name="SimpleDistanceCV",
            cv_init_args=cv.get_init_args(),
        )

        assert script_path.exists()

        content = script_path.read_text()

        # Verify the adapter script imports the selector
        assert "from hillclimber import SMARTSSelector" in content

        # Verify the adapter script contains the selector in CV instantiation
        assert "[O]" in content
        assert "_CV_INSTANCE = SimpleDistanceCV" in content
        assert "plumedCalculate" in content

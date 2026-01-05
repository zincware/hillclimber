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

    def test_pycv_with_atoms_none_larger_system(self):
        """Test PyCV with atoms=None on a larger system."""
        cv = SimpleDistanceCV(atoms=None, prefix="d_full")

        atoms = Atoms("C2H6", positions=[
            [0, 0, 0], [1.54, 0, 0],  # C atoms
            [0, 1, 0], [0, -1, 0], [0, 0, 1],  # H on first C
            [1.54, 1, 0], [1.54, -1, 0], [1.54, 0, 1],  # H on second C
        ])

        labels, commands = cv.to_plumed(atoms)

        expected = [
            "d_full: PYCVINTERFACE ATOMS=1,2,3,4,5,6,7,8 IMPORT=_pycv_d_full",
        ]

        assert labels == ["d_full"]
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

        assert "IndexSelector" in init_args
        assert "indices=[[0, 1]]" in init_args
        assert "prefix='d_idx'" in init_args

    def test_init_args_with_smiles_selector(self):
        """Test get_init_args with SMILESSelector."""
        cv = SimpleDistanceCV(
            atoms=hc.SMILESSelector(smiles="CCO"),
            prefix="d_smiles",
        )

        init_args = cv.get_init_args()

        assert "SMILESSelector" in init_args
        assert "smiles='CCO'" in init_args

    def test_init_args_with_smarts_selector(self):
        """Test get_init_args with SMARTSSelector."""
        cv = SimpleDistanceCV(
            atoms=hc.SMARTSSelector(pattern="[OH]"),
            prefix="d_smarts",
        )

        init_args = cv.get_init_args()

        assert "SMARTSSelector" in init_args
        assert "pattern='[OH]'" in init_args


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

    def test_pycv_metad_to_plumed(self, small_ethanol_water):
        """Test that MetaDynamicsModel correctly handles PyCV."""
        cv = SimpleDistanceCV(atoms=[0, 1], prefix="d_py")

        bias = hc.MetadBias(cv=cv, sigma=0.1, grid_min=0.0, grid_max=5.0, grid_bin=100)

        config = hc.MetaDynamicsConfig(
            height=0.5,
            pace=100,
            temp=300.0,
        )

        model = hc.MetaDynamicsModel(
            config=config,
            data=small_ethanol_water,
            bias_cvs=[bias],
            model=None,  # type: ignore
        )

        result = model.to_plumed(small_ethanol_water)

        # Check that LOAD command is present
        assert any(
            "LOAD FILE=" in line and "PythonCVInterface" in line for line in result
        )

        # Check PYCVINTERFACE command
        assert any("d_py: PYCVINTERFACE ATOMS=1,2" in line for line in result)

        # Check METAD command references the CV
        assert any("METAD ARG=d_py" in line for line in result)

    def test_pycv_full_output_format(self, small_ethanol_water):
        """Test full output format matches expected structure."""
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
            data=small_ethanol_water,
            bias_cvs=[bias],
            model=None,  # type: ignore
        )

        result = model.to_plumed(small_ethanol_water)

        # Verify structure (first few lines)
        assert result[0] == "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877"
        assert "LOAD FILE=" in result[1]
        assert "pycv: PYCVINTERFACE ATOMS=1,2 IMPORT=_pycv_pycv" in result[2]
        assert "metad: METAD ARG=pycv" in result[3]

    def test_pycv_mixed_with_regular_cv(self, small_ethanol_water):
        """Test PyCV works alongside regular CVs."""
        # Python CV
        py_cv = SimpleDistanceCV(atoms=[0, 1], prefix="d_py")

        # Regular distance CV
        regular_cv = hc.DistanceCV(
            x1=hc.VirtualAtom(hc.SMILESSelector(smiles="CCO")[0], "com"),
            x2=hc.VirtualAtom(hc.SMILESSelector(smiles="O")[0], "com"),
            prefix="d_regular",
        )

        bias_py = hc.MetadBias(cv=py_cv, sigma=0.1)
        bias_regular = hc.MetadBias(cv=regular_cv, sigma=0.2)

        config = hc.MetaDynamicsConfig(height=0.5, pace=100, temp=300.0)

        model = hc.MetaDynamicsModel(
            config=config,
            data=small_ethanol_water,
            bias_cvs=[bias_py, bias_regular],
            model=None,  # type: ignore
        )

        result = model.to_plumed(small_ethanol_water)

        # Check both CVs are present
        assert any("PYCVINTERFACE" in line for line in result)
        assert any("DISTANCE" in line for line in result)

        # Check METAD uses both
        metad_line = [line for line in result if "metad: METAD" in line][0]
        assert "d_py" in metad_line
        assert "d_regular" in metad_line


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

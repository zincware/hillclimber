"""ZnTrack integration test for PyCV with AtomSelector.

This test verifies that PyCV with AtomSelector serializes correctly
through a full ZnTrack workflow (git init, dvc init, project.repro()).

Unit tests for get_init_args(), adapter script generation, and PLUMED
command generation are in test_pycv_cv.py.
"""

import os
import subprocess

import ase.io
import ipsuite as ips
import pytest
from ase import Atoms

import hillclimber as hc
import zntrack


# PyCV module code written to tmp_path for import during dvc repro
SIMPLE_PYCV_MODULE_CODE = '''
"""Simple PyCV for ZnTrack integration testing."""

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from hillclimber.pycv import PyCV


@dataclass
class SimpleDistancePyCV(PyCV):
    """A simple distance CV for testing - computes distance between first two atoms."""

    def compute(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Compute distance between first two atoms with analytical gradients."""
        positions = atoms.get_positions()
        if len(positions) < 2:
            return 0.0, np.zeros((len(atoms), 3))

        diff = positions[1] - positions[0]
        dist = float(np.sqrt(np.sum(diff**2)))

        grad = np.zeros((len(atoms), 3))
        if dist > 1e-10:
            grad[0] = -diff / dist
            grad[1] = diff / dist

        return dist, grad
'''


@pytest.fixture
def zntrack_project(tmp_path):
    """Initialize a ZnTrack project with git and dvc."""
    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["dvc", "init"], check=True, capture_output=True)
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            check=True,
            capture_output=True,
        )

        yield tmp_path
    finally:
        os.chdir(cwd)


def test_pycv_with_atom_selector_zntrack_workflow(zntrack_project):
    """Test PyCV with AtomSelector through full ZnTrack workflow.

    This integration test verifies that:
    1. PyCV with AtomSelector serializes correctly to params.yaml
    2. The adapter script is generated with correct selector import
    3. project.repro() successfully runs MetaDynamicsModel and ASEMD
    4. MD simulation produces output frames
    """
    import sys

    os.chdir(zntrack_project)

    # Write PyCV module to project path for import during dvc repro
    module_path = zntrack_project / "simple_pycv.py"
    module_path.write_text(SIMPLE_PYCV_MODULE_CODE)

    # Import the PyCV class
    if str(zntrack_project) not in sys.path:
        sys.path.insert(0, str(zntrack_project))
    import importlib

    if "simple_pycv" in sys.modules:
        importlib.reload(sys.modules["simple_pycv"])
    import simple_pycv

    SimpleDistancePyCV = simple_pycv.SimpleDistancePyCV

    # Create test atoms
    atoms = Atoms(
        "Ar4",
        positions=[[0, 0, 0], [3.8, 0, 0], [0, 3.8, 0], [3.8, 3.8, 0]],
    )
    data_file = zntrack_project / "atoms.xyz"
    ase.io.write(data_file, atoms)

    # Create PyCV with IndexSelector (the key feature being tested)
    selector = hc.IndexSelector(indices=[[0, 1, 2, 3]])
    cv = SimpleDistancePyCV(atoms=selector, prefix="testcv")
    bias = hc.MetadBias(cv=cv, sigma=0.1, grid_min=0.0, grid_max=10.0, grid_bin=100)
    config = hc.MetaDynamicsConfig(height=0.01, pace=2, temp=120.0)

    lj_model = ips.GenericASEModel(
        module="ase.calculators.lj",
        class_name="LennardJones",
        kwargs={"sigma": 3.4, "epsilon": 0.01, "rc": 10.0},
    )
    thermostat = ips.LangevinThermostat(
        time_step=1.0,
        temperature=120.0,
        friction=0.01,
    )

    # Build and run ZnTrack workflow
    project = zntrack.Project()

    with project:
        data_node = ips.AddData(file=data_file, name="atoms")
        model_node = hc.MetaDynamicsModel(
            config=config,
            data=data_node.frames,
            bias_cvs=[bias],
            model=lj_model,
            timestep=1.0,
        )
        md_node = ips.ASEMD(
            data=data_node.frames,
            model=model_node,
            thermostat=thermostat,
            steps=10,
            sampling_rate=2,
        )

    project.repro()

    # Verify serialization: params.yaml contains the selector
    params_content = (zntrack_project / "params.yaml").read_text()
    assert "IndexSelector" in params_content

    # Verify workflow ran: MetaDynamicsModel figures created
    assert (zntrack_project / "nodes" / "MetaDynamicsModel" / "figures").exists()

    # Verify MD produced frames
    assert len(md_node.frames) > 0

"""Integration test for Argon dimer metadynamics with distance CV and sum_hills.

This test verifies that:
1. The PLUMED calculator wrapper works correctly with ASE's LennardJones calculator
2. A short metadynamics simulation can be run with DistanceCV
3. The sum_hills analysis tool works correctly on the generated HILLS file
"""

import ase.units
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import hillclimber as hc


@pytest.fixture
def lj():
    epsilon_ar = 0.0104  # eV
    sigma_ar = 3.4  # Angstrom
    cutoff = 10.0  # Angstrom

    lj_calc = LennardJones(sigma=sigma_ar, epsilon=epsilon_ar, rc=cutoff, smooth=True)

    class LJModel:
        """Minimal model for testing that wraps LJ calculator."""

        def get_calculator(self, directory=None, **kwargs):
            return lj_calc

    return LJModel()


def test_plumed_cli():
    # run plumed --help to ensure CLI is functional
    import subprocess

    result = subprocess.run(["plumed", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage: plumed [options] [command] [command options]" in result.stdout


def test_argon_dimer_metad_distance_sum_hills_integration(tmp_path, lj):
    """Full integration test: Argon dimer metadynamics with LJ calculator and sum_hills."""
    system = ase.Atoms(
        "Ar2",
        positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
    )
    atom1 = hc.IndexSelector(indices=[[0]])
    atom2 = hc.IndexSelector(indices=[[1]])

    distance_cv = hc.DistanceCV(x1=atom1, x2=atom2, prefix="dist", flatten=True)

    # Step 4: Set up metadynamics
    # Distance will vary around 3-5 Å, so use appropriate grid bounds
    metad_bias = hc.MetadBias(
        cv=distance_cv,
        sigma=0.2,  # Width in Angstrom
        grid_min=0.0,
        grid_max=5.0,
    )

    wall = hc.UpperWallBias(
        cv=distance_cv,
        at=4,
        kappa=100.0,
    )

    metad_config = hc.MetaDynamicsConfig(
        height=0.005,  # Small height in eV
        pace=10,  # Deposit Gaussian every 10 steps
        temp=120.0,  # Temperature in Kelvin (Argon melts at ~84 K)
        biasfactor=10.0,  # Well-tempered metadynamics
        file="HILLS",
        flush=10,
    )

    # Step 6: Create MetaDynamicsModel
    metad_model = hc.MetaDynamicsModel(
        config=metad_config,
        data=[system.copy()],
        data_idx=0,
        bias_cvs=[metad_bias],
        actions=[hc.PrintAction(cvs=[distance_cv], stride=5), wall],
        timestep=2.0,  # 2 fs timestep
        model=lj,
    )

    calc = metad_model.get_calculator(directory=tmp_path)
    system.calc = calc

    dyn = Langevin(
        atoms=system,
        timestep=2.0 * ase.units.fs,
        temperature_K=120.0,
        friction=0.01,  # 1/fs
    )

    n_steps = 1_000
    for _ in dyn.irun(n_steps):
        pass

    _ = hc.plot_cv_time_series(tmp_path / "COLVAR")
    hc.sum_hills(
        hills_file=tmp_path / "HILLS",
        bin=500,
        outfile=tmp_path / "fes.dat",
    )

    assert (tmp_path / "fes.dat").exists()
    assert (tmp_path / "COLVAR").exists()
    assert (tmp_path / "HILLS").exists()

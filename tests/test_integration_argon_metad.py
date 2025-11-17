"""Integration test for Argon dimer metadynamics with distance CV and sum_hills.

This test verifies that:
1. The PLUMED calculator wrapper works correctly with ASE's LennardJones calculator
2. A short metadynamics simulation can be run with DistanceCV
3. The sum_hills analysis tool works correctly on the generated HILLS file
"""

import ase.units
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import hillclimber as hc


def test_argon_dimer_metad_distance_sum_hills_integration(tmp_path):
    """Full integration test: Argon dimer metadynamics with LJ calculator and sum_hills.

    This test:
    - Creates a simple 2-Argon system (dimer)
    - Sets up LennardJones calculator with Argon parameters
    - Wraps it with PLUMED metadynamics using DistanceCV
    - Runs a short MD simulation (50 steps)
    - Verifies HILLS and COLVAR files are generated
    - Tests sum_hills functionality on the generated data
    - Verifies calculator preserves model results with 'model_' prefix
    """
    # Step 1: Create simple 2-Argon system (dimer)
    # Place atoms at equilibrium distance for Argon (~3.8 Å)
    # Equilibrium for LJ is at r = 2^(1/6) * sigma ≈ 1.122 * 3.4 ≈ 3.8 Å
    atoms = Atoms(
        'Ar2',
        positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],  # Large box to avoid periodic interactions
        pbc=True
    )

    # Step 2: Set up LennardJones calculator for Argon
    # Standard Argon Lennard-Jones parameters:
    # epsilon = 0.0104 eV (~120 K), sigma = 3.4 Å
    epsilon_ar = 0.0104  # eV
    sigma_ar = 3.4  # Angstrom
    cutoff = 10.0  # Angstrom

    lj_calc = LennardJones(
        sigma=sigma_ar,
        epsilon=epsilon_ar,
        rc=cutoff,
        smooth=True
    )

    # Step 3: Define distance CV between the two Argon atoms
    atom1 = hc.IndexSelector(indices=[[0]])
    atom2 = hc.IndexSelector(indices=[[1]])

    distance_cv = hc.DistanceCV(
        x1=atom1,
        x2=atom2,
        prefix="dist",
        flatten=True
    )

    # Step 4: Set up metadynamics
    # Distance will vary around 3-5 Å, so use appropriate grid bounds
    metad_bias = hc.MetadBias(
        cv=distance_cv,
        sigma=0.1,  # Width in Angstrom
        grid_min=2.0,
        grid_max=8.0,
        grid_bin=120
    )

    metad_config = hc.MetaDynamicsConfig(
        height=0.05,  # Small height in eV
        pace=10,  # Deposit Gaussian every 10 steps
        temp=120.0,  # Temperature in Kelvin (Argon melts at ~84 K)
        biasfactor=10.0,  # Well-tempered metadynamics
        file="HILLS",
        flush=10
    )

    # Step 5: Create minimal model class that provides the LJ calculator
    class LJModel:
        """Minimal model for testing that wraps LJ calculator."""
        def get_calculator(self, directory=None, **kwargs):
            return lj_calc

    # Step 6: Create MetaDynamicsModel
    metad_model = hc.MetaDynamicsModel(
        config=metad_config,
        data=[atoms.copy()],
        data_idx=0,
        bias_cvs=[metad_bias],
        actions=[hc.PrintAction(cvs=[distance_cv], stride=5)],
        timestep=2.0,  # 2 fs timestep
        model=LJModel()
    )

    # Create working directory
    work_dir = tmp_path / "argon_metad"
    work_dir.mkdir()

    # Step 7: Get PLUMED-wrapped calculator
    calc = metad_model.get_calculator(directory=work_dir)

    # Verify calculator structure
    assert hasattr(calc, 'calc'), "PLUMED calculator should wrap base calculator"
    assert calc.calc is lj_calc, "Wrapped calculator should be LennardJones"

    # Verify PLUMED input file was created
    plumed_dat = work_dir / "plumed.dat"
    assert plumed_dat.exists(), "plumed.dat should be created"

    with open(plumed_dat, 'r') as f:
        plumed_content = f.read()

    assert "DISTANCE" in plumed_content
    assert "METAD" in plumed_content
    assert "PRINT" in plumed_content
    assert "HEIGHT=0.05" in plumed_content
    assert "PACE=10" in plumed_content
    assert "dist:" in plumed_content

    # Step 8: Run short MD simulation
    md_atoms = atoms.copy()
    md_atoms.calc = calc

    # Initialize velocities at target temperature
    MaxwellBoltzmannDistribution(md_atoms, temperature_K=120.0, rng=np.random.RandomState(42))

    # Run MD with Langevin thermostat (NVT ensemble)
    dyn = Langevin(
        md_atoms,
        timestep=2.0 * ase.units.fs,
        temperature_K=120.0,
        friction=0.01,  # 1/fs
        trajectory=str(work_dir / "md.traj")
    )

    # Run 50 steps (100 fs total)
    # With pace=10, this deposits 5 Gaussians
    n_steps = 50
    dyn.run(n_steps)

    # Step 9: Verify output files
    hills_file = work_dir / "HILLS"
    colvar_file = work_dir / "COLVAR"

    assert hills_file.exists(), "HILLS file should be created"
    assert colvar_file.exists(), "COLVAR file should be created"

    # Check HILLS file content
    with open(hills_file, 'r') as f:
        hills_lines = f.readlines()

    hills_data_lines = [line for line in hills_lines if not line.startswith('#')]
    assert len(hills_data_lines) >= 3, \
        f"Expected at least 3 hills deposited, got {len(hills_data_lines)}"

    # Step 10: Test sum_hills functionality
    fes_file = work_dir / "fes.dat"

    # Run sum_hills
    _ = hc.sum_hills(
        hills_file=hills_file,
        outfile=fes_file,
        bin=120,
        min_bounds=2.0,
        max_bounds=8.0,
        verbose=True,
        check=True
    )

    # Verify FES file
    assert fes_file.exists(), "Free energy surface file should be created"

    fes_data = np.loadtxt(fes_file)
    assert fes_data.ndim == 2, "FES should be 2D array"
    # PLUMED sum_hills outputs 3 columns: CV, FES, derivative
    assert fes_data.shape[1] == 3, "FES should have columns: [dist, FES, derivative]"
    assert len(fes_data) > 0, "FES should contain data"

    # Verify FES values are reasonable (column 1 is the FES)
    # Note: With only 5 Gaussians, FES might be mostly zeros except near deposited points
    assert not np.any(np.isnan(fes_data[:, 1])), "FES should not contain NaN"

    # Step 11: Test read_colvar functionality
    colvar_data = hc.read_colvar(colvar_file)

    assert 'time' in colvar_data, "COLVAR should contain time"
    assert 'dist' in colvar_data, "COLVAR should contain distance CV"
    assert len(colvar_data['time']) > 0, "COLVAR should have data"

    # With stride=5 and 50 steps, expect ~10 COLVAR entries
    expected_prints = n_steps // 5
    assert len(colvar_data['time']) >= expected_prints - 2, \
        f"Expected at least {expected_prints - 2} COLVAR entries, got {len(colvar_data['time'])}"

    # Verify distances are in reasonable range for Argon dimer
    dist_values = colvar_data['dist']
    assert np.all(dist_values > 0), "Distances should be positive"
    assert np.all(dist_values < 10.0), "Distances should be less than box size cutoff"

    # Step 12: Verify NonOverwritingPlumed preserves model results
    # Check that the calculator stores original LJ results with 'model_' prefix
    results = calc.results

    assert 'energy' in results, "Should have biased energy"
    assert 'forces' in results, "Should have biased forces"
    assert 'model_energy' in results, "Should preserve model energy with 'model_' prefix"
    assert 'model_forces' in results, "Should preserve model forces with 'model_' prefix"

    # Model results should be numeric and have correct shape
    assert isinstance(results['model_energy'], (float, np.floating)), \
        "Model energy should be a scalar"
    assert results['model_forces'].shape == results['forces'].shape, \
        "Model forces shape should match biased forces shape"

    # Test passed - print summary
    print("\n" + "="*60)
    print("✓ Integration test PASSED!")
    print("="*60)
    print(f"System: {len(atoms)} Argon atoms (dimer)")
    print(f"MD steps: {n_steps} ({n_steps * 2} fs)")
    print(f"Hills deposited: {len(hills_data_lines)}")
    print(f"COLVAR entries: {len(colvar_data['time'])}")
    if fes_file.exists():
        print(f"FES grid points: {len(fes_data)}")
        print(f"Distance range: [{dist_values.min():.2f}, {dist_values.max():.2f}] Å")
    print("="*60)

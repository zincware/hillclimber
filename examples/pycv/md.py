import ase
import ase.units
from ase.calculators.lj import LennardJones
from ase.calculators.plumed import Plumed
from ase.md.langevin import Langevin

import plumed

# Create Argon dimer system
system = ase.Atoms(
    "Ar2",
    positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
)

# Set up Lennard-Jones calculator for Argon
epsilon_ar = 0.0104  # eV
sigma_ar = 3.4  # Angstrom
cutoff = 10.0  # Angstrom

lj_calc = LennardJones(sigma=sigma_ar, epsilon=epsilon_ar, rc=cutoff, smooth=True)

# Get path to PYCV library
pycv_path = plumed.get_pycv_path()
plumed_input = [
    f"LOAD FILE={pycv_path}",
    "cvPY: PYCVINTERFACE ATOMS=1,2 IMPORT=cv CALCULATE=cv1",
    "PRINT FILE=colvar.out ARG=cvPY STRIDE=1",
]

print("Summary of PLUMED input:")
for line in plumed_input:
    print(f"  {line}")
print("")

# Wrap LJ calculator with PLUMED
calc = Plumed(
    calc=lj_calc,
    input=plumed_input,
    timestep=2.0 * ase.units.fs,
    atoms=system,
    kT=120.0 * ase.units.kB,  # Temperature in eV
)

system.calc = calc

# Set up Langevin dynamics
dyn = Langevin(
    atoms=system,
    timestep=2.0 * ase.units.fs,
    temperature_K=120.0,
    friction=0.01,  # 1/fs
    trajectory=str("md.traj"),
)

dyn.run(steps=1000)  # Run 1000 MD steps

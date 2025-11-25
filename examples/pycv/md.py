"""Example MD simulation using custom PYCV collective variable with vanilla ASE.

This script demonstrates how to:
1. Create an Argon dimer system
2. Use a custom Python-based CV (distance) defined in cv.py
3. Run Langevin MD with PLUMED using vanilla ASE
"""
import sys
import os

# FORCE GLOBAL SYMBOLS
# This ensures that C++ extensions (like PLUMED and PyCV) can share 
# runtime type information and memory structures.
try:
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
except AttributeError:
    pass # Not a posix system, but since you are on mac, this will run.

# Ensure current directory is in path so PLUMED can find 'pyinfcv'
sys.path.insert(0, os.getcwd())

from pathlib import Path

import ase
import ase.units
from ase.calculators.lj import LennardJones
from ase.md.langevin import Langevin
from ase.calculators.plumed import Plumed

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

# Get absolute path to cv.py module (in same directory as this script)
cv_module_path = Path(__file__).parent / "cv.py"
cv_module_name = cv_module_path.stem  # "cv"

# Create PLUMED input as list of strings (required by ASE Plumed calculator)
# Does not work?!
plumed_input = [
    f"LOAD FILE={pycv_path}",
    "cvPY: PYCVINTERFACE ATOMS=1,2 IMPORT=pyinfcv CALCULATE=cv1",
    "PRINT FILE=colvar.out ARG=cvPY STRIDE=1",
]

# WORKS!!
# plumed_input = [
#     f"LOAD FILE={pycv_path}",
#     "cvPY: PYFUNCTION IMPORT=pyfn",
#     "PRINT FILE=colvar.out ARG=*",
# ]

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

"""Test PYCVINTERFACE"""
from pathlib import Path
import ase
import ase.units
from ase.calculators.lj import LennardJones
from ase.md.langevin import Langevin
from ase.calculators.plumed import Plumed
import plumed

system = ase.Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
epsilon_ar = 0.0104
sigma_ar = 3.4
cutoff = 10.0
lj_calc = LennardJones(sigma=sigma_ar, epsilon=epsilon_ar, rc=cutoff, smooth=True)
pycv_path = plumed.get_pycv_path()

plumed_input = [
    f"LOAD FILE={pycv_path}",
    "cvPY: PYCVINTERFACE ATOMS=1,2 IMPORT=pyinfcv CALCULATE=cv1",
    "PRINT FILE=colvar_test.out ARG=cvPY STRIDE=1",
]

print("PYCVINTERFACE Test:")
for line in plumed_input:
    print(f"  {line}")

calc = Plumed(
    calc=lj_calc,
    input=plumed_input,
    timestep=2.0 * ase.units.fs,
    atoms=system,
    kT=120.0 * ase.units.kB,
)

system.calc = calc
dyn = Langevin(
    atoms=system,
    timestep=2.0 * ase.units.fs,
    temperature_K=120.0,
    friction=0.01,
    trajectory=str("test.traj"),
)

dyn.run(steps=10)

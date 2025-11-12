from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.plumed import Plumed
import ase
class NonOverwritingPlumed(Plumed):
    def __init__(self, *args, **kwargs):
        self.wrap = kwargs.pop("wrap", False)
        super().__init__(*args, **kwargs)
    def calculate(self, atoms: ase.Atoms| None=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy", "forces"]

        if self.wrap:
            if atoms is not None:
                atoms.wrap()
            if self.atoms is not None:
                self.atoms.wrap()
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self.compute_energy_and_forces(
            self.atoms.get_positions(), self.istep
        )
        self.istep += 1
        self.results = {f"model_{k}": v for k, v in self.calc.results.items()}
        self.results["energy"], self.results["forces"] = energy, forces

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.plumed import Plumed


class NonOverwritingPlumed(Plumed):
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy", "forces"]
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self.compute_energy_and_forces(
            self.atoms.get_positions(), self.istep
        )
        self.istep += 1
        self.results = {f"model_{k}": v for k, v in self.calc.results.items()}
        self.results["energy"], self.results["forces"] = energy, forces

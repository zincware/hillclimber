import dataclasses

import ase

from plumed_nodes.interfaces import CollectiveVariable


@dataclasses.dataclass
class PrintCVAction:
    """Node for PRINT action."""

    cvs: list[CollectiveVariable]
    stride: int = 1
    file: str = "COLVAR"

    def to_plumed(self, atoms: ase.Atoms) -> list[str]:
        """Convert the action node to a PLUMED input string."""
        commands = []
        for cv in self.cvs:
            labels, commands = cv.to_plumed(atoms)
            if not commands:
                raise ValueError(f"Empty PLUMED commands for CV {cv.prefix}")

            print_command = (
                f"PRINT ARG={','.join(labels)} STRIDE={self.stride} FILE={self.file}"
            )
            commands.append(print_command)
        return commands

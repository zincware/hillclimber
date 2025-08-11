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
        all_labels = set()
        for cv in self.cvs:
            labels, _ = cv.to_plumed(atoms)
            all_labels.update(labels)

        # Create the PRINT command with the unique labels
        print_command = (
            f"PRINT ARG={','.join(sorted(all_labels))} STRIDE={self.stride} FILE={self.file}"
        )

        # Return the command as a list
        return [print_command]


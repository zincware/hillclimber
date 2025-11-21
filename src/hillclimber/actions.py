import dataclasses

import ase

from hillclimber.interfaces import CollectiveVariable, PlumedGenerator


@dataclasses.dataclass
class PrintAction(PlumedGenerator):
    """PLUMED PRINT action for outputting collective variables.

    This action prints the values of collective variables to a file during
    the simulation. Multiple CVs can be printed to the same file.

    Parameters
    ----------
    cvs : list[CollectiveVariable]
        List of collective variables to print.
    stride : int, optional
        Print every N steps, by default 1.
    file : str, optional
        Output file name, by default "COLVAR".

    Examples
    --------
    >>> import hillclimber as hc
    >>> print_action = hc.PrintAction(
    ...     cvs=[cv1, cv2, cv3],
    ...     stride=100,
    ...     file="COLVAR"
    ... )

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/PRINT/
    """

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
        print_command = f"PRINT ARG={','.join(sorted(all_labels))} STRIDE={self.stride} FILE={self.file}"

        # Return the command as a list
        return [print_command]

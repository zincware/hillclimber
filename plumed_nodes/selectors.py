import dataclasses
import typing as tp

import ase
import rdkit2ase

from plumed_nodes.interfaces import AtomSelector


@dataclasses.dataclass
class IndexSelector(AtomSelector):
    """Select atoms based on grouped indices.

    Parameters
    ----------
    indices : list[list[int]]
        A list of atom index groups to select. Each inner list represents
        a group of atoms (e.g., a molecule). For example:
        - [[0, 1], [2, 3]] selects two groups: atoms [0,1] and atoms [2,3]
        - [[0], [1]] selects two single-atom groups
    """

    # mostly used for debugging
    indices: list[list[int]]

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        return self.indices


@dataclasses.dataclass
class SMILESSelector(AtomSelector):
    """Select atoms based on a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string to use for selection.
    """

    smiles: str

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        matches = rdkit2ase.match_substructure(atoms, smiles=self.smiles)
        return [list(match) for match in matches]


@dataclasses.dataclass
class SMARTSSelector(AtomSelector):
    """Select atoms based on SMARTS or mapped SMILES patterns.

    This selector uses RDKit's substructure matching to identify atoms
    matching a given SMARTS pattern or mapped SMILES string. It supports
    flexible hydrogen handling and can work with mapped atoms for
    precise selection.

    Note
    ----
    The selector is applied only to the first trajectory frame.  
    Since indices can change during e.g. proton transfer, biasing specific groups (e.g. `[OH-]`) may fail.  
    In such cases, select all `[OH2]` and `[OH-]` groups and use CoordinationNumber CVs.
    Account for this method with all changes in chemical structure.

    Parameters
    ----------
    pattern : str
        SMARTS pattern (e.g., "[F]", "[OH]", "C(=O)O") or SMILES with
        atom maps (e.g., "C1[C:1]OC(=[O:1])O1"). If atom maps are present,
        only the mapped atoms are selected.
    hydrogens : {'exclude', 'include', 'isolated'}, default='exclude'
        How to handle hydrogen atoms in the selection:
        - 'exclude': Remove all hydrogens from the selection
        - 'include': Include hydrogens bonded to selected heavy atoms
        - 'isolated': Select only hydrogens bonded to selected heavy atoms

    Examples
    --------
    >>> # Select all fluorine atoms
    >>> selector = SMARTSSelection(pattern="[F]")

    >>> # Select carboxylic acid groups including hydrogens
    >>> selector = SMARTSSelection(pattern="C(=O)O", hydrogens="include")

    >>> # Select only specific mapped atoms
    >>> selector = SMARTSSelection(pattern="C1[C:1]OC(=[O:1])O1")

    >>> # Select 4 elements in order to define an angle
    >>> selector = SMARTSSelection(pattern="CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C")
    """

    pattern: str
    hydrogens: tp.Literal["include", "exclude", "isolated"] = "exclude"

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        return rdkit2ase.select_atoms_grouped(
            rdkit2ase.ase2rdkit(atoms), self.pattern, self.hydrogens
        )

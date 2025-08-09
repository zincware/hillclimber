import dataclasses
import typing as tp

import ase
import rdkit.Chem.rdchem as rdchem
import rdkit2ase
from rdkit import Chem

from plumed_nodes.interfaces import AtomSelector


@dataclasses.dataclass
class IndexSelector(AtomSelector):
    indices: list[int]

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        return [[x] for x in self.indices]


@dataclasses.dataclass
class SMILESSelector(AtomSelector):
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
    """

    pattern: str
    hydrogens: tp.Literal["include", "exclude", "isolated"] = "exclude"

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        return rdkit2ase.select_atoms_grouped(
            rdkit2ase.ase2rdkit(atoms), self.pattern, self.hydrogens
        )

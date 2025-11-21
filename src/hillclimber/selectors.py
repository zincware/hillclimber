import dataclasses
import typing as tp

import ase
import molify
import rdkit2ase

from hillclimber.interfaces import AtomSelector

# --- Indexable Selector Wrappers ---


@dataclasses.dataclass
class _GroupIndexedSelector(AtomSelector):
    """Selector with group-level indexing applied.

    This is an internal class created when you index a selector at the group level.
    For example: water_sel[0] or water_sel[0:2]
    """

    selector: AtomSelector
    group_index: int | slice | list[int]

    def __getitem__(self, idx: int | slice | list[int]) -> AtomSelector:
        """Atom-level indexing.

        After group indexing, this applies atom-level indexing.
        For example: water_sel[0:2][1:3] selects atoms 1-2 from groups 0-1.
        """
        return _AtomIndexedSelector(self, idx)

    def __add__(self, other: AtomSelector) -> AtomSelector:
        """Combine two selectors."""
        return _CombinedSelector([self, other])

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        """Apply group indexing to the underlying selector."""
        groups = self.selector.select(atoms)

        # Apply group indexing (supports negative indices)
        if isinstance(self.group_index, int):
            return [groups[self.group_index]]  # Python handles negative indices
        elif isinstance(self.group_index, slice):
            return groups[self.group_index]  # Python handles negative indices in slices
        else:  # list[int]
            return [
                groups[i] for i in self.group_index
            ]  # Negative indices work here too


@dataclasses.dataclass
class _AtomIndexedSelector(AtomSelector):
    """Selector with both group and atom-level indexing applied.

    This is an internal class created when you apply two levels of indexing.
    For example: water_sel[0][0] or water_sel[0:2][1:3]
    """

    group_selector: _GroupIndexedSelector
    atom_index: int | slice | list[int]

    def __getitem__(self, idx) -> AtomSelector:
        """Prevent three-level indexing."""
        raise ValueError("Cannot index beyond 2 levels (group, then atom)")

    def __add__(self, other: AtomSelector) -> AtomSelector:
        """Combine two selectors."""
        return _CombinedSelector([self, other])

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        """Apply atom-level indexing to each group."""
        groups = self.group_selector.select(atoms)

        # Apply atom-level indexing to each group (supports negative indices)
        result = []
        for group in groups:
            if isinstance(self.atom_index, int):
                result.append([group[self.atom_index]])  # Negative indices work
            elif isinstance(self.atom_index, slice):
                result.append(group[self.atom_index])  # Negative indices in slices work
            else:  # list[int]
                result.append(
                    [group[i] for i in self.atom_index]
                )  # Negative indices work

        return result


@dataclasses.dataclass
class _CombinedSelector(AtomSelector):
    """Selector that combines multiple selectors.

    This is an internal class created when you combine selectors with +.
    For example: water_sel + ethanol_sel
    """

    selectors: list[AtomSelector]

    def __getitem__(self, idx: int | slice | list[int]) -> AtomSelector:
        """Group-level indexing on combined result."""
        return _GroupIndexedSelector(self, idx)

    def __add__(self, other: AtomSelector) -> AtomSelector:
        """Combine with another selector."""
        # Flatten if other is also a CombinedSelector
        if isinstance(other, _CombinedSelector):
            return _CombinedSelector(self.selectors + other.selectors)
        return _CombinedSelector(self.selectors + [other])

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        """Concatenate all groups from all selectors."""
        result = []
        for selector in self.selectors:
            result.extend(selector.select(atoms))
        return result


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

    def __getitem__(self, idx: int | slice | list[int]) -> AtomSelector:
        """Group-level indexing."""
        return _GroupIndexedSelector(self, idx)

    def __add__(self, other: AtomSelector) -> AtomSelector:
        """Combine two selectors."""
        return _CombinedSelector([self, other])

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

    def __getitem__(self, idx: int | slice | list[int]) -> AtomSelector:
        """Group-level indexing."""
        return _GroupIndexedSelector(self, idx)

    def __add__(self, other: AtomSelector) -> AtomSelector:
        """Combine two selectors."""
        return _CombinedSelector([self, other])

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        # TODO: switch to molify once available
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

    def __getitem__(self, idx: int | slice | list[int]) -> AtomSelector:
        """Group-level indexing."""
        return _GroupIndexedSelector(self, idx)

    def __add__(self, other: AtomSelector) -> AtomSelector:
        """Combine two selectors."""
        return _CombinedSelector([self, other])

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        # TODO: switch to molify once available
        return rdkit2ase.select_atoms_grouped(
            molify.ase2rdkit(atoms), self.pattern, self.hydrogens
        )

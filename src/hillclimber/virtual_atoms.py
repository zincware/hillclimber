"""Virtual atom definitions for creating points in space from atom groups.

Virtual atoms reduce groups of atoms to single points (or multiple points, one per group)
using strategies like center of mass (COM), center of geometry (COG), or first atom.

Resources
---------
- https://www.plumed.org/doc-master/user-doc/html/COM
- https://www.plumed.org/doc-master/user-doc/html/CENTER
"""

from __future__ import annotations

import dataclasses
import typing as tp

import ase

from hillclimber.interfaces import AtomSelector


@dataclasses.dataclass
class VirtualAtom:
    """Creates virtual atom(s) from atom groups.

    Creates ONE virtual atom for EACH group returned by the selector.
    To select specific groups, use selector indexing before creating the VirtualAtom.

    Parameters
    ----------
    atoms : AtomSelector | VirtualAtom
        Source atoms or nested virtual atoms.
        Use selector indexing to choose specific groups: VirtualAtom(selector[0], "com")
    reduction : {"com", "cog", "first", "flatten"}, default="com"
        How to reduce atoms to points:

        - "com": Center of mass for each group (creates virtual site)
        - "cog": Center of geometry for each group (creates virtual site)
        - "first": First atom of each group (no virtual site needed)
        - "flatten": Combine all groups into one, then apply reduction
    label : str | None, default=None
        Optional label for PLUMED virtual site commands.

    Examples
    --------
    Create COM for each molecule:

    >>> import hillclimber as hc
    >>> # Two ethanols → two virtual atoms (COMs)
    >>> selector = hc.SMARTSSelector("CCO")
    >>> ethanols = hc.VirtualAtom(selector, reduction="com")

    Select specific virtual atom using selector indexing:

    >>> # Select first ethanol's COM
    >>> ethanol_0 = hc.VirtualAtom(selector[0], "com")

    Select multiple virtual atoms using selector slicing:

    >>> # Select first two ethanols
    >>> first_two = hc.VirtualAtom(selector[0:2], "com")
    >>> # Select every other water
    >>> water_sel = hc.SMARTSSelector("O")
    >>> every_other = hc.VirtualAtom(water_sel[::2], "com")

    Select specific virtual atoms using selector list indexing:

    >>> # Select specific ethanols by indices
    >>> selected = hc.VirtualAtom(selector[[0, 2, 4]], "com")

    Flatten all groups into one virtual atom:

    >>> # All water oxygens → one COM
    >>> all_waters = hc.VirtualAtom(hc.SMARTSSelector("O"), reduction="flatten")

    Nested virtual atoms (COM of COMs):

    >>> # First: create COM for each water
    >>> water_coms = hc.VirtualAtom(hc.SMARTSSelector("O"), reduction="com")
    >>> # Then: create COM of all those COMs
    >>> center = hc.VirtualAtom(water_coms, reduction="flatten")

    Use in distance CVs:

    >>> # Single distance (using selector indexing)
    >>> ethanol_sel = hc.SMARTSSelector("CCO")
    >>> water_sel = hc.SMARTSSelector("O")
    >>> dist = hc.DistanceCV(
    ...     x1=hc.VirtualAtom(ethanol_sel[0], "com"),
    ...     x2=hc.VirtualAtom(water_sel[0], "com"),
    ...     prefix="d"
    ... )
    >>>
    >>> # Multiple distances (first ethanol to all waters)
    >>> dist = hc.DistanceCV(
    ...     x1=hc.VirtualAtom(ethanol_sel[0], "com"),
    ...     x2=hc.VirtualAtom(water_sel, "com"),
    ...     prefix="d"
    ... )
    >>>
    >>> # Multiple distances (first two ethanols to first three waters)
    >>> dist = hc.DistanceCV(
    ...     x1=hc.VirtualAtom(ethanol_sel[0:2], "com"),
    ...     x2=hc.VirtualAtom(water_sel[0:3], "com"),
    ...     prefix="d"
    ... )
    >>>
    >>> # All pairwise distances
    >>> dist = hc.DistanceCV(
    ...     x1=hc.VirtualAtom(water_sel, "com"),
    ...     x2=hc.VirtualAtom(water_sel, "com"),
    ...     prefix="d"
    ... )

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/COM
    - https://www.plumed.org/doc-master/user-doc/html/CENTER
    """

    atoms: AtomSelector | "VirtualAtom"
    reduction: tp.Literal["com", "cog", "first", "flatten"] = "com"
    label: str | None = None

    def __add__(self, other: "VirtualAtom") -> "VirtualAtom":
        """Combine two VirtualAtoms.

        Returns a new VirtualAtom that represents both sets of virtual sites.
        The underlying selectors are combined using the selector's __add__ method.

        Parameters
        ----------
        other : VirtualAtom
            Another VirtualAtom to combine with this one.

        Returns
        -------
        VirtualAtom
            New VirtualAtom with combined underlying selectors.

        Raises
        ------
        ValueError
            If the two VirtualAtoms have different reduction strategies.

        Examples
        --------
        >>> water_coms = hc.VirtualAtom(water_sel, "com")
        >>> ethanol_coms = hc.VirtualAtom(ethanol_sel, "com")
        >>> all_coms = water_coms + ethanol_coms
        >>>
        >>> # Combine with indexed selectors
        >>> first_water = hc.VirtualAtom(water_sel[0], "com")
        >>> first_ethanol = hc.VirtualAtom(ethanol_sel[0], "com")
        >>> combined = first_water + first_ethanol  # 2 COMs
        """
        # Reduction must be the same
        if self.reduction != other.reduction:
            raise ValueError(
                f"Cannot combine VirtualAtoms with different reductions: "
                f"{self.reduction} vs {other.reduction}"
            )

        # Combine the underlying selectors
        combined_atoms = self.atoms + other.atoms

        return VirtualAtom(
            atoms=combined_atoms,
            reduction=self.reduction,
            label=None,  # Reset label for combined VirtualAtom
        )

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        """Select atom groups for virtual atom creation.

        Returns list of atom groups. Each group represents the atoms for
        one virtual site.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to select from.

        Returns
        -------
        list[list[int]]
            Groups of atom indices. Each inner list is one group.
        """
        # Get groups from source
        if isinstance(self.atoms, VirtualAtom):
            # Nested: inner VirtualAtom returns groups
            groups = self.atoms.select(atoms)
        else:
            groups = self.atoms.select(atoms)

        # Apply reduction strategy
        if self.reduction == "flatten":
            # Combine all groups into one
            flat = [idx for group in groups for idx in group]
            return [flat]
        else:
            # com, cog, first: keep groups separate
            # Actual point creation happens in to_plumed()
            return groups

    def to_plumed(self, atoms: ase.Atoms) -> tuple[list[str], list[str]]:
        """Generate PLUMED virtual site commands.

        Handles nested VirtualAtoms by first generating commands for the inner
        VirtualAtom, then using those labels to create the outer virtual site.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to use for generating commands.

        Returns
        -------
        labels : list[str]
            Labels for the virtual sites/atoms created. These can be used
            in subsequent PLUMED commands.
        commands : list[str]
            PLUMED command strings for creating virtual sites.
            Empty if reduction="first" (no virtual sites needed).

        Examples
        --------
        >>> va = hc.VirtualAtom(hc.SMARTSSelector("O"), reduction="com")
        >>> labels, commands = va.to_plumed(atoms)
        >>> print(labels)
        ['vsite_123_0', 'vsite_123_1']
        >>> print(commands)
        ['vsite_123_0: COM ATOMS=1,2,3', 'vsite_123_1: COM ATOMS=4,5,6']

        >>> # Nested VirtualAtom (COM of COMs)
        >>> water_coms = hc.VirtualAtom(water_sel, "com")
        >>> center = hc.VirtualAtom(water_coms, "com")
        >>> labels, commands = center.to_plumed(atoms)
        >>> # Commands will include both the individual COMs and the center COM
        """
        # Check if this is a nested VirtualAtom
        if isinstance(self.atoms, VirtualAtom):
            # First, generate commands for the inner VirtualAtom
            inner_labels, inner_commands = self.atoms.to_plumed(atoms)

            # Now create virtual site using the inner labels
            commands = list(inner_commands)  # Copy inner commands
            labels = []

            if self.reduction == "first":
                # Just use the first inner label
                labels = [inner_labels[0]]
            elif self.reduction == "flatten":
                # Create COM/COG of all inner virtual sites
                base_label = self.label or f"vsite_{id(self)}"
                cmd_type = "COM" if self.reduction == "com" else "CENTER"
                atom_list = ",".join(inner_labels)
                commands.append(f"{base_label}: {cmd_type} ATOMS={atom_list}")
                labels = [base_label]
            elif self.reduction in ["com", "cog"]:
                # For COM/COG with nested VirtualAtoms, we create a single
                # virtual site from all inner labels (similar to flatten)
                base_label = self.label or f"vsite_{id(self)}"
                cmd_type = "COM" if self.reduction == "com" else "CENTER"
                atom_list = ",".join(inner_labels)
                commands.append(f"{base_label}: {cmd_type} ATOMS={atom_list}")
                labels = [base_label]
            else:
                # This shouldn't typically happen, but handle it
                # Just pass through the inner labels
                labels = inner_labels

            return labels, commands

        # Non-nested case: regular selector
        groups = self.select(atoms)
        labels = []
        commands = []

        for i, group in enumerate(groups):
            if self.reduction == "first" or (
                self.reduction == "flatten" and len(group) == 1
            ):
                # No virtual site needed, use atom index directly
                labels.append(str(group[0] + 1))
            else:
                # Create virtual site (COM or CENTER)
                base_label = self.label or f"vsite_{id(self)}"
                label = base_label if len(groups) == 1 else f"{base_label}_{i}"

                cmd_type = "COM" if self.reduction == "com" else "CENTER"
                atom_list = ",".join(str(idx + 1) for idx in group)
                commands.append(f"{label}: {cmd_type} ATOMS={atom_list}")
                labels.append(label)

        return labels, commands

    def count(self, atoms: ase.Atoms) -> int:
        """Return number of virtual atoms this represents.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to count groups in.

        Returns
        -------
        int
            Number of virtual atoms (groups) this will create.

        Examples
        --------
        >>> water_sel = hc.SMARTSSelector("O")
        >>> va = hc.VirtualAtom(water_sel, reduction="com")
        >>> va.count(atoms)  # Number of water molecules
        3
        >>> # Use selector indexing to get single group
        >>> va_single = hc.VirtualAtom(water_sel[0], reduction="com")
        >>> va_single.count(atoms)  # Single water - always 1
        1
        >>> va_flat = hc.VirtualAtom(water_sel, reduction="flatten")
        >>> va_flat.count(atoms)  # Flattened - always 1
        1
        >>> # Nested VirtualAtom
        >>> water_coms = hc.VirtualAtom(water_sel, "com")  # 3 COMs
        >>> center = hc.VirtualAtom(water_coms, "com")  # 1 COM of 3 COMs
        >>> center.count(atoms)
        1
        """
        # For nested VirtualAtoms with com/cog/flatten reduction, we create a single virtual site
        if isinstance(self.atoms, VirtualAtom) and self.reduction in [
            "com",
            "cog",
            "flatten",
        ]:
            return 1
        # For nested VirtualAtoms with "first" reduction, pass through
        elif isinstance(self.atoms, VirtualAtom) and self.reduction == "first":
            return 1
        # For non-nested or other cases, count the groups
        return len(self.select(atoms))

# --- IMPORTS ---
# Standard library
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

# Third-party
import molify
from ase import Atoms
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

# Local
from hillclimber.interfaces import AtomSelector, CollectiveVariable
from hillclimber.virtual_atoms import VirtualAtom

# --- TYPE HINTS ---
GroupReductionStrategyType = Literal[
    "com", "cog", "first", "all", "com_per_group", "cog_per_group"
]
SiteIdentifier = Union[str, List[int]]
ColorTuple = Tuple[float, float, float]
AtomHighlightMap = Dict[int, ColorTuple]


# --- BASE CLASS FOR SHARED LOGIC ---
class _BasePlumedCV(CollectiveVariable):
    """An abstract base class for PLUMED CVs providing shared utilities."""

    prefix: str

    def _get_atom_highlights(
        self, atoms: Atoms, **kwargs
    ) -> Optional[AtomHighlightMap]:
        """
        Get atom indices and colors for visualization.

        This abstract method must be implemented by subclasses to define which atoms
        to highlight and with which colors.

        Args:
            atoms: The ASE Atoms object.
            **kwargs: Additional keyword arguments for specific implementations.

        Returns:
            A dictionary mapping global atom indices to their RGB highlight color,
            or None if selection fails.
        """
        raise NotImplementedError

    def get_img(self, atoms: Atoms, **kwargs) -> Image.Image:
        """
        Generates an image of the molecule(s) with selected atoms highlighted.

        This method uses RDKit to render the image. It automatically identifies
        molecular fragments containing highlighted atoms and draws them in a row.

        Args:
            atoms: The ASE Atoms object to visualize.
            **kwargs: Additional arguments passed to _get_atom_highlights.

        Returns:
            A PIL Image object of the visualization.
        """
        highlight_map = self._get_atom_highlights(atoms, **kwargs)
        try:
            mol = molify.ase2rdkit(atoms)
        except ValueError:
            # Bond determination failed (e.g., jittered positions without connectivity)
            # Return a placeholder image
            return Image.new("RGB", (400, 400), color=(255, 255, 255))

        if not highlight_map:
            return Draw.MolsToGridImage(
                [mol],
                molsPerRow=1,
                subImgSize=(400, 400),
                useSVG=False,
            )

        mol_frags = Chem.GetMolFrags(mol, asMols=True)
        frag_indices_list = Chem.GetMolFrags(mol, asMols=False)

        mols_to_draw, highlights_to_draw, colors_to_draw = [], [], []
        seen_molecules = set()

        for frag_mol, frag_indices in zip(mol_frags, frag_indices_list):
            local_idx_map = {
                global_idx: local_idx
                for local_idx, global_idx in enumerate(frag_indices)
            }
            current_highlights = {
                local_idx_map[g_idx]: color
                for g_idx, color in highlight_map.items()
                if g_idx in local_idx_map
            }

            if current_highlights:
                # Create unique identifier: canonical SMILES + highlighted local indices
                canonical_smiles = Chem.MolToSmiles(frag_mol)
                highlighted_local_indices = tuple(sorted(current_highlights.keys()))
                molecule_signature = (canonical_smiles, highlighted_local_indices)

                if molecule_signature not in seen_molecules:
                    seen_molecules.add(molecule_signature)
                    mols_to_draw.append(frag_mol)
                    highlights_to_draw.append(list(current_highlights.keys()))
                    colors_to_draw.append(current_highlights)

        if not mols_to_draw:
            return Draw.MolsToGridImage(
                [mol],
                molsPerRow=1,
                subImgSize=(400, 400),
                useSVG=False,
            )

        return Draw.MolsToGridImage(
            mols_to_draw,
            molsPerRow=len(mols_to_draw),
            subImgSize=(400, 400),
            highlightAtomLists=highlights_to_draw,
            highlightAtomColors=colors_to_draw,
            useSVG=False,
        )

    @staticmethod
    def _extract_labels(commands: List[str], prefix: str, cv_keyword: str) -> List[str]:
        """Extracts generated CV labels from a list of PLUMED commands."""
        return [
            cmd.split(":", 1)[0].strip()
            for cmd in commands
            if cv_keyword in cmd and cmd.strip().startswith((prefix, f"{prefix}_"))
        ]

    @staticmethod
    def _create_virtual_site_command(
        group: List[int], strategy: Literal["com", "cog"], label: str
    ) -> str:
        """Creates a PLUMED command for a COM or CENTER virtual site."""
        if not group:
            raise ValueError("Cannot create a virtual site for an empty group.")
        atom_list = ",".join(str(idx + 1) for idx in group)
        cmd_keyword = "COM" if strategy == "com" else "CENTER"
        return f"{label}: {cmd_keyword} ATOMS={atom_list}"


# --- REFACTORED CV CLASSES ---
@dataclass
class DistanceCV(_BasePlumedCV):
    """
    PLUMED DISTANCE collective variable.

    Calculates the distance between two atoms, groups of atoms, or virtual sites.
    Supports flexible flattening and pairing strategies for multiple groups.

    Parameters
    ----------
    x1 : AtomSelector | VirtualAtom
        First atom/group or virtual site.
    x2 : AtomSelector | VirtualAtom
        Second atom/group or virtual site.
    prefix : str
        Label prefix for generated PLUMED commands.
    flatten : bool, default=True
        For AtomSelectors only: If True, flatten all groups into single atom list.
        If False, create PLUMED GROUP for each group. VirtualAtoms are never flattened.
    pairwise : {"all", "diagonal", "none"}, default="all"
        Strategy for pairing multiple groups:
        - "all": Create all N×M pair combinations (can create many CVs!)
        - "diagonal": Pair corresponding indices only (creates min(N,M) CVs)
        - "none": Error if both sides have multiple groups (safety check)

    Examples
    --------
    >>> # Distance between two specific atoms
    >>> dist = hc.DistanceCV(
    ...     x1=ethanol_sel[0][0],  # First atom of first ethanol
    ...     x2=water_sel[0][0],    # First atom of first water
    ...     prefix="d_atoms"
    ... )

    >>> # Distance between molecule COMs
    >>> dist = hc.DistanceCV(
    ...     x1=hc.VirtualAtom(ethanol_sel[0], "com"),
    ...     x2=hc.VirtualAtom(water_sel[0], "com"),
    ...     prefix="d_com"
    ... )

    >>> # One-to-many: First ethanol COM to all water COMs
    >>> dist = hc.DistanceCV(
    ...     x1=hc.VirtualAtom(ethanol_sel[0], "com"),
    ...     x2=hc.VirtualAtom(water_sel, "com"),
    ...     prefix="d",
    ...     pairwise="all"  # Creates 3 CVs
    ... )

    >>> # Diagonal pairing (avoid explosion)
    >>> dist = hc.DistanceCV(
    ...     x1=hc.VirtualAtom(water_sel, "com"),     # 3 waters
    ...     x2=hc.VirtualAtom(ethanol_sel, "com"),   # 2 ethanols
    ...     prefix="d",
    ...     pairwise="diagonal"  # Creates only 2 CVs: d_0, d_1
    ... )

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/DISTANCE.html

    Notes
    -----
    For backwards compatibility, old parameters are still supported but deprecated:
    - `group_reduction` → Use VirtualAtom instead
    - `multi_group` → Use `pairwise` parameter
    """

    x1: AtomSelector | VirtualAtom
    x2: AtomSelector | VirtualAtom
    prefix: str
    flatten: bool = True
    pairwise: Literal["all", "diagonal", "none"] = "all"

    def _get_atom_highlights(
        self, atoms: Atoms, **kwargs
    ) -> Optional[AtomHighlightMap]:
        """Get atom highlights for visualization."""
        # Skip for VirtualAtom inputs
        if isinstance(self.x1, VirtualAtom) or isinstance(self.x2, VirtualAtom):
            return None

        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)

        if not groups1 or not groups2:
            return None

        # Highlight all atoms from both selections
        indices1 = {idx for group in groups1 for idx in group}
        indices2 = {idx for group in groups2 for idx in group}

        if not indices1 and not indices2:
            return None

        # Color atoms based on group membership
        highlights: AtomHighlightMap = {}
        red, blue, purple = (1.0, 0.2, 0.2), (0.2, 0.2, 1.0), (1.0, 0.2, 1.0)
        for idx in indices1.union(indices2):
            in1, in2 = idx in indices1, idx in indices2
            if in1 and in2:
                highlights[idx] = purple
            elif in1:
                highlights[idx] = red
            elif in2:
                highlights[idx] = blue
        return highlights

    def to_plumed(self, atoms: Atoms) -> Tuple[List[str], List[str]]:
        """Generate PLUMED input strings for the DISTANCE CV.

        Returns
        -------
        labels : list[str]
            List of CV labels generated.
        commands : list[str]
            List of PLUMED command strings.
        """
        commands = []

        # Process x1
        labels1, cmds1 = self._process_input(self.x1, atoms, "x1")
        commands.extend(cmds1)

        # Process x2
        labels2, cmds2 = self._process_input(self.x2, atoms, "x2")
        commands.extend(cmds2)

        # Check for empty selections
        if not labels1 or not labels2:
            raise ValueError(f"Empty selection for distance CV '{self.prefix}'")

        # Generate distance CVs based on pairwise strategy
        cv_labels, cv_commands = self._generate_distance_cvs(labels1, labels2)
        commands.extend(cv_commands)

        return cv_labels, commands

    def _process_input(
        self, input_obj: AtomSelector | VirtualAtom, atoms: Atoms, label_prefix: str
    ) -> Tuple[List[str], List[str]]:
        """Process an input (AtomSelector or VirtualAtom) and return labels and commands.

        Returns
        -------
        labels : list[str]
            List of labels for this input (either virtual site labels or GROUP labels).
        commands : list[str]
            PLUMED commands to create the labels.
        """
        if isinstance(input_obj, VirtualAtom):
            # VirtualAtom: set deterministic label if not already set
            if input_obj.label is None:
                # Set label based on prefix and label_prefix (x1 or x2)
                labeled_va = dataclasses.replace(
                    input_obj, label=f"{self.prefix}_{label_prefix}"
                )
                return labeled_va.to_plumed(atoms)
            else:
                return input_obj.to_plumed(atoms)
        else:
            # AtomSelector: handle based on flatten parameter
            groups = input_obj.select(atoms)
            if not groups:
                return [], []

            if self.flatten:
                # Flatten all groups into single list
                flat_atoms = [idx for group in groups for idx in group]
                atom_list = ",".join(str(idx + 1) for idx in flat_atoms)
                # Return as pseudo-label (will be used directly in DISTANCE command)
                return [atom_list], []
            else:
                # Smart GROUP creation: only create GROUP for multi-atom groups
                labels = []
                commands = []
                for i, group in enumerate(groups):
                    if len(group) == 1:
                        # Single atom: use directly (no GROUP needed)
                        labels.append(str(group[0] + 1))
                    else:
                        # Multi-atom group: create GROUP
                        group_label = f"{self.prefix}_{label_prefix}_g{i}"
                        atom_list = ",".join(str(idx + 1) for idx in group)
                        commands.append(f"{group_label}: GROUP ATOMS={atom_list}")
                        labels.append(group_label)
                return labels, commands

    def _generate_distance_cvs(
        self, labels1: List[str], labels2: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Generate DISTANCE CV commands based on pairwise strategy."""
        n1, n2 = len(labels1), len(labels2)

        # Determine which pairs to create based on pairwise strategy
        if n1 == 1 and n2 == 1:
            # One-to-one: always create single CV
            pairs = [(0, 0)]
        elif n1 == 1:
            # One-to-many: pair first of x1 with all of x2
            pairs = [(0, j) for j in range(n2)]
        elif n2 == 1:
            # Many-to-one: pair all of x1 with first of x2
            pairs = [(i, 0) for i in range(n1)]
        else:
            # Many-to-many: apply pairwise strategy
            if self.pairwise == "all":
                pairs = [(i, j) for i in range(n1) for j in range(n2)]
            elif self.pairwise == "diagonal":
                n_pairs = min(n1, n2)
                pairs = [(i, i) for i in range(n_pairs)]
            elif self.pairwise == "none":
                raise ValueError(
                    f"Both x1 and x2 have multiple groups ({n1} and {n2}). "
                    f"Use pairwise='all' or 'diagonal', or select specific groups with indexing."
                )
            else:
                raise ValueError(f"Unknown pairwise strategy: {self.pairwise}")

        # Generate DISTANCE commands
        cv_labels = []
        commands = []
        for idx, (i, j) in enumerate(pairs):
            if len(pairs) == 1:
                label = self.prefix
            else:
                label = f"{self.prefix}_{idx}"

            # Create DISTANCE command
            cmd = f"{label}: DISTANCE ATOMS={labels1[i]},{labels2[j]}"
            commands.append(cmd)
            cv_labels.append(label)

        return cv_labels, commands


@dataclass
class AngleCV(_BasePlumedCV):
    """
    PLUMED ANGLE collective variable.

    Calculates the angle formed by three atoms or groups of atoms using the new
    VirtualAtom API. The angle is computed as the angle between the vectors
    (x1-x2) and (x3-x2), where x2 is the vertex of the angle.

    Parameters
    ----------
    x1 : AtomSelector | VirtualAtom
        First position. Can be an AtomSelector or VirtualAtom.
    x2 : AtomSelector | VirtualAtom
        Vertex position (center of the angle). Can be an AtomSelector or VirtualAtom.
    x3 : AtomSelector | VirtualAtom
        Third position. Can be an AtomSelector or VirtualAtom.
    prefix : str
        Label prefix for the generated PLUMED commands.
    flatten : bool, default=True
        How to handle AtomSelector inputs:
        - True: Flatten all groups into a single list
        - False: Create GROUP for each selector group (not typically used for ANGLE)
    strategy : {"first", "all", "diagonal", "none"}, default="first"
        Strategy for creating multiple angles from multiple groups:
        - "first": Use first group from each selector (1 angle)
        - "all": All combinations (N×M×P angles)
        - "diagonal": Pair by index (min(N,M,P) angles)
        - "none": Raise error if any selector has multiple groups

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/ANGLE/
    """

    x1: AtomSelector | VirtualAtom
    x2: AtomSelector | VirtualAtom
    x3: AtomSelector | VirtualAtom
    prefix: str
    flatten: bool = True
    strategy: Literal["first", "all", "diagonal", "none"] = "first"

    def _get_atom_highlights(
        self, atoms: Atoms, **kwargs
    ) -> Optional[AtomHighlightMap]:
        """Get atom highlights for visualization."""
        # Skip for VirtualAtom inputs
        if (
            isinstance(self.x1, VirtualAtom)
            or isinstance(self.x2, VirtualAtom)
            or isinstance(self.x3, VirtualAtom)
        ):
            return None

        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)
        groups3 = self.x3.select(atoms)

        if not groups1 or not groups2 or not groups3:
            return None

        # Highlight all atoms from all three selections
        indices1 = {idx for group in groups1 for idx in group}
        indices2 = {idx for group in groups2 for idx in group}
        indices3 = {idx for group in groups3 for idx in group}

        if not indices1 and not indices2 and not indices3:
            return None

        # Color atoms: red for x1, green for x2 (vertex), blue for x3
        highlights: AtomHighlightMap = {}
        red, green, blue = (1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)

        # Handle overlaps by prioritizing vertex (x2) coloring
        all_indices = indices1.union(indices2).union(indices3)
        for idx in all_indices:
            in1, in2, in3 = idx in indices1, idx in indices2, idx in indices3
            if in2:  # Vertex gets priority
                highlights[idx] = green
            elif in1 and in3:  # Overlap between x1 and x3
                highlights[idx] = (0.5, 0.2, 0.6)  # Purple
            elif in1:
                highlights[idx] = red
            elif in3:
                highlights[idx] = blue
        return highlights

    def to_plumed(self, atoms: Atoms) -> Tuple[List[str], List[str]]:
        """Generate PLUMED ANGLE command(s).

        Returns
        -------
        labels : list[str]
            List of CV labels created.
        commands : list[str]
            List of PLUMED commands.

        Raises
        ------
        ValueError
            If any selector returns empty selection.
        """
        # Process all three inputs
        labels1, cmds1 = self._process_input(self.x1, atoms, "x1")
        labels2, cmds2 = self._process_input(self.x2, atoms, "x2")
        labels3, cmds3 = self._process_input(self.x3, atoms, "x3")

        # Check for empty selections
        if not labels1 or not labels2 or not labels3:
            raise ValueError(f"Empty selection for angle CV '{self.prefix}'")

        commands = []
        commands.extend(cmds1)
        commands.extend(cmds2)
        commands.extend(cmds3)

        # Generate ANGLE commands
        cv_labels, cv_commands = self._generate_angle_cvs(labels1, labels2, labels3)
        commands.extend(cv_commands)

        return cv_labels, commands

    def _process_input(
        self, input_obj: AtomSelector | VirtualAtom, atoms: Atoms, label_prefix: str
    ) -> Tuple[List[str], List[str]]:
        """Process input (AtomSelector or VirtualAtom) and return labels and commands.

        Same as DistanceCV._process_input() method.

        Returns
        -------
        labels : list[str]
            List of labels for this input (either virtual site labels or atom lists).
        commands : list[str]
            PLUMED commands to create the labels.
        """
        if isinstance(input_obj, VirtualAtom):
            # VirtualAtom: set deterministic label if not already set
            if input_obj.label is None:
                labeled_va = dataclasses.replace(
                    input_obj, label=f"{self.prefix}_{label_prefix}"
                )
                return labeled_va.to_plumed(atoms)
            else:
                return input_obj.to_plumed(atoms)
        else:
            # AtomSelector: handle based on flatten parameter
            groups = input_obj.select(atoms)
            if not groups:
                return [], []

            if self.flatten:
                # Flatten all groups into single list
                flat_atoms = [idx for group in groups for idx in group]
                atom_list = ",".join(str(idx + 1) for idx in flat_atoms)
                # Return as pseudo-label (will be used directly in ANGLE command)
                return [atom_list], []
            else:
                # Smart GROUP creation: only create GROUP for multi-atom groups
                labels = []
                commands = []
                for i, group in enumerate(groups):
                    if len(group) == 1:
                        # Single atom: use directly (no GROUP needed)
                        labels.append(str(group[0] + 1))
                    else:
                        # Multi-atom group: create GROUP
                        group_label = f"{self.prefix}_{label_prefix}_g{i}"
                        atom_list = ",".join(str(idx + 1) for idx in group)
                        commands.append(f"{group_label}: GROUP ATOMS={atom_list}")
                        labels.append(group_label)
                return labels, commands

    def _generate_angle_cvs(
        self, labels1: List[str], labels2: List[str], labels3: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Generate ANGLE CV commands based on strategy.

        Parameters
        ----------
        labels1, labels2, labels3 : list[str]
            Labels or atom lists for the three angle positions.

        Returns
        -------
        cv_labels : list[str]
            Labels for the ANGLE CVs created.
        commands : list[str]
            ANGLE command strings.
        """
        n1, n2, n3 = len(labels1), len(labels2), len(labels3)

        # Determine which triplets to create based on strategy
        if n1 == 1 and n2 == 1 and n3 == 1:
            # One-to-one-to-one: always create single CV
            triplets = [(0, 0, 0)]
        elif n1 == 1 and n2 == 1:
            # One-one-to-many: pair first of x1/x2 with all of x3
            triplets = [(0, 0, k) for k in range(n3)]
        elif n1 == 1 and n3 == 1:
            # One-many-to-one: pair first of x1/x3 with all of x2
            triplets = [(0, j, 0) for j in range(n2)]
        elif n2 == 1 and n3 == 1:
            # Many-to-one-one: pair all of x1 with first of x2/x3
            triplets = [(i, 0, 0) for i in range(n1)]
        else:
            # Multi-way: apply strategy
            if self.strategy == "first":
                triplets = [(0, 0, 0)] if n1 > 0 and n2 > 0 and n3 > 0 else []
            elif self.strategy == "all":
                triplets = [
                    (i, j, k) for i in range(n1) for j in range(n2) for k in range(n3)
                ]
            elif self.strategy == "diagonal":
                n_triplets = min(n1, n2, n3)
                triplets = [(i, i, i) for i in range(n_triplets)]
            elif self.strategy == "none":
                raise ValueError(
                    f"Multiple groups in x1/x2/x3 ({n1}, {n2}, {n3}). "
                    f"Use strategy='all' or 'diagonal', or select specific groups with indexing."
                )
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        # Generate ANGLE commands
        cv_labels = []
        commands = []
        for idx, (i, j, k) in enumerate(triplets):
            if len(triplets) == 1:
                label = self.prefix
            else:
                label = f"{self.prefix}_{i}_{j}_{k}"

            # Create ANGLE command (ATOMS=x1,x2,x3 where x2 is vertex)
            cmd = f"{label}: ANGLE ATOMS={labels1[i]},{labels2[j]},{labels3[k]}"
            commands.append(cmd)
            cv_labels.append(label)

        return cv_labels, commands


@dataclass
class CoordinationNumberCV(_BasePlumedCV):
    """
    PLUMED COORDINATION collective variable.

    Calculates a coordination number based on a switching function using the new
    VirtualAtom API. The coordination number is computed between two groups of atoms
    using a switching function.

    Parameters
    ----------
    x1 : AtomSelector | VirtualAtom
        First group of atoms. Can be an AtomSelector or VirtualAtom.
    x2 : AtomSelector | VirtualAtom
        Second group of atoms. Can be an AtomSelector or VirtualAtom.
    prefix : str
        Label prefix for the generated PLUMED commands.
    r_0 : float
        Reference distance for the switching function (in Angstroms).
    nn : int, default=6
        Exponent for the switching function numerator.
    mm : int, default=0
        Exponent for the switching function denominator.
    d_0 : float, default=0.0
        Offset for the switching function (in Angstroms).
    flatten : bool, default=True
        How to handle AtomSelector inputs:
        - True: Flatten all groups into a single GROUP
        - False: Create a GROUP for each selector group
    pairwise : {"all", "diagonal", "none"}, default="all"
        Strategy for pairing multiple groups:
        - "all": All pairwise combinations (N×M CVs)
        - "diagonal": Pair by index (min(N,M) CVs)
        - "none": Raise error if both have multiple groups

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/COORDINATION
    - https://www.plumed.org/doc-master/user-doc/html/GROUP
    """

    x1: AtomSelector | VirtualAtom
    x2: AtomSelector | VirtualAtom
    prefix: str
    r_0: float
    nn: int = 6
    mm: int = 0
    d_0: float = 0.0
    flatten: bool = True
    pairwise: Literal["all", "diagonal", "none"] = "all"

    def _get_atom_highlights(
        self, atoms: Atoms, **kwargs
    ) -> Optional[AtomHighlightMap]:
        """Get atom highlights for visualization."""
        # Skip for VirtualAtom inputs
        if isinstance(self.x1, VirtualAtom) or isinstance(self.x2, VirtualAtom):
            return None

        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)

        if not groups1 or not groups2:
            return None

        # Highlight all atoms from both selections
        indices1 = {idx for group in groups1 for idx in group}
        indices2 = {idx for group in groups2 for idx in group}

        if not indices1 and not indices2:
            return None

        # Color atoms based on group membership
        highlights: AtomHighlightMap = {}
        red, blue, purple = (1.0, 0.2, 0.2), (0.2, 0.2, 1.0), (1.0, 0.2, 1.0)
        for idx in indices1.union(indices2):
            in1, in2 = idx in indices1, idx in indices2
            if in1 and in2:
                highlights[idx] = purple
            elif in1:
                highlights[idx] = red
            elif in2:
                highlights[idx] = blue
        return highlights

    def to_plumed(self, atoms: Atoms) -> Tuple[List[str], List[str]]:
        """Generate PLUMED COORDINATION command(s).

        Returns
        -------
        labels : list[str]
            List of CV labels created.
        commands : list[str]
            List of PLUMED commands.
        """
        # Process both inputs to get group labels
        labels1, cmds1 = self._process_coordination_input(self.x1, atoms, "x1")
        labels2, cmds2 = self._process_coordination_input(self.x2, atoms, "x2")

        commands = []
        commands.extend(cmds1)
        commands.extend(cmds2)

        # Generate COORDINATION commands
        cv_labels, cv_commands = self._generate_coordination_cvs(labels1, labels2)
        commands.extend(cv_commands)

        return cv_labels, commands

    def _process_coordination_input(
        self, input_obj: AtomSelector | VirtualAtom, atoms: Atoms, label_prefix: str
    ) -> Tuple[List[str], List[str]]:
        """Process input for COORDINATION and return group labels/commands.

        For COORDINATION, we need groups (not individual points), so the processing
        is different from DistanceCV:
        - VirtualAtom with multiple sites → create GROUP of those sites
        - VirtualAtom with single site → use site directly
        - AtomSelector with flatten=True → create single group with all atoms
        - AtomSelector with flatten=False → create GROUP for each selector group

        Returns
        -------
        labels : list[str]
            Group labels that can be used in COORDINATION GROUPA/GROUPB.
        commands : list[str]
            PLUMED commands to create those groups.
        """
        if isinstance(input_obj, VirtualAtom):
            # Set deterministic label if not already set
            if input_obj.label is None:
                labeled_va = dataclasses.replace(
                    input_obj, label=f"{self.prefix}_{label_prefix}"
                )
            else:
                labeled_va = input_obj

            # Get virtual site labels
            vsite_labels, vsite_commands = labeled_va.to_plumed(atoms)

            # If multiple virtual sites, create a GROUP of them
            if len(vsite_labels) > 1:
                group_label = f"{self.prefix}_{label_prefix}_group"
                group_cmd = f"{group_label}: GROUP ATOMS={','.join(vsite_labels)}"
                return [group_label], vsite_commands + [group_cmd]
            else:
                # Single virtual site, use directly
                return vsite_labels, vsite_commands
        else:
            # AtomSelector: create group(s) based on flatten parameter
            groups = input_obj.select(atoms)
            if not groups:
                return [], []

            if self.flatten:
                # Flatten all groups into single group
                flat_atoms = [idx for group in groups for idx in group]
                # Return as list of atom indices (will be formatted in COORDINATION command)
                return [flat_atoms], []
            else:
                # Smart GROUP creation: only create GROUP for multi-atom groups
                labels = []
                commands = []
                for i, group in enumerate(groups):
                    if len(group) == 1:
                        # Single atom: use directly (no GROUP needed)
                        labels.append(str(group[0] + 1))
                    else:
                        # Multi-atom group: create GROUP
                        group_label = f"{self.prefix}_{label_prefix}_g{i}"
                        atom_list = ",".join(str(idx + 1) for idx in group)
                        commands.append(f"{group_label}: GROUP ATOMS={atom_list}")
                        labels.append(group_label)

                # If multiple groups, create a parent GROUP
                if len(labels) > 1:
                    parent_label = f"{self.prefix}_{label_prefix}_group"
                    parent_cmd = f"{parent_label}: GROUP ATOMS={','.join(labels)}"
                    return [parent_label], commands + [parent_cmd]
                else:
                    return labels, commands

    def _generate_coordination_cvs(
        self, labels1: List[str | List[int]], labels2: List[str | List[int]]
    ) -> Tuple[List[str], List[str]]:
        """Generate COORDINATION CV commands.

        Parameters
        ----------
        labels1, labels2 : list[str | list[int]]
            Group labels or atom index lists for GROUPA and GROUPB.

        Returns
        -------
        cv_labels : list[str]
            Labels for the COORDINATION CVs created.
        commands : list[str]
            COORDINATION command strings.
        """
        n1, n2 = len(labels1), len(labels2)

        # Determine which pairs to create based on pairwise strategy
        if n1 == 1 and n2 == 1:
            # One-to-one: always create single CV
            pairs = [(0, 0)]
        elif n1 == 1:
            # One-to-many: pair first of x1 with all of x2
            pairs = [(0, j) for j in range(n2)]
        elif n2 == 1:
            # Many-to-one: pair all of x1 with first of x2
            pairs = [(i, 0) for i in range(n1)]
        else:
            # Many-to-many: apply pairwise strategy
            if self.pairwise == "all":
                pairs = [(i, j) for i in range(n1) for j in range(n2)]
            elif self.pairwise == "diagonal":
                n_pairs = min(n1, n2)
                pairs = [(i, i) for i in range(n_pairs)]
            elif self.pairwise == "none":
                raise ValueError(
                    f"Both x1 and x2 have multiple groups ({n1} and {n2}). "
                    f"Use pairwise='all' or 'diagonal', or select specific groups with indexing."
                )
            else:
                raise ValueError(f"Unknown pairwise strategy: {self.pairwise}")

        # Generate COORDINATION commands
        cv_labels = []
        commands = []
        for idx, (i, j) in enumerate(pairs):
            if len(pairs) == 1:
                label = self.prefix
            else:
                label = f"{self.prefix}_{idx}"

            # Format group labels for COORDINATION
            def format_group(g):
                if isinstance(g, list):  # List of atom indices
                    return ",".join(str(idx + 1) for idx in g)
                else:  # String label
                    return g

            g_a = format_group(labels1[i])
            g_b = format_group(labels2[j])

            # Create COORDINATION command
            cmd = f"{label}: COORDINATION GROUPA={g_a}"
            if g_a != g_b:  # Omit GROUPB for self-coordination
                cmd += f" GROUPB={g_b}"

            # Add parameters
            cmd += f" R_0={self.r_0} NN={self.nn} D_0={self.d_0}"
            if self.mm != 0:
                cmd += f" MM={self.mm}"

            commands.append(cmd)
            cv_labels.append(label)

        return cv_labels, commands


@dataclass
class TorsionCV(_BasePlumedCV):
    """
    PLUMED TORSION collective variable.

    Calculates the torsional (dihedral) angle defined by four atoms. Each group
    provided by the selector must contain exactly four atoms.

    Parameters
    ----------
    atoms : AtomSelector
        Selector for one or more groups of 4 atoms. Each group must contain exactly 4 atoms.
    prefix : str
        Label prefix for the generated PLUMED commands.
    strategy : {"first", "all"}, default="first"
        Strategy for handling multiple groups from the selector:
        - "first": Process only the first group (creates 1 CV)
        - "all": Process all groups independently (creates N CVs)

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/TORSION
    """

    atoms: AtomSelector
    prefix: str
    strategy: Literal["first", "all"] = "first"

    def _get_atom_highlights(
        self, atoms: Atoms, **kwargs
    ) -> Optional[AtomHighlightMap]:
        groups = self.atoms.select(atoms)
        if not groups or len(groups[0]) != 4:
            print("Warning: Torsion CV requires a group of 4 atoms for visualization.")
            return None

        # Highlight the first 4-atom group with a color sequence.
        torsion_atoms = groups[0]
        colors = [
            (1.0, 0.2, 0.2),  # Red
            (1.0, 0.6, 0.2),  # Orange
            (1.0, 1.0, 0.2),  # Yellow
            (0.2, 1.0, 0.2),  # Green
        ]
        return {atom_idx: color for atom_idx, color in zip(torsion_atoms, colors)}

    def to_plumed(self, atoms: Atoms) -> Tuple[List[str], List[str]]:
        """
        Generates PLUMED input strings for the TORSION CV.

        Returns:
            A tuple containing a list of CV labels and a list of PLUMED commands.
        """
        groups = self.atoms.select(atoms)
        if not groups:
            raise ValueError(f"Empty selection for torsion CV '{self.prefix}'")

        for i, group in enumerate(groups):
            if len(group) != 4:
                raise ValueError(
                    f"Torsion CV requires 4 atoms per group, but group {i} has {len(group)}."
                )

        commands = self._generate_commands(groups)
        labels = self._extract_labels(commands, self.prefix, "TORSION")
        return labels, commands

    def _generate_commands(self, groups: List[List[int]]) -> List[str]:
        """Generates all necessary PLUMED commands."""
        # Determine which groups to process based on strategy
        if self.strategy == "first" and groups:
            indices_to_process = [0]
        else:  # "all" - process all groups independently
            indices_to_process = list(range(len(groups)))

        commands = []
        for i in indices_to_process:
            label = (
                self.prefix if len(indices_to_process) == 1 else f"{self.prefix}_{i}"
            )
            atom_list = ",".join(str(idx + 1) for idx in groups[i])
            commands.append(f"{label}: TORSION ATOMS={atom_list}")
        return commands


# TODO: we might need to set weights because plumed does not know about the atomistic weights?
@dataclass
class RadiusOfGyrationCV(_BasePlumedCV):
    """
    PLUMED GYRATION collective variable.

    Calculates the radius of gyration of a group of atoms. The radius of gyration
    is a measure of the size of a molecular system.

    Parameters
    ----------
    atoms : AtomSelector
        Selector for the atoms to include in the gyration calculation.
    prefix : str
        Label prefix for the generated PLUMED commands.
    flatten : bool, default=False
        How to handle multiple groups from the selector:
        - True: Combine all groups into one and calculate single Rg (creates 1 CV)
        - False: Keep groups separate, use strategy to determine which to process
    strategy : {"first", "all"}, default="first"
        Strategy for handling multiple groups when flatten=False:
        - "first": Process only the first group (creates 1 CV)
        - "all": Process all groups independently (creates N CVs)
    type : str, default="RADIUS"
        The type of gyration tensor to use.
        Options: "RADIUS", "GTPC_1", "GTPC_2", "GTPC_3", "ASPHERICITY", "ACYLINDRICITY", "KAPPA2", etc.

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/GYRATION/
    """

    atoms: AtomSelector
    prefix: str
    flatten: bool = False
    strategy: Literal["first", "all"] = "first"
    type: str = "RADIUS"  # Options: RADIUS, GTPC_1, GTPC_2, GTPC_3, ASPHERICITY, ACYLINDRICITY, KAPPA2, etc.

    def _get_atom_highlights(
        self, atoms: Atoms, **kwargs
    ) -> Optional[AtomHighlightMap]:
        groups = self.atoms.select(atoms)
        if not groups or not groups[0]:
            return None

        # Highlight all atoms in the first group with a single color
        group = groups[0]
        return {atom_idx: (0.2, 0.8, 0.2) for atom_idx in group}  # Green

    def to_plumed(self, atoms: Atoms) -> Tuple[List[str], List[str]]:
        """
        Generates PLUMED input strings for the GYRATION CV.

        Returns:
            A tuple containing a list of CV labels and a list of PLUMED commands.
        """
        groups = self.atoms.select(atoms)
        if not groups:
            raise ValueError(f"Empty selection for gyration CV '{self.prefix}'")

        commands = self._generate_commands(groups)
        labels = self._extract_labels(commands, self.prefix, "GYRATION")
        return labels, commands

    def _generate_commands(self, groups: List[List[int]]) -> List[str]:
        """Generates all necessary PLUMED commands."""
        commands = []

        if self.flatten:
            # Combine all groups into single atom list
            flat_atoms = [idx for group in groups for idx in group]
            atom_list = ",".join(str(idx + 1) for idx in flat_atoms)
            command = f"{self.prefix}: GYRATION ATOMS={atom_list}"
            if self.type != "RADIUS":
                command += f" TYPE={self.type}"
            commands.append(command)
        else:
            # Keep groups separate and use strategy to determine which to process
            if self.strategy == "first" and groups:
                indices_to_process = [0]
            else:  # "all" - process all groups independently
                indices_to_process = list(range(len(groups)))

            for i in indices_to_process:
                label = (
                    self.prefix
                    if len(indices_to_process) == 1
                    else f"{self.prefix}_{i}"
                )
                atom_list = ",".join(str(idx + 1) for idx in groups[i])
                command = f"{label}: GYRATION ATOMS={atom_list}"
                if self.type != "RADIUS":
                    command += f" TYPE={self.type}"
                commands.append(command)

        return commands

import dataclasses
from typing import Literal, Union

import rdkit2ase
from ase import Atoms
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from plumed_nodes.interfaces import AtomSelector, CollectiveVariable

# Define allowed string literals for type hints
GroupReductionStrategyType = Literal["com", "cog", "first", "all"]
MultiGroupStrategyType = Literal["first", "all_pairs", "corresponding", "first_to_all"]


@dataclasses.dataclass
class DistanceCV(CollectiveVariable):
    """
    PLUMED DISTANCE collective variable.
    """

    x1: AtomSelector
    x2: AtomSelector
    prefix: str
    group_reduction: GroupReductionStrategyType = "com"
    multi_group: MultiGroupStrategyType = "first"
    create_virtual_sites: bool = True

    def get_img(self, atoms: Atoms) -> Image.Image | None:
        """
        Generates an image of the molecule(s) with the selected atoms for the distance CV highlighted.

        - The first atom from the first selector (`x1`) is colored red.
        - The first atom from the second selector (`x2`) is colored blue.

        If the atoms are in different molecular fragments, both fragments are shown side-by-side.

        Returns
        -------
        PIL.Image.Image | None
            A PIL Image object of the visualization, or None if atoms could not be selected.
        """
        # --- Start of new/modified code ---

        # 1. Get atom groups from each selector
        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)

        if not groups1 or not groups1[0] or not groups2 or not groups2[0]:
            print("Warning: One or both selectors returned no atoms for visualization.")
            return None

        # For visualization, we only highlight the first atom of the first group from each selector
        atom1_idx = groups1[0][0]
        atom2_idx = groups2[0][0]

        # 2. Convert ASE object to RDKit and identify molecular fragments
        mol = rdkit2ase.ase2rdkit(atoms)
        mol_frags = Chem.GetMolFrags(mol, asMols=True)
        frag_indices_list = Chem.GetMolFrags(mol, asMols=False)

        # 3. Find which fragments contain the selected atoms and prepare for drawing
        mols_to_draw = []
        highlights_to_draw = []
        colors_to_draw = []
        red = (1.0, 0.2, 0.2)
        blue = (0.2, 0.2, 1.0)

        for i, frag_mol in enumerate(mol_frags):
            # Create a map of global (original) index -> local (fragment) index
            local_idx_map = {
                global_idx: local_idx
                for local_idx, global_idx in enumerate(frag_indices_list[i])
            }

            current_highlights = {}

            # Check if our selected atoms are in this fragment
            if atom1_idx in local_idx_map:
                current_highlights[local_idx_map[atom1_idx]] = red
            if atom2_idx in local_idx_map:
                current_highlights[local_idx_map[atom2_idx]] = blue

            # If this fragment contains at least one of the selected atoms, add it for drawing
            if current_highlights:
                mols_to_draw.append(frag_mol)
                highlights_to_draw.append(list(current_highlights.keys()))
                colors_to_draw.append(current_highlights)

        if not mols_to_draw:
            return None

        # 4. Generate the image using RDKit's grid drawing utility
        img = Draw.MolsToGridImage(
            mols_to_draw,
            molsPerRow=len(mols_to_draw),  # Places all fragments in a single row
            subImgSize=(400, 400),
            highlightAtomLists=highlights_to_draw,
            highlightAtomColors=colors_to_draw,
            useSVG=False,  # Ensures a PIL Image object is returned
        )
        return img

    def to_plumed(self, atoms: Atoms) -> tuple[list[str], list[str]]:
        """Generate PLUMED input string(s) for DISTANCE.

        Returns
        -------
        - list of distance labels
        - PLUMED input string
        """
        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)

        if not groups1 or not groups2:
            raise ValueError(f"Empty selection for distance CV {self.prefix}")

        # Check for overlaps
        overlaps = self._check_overlaps(groups1, groups2)
        if overlaps and self.group_reduction not in ["com", "cog"]:
            raise ValueError(
                f"Overlapping atoms found: {overlaps}. "
                "This is only valid with CENTER_OF_MASS or CENTER_OF_GEOMETRY reduction."
            )

        commands = self._generate_commands(groups1, groups2, atoms)

        # Extract labels from commands
        labels = []
        for cmd in commands:
            if ":" in cmd and cmd.strip().startswith((self.prefix, f"{self.prefix}_")):
                label_part = cmd.split(":")[0].strip()
                if "DISTANCE" in cmd:
                    labels.append(label_part)

        return labels, commands

    def _check_overlaps(
        self, groups1: list[list[int]], groups2: list[list[int]]
    ) -> set[int]:
        """Check for overlapping indices between groups."""
        flat1 = {idx for group in groups1 for idx in group}
        flat2 = {idx for group in groups2 for idx in group}
        return flat1.intersection(flat2)

    def _generate_commands(
        self, groups1: list[list[int]], groups2: list[list[int]], atoms: Atoms
    ) -> list[str]:
        """Generate PLUMED commands based on the strategies."""
        commands = []

        # Determine which groups to process based on multi_group strategy
        if self.multi_group == "first":
            # Only process first groups
            process_groups1 = [groups1[0]]
            process_groups2 = [groups2[0]]
            group_pairs = [(0, 0)]
        elif self.multi_group == "all_pairs":
            process_groups1 = groups1
            process_groups2 = groups2
            group_pairs = [
                (i, j) for i in range(len(groups1)) for j in range(len(groups2))
            ]
        elif self.multi_group == "corresponding":
            n = min(len(groups1), len(groups2))
            process_groups1 = groups1[:n]
            process_groups2 = groups2[:n]
            group_pairs = [(i, i) for i in range(n)]
        elif self.multi_group == "first_to_all":
            process_groups1 = [groups1[0]]
            process_groups2 = groups2
            group_pairs = [(0, j) for j in range(len(groups2))]

        # Create virtual sites for the groups we're actually using
        sites1 = {}
        sites2 = {}

        for i, group in enumerate(process_groups1):
            site, site_commands = self._reduce_group(
                group, f"{self.prefix}_g1_{i}", atoms
            )
            sites1[i] = site
            commands.extend(site_commands)

        for j, group in enumerate(process_groups2):
            site, site_commands = self._reduce_group(
                group, f"{self.prefix}_g2_{j}", atoms
            )
            sites2[j] = site
            commands.extend(site_commands)

        # Create distance commands for the specified pairs
        for i, j in group_pairs:
            if len(group_pairs) == 1:
                dist_label = self.prefix
            else:
                dist_label = f"{self.prefix}_{i}_{j}"

            commands.append(
                self._make_distance_command(sites1[i], sites2[j], dist_label)
            )

        return commands

    def _reduce_group(
        self, group: list[int], prefix: str, atoms: Atoms
    ) -> tuple[Union[str, list[int]], list[str]]:
        """
        Reduce a group to a single point based on reduction strategy.

        Returns:
            - Site identifier (atom index, virtual site label, or group)
            - List of PLUMED commands to create virtual sites
        """
        commands = []

        if len(group) == 1:
            # Single atom - no reduction needed
            return str(group[0] + 1), commands

        if self.group_reduction == "first":
            return str(group[0] + 1), commands

        if self.group_reduction == "com":
            if self.create_virtual_sites:
                site_label = f"{prefix}_com"
                atom_list = ",".join(str(idx + 1) for idx in group)
                commands.append(f"{site_label}: COM ATOMS={atom_list}")
                return site_label, commands
            else:
                return group, commands

        if self.group_reduction == "cog":
            site_label = f"{prefix}_cog"
            atom_list = ",".join(str(idx + 1) for idx in group)
            commands.append(f"{site_label}: CENTER ATOMS={atom_list}")
            return site_label, commands

        if self.group_reduction == "all":
            return group, commands

        raise ValueError(f"Unknown group reduction strategy: {self.group_reduction}")

    def _make_distance_command(
        self, site1: Union[str, list], site2: Union[str, list], label: str
    ) -> str:
        """Create a single DISTANCE command."""
        if isinstance(site1, str) and isinstance(site2, str):
            return f"{label}: DISTANCE ATOMS={site1},{site2}"
        elif isinstance(site1, list) and isinstance(site2, list):
            atoms1 = ",".join(str(idx + 1) for idx in site1)
            atoms2 = ",".join(str(idx + 1) for idx in site2)
            return f"{label}: DISTANCE ATOMS1={atoms1} ATOMS2={atoms2}"
        else:
            if isinstance(site1, list):
                atoms1 = ",".join(str(idx + 1) for idx in site1)
                return f"{label}: DISTANCE ATOMS1={atoms1} ATOMS2={site2}"
            else:
                atoms2 = ",".join(str(idx + 1) for idx in site2)
                return f"{label}: DISTANCE ATOMS1={site1} ATOMS2={atoms2}"

# --- IMPORTS ---
# Standard library
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

# Third-party
import rdkit2ase
from ase import Atoms
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

# Local
from hillclimber.interfaces import AtomSelector, CollectiveVariable


# --- TYPE HINTS ---
GroupReductionStrategyType = Literal[
    "com", "cog", "first", "all", "com_per_group", "cog_per_group"
]
MultiGroupStrategyType = Literal["first", "all_pairs", "corresponding", "first_to_all"]
SiteIdentifier = Union[str, List[int]]
ColorTuple = Tuple[float, float, float]
AtomHighlightMap = Dict[int, ColorTuple]


# --- BASE CLASS FOR SHARED LOGIC ---
class _BasePlumedCV(CollectiveVariable):
    """An abstract base class for PLUMED CVs providing shared utilities."""

    prefix: str

    def _get_atom_highlights(self, atoms: Atoms, **kwargs) -> Optional[AtomHighlightMap]:
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
        mol = rdkit2ase.ase2rdkit(atoms)

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
    def _extract_labels(
        commands: List[str], prefix: str, cv_keyword: str
    ) -> List[str]:
        """Extracts generated CV labels from a list of PLUMED commands."""
        return [
            cmd.split(":", 1)[0].strip()
            for cmd in commands
            if cv_keyword in cmd and cmd.strip().startswith((prefix, f"{prefix}_"))
        ]

    @staticmethod
    def _get_index_pairs(
        len1: int, len2: int, strategy: MultiGroupStrategyType
    ) -> List[Tuple[int, int]]:
        """Determines pairs of group indices based on the multi-group strategy."""
        if strategy == "first":
            return [(0, 0)] if len1 > 0 and len2 > 0 else []
        if strategy == "all_pairs":
            return [(i, j) for i in range(len1) for j in range(len2)]
        if strategy == "corresponding":
            n = min(len1, len2)
            return [(i, i) for i in range(n)]
        if strategy == "first_to_all":
            return [(0, j) for j in range(len2)] if len1 > 0 else []
        raise ValueError(f"Unknown multi-group strategy: {strategy}")

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

    Calculates the distance between two atoms or groups of atoms. This CV supports
    various strategies for reducing groups to single points (e.g., center of mass)
    and for pairing multiple groups.

    Attributes:
        x1: Selector for the first atom/group.
        x2: Selector for the second atom/group.
        prefix: Label prefix for the generated PLUMED commands.
        group_reduction: Strategy to reduce an atom group to a single point.
        multi_group: Strategy for handling multiple groups from selectors.
        create_virtual_sites: If True, create explicit virtual sites for COM/COG.

    Resources:
        - https://www.plumed.org/doc-master/user-doc/html/DISTANCE.html
    """

    x1: AtomSelector
    x2: AtomSelector
    prefix: str
    group_reduction: GroupReductionStrategyType = "com"
    multi_group: MultiGroupStrategyType = "first"
    create_virtual_sites: bool = True

    def _get_atom_highlights(
        self, atoms: Atoms, **kwargs
    ) -> Optional[AtomHighlightMap]:
        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)

        if not groups1 or not groups2:
            return None

        index_pairs = self._get_index_pairs(len(groups1), len(groups2), self.multi_group)
        if not index_pairs:
            return None

        # Correctly select atoms based on the group_reduction strategy
        indices1, indices2 = set(), set()
        for i, j in index_pairs:
            # Handle the 'first' atom case specifically for highlighting
            if self.group_reduction == "first":
                # Ensure the group is not empty before accessing the first element
                if groups1[i]:
                    indices1.add(groups1[i][0])
                if groups2[j]:
                    indices2.add(groups2[j][0])
            # For other strategies (com, cog, all), highlight the whole group
            else:
                indices1.update(groups1[i])
                indices2.update(groups2[j])

        if not indices1 and not indices2:
            return None

        # Color atoms based on group membership, with purple for overlaps.
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
        """
        Generates PLUMED input strings for the DISTANCE CV.

        Returns:
            A tuple containing a list of CV labels and a list of PLUMED commands.
        """
        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)

        if not groups1 or not groups2:
            raise ValueError(f"Empty selection for distance CV '{self.prefix}'")

        flat1 = {idx for group in groups1 for idx in group}
        flat2 = {idx for group in groups2 for idx in group}
        if flat1.intersection(flat2) and self.group_reduction not in ["com", "cog"]:
            raise ValueError(
                "Overlapping atoms found. This is only valid with 'com' or 'cog' reduction."
            )

        commands = self._generate_commands(groups1, groups2)
        labels = self._extract_labels(commands, self.prefix, "DISTANCE")
        return labels, commands

    def _generate_commands(
        self, groups1: List[List[int]], groups2: List[List[int]]
    ) -> List[str]:
        """Generates all necessary PLUMED commands."""
        commands = []
        index_pairs = self._get_index_pairs(
            len(groups1), len(groups2), self.multi_group
        )

        # Efficiently create virtual sites only for groups that will be used.
        sites1, sites2 = {}, {}
        unique_indices1 = sorted({i for i, j in index_pairs})
        unique_indices2 = sorted({j for i, j in index_pairs})

        for i in unique_indices1:
            site, site_cmds = self._reduce_group(groups1[i], f"{self.prefix}_g1_{i}")
            sites1[i] = site
            commands.extend(site_cmds)
        for j in unique_indices2:
            site, site_cmds = self._reduce_group(groups2[j], f"{self.prefix}_g2_{j}")
            sites2[j] = site
            commands.extend(site_cmds)

        # Create the final DISTANCE commands.
        for i, j in index_pairs:
            label = self.prefix if len(index_pairs) == 1 else f"{self.prefix}_{i}_{j}"
            cmd = self._make_distance_command(sites1[i], sites2[j], label)
            commands.append(cmd)

        return commands

    def _reduce_group(
        self, group: List[int], site_prefix: str
    ) -> Tuple[SiteIdentifier, List[str]]:
        """Reduces a single atom group to a site identifier based on strategy."""
        if len(group) == 1 or self.group_reduction == "first":
            return str(group[0] + 1), []
        if self.group_reduction == "all":
            return group, []

        if self.group_reduction in ["com", "cog"]:
            if self.create_virtual_sites:
                label = f"{site_prefix}_{self.group_reduction}"
                cmd = self._create_virtual_site_command(
                    group, self.group_reduction, label
                )
                return label, [cmd]
            return group, []  # Use group directly if not creating virtual sites

        raise ValueError(f"Unknown group reduction strategy: {self.group_reduction}")

    def _make_distance_command(
        self, site1: SiteIdentifier, site2: SiteIdentifier, label: str
    ) -> str:
        """Creates a single PLUMED DISTANCE command string."""

        def _format(site):
            return ",".join(map(str, (s + 1 for s in site))) if isinstance(site, list) else site

        s1_str, s2_str = _format(site1), _format(site2)
        # Use ATOMS for point-like sites, ATOMS1/ATOMS2 for group-based distances
        if isinstance(site1, str) and isinstance(site2, str):
            return f"{label}: DISTANCE ATOMS={s1_str},{s2_str}"
        return f"{label}: DISTANCE ATOMS1={s1_str} ATOMS2={s2_str}"


@dataclass
class CoordinationNumberCV(_BasePlumedCV):
    """
    PLUMED COORDINATION collective variable.

    Calculates a coordination number based on a switching function. It supports
    complex group definitions, including groups of virtual sites.

    Attributes:
        x1, x2: Selectors for the two groups of atoms.
        prefix: Label prefix for the generated PLUMED commands.
        r_0: The reference distance for the switching function (in Angstroms).
        nn, mm, d_0: Parameters for the switching function.
        group_reduction_1, group_reduction_2: Reduction strategies for each group.
        multi_group: Strategy for handling multiple groups from selectors.
        create_virtual_sites: If True, create explicit virtual sites for COM/COG.

    Resources:
        - https://www.plumed.org/doc-master/user-doc/html/COORDINATION.html
        - https://www.plumed.org/doc-master/user-doc/html/GROUP.html
    """

    x1: AtomSelector
    x2: AtomSelector
    prefix: str
    r_0: float
    nn: int = 6
    mm: int = 0
    d_0: float = 0.0
    group_reduction_1: GroupReductionStrategyType = "all"
    group_reduction_2: GroupReductionStrategyType = "all"
    multi_group: MultiGroupStrategyType = "first"
    create_virtual_sites: bool = True

    def _get_atom_highlights(
        self, atoms: Atoms, **kwargs
    ) -> Optional[AtomHighlightMap]:
        highlight_hydrogens = kwargs.get("highlight_hydrogens", False)
        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)

        if not groups1 or not groups2:
            return None

        # Flatten groups and optionally filter out hydrogens.
        indices1 = {idx for g in groups1 for idx in g}
        indices2 = {idx for g in groups2 for idx in g}
        if not highlight_hydrogens:
            indices1 = {i for i in indices1 if atoms[i].symbol != "H"}
            indices2 = {i for i in indices2 if atoms[i].symbol != "H"}

        if not indices1 and not indices2:
            return None

        # Color atoms based on group membership, with purple for overlaps.
        highlights: AtomHighlightMap = {}
        red, blue, purple = (1.0, 0.5, 0.5), (0.5, 0.5, 1.0), (1.0, 0.5, 1.0)
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
        """
        Generates PLUMED input strings for the COORDINATION CV.

        Returns:
            A tuple containing a list of CV labels and a list of PLUMED commands.
        """
        groups1 = self.x1.select(atoms)
        groups2 = self.x2.select(atoms)

        if not groups1 or not groups2:
            raise ValueError(f"Empty selection for coordination CV '{self.prefix}'")

        commands = self._generate_commands(groups1, groups2)
        labels = self._extract_labels(commands, self.prefix, "COORDINATION")
        return labels, commands

    def _generate_commands(
        self, groups1: List[List[int]], groups2: List[List[int]]
    ) -> List[str]:
        """Generates all necessary PLUMED commands."""
        commands: List[str] = []

        sites1 = self._reduce_groups(
            groups1, self.group_reduction_1, f"{self.prefix}_g1", commands
        )
        sites2 = self._reduce_groups(
            groups2, self.group_reduction_2, f"{self.prefix}_g2", commands
        )

        # Get site pairs using a simplified helper
        site_pairs = []
        if self.multi_group == "first":
            site_pairs = [(sites1[0], sites2[0])] if sites1 and sites2 else []
        elif self.multi_group == "all_pairs":
            site_pairs = [(s1, s2) for s1 in sites1 for s2 in sites2]
        elif self.multi_group == "corresponding":
            n = min(len(sites1), len(sites2))
            site_pairs = [(sites1[i], sites2[i]) for i in range(n)]
        elif self.multi_group == "first_to_all":
            site_pairs = [(sites1[0], s2) for s2 in sites2] if sites1 else []

        for i, (s1, s2) in enumerate(site_pairs):
            label = self.prefix if len(site_pairs) == 1 else f"{self.prefix}_{i}"
            commands.append(self._make_coordination_command(s1, s2, label))

        return commands

    def _reduce_groups(
        self,
        groups: List[List[int]],
        strategy: GroupReductionStrategyType,
        site_prefix: str,
        commands: List[str],
    ) -> List[SiteIdentifier]:
        """Reduces a list of atom groups into a list of site identifiers."""
        if strategy in ["com_per_group", "cog_per_group"]:
            if not self.create_virtual_sites:
                raise ValueError(f"'{strategy}' requires create_virtual_sites=True")

            reduction_type = "COM" if strategy == "com_per_group" else "CENTER"
            vsite_labels = []
            for i, group in enumerate(groups):
                if not group:
                    continue
                vsite_label = f"{site_prefix}_{i}"
                atom_list = ",".join(str(idx + 1) for idx in group)
                commands.append(f"{vsite_label}: {reduction_type} ATOMS={atom_list}")
                vsite_labels.append(vsite_label)

            group_label = f"{site_prefix}_group"
            commands.append(f"{group_label}: GROUP ATOMS={','.join(vsite_labels)}")
            return [group_label]

        if strategy == "all":
            return [sorted({idx for group in groups for idx in group})]

        # Handle other strategies by reducing each group individually.
        sites: List[SiteIdentifier] = []
        for i, group in enumerate(groups):
            if len(group) == 1 or strategy == "first":
                sites.append(str(group[0] + 1))
            elif strategy in ["com", "cog"]:
                if self.create_virtual_sites:
                    label = f"{site_prefix}_{i}_{strategy}"
                    cmd = self._create_virtual_site_command(group, strategy, label)
                    commands.append(cmd)
                    sites.append(label)
                else:
                    sites.append(group)
            else:
                raise ValueError(f"Unsupported reduction strategy: {strategy}")
        return sites

    def _make_coordination_command(
        self, site1: SiteIdentifier, site2: SiteIdentifier, label: str
    ) -> str:
        """Creates a single PLUMED COORDINATION command string."""

        def _format(site):
            return ",".join(map(str, (s + 1 for s in site))) if isinstance(site, list) else site

        g_a, g_b = _format(site1), _format(site2)
        base_cmd = f"{label}: COORDINATION GROUPA={g_a}"
        if g_a != g_b:  # Omit GROUPB for self-coordination
            base_cmd += f" GROUPB={g_b}"

        params = f" R_0={self.r_0} NN={self.nn} D_0={self.d_0}"
        if self.mm != 0:
            params += f" MM={self.mm}"

        return base_cmd + params


@dataclass
class TorsionCV(_BasePlumedCV):
    """
    PLUMED TORSION collective variable.

    Calculates the torsional (dihedral) angle defined by four atoms. Each group
    provided by the selector must contain exactly four atoms.

    Attributes:
        atoms: Selector for one or more groups of 4 atoms.
        prefix: Label prefix for the generated PLUMED commands.
        multi_group: Strategy for handling multiple groups from the selector.

    Resources:
        - https://www.plumed.org/doc-master/user-doc/html/TORSION.html
    """

    atoms: AtomSelector
    prefix: str
    multi_group: MultiGroupStrategyType = "first"

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
        # For torsions, 'multi_group' determines how many groups to process.
        if self.multi_group in ["first", "first_to_all"] and groups:
            indices_to_process = [0]
        else:  # "all_pairs" and "corresponding" imply processing all independent groups.
            indices_to_process = list(range(len(groups)))

        commands = []
        for i in indices_to_process:
            label = self.prefix if len(indices_to_process) == 1 else f"{self.prefix}_{i}"
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

    Attributes:
        atoms: Selector for the atoms to include in the gyration calculation.
        prefix: Label prefix for the generated PLUMED commands.
        multi_group: Strategy for handling multiple groups from the selector.
        type: The type of gyration tensor to use ("RADIUS" for scalar Rg, "GTPC_1", etc.)

    Resources:
        - https://www.plumed.org/doc-master/user-doc/html/GYRATION/
    """

    atoms: AtomSelector
    prefix: str
    multi_group: MultiGroupStrategyType = "first"
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
        # For gyration, 'multi_group' determines how many groups to process.
        if self.multi_group in ["first", "first_to_all"] and groups:
            indices_to_process = [0]
        else:  # "all_pairs" and "corresponding" imply processing all independent groups.
            indices_to_process = list(range(len(groups)))

        commands = []
        for i in indices_to_process:
            label = self.prefix if len(indices_to_process) == 1 else f"{self.prefix}_{i}"
            atom_list = ",".join(str(idx + 1) for idx in groups[i])
            command = f"{label}: GYRATION ATOMS={atom_list}"
            if self.type != "RADIUS":
                command += f" TYPE={self.type}"
            commands.append(command)
        return commands

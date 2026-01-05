"""Python Collective Variables (PyCV) for PLUMED.

This module provides a base class for implementing custom collective variables
in Python that integrate with PLUMED via the PYCVINTERFACE mechanism.

Resources
---------
- https://www.plumed-tutorials.org/lessons/24/015/data/GAT_SAFE_README.html
- https://joss.theoj.org/papers/10.21105/joss.01773
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase import Atoms
from PIL import Image

from hillclimber.interfaces import AtomSelector


@dataclass
class PyCV(ABC):
    """Base class for user-defined Python Collective Variables.

    Users subclass this and implement the `compute()` method.
    The CV is evaluated via PLUMED's PYCVINTERFACE mechanism.

    Parameters
    ----------
    atoms : AtomSelector | list[int] | None
        Atoms to pass to the CV. Either an AtomSelector, direct indices (0-based),
        or None to select all atoms.
    prefix : str
        Label prefix for PLUMED commands.

    Examples
    --------
    >>> import hillclimber as hc
    >>> import numpy as np
    >>> from ase import Atoms
    >>>
    >>> class MyDistanceCV(hc.PyCV):
    ...     def compute(self, atoms: Atoms) -> tuple[float, np.ndarray]:
    ...         positions = atoms.get_positions()
    ...         diff = positions[1] - positions[0]
    ...         dist = np.sqrt(np.sum(diff**2))
    ...         grad = np.zeros((len(atoms), 3))
    ...         grad[0] = -diff / dist
    ...         grad[1] = diff / dist
    ...         return dist, grad
    >>>
    >>> cv = MyDistanceCV(
    ...     atoms=hc.IndexSelector(indices=[[0, 1]]),
    ...     prefix="my_dist"
    ... )

    Resources
    ---------
    - https://www.plumed-tutorials.org/lessons/24/015/data/GAT_SAFE_README.html
    - https://joss.theoj.org/papers/10.21105/joss.01773

    Notes
    -----
    The `compute()` method receives an ASE Atoms object with positions in
    the same units as specified in the PLUMED UNITS line. When used with
    hillclimber's MetaDynamicsModel, this is Angstrom (ASE default units).

    If `compute()` returns gradients, the CV can be used for biasing
    (metadynamics, restraints, etc.). If only a scalar is returned,
    the CV can only be printed/monitored.
    """

    atoms: AtomSelector | list[int] | None
    prefix: str

    @abstractmethod
    def compute(
        self,
        atoms: Atoms,
    ) -> tuple[float, npt.NDArray[np.float64]] | float:
        """Compute the CV value and optionally gradients.

        Parameters
        ----------
        atoms : ase.Atoms
            The selected atoms with their current positions from the simulation.
            Positions are in the units specified by PLUMED's UNITS line
            (Angstrom when using hillclimber's default configuration).
            Species, masses, and other properties are preserved from the
            original system definition.

        Returns
        -------
        value : float
            The CV value.
        gradients : np.ndarray, optional
            Gradients of shape (n_atoms, 3) where n_atoms is the number of
            atoms in the input. If not returned, derivatives will not be
            available and the CV cannot be biased.

        Examples
        --------
        >>> def compute(self, atoms):
        ...     # Simple distance between first two atoms
        ...     pos = atoms.get_positions()
        ...     diff = pos[1] - pos[0]
        ...     dist = np.linalg.norm(diff)
        ...
        ...     # Compute gradients
        ...     grad = np.zeros((len(atoms), 3))
        ...     grad[0] = -diff / dist
        ...     grad[1] = diff / dist
        ...
        ...     return dist, grad
        """
        ...

    def to_plumed(self, atoms: Atoms) -> tuple[list[str], list[str]]:
        """Generate PLUMED commands for this CV.

        Parameters
        ----------
        atoms : ase.Atoms
            Reference structure for atom selection.

        Returns
        -------
        labels : list[str]
            CV labels generated (single element list with self.prefix).
        commands : list[str]
            PLUMED command strings for PYCVINTERFACE.

        Notes
        -----
        The actual LOAD command for the pycv plugin is added by
        MetaDynamicsModel, not here.
        """
        # Get atom indices (0-based)
        indices = self._get_atom_indices(atoms)

        if not indices:
            raise ValueError(f"Empty atom selection for PyCV '{self.prefix}'")

        # Generate module name based on prefix
        module_name = f"_pycv_{self.prefix}"

        # Convert to 1-based indices for PLUMED
        atom_list = ",".join(str(i + 1) for i in indices)

        commands = [
            f"{self.prefix}: PYCVINTERFACE ATOMS={atom_list} IMPORT={module_name}",
        ]

        return [self.prefix], commands

    def get_img(self, atoms: Atoms) -> Image.Image:
        """Generate visualization for this CV.

        For PyCV, returns a placeholder gray image since the CV logic
        is user-defined and may not have a simple visual representation.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure.

        Returns
        -------
        Image.Image
            A placeholder gray image.
        """
        # Create a simple placeholder image
        img = Image.new("RGB", (400, 400), color=(200, 200, 200))
        return img

    def _get_atom_indices(self, atoms: Atoms) -> list[int]:
        """Get flat list of atom indices (0-based).

        Parameters
        ----------
        atoms : ase.Atoms
            Reference structure for atom selection.

        Returns
        -------
        list[int]
            Flat list of 0-based atom indices.
        """
        if self.atoms is None:
            # None means all atoms
            return list(range(len(atoms)))
        elif isinstance(self.atoms, list):
            return self.atoms
        else:
            # AtomSelector returns list[list[int]], flatten it
            groups = self.atoms.select(atoms)
            return [idx for group in groups for idx in group]

    def _get_symbols(self, atoms: Atoms) -> list[str]:
        """Get atomic symbols for the selected atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            Reference structure.

        Returns
        -------
        list[str]
            List of chemical symbols in selection order.
        """
        indices = self._get_atom_indices(atoms)
        all_symbols = atoms.get_chemical_symbols()
        return [all_symbols[i] for i in indices]

    def get_init_args(self) -> str:
        """Generate Python code to reconstruct this PyCV's initialization arguments.

        This method serializes the atoms and prefix arguments so the PyCV can be
        reconstructed in the PLUMED adapter script.

        Returns
        -------
        str
            Python code string for initializing this PyCV instance.
        """
        # Serialize atoms argument
        if self.atoms is None:
            atoms_repr = "None"
        elif isinstance(self.atoms, list):
            atoms_repr = repr(self.atoms)
        else:
            # AtomSelector - use dataclass fields for serialization
            selector = self.atoms
            selector_class = type(selector).__name__

            if dataclasses.is_dataclass(selector):
                fields = dataclasses.fields(selector)
                args = ", ".join(
                    f"{f.name}={repr(getattr(selector, f.name))}" for f in fields
                )
                atoms_repr = f"{selector_class}({args})"
            else:
                raise ValueError(
                    f"Cannot serialize AtomSelector of type {selector_class}. "
                    "AtomSelector must be a dataclass."
                )

        return f"atoms={atoms_repr}, prefix={repr(self.prefix)}"

    def _generate_adapter_script(
        self,
        symbols: list[str],
        cv_class_module: str,
        cv_class_name: str,
        cv_init_args: str,
    ) -> str:
        """Generate the Python adapter script content for PLUMED.

        Parameters
        ----------
        symbols : list[str]
            Atomic symbols for reconstructing the Atoms object.
        cv_class_module : str
            Module path where the PyCV subclass is defined.
        cv_class_name : str
            Name of the PyCV subclass.
        cv_init_args : str
            Python code to reconstruct the CV initialization arguments.

        Returns
        -------
        str
            Python script content.
        """
        symbols_repr = repr(symbols)

        return f'''"""Auto-generated PYCV adapter script for {self.prefix}.

This script bridges PLUMED's PYCVINTERFACE to the user's PyCV.compute() method.
Generated by hillclimber.
"""
import numpy as np
import plumedCommunications as PLMD
from ase import Atoms

# Atomic symbols for reconstructing Atoms object
_SYMBOLS = {symbols_repr}

# CV class import and instantiation
from {cv_class_module} import {cv_class_name}
_CV_INSTANCE = {cv_class_name}({cv_init_args})

# PLUMED initialization - request derivatives
plumedInit = {{"Value": PLMD.defaults.COMPONENT}}


def plumedCalculate(action: PLMD.PythonCVInterface):
    """PLUMED callback for CV calculation."""
    # Get positions from PLUMED (shape: n_atoms x 3)
    positions = action.getPositions()

    # Reconstruct ASE Atoms object with species info
    atoms = Atoms(symbols=_SYMBOLS, positions=positions)

    # Call user's compute method
    result = _CV_INSTANCE.compute(atoms)

    # Handle return value
    if isinstance(result, tuple):
        value, grad = result
        # Box gradients (zeros - no contribution to virial)
        box_grad = np.zeros((3, 3))
        return float(value), np.asarray(grad, dtype=np.float64), box_grad
    else:
        # No gradients provided - return value only
        # Note: This means the CV cannot be biased
        return float(result)
'''

    def write_adapter_script(
        self,
        directory: Path,
        atoms: Atoms,
        cv_class_module: str,
        cv_class_name: str,
        cv_init_args: str,
    ) -> Path:
        """Write the Python adapter script for PLUMED.

        Parameters
        ----------
        directory : Path
            Directory to write the script to.
        atoms : ase.Atoms
            Reference structure for extracting symbols.
        cv_class_module : str
            Module path where the PyCV subclass is defined.
        cv_class_name : str
            Name of the PyCV subclass.
        cv_init_args : str
            Python code to reconstruct the CV initialization arguments.

        Returns
        -------
        Path
            Path to the written script file.
        """
        symbols = self._get_symbols(atoms)
        module_name = f"_pycv_{self.prefix}"
        script_content = self._generate_adapter_script(
            symbols=symbols,
            cv_class_module=cv_class_module,
            cv_class_name=cv_class_name,
            cv_init_args=cv_init_args,
        )
        script_path = directory / f"{module_name}.py"
        script_path.write_text(script_content)
        return script_path


__all__ = ["PyCV"]

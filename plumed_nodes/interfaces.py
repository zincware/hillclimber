import pathlib
from typing import Protocol

import ase
from ase.calculators.calculator import Calculator
from PIL import Image


class NodeWithCalculator(Protocol):
    """Any class with a `get_calculator` method returning an ASE Calculator."""

    def get_calculator(
        self, *, directory: str | pathlib.Path | None = None, **kwargs
    ) -> Calculator: ...


class AtomSelector(Protocol):
    """Protocol for selecting atoms within a single ASE Atoms object.

    This interface defines the contract for selecting atoms based on various
    criteria within an individual frame/structure.
    """

    def select(self, atoms: ase.Atoms) -> list[list[int]]:
        """Select atoms based on the implemented criteria.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to select from.

        Returns
        -------
        list[list[int]]
            Groups of indices of selected atoms. All indices in the inner lists
            are representative of the same group, e.g. one molecule.
        """
        ...


class PlumedGenerator(Protocol):
    """Protocol for generating PLUMED strings from collective variables."""

    def to_plumed(self, atoms: ase.Atoms) -> list[str]: ...


class CollectiveVariable(Protocol):
    """Protocol for collective variables (CVs) that can be used in PLUMED."""

    prefix: str

    def get_img(self, atoms: ase.Atoms) -> Image.Image: ...

    def to_plumed(self, atoms: ase.Atoms) -> tuple[list[str], list[str]]:
        """
        Convert the collective variable to a PLUMED string.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to use for generating the PLUMED string.

        Returns
        -------
        tuple[list[str], str]
            - List of distance labels.
            - list of PLUMED strings representing the CV.
        """
        ...


class MetadynamicsBiasCollectiveVariable(Protocol):
    """Protocol for metadata associated with a bias in PLUMED."""

    cv: CollectiveVariable
    sigma: float | None = None
    grid_min: float | None = None
    grid_max: float | None = None
    grid_bin: int | None = None


__all__ = [
    "NodeWithCalculator",
    "AtomSelector",
    "PlumedGenerator",
    "CollectiveVariable",
    "MetadynamicsBiasCollectiveVariable",
]

def interfaces() -> dict[str, list[str]]:
    """Return a dictionary of available interfaces."""
    return {"plumed-nodes": __all__}

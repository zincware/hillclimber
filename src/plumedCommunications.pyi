"""Type stubs for plumedCommunications module.

This module is a pybind11 C++ extension that provides Python bindings
for PLUMED's PythonCVInterface and PythonFunction classes.

See: https://joss.theoj.org/papers/10.21105/joss.01773
"""

from typing import Any

import numpy as np
import numpy.typing as npt

class defaults:
    """Default definitions for PLUMED components."""

    COMPONENT: dict[str, Any]
    """Default component settings with derivative=True and period=None."""

    COMPONENT_NODEV: dict[str, Any]
    """Component settings with derivative=False and period=None."""

class Pbc:
    """Interface to PLUMED periodic boundary conditions."""

    def apply(self, deltas: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply PBC to a set of positions or distance vectors.

        Parameters
        ----------
        deltas : np.ndarray
            Array of shape (n, 3) containing distance vectors.

        Returns
        -------
        np.ndarray
            The input array with PBC applied (modified in-place).
        """
        ...

    def getBox(self) -> npt.NDArray[np.float64]:
        """Get the simulation box vectors.

        Returns
        -------
        np.ndarray
            Array of shape (3, 3) with the box vectors.
        """
        ...

    def getInvBox(self) -> npt.NDArray[np.float64]:
        """Get the inverted box vectors.

        Returns
        -------
        np.ndarray
            Array of shape (3, 3) with the inverted box vectors.
        """
        ...

class NeighborList:
    """Interface to PLUMED neighbor list."""

    @property
    def size(self) -> int:
        """The number of atom pairs in the neighbor list."""
        ...

    def __len__(self) -> int:
        """Return the number of atom pairs."""
        ...

    def getClosePairs(self) -> npt.NDArray[np.uint64]:
        """Get the list of close atom pairs.

        Returns
        -------
        np.ndarray
            Array of shape (n_pairs, 2) with atom pair indices.
        """
        ...

class PythonCVInterface:
    """Interface for defining collective variables in Python.

    This class provides access to atomic data and simulation information
    when implementing custom collective variables via PYCVINTERFACE.
    """

    data: dict[str, Any]
    """Persistent dictionary that survives across simulation steps."""

    @property
    def label(self) -> str:
        """The label of this action."""
        ...

    @property
    def nat(self) -> int:
        """The number of atoms requested by this action."""
        ...

    def getStep(self) -> int:
        """Get the current simulation step.

        Returns
        -------
        int
            The current step number.
        """
        ...

    def getTime(self) -> float:
        """Get the current simulation time.

        Returns
        -------
        float
            The current time.
        """
        ...

    def getTimeStep(self) -> float:
        """Get the simulation timestep.

        Returns
        -------
        float
            The timestep.
        """
        ...

    def isRestart(self) -> bool:
        """Check if this is a restart simulation.

        Returns
        -------
        bool
            True if restarting from a previous simulation.
        """
        ...

    def isExchangeStep(self) -> bool:
        """Check if this is a replica exchange step.

        Returns
        -------
        bool
            True if on an exchange step.
        """
        ...

    def log(self, s: object) -> None:
        """Write a message to the PLUMED log.

        Parameters
        ----------
        s : object
            Message to write (will be converted to string).
        """
        ...

    def lognl(self, s: object) -> None:
        """Write a message to the PLUMED log with newline.

        Parameters
        ----------
        s : object
            Message to write (will be converted to string).
        """
        ...

    def getPosition(self, i: int) -> npt.NDArray[np.float64]:
        """Get the position of atom i.

        Parameters
        ----------
        i : int
            Atom index (0-based, relative to atoms requested by action).

        Returns
        -------
        np.ndarray
            Array of shape (3,) with x, y, z coordinates.
        """
        ...

    def getPositions(self) -> npt.NDArray[np.float64]:
        """Get positions of all atoms requested by this action.

        Returns
        -------
        np.ndarray
            Array of shape (n_atoms, 3) with atomic positions.
        """
        ...

    def getPbc(self) -> Pbc:
        """Get the periodic boundary conditions interface.

        Returns
        -------
        Pbc
            Interface to the current PBC.
        """
        ...

    def getNeighbourList(self) -> NeighborList:
        """Get the neighbor list interface.

        Returns
        -------
        NeighborList
            Interface to the current neighbor list.
        """
        ...

    def makeWhole(self) -> None:
        """Make atoms whole across periodic boundaries.

        Assumes atoms are in proper order.
        """
        ...

    def absoluteIndexes(self) -> npt.NDArray[np.uint32]:
        """Get the absolute atom indices.

        Returns
        -------
        np.ndarray
            Array of absolute atom indices in the full system.
        """
        ...

    def charge(self, i: int) -> float:
        """Get the charge of atom i.

        Parameters
        ----------
        i : int
            Atom index.

        Returns
        -------
        float
            The atomic charge.
        """
        ...

    def mass(self, i: int) -> float:
        """Get the mass of atom i.

        Parameters
        ----------
        i : int
            Atom index.

        Returns
        -------
        float
            The atomic mass.
        """
        ...

    def masses(self) -> npt.NDArray[np.float64]:
        """Get masses of all atoms.

        Returns
        -------
        np.ndarray
            Array of atomic masses.
        """
        ...

    def charges(self) -> npt.NDArray[np.float64]:
        """Get charges of all atoms.

        Returns
        -------
        np.ndarray
            Array of atomic charges.
        """
        ...

class PythonFunction:
    """Interface for defining functions of collective variables in Python.

    This class provides access to CV arguments when implementing
    custom functions via PYFUNCTION.
    """

    @property
    def label(self) -> str:
        """The label of this action."""
        ...

    @property
    def nargs(self) -> int:
        """The number of arguments passed to this function."""
        ...

    def getStep(self) -> int:
        """Get the current simulation step.

        Returns
        -------
        int
            The current step number.
        """
        ...

    def getTime(self) -> float:
        """Get the current simulation time.

        Returns
        -------
        float
            The current time.
        """
        ...

    def getTimeStep(self) -> float:
        """Get the simulation timestep.

        Returns
        -------
        float
            The timestep.
        """
        ...

    def isRestart(self) -> bool:
        """Check if this is a restart simulation.

        Returns
        -------
        bool
            True if restarting from a previous simulation.
        """
        ...

    def isExchangeStep(self) -> bool:
        """Check if this is a replica exchange step.

        Returns
        -------
        bool
            True if on an exchange step.
        """
        ...

    def log(self, s: object) -> None:
        """Write a message to the PLUMED log.

        Parameters
        ----------
        s : object
            Message to write (will be converted to string).
        """
        ...

    def lognl(self, s: object) -> None:
        """Write a message to the PLUMED log with newline.

        Parameters
        ----------
        s : object
            Message to write (will be converted to string).
        """
        ...

    def argument(self, i: int) -> float:
        """Get the value of argument i.

        Parameters
        ----------
        i : int
            Argument index.

        Returns
        -------
        float
            The argument value.
        """
        ...

    def arguments(self) -> npt.NDArray[np.float64]:
        """Get all argument values.

        Returns
        -------
        np.ndarray
            Array of argument values.
        """
        ...

    def difference(self, i: int, x: float, y: float) -> float:
        """Compute difference accounting for periodicity.

        Parameters
        ----------
        i : int
            Argument index (for periodicity info).
        x : float
            First value.
        y : float
            Second value.

        Returns
        -------
        float
            The difference y - x accounting for periodicity.
        """
        ...

    def bringBackInPbc(self, i: int, x: float) -> float:
        """Bring a value back into the periodic domain.

        Parameters
        ----------
        i : int
            Argument index (for periodicity info).
        x : float
            Value to wrap.

        Returns
        -------
        float
            The value wrapped into the periodic domain.
        """
        ...

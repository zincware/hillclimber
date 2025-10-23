"""Bias potentials and restraints for collective variables.

This module provides classes for applying bias potentials to collective variables,
including harmonic restraints and wall potentials.
"""

# --- IMPORTS ---
import dataclasses

import ase

from hillclimber.interfaces import BiasProtocol, CollectiveVariable


# --- RESTRAINT ---
@dataclasses.dataclass
class RestraintBias(BiasProtocol):
    """Apply a harmonic restraint to a collective variable.

    This class implements the BiasProtocol interface and can be used in the
    `actions` parameter of MetaDynamicsModel.

    The restraint creates a harmonic potential that restrains a collective variable
    around a specified value. The restraint potential has the form:
    V(s) = (1/2) * kappa * (s - at)^2

    Parameters
    ----------
    cv : CollectiveVariable
        The collective variable to restrain.
    kappa : float
        The force constant of the restraint in eV/unit^2, where unit
        depends on the CV (e.g., eV/Å² for distances, eV/rad² for angles).
    at : float
        The center/target value of the restraint in CV units (e.g., Å for distances).
    label : str, optional
        A custom label for this restraint. If not provided, uses cv.prefix + "_restraint".

    Examples
    --------
    >>> import hillclimber as hc
    >>> # Restrain a distance around 2.5 Å with force constant 200 eV/Å²
    >>> distance_cv = hc.DistanceCV(...)
    >>> restraint = hc.RestraintBias(cv=distance_cv, kappa=200.0, at=2.5)

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/RESTRAINT/

    Notes
    -----
    Due to the UNITS line, kappa is in eV/unit^2 (ASE energy units). For typical CVs:
    - Distances (Å): kappa in eV/Å²
    - Angles (radians): kappa in eV/rad²
    - Dimensionless CVs: kappa in eV

    The restraint creates a bias force: F = -kappa * (s - at)
    """

    cv: CollectiveVariable
    kappa: float
    at: float
    label: str | None = None

    def to_plumed(self, atoms: ase.Atoms) -> list[str]:
        """Generate PLUMED input strings for the harmonic restraint.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to use for generating the PLUMED string.

        Returns
        -------
        list[str]
            List of PLUMED command strings. First defines the CV (if needed),
            then applies the restraint.
        """
        # Get CV definition
        cv_labels, cv_commands = self.cv.to_plumed(atoms)

        # Determine restraint label
        restraint_label = self.label if self.label else f"{self.cv.prefix}_restraint"

        # Build restraint command
        # PLUMED expects ARG to be comma-separated CV labels
        arg_str = ",".join(cv_labels)
        restraint_cmd = f"{restraint_label}: RESTRAINT ARG={arg_str} KAPPA={self.kappa} AT={self.at}"

        # Combine CV commands and restraint
        commands = cv_commands + [restraint_cmd]
        return commands


# --- WALLS ---
@dataclasses.dataclass
class UpperWallBias(BiasProtocol):
    """Apply an upper wall restraint to a collective variable.

    This class implements the BiasProtocol interface and can be used in the
    `actions` parameter of MetaDynamicsModel.

    The wall creates a repulsive bias potential that acts when a collective
    variable exceeds a specified threshold. The wall potential has the form:
    V(s) = kappa * ((s - at) / offset)^exp  if s > at, else 0

    Parameters
    ----------
    cv : CollectiveVariable
        The collective variable to apply the wall to.
    at : float
        The position of the wall (threshold value) in CV units.
    kappa : float
        The force constant of the wall in eV.
    exp : int, optional
        The exponent of the wall potential. Default is 2 (harmonic).
        Higher values create steeper walls.
    eps : float, optional
        A small offset to avoid numerical issues. Default is 0.0.
    offset : float, optional
        The offset parameter for the wall. Default is 0.0.
    label : str, optional
        A custom label for this wall. If not provided, uses cv.prefix + "_uwall".

    Examples
    --------
    >>> import hillclimber as hc
    >>> # Prevent distance from exceeding 3.0 Å with force constant 100 eV
    >>> distance_cv = hc.DistanceCV(...)
    >>> upper_wall = hc.UpperWallBias(cv=distance_cv, at=3.0, kappa=100.0, exp=2)

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/UPPER_WALLS/

    Notes
    -----
    The wall only acts when the CV value exceeds 'at'. The steepness of the
    wall is controlled by both 'kappa' and 'exp':
    - exp=2: harmonic wall (smooth)
    - exp=4,6,...: steeper walls
    - Higher exp values create harder walls but may cause numerical issues
    """

    cv: CollectiveVariable
    at: float
    kappa: float
    exp: int = 2
    eps: float = 0.0
    offset: float = 0.0
    label: str | None = None

    def to_plumed(self, atoms: ase.Atoms) -> list[str]:
        """Generate PLUMED input strings for the upper wall.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to use for generating the PLUMED string.

        Returns
        -------
        list[str]
            List of PLUMED command strings. First defines the CV (if needed),
            then applies the upper wall.
        """
        # Get CV definition
        cv_labels, cv_commands = self.cv.to_plumed(atoms)

        # Determine wall label
        wall_label = self.label if self.label else f"{self.cv.prefix}_uwall"

        # Build wall command
        arg_str = ",".join(cv_labels)
        wall_parts = [
            f"{wall_label}: UPPER_WALLS",
            f"ARG={arg_str}",
            f"AT={self.at}",
            f"KAPPA={self.kappa}",
            f"EXP={self.exp}",
        ]

        # Add optional parameters only if non-zero
        if self.eps != 0.0:
            wall_parts.append(f"EPS={self.eps}")
        if self.offset != 0.0:
            wall_parts.append(f"OFFSET={self.offset}")

        wall_cmd = " ".join(wall_parts)

        # Combine CV commands and wall
        commands = cv_commands + [wall_cmd]
        return commands


@dataclasses.dataclass
class LowerWallBias(BiasProtocol):
    """Apply a lower wall restraint to a collective variable.

    This class implements the BiasProtocol interface and can be used in the
    `actions` parameter of MetaDynamicsModel.

    The wall creates a repulsive bias potential that acts when a collective
    variable falls below a specified threshold. The wall potential has the form:
    V(s) = kappa * ((at - s) / offset)^exp  if s < at, else 0

    Parameters
    ----------
    cv : CollectiveVariable
        The collective variable to apply the wall to.
    at : float
        The position of the wall (threshold value) in CV units.
    kappa : float
        The force constant of the wall in eV.
    exp : int, optional
        The exponent of the wall potential. Default is 2 (harmonic).
        Higher values create steeper walls.
    eps : float, optional
        A small offset to avoid numerical issues. Default is 0.0.
    offset : float, optional
        The offset parameter for the wall. Default is 0.0.
    label : str, optional
        A custom label for this wall. If not provided, uses cv.prefix + "_lwall".

    Examples
    --------
    >>> import hillclimber as hc
    >>> # Prevent distance from going below 1.0 Å with force constant 100 eV
    >>> distance_cv = hc.DistanceCV(...)
    >>> lower_wall = hc.LowerWallBias(cv=distance_cv, at=1.0, kappa=100.0, exp=2)

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/LOWER_WALLS

    Notes
    -----
    The wall only acts when the CV value falls below 'at'. The steepness of the
    wall is controlled by both 'kappa' and 'exp':
    - exp=2: harmonic wall (smooth)
    - exp=4,6,...: steeper walls
    - Higher exp values create harder walls but may cause numerical issues
    """

    cv: CollectiveVariable
    at: float
    kappa: float
    exp: int = 2
    eps: float = 0.0
    offset: float = 0.0
    label: str | None = None

    def to_plumed(self, atoms: ase.Atoms) -> list[str]:
        """Generate PLUMED input strings for the lower wall.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to use for generating the PLUMED string.

        Returns
        -------
        list[str]
            List of PLUMED command strings. First defines the CV (if needed),
            then applies the lower wall.
        """
        # Get CV definition
        cv_labels, cv_commands = self.cv.to_plumed(atoms)

        # Determine wall label
        wall_label = self.label if self.label else f"{self.cv.prefix}_lwall"

        # Build wall command
        arg_str = ",".join(cv_labels)
        wall_parts = [
            f"{wall_label}: LOWER_WALLS",
            f"ARG={arg_str}",
            f"AT={self.at}",
            f"KAPPA={self.kappa}",
            f"EXP={self.exp}",
        ]

        # Add optional parameters only if non-zero
        if self.eps != 0.0:
            wall_parts.append(f"EPS={self.eps}")
        if self.offset != 0.0:
            wall_parts.append(f"OFFSET={self.offset}")

        wall_cmd = " ".join(wall_parts)

        # Combine CV commands and wall
        commands = cv_commands + [wall_cmd]
        return commands

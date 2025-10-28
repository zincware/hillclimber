import dataclasses
import typing as t
from pathlib import Path

import ase.units
import zntrack

from hillclimber.calc import NonOverwritingPlumed
from hillclimber.interfaces import (
    CollectiveVariable,
    NodeWithCalculator,
    PlumedGenerator,
)


@dataclasses.dataclass
class MetadBias:
    """Metadynamics bias configuration for a collective variable.

    Parameters
    ----------
    cv : CollectiveVariable
        The collective variable to bias.
    sigma : float, optional
        The width of the Gaussian potential in the same units as the CV
        (e.g., Å for distances, radians for angles), by default None.
    grid_min : float | str, optional
        The minimum value of the grid in CV units (or PLUMED expression like "-pi"),
        by default None.
    grid_max : float | str, optional
        The maximum value of the grid in CV units (or PLUMED expression like "pi"),
        by default None.
    grid_bin : int, optional
        The number of bins in the grid, by default None.

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/METAD/
    """

    cv: CollectiveVariable
    sigma: float | None = None
    grid_min: float | str | None = None
    grid_max: float | str | None = None
    grid_bin: int | None = None


@dataclasses.dataclass
class MetaDynamicsConfig:
    """Base configuration for metadynamics.

    This contains only the global parameters that apply to all CVs.

    Units
    -----
    hillclimber uses ASE units throughout. The UNITS line in the PLUMED input tells
    PLUMED to interpret all values in ASE units:
    - Distances: Ångström (Å)
    - Energies: electronvolt (eV) - including HEIGHT, SIGMA for energy-based CVs, etc.
    - Time: femtoseconds (fs)
    - Temperature: Kelvin (K)

    Parameters
    ----------
    height : float, optional
        The height of the Gaussian potential in eV, by default 1.0.
    pace : int, optional
        The frequency of Gaussian deposition in MD steps, by default 500.
    biasfactor : float, optional
        The bias factor for well-tempered metadynamics, by default None.
    temp : float, optional
        The temperature of the system in Kelvin, by default 300.0.
    file : str, optional
        The name of the hills file, by default "HILLS".
    adaptive : t.Literal["GEOM", "DIFF"] | None, optional
        The adaptive scheme to use, by default None.
        If None, no ADAPTIVE parameter is written to PLUMED.
    flush : int | None
        The frequency of flushing the output files in MD steps.
        If None, uses the plumed default.

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/METAD/
    - https://www.plumed.org/doc-master/user-doc/html/FLUSH/
    """

    height: float = 1.0  # kJ/mol
    pace: int = 500
    biasfactor: float | None = None
    temp: float = 300.0
    file: str = "HILLS"
    adaptive: t.Literal["GEOM", "DIFF"] | None = None
    flush: int | None = None


# THERE SEEMS to be an issue with MetaDynamicsModel changing but ASEMD not updating?!?!


class MetaDynamicsModel(zntrack.Node, NodeWithCalculator):
    """A metadynamics model.

    Parameters
    ----------
    config : MetaDynamicsConfig
        The configuration for the metadynamics simulation.
    data : list[ase.Atoms]
        The input data for the simulation.
    data_idx : int, optional
        The index of the data to use, by default -1.
    bias_cvs : list[MetaDBiasCV], optional
        The collective variables to bias, by default [].
    actions : list[PlumedGenerator], optional
        A list of actions to perform during the simulation, by default [].
    timestep : float, optional
        The timestep of the simulation in fs, by default 1.0.
    model : NodeWithCalculator
        The model to use for the simulation.

    Example
    -------
    >>> import hillclimber as pn
    >>> import ipsuite as ips
    >>>
    >>> data = ips.AddData("seed.xyz")
    >>> cv1 = pn.DistanceCV(
    ...     x1=pn.SMARTSSelector(pattern="[H]O[H]"),
    ...     x2=pn.SMARTSSelector(pattern="CO[C:1]"),
    ...     prefix="d",
    ... )
    >>> metad_cv1 = pn.MetadBias(
    ...     cv=cv1, sigma=0.1, grid_min=0.0, grid_max=2.0, grid_bin=200
    ... )
    >>> model = pn.MetaDynamicsModel(
    ...     config=pn.MetaDynamicsConfig(height=0.25, temp=300, pace=2000, biasfactor=10),
    ...     bias_cvs=[metad_cv1],
    ...     data=data.frames,
    ...     model=ips.MACEMP(),
    ...     timestep=0.5
    ... )
    >>> md = ips.ASEMD(
    ...     model=model, data=data.frames, ...
    ... )
    """

    config: MetaDynamicsConfig = zntrack.deps()
    data: list[ase.Atoms] = zntrack.deps()
    data_idx: int = zntrack.params(-1)
    bias_cvs: list[MetadBias] = zntrack.deps(default_factory=list)
    actions: list[PlumedGenerator] = zntrack.deps(default_factory=list)
    timestep: float = zntrack.params(1.0)  # in fs, default is 1 fs
    model: NodeWithCalculator = zntrack.deps()

    figures: Path = zntrack.outs_path(zntrack.nwd / "figures", independent=True)

    def run(self):
        self.figures.mkdir(parents=True, exist_ok=True)
        for cv in self.bias_cvs:
            img = cv.cv.get_img(self.data[self.data_idx])
            img.save(self.figures / f"{cv.cv.prefix}.png")

    def get_calculator(
        self, *, directory: str | Path | None = None, **kwargs
    ) -> NonOverwritingPlumed:
        if directory is None:
            raise ValueError("Directory must be specified for PLUMED input files.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        lines = self.to_plumed(self.data[self.data_idx])
        # replace FILE= with f"FILE={directory}/" inside config
        lines = [line.replace("FILE=", f"FILE={directory}/") for line in lines]

        # Write plumed input file
        with (directory / "plumed.dat").open("w") as file:
            for line in lines:
                file.write(line + "\n")

        kT = ase.units.kB * self.config.temp

        return NonOverwritingPlumed(
            calc=self.model.get_calculator(directory=directory),
            atoms=self.data[self.data_idx],
            input=lines,
            timestep=float(self.timestep * ase.units.fs),
            kT=float(kT),
            log=(directory / "plumed.log").as_posix(),
        )

    def to_plumed(self, atoms: ase.Atoms) -> list[str]:
        """Generate PLUMED input string for the metadynamics model."""
        # check for duplicate CV prefixes
        cv_labels = set()
        for bias_cv in self.bias_cvs:
            if bias_cv.cv.prefix in cv_labels:
                raise ValueError(f"Duplicate CV prefix found: {bias_cv.cv.prefix}")
            cv_labels.add(bias_cv.cv.prefix)

        plumed_lines = []
        all_labels = []

        sigmas, grid_mins, grid_maxs, grid_bins = [], [], [], []

        # PLUMED UNITS line specifies conversion factors from ASE units to PLUMED's native units:
        # - LENGTH=A: ASE uses Ångström (A), PLUMED native is nm → A is a valid PLUMED unit
        # - TIME: ASE uses fs, PLUMED native is ps → 1 fs = 0.001 ps
        # - ENERGY: ASE uses eV, PLUMED native is kJ/mol → 1 eV = 96.485 kJ/mol
        # See: https://www.plumed.org/doc-master/user-doc/html/ (MD engine integration docs)
        plumed_lines.append(
            f"UNITS LENGTH=A TIME={1 / 1000} ENERGY={ase.units.mol / ase.units.kJ}"
        )

        for bias_cv in self.bias_cvs:
            labels, cv_str = bias_cv.cv.to_plumed(atoms)
            plumed_lines.extend(cv_str)
            all_labels.extend(labels)

            # Collect per-CV parameters for later - repeat for each label
            # PLUMED requires one parameter value per ARG, so if a CV generates
            # multiple labels, we need to repeat the parameter values
            for _ in labels:
                sigmas.append(str(bias_cv.sigma) if bias_cv.sigma is not None else None)
                grid_mins.append(
                    str(bias_cv.grid_min) if bias_cv.grid_min is not None else None
                )
                grid_maxs.append(
                    str(bias_cv.grid_max) if bias_cv.grid_max is not None else None
                )
                grid_bins.append(
                    str(bias_cv.grid_bin) if bias_cv.grid_bin is not None else None
                )

        metad_parts = [
            "METAD",
            f"ARG={','.join(all_labels)}",
            f"HEIGHT={self.config.height}",
            f"PACE={self.config.pace}",
            f"TEMP={self.config.temp}",
            f"FILE={self.config.file}",
        ]
        if self.config.adaptive is not None:
            metad_parts.append(f"ADAPTIVE={self.config.adaptive}")
        if self.config.biasfactor is not None:
            metad_parts.append(f"BIASFACTOR={self.config.biasfactor}")

        # Add SIGMA, GRID_MIN, GRID_MAX, GRID_BIN only if any value is set
        if any(v is not None for v in sigmas):
            # When using ADAPTIVE, PLUMED requires only one sigma value
            if self.config.adaptive is not None:
                # Validate that all sigma values are the same when adaptive is set
                unique_sigmas = set(v for v in sigmas if v is not None)
                if len(unique_sigmas) > 1:
                    raise ValueError(
                        f"When using ADAPTIVE={self.config.adaptive}, all CVs must have the same sigma value. "
                        f"Found different sigma values: {unique_sigmas}"
                    )
                # Use the first non-None sigma value
                sigma_value = next(v for v in sigmas if v is not None)
                metad_parts.append(f"SIGMA={sigma_value}")
            else:
                # Standard mode: one sigma per CV
                metad_parts.append(
                    f"SIGMA={','.join(v if v is not None else '0.0' for v in sigmas)}"
                )
        if any(v is not None for v in grid_mins):
            metad_parts.append(
                f"GRID_MIN={','.join(v if v is not None else '0.0' for v in grid_mins)}"
            )
        if any(v is not None for v in grid_maxs):
            metad_parts.append(
                f"GRID_MAX={','.join(v if v is not None else '0.0' for v in grid_maxs)}"
            )
        if any(v is not None for v in grid_bins):
            metad_parts.append(
                f"GRID_BIN={','.join(v if v is not None else '0' for v in grid_bins)}"
            )

        plumed_lines.append(f"metad: {' '.join(metad_parts)}")

        # Track defined commands to detect duplicates and conflicts
        # Map label -> full command for labeled commands (e.g., "d: DISTANCE ...")
        defined_commands = {}
        for line in plumed_lines:
            # Check if this is a labeled command (format: "label: ACTION ...")
            if ": " in line:
                label = line.split(": ", 1)[0]
                defined_commands[label] = line

        # Add any additional actions (restraints, walls, print actions, etc.)
        for action in self.actions:
            action_lines = action.to_plumed(atoms)

            # Filter out duplicate CV definitions, but detect conflicts
            filtered_lines = []
            for line in action_lines:
                # Check if this is a labeled command
                if ": " in line:
                    label = line.split(": ", 1)[0]

                    # Check if this label was already defined
                    if label in defined_commands:
                        # If the command is identical, skip (deduplication)
                        if defined_commands[label] == line:
                            continue
                        # If the command is different, raise error (conflict)
                        else:
                            raise ValueError(
                                f"Conflicting definitions for label '{label}':\n"
                                f"  Already defined: {defined_commands[label]}\n"
                                f"  New definition:  {line}"
                            )
                    else:
                        # New labeled command, track it
                        defined_commands[label] = line
                        filtered_lines.append(line)
                else:
                    # Unlabeled command, always add
                    filtered_lines.append(line)

            plumed_lines.extend(filtered_lines)

        # Add FLUSH if configured
        if self.config.flush is not None:
            plumed_lines.append(f"FLUSH STRIDE={self.config.flush}")
        return plumed_lines

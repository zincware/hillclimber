import dataclasses
from pathlib import Path

import ase.units
import zntrack

from hillclimber.calc import NonOverwritingPlumed
from hillclimber.interfaces import (
    CollectiveVariable,
    MetadynamicsBiasCollectiveVariable,
    NodeWithCalculator,
    PlumedGenerator,
)


@dataclasses.dataclass
class MetaDBiasCV(MetadynamicsBiasCollectiveVariable):
    """Metadynamics bias on a collective variable.

    Parameters
    ----------
    cv : CollectiveVariable
        The collective variable to bias.
    sigma : float, optional
        The width of the Gaussian potential, by default None.
    grid_min : float | str, optional
        The minimum value of the grid, by default None.
    grid_max : float | str, optional
        The maximum value of the grid, by default None.
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

    Parameters
    ----------
    height : float, optional
        The height of the Gaussian potential in kJ/mol, by default 1.0.
    pace : int, optional
        The frequency of Gaussian deposition, by default 500.
    biasfactor : float, optional
        The bias factor for well-tempered metadynamics, by default None.
    temp : float, optional
        The temperature of the system in Kelvin, by default 300.0.
    file : str, optional
        The name of the hills file, by default "HILLS".
    adaptive : str, optional
        The adaptive scheme to use, by default "NONE".
    flush : int | None
        The frequency of flushing the output files. 
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
    adaptive: str = "NONE"  # NONE, DIFF, GEOM
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
    >>> metad_cv1 = pn.MetaDBiasCV(
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
    bias_cvs: list[MetaDBiasCV] = zntrack.deps(default_factory=list)
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

        plumed_lines.append(
            f"UNITS LENGTH=A TIME={1 / (1000 * ase.units.fs)} ENERGY={ase.units.mol / ase.units.kJ}"
        )

        for bias_cv in self.bias_cvs:
            labels, cv_str = bias_cv.cv.to_plumed(atoms)
            plumed_lines.extend(cv_str)
            all_labels.extend(labels)

            # Collect per-CV parameters for later
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
            f"ADAPTIVE={self.config.adaptive}",
        ]
        if self.config.biasfactor is not None:
            metad_parts.append(f"BIASFACTOR={self.config.biasfactor}")

        # Add SIGMA, GRID_MIN, GRID_MAX, GRID_BIN only if any value is set
        if any(v is not None for v in sigmas):
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
        # Temporary until https://github.com/zincware/ZnTrack/issues/936
        from hillclimber.actions import PrintCVAction
        lines = PrintCVAction(cvs=[x.cv for x in self.bias_cvs], stride=100).to_plumed(atoms)
        plumed_lines.extend(lines)
        if self.config.flush is not None:
            plumed_lines.append(f"FLUSH STRIDE={self.config.flush}")
        return plumed_lines

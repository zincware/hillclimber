import dataclasses
from pathlib import Path

import ase.units
import zntrack

from plumed_nodes.calc import NonOverwritingPlumed
from plumed_nodes.interfaces import (
    CollectiveVariable,
    MetadynamicsBiasCollectiveVariable,
    NodeWithCalculator,
    PlumedGenerator,
)


@dataclasses.dataclass
class MetaDBiasCV(MetadynamicsBiasCollectiveVariable):
    """
    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/METAD/
    """

    cv: CollectiveVariable
    sigma: float | None = None
    grid_min: float | None = None
    grid_max: float | None = None
    grid_bin: int | None = None


@dataclasses.dataclass
class MetaDynamicsConfig:
    """
    Base configuration for metadynamics.
    This contains only the global parameters that apply to all CVs.
    """

    height: float = 1.0  # kJ/mol
    pace: int = 500
    biasfactor: float | None = None
    temp: float = 300.0
    file: str = "HILLS"
    adaptive: str = "NONE"  # NONE, DIFF, GEOM


class MetaDynamicsModel(zntrack.Node, NodeWithCalculator):
    config: MetaDynamicsConfig = zntrack.deps()
    data: list[ase.Atoms] = zntrack.deps()
    data_idx: int = zntrack.params(-1)
    bias_cvs: list[MetaDBiasCV] = zntrack.deps(default_factory=list)
    actions: list[PlumedGenerator] = zntrack.deps(default_factory=list)
    timestep: float = zntrack.params(1.0)  # in fs, default is 1 fs
    model: NodeWithCalculator = zntrack.deps()

    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

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
        # Add actions
        for action in self.actions:
            action_lines = action.to_plumed(atoms)
            if not action_lines:
                raise ValueError(f"Empty PLUMED commands for action {action}")
            plumed_lines.extend(action_lines)
        return plumed_lines

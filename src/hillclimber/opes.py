"""OPES (On-the-fly Probability Enhanced Sampling) methods.

This module provides classes for OPES enhanced sampling, a modern alternative
to traditional metadynamics with improved convergence properties.
"""

import dataclasses
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
class OPESBias:
    """OPES bias configuration for a collective variable.

    Parameters
    ----------
    cv : CollectiveVariable
        The collective variable to bias.
    sigma : float | str, optional
        Initial kernel width in CV units (e.g., Å for distances, radians for angles).
        Use "ADAPTIVE" for automatic adaptation (recommended).
        If numeric, specifies the initial width.
        Default: "ADAPTIVE".

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/OPES_METAD/

    Notes
    -----
    The ADAPTIVE sigma option automatically adjusts kernel widths based on
    CV fluctuations, which is usually the best choice. The automatic width
    is measured every ADAPTIVE_SIGMA_STRIDE steps (default: 10×PACE).
    """

    cv: CollectiveVariable
    sigma: float | str = "ADAPTIVE"


@dataclasses.dataclass
class OPESConfig:
    """Configuration for OPES_METAD and OPES_METAD_EXPLORE.

    OPES (On-the-fly Probability Enhanced Sampling) is a modern enhanced
    sampling method that samples well-tempered target distributions.

    Units
    -----
    hillclimber uses ASE units throughout. The UNITS line in the PLUMED input tells
    PLUMED to interpret all values in ASE units:
    - Distances: Ångström (Å)
    - Energies: electronvolt (eV) - including BARRIER, SIGMA_MIN, etc.
    - Time: femtoseconds (fs)
    - Temperature: Kelvin (K)

    Parameters
    ----------
    barrier : float
        Highest free energy barrier to overcome (eV). This is the key
        parameter that determines sampling efficiency.
    pace : int, optional
        Frequency of kernel deposition in MD steps (default: 500).
    temp : float, optional
        Temperature in Kelvin (default: 300.0). If -1, retrieved from MD engine.
    explore_mode : bool, optional
        If True, uses OPES_METAD_EXPLORE which estimates target distribution
        directly (better exploration, slower reweighting convergence).
        If False, uses OPES_METAD which estimates unbiased distribution
        (faster convergence, less exploration). Default: False.
    biasfactor : float, optional
        Well-tempered gamma factor. If not specified, uses default behavior.
        Set to inf for custom target distributions.
    compression_threshold : float, optional
        Merge kernels if closer than this threshold in sigma units (default: 1.0).
    file : str, optional
        File to store deposited kernels (default: "KERNELS").
    adaptive_sigma_stride : int, optional
        MD steps between adaptive sigma measurements. If not set, uses 10×PACE.
    sigma_min : float, optional
        Minimum allowable sigma value for adaptive sigma in CV units.
    state_wfile : str, optional
        State file for writing exact restart information.
    state_rfile : str, optional
        State file for reading restart information.
    state_wstride : int, optional
        Frequency of STATE file writing in number of kernel depositions.
    walkers_mpi : bool, optional
        Enable multiple walker mode with MPI communication (default: False).
    calc_work : bool, optional
        Calculate and output accumulated work (default: False).
    flush : int, optional
        Frequency of flushing output files in MD steps.

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/OPES_METAD/
    - https://www.plumed.org/doc-master/user-doc/html/OPES_METAD_EXPLORE/
    - Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020

    Notes
    -----
    **When to use OPES_METAD vs OPES_METAD_EXPLORE:**

    - OPES_METAD: Use when you want quick convergence of reweighted free energy.
      Estimates unbiased distribution P(s).

    - OPES_METAD_EXPLORE: Use for systems with unknown barriers or when testing
      new CVs. Allows more exploration but slower reweighting convergence.
      Estimates target distribution p^WT(s) directly.

    Both methods converge to the same bias given enough time, but approach it
    differently. OPES is more sensitive to degenerate CVs than standard METAD.
    """

    barrier: float  # kJ/mol
    pace: int = 500
    temp: float = 300.0
    explore_mode: bool = False  # False=OPES_METAD, True=OPES_METAD_EXPLORE
    biasfactor: float | None = None
    compression_threshold: float = 1.0
    file: str = "KERNELS"
    adaptive_sigma_stride: int | None = None
    sigma_min: float | None = None
    state_wfile: str | None = None
    state_rfile: str | None = None
    state_wstride: int | None = None
    walkers_mpi: bool = False
    calc_work: bool = False
    flush: int | None = None


class OPESModel(zntrack.Node, NodeWithCalculator):
    """OPES (On-the-fly Probability Enhanced Sampling) model.

    Implements OPES_METAD and OPES_METAD_EXPLORE enhanced sampling methods.
    OPES samples well-tempered target distributions and provides better
    convergence properties than traditional metadynamics.

    Parameters
    ----------
    config : OPESConfig
        Configuration for the OPES simulation.
    data : list[ase.Atoms]
        Input data for simulation.
    data_idx : int, optional
        Index of data to use (default: -1).
    bias_cvs : list[OPESBias], optional
        Collective variables to bias (default: []).
    actions : list[PlumedGenerator], optional
        Additional actions like restraints, walls, print (default: []).
    timestep : float, optional
        Timestep in fs (default: 1.0).
    model : NodeWithCalculator
        Underlying force field model.

    Examples
    --------
    >>> import hillclimber as hc
    >>>
    >>> # Define collective variables
    >>> phi = hc.TorsionCV(...)
    >>> psi = hc.TorsionCV(...)
    >>>
    >>> # OPES configuration (standard mode)
    >>> config = hc.OPESConfig(
    ...     barrier=40.0,  # kJ/mol
    ...     pace=500,
    ...     temp=300.0,
    ...     explore_mode=False  # Use OPES_METAD
    ... )
    >>>
    >>> # Bias configuration with adaptive sigma
    >>> bias1 = hc.OPESBias(cv=phi, sigma="ADAPTIVE")
    >>> bias2 = hc.OPESBias(cv=psi, sigma="ADAPTIVE")
    >>>
    >>> # Create OPES model
    >>> opes = hc.OPESModel(
    ...     config=config,
    ...     bias_cvs=[bias1, bias2],
    ...     data=data.frames,
    ...     model=force_field,
    ...     timestep=0.5
    ... )
    >>>
    >>> # For exploration mode, set explore_mode=True
    >>> explore_config = hc.OPESConfig(
    ...     barrier=40.0,
    ...     pace=500,
    ...     explore_mode=True  # Use OPES_METAD_EXPLORE
    ... )

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/OPES_METAD/
    - https://www.plumed.org/doc-master/user-doc/html/OPES_METAD_EXPLORE/
    - https://www.plumed.org/doc-master/user-doc/html/masterclass-22-03.html
    - Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020

    Notes
    -----
    **Output Components:**
    OPES provides several diagnostic outputs (accessible via PrintAction):
    - opes.bias: Instantaneous bias potential value
    - opes.rct: Convergence indicator (should flatten at convergence)
    - opes.zed: Normalization estimate (should stabilize)
    - opes.neff: Effective sample size
    - opes.nker: Number of compressed kernels
    - opes.work: Accumulated work (if calc_work=True)

    **Advantages over Metadynamics:**
    - Better convergence properties
    - Automatic variance adaptation
    - Lower systematic error
    - More sensitive to degenerate CVs (helps identify CV problems)
    """

    config: OPESConfig = zntrack.deps()
    data: list[ase.Atoms] = zntrack.deps()
    data_idx: int = zntrack.params(-1)
    bias_cvs: list[OPESBias] = zntrack.deps(default_factory=list)
    actions: list[PlumedGenerator] = zntrack.deps(default_factory=list)
    timestep: float = zntrack.params(1.0)
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
        """Generate PLUMED input string for the OPES model."""
        # check for duplicate CV prefixes
        cv_labels = set()
        for bias_cv in self.bias_cvs:
            if bias_cv.cv.prefix in cv_labels:
                raise ValueError(f"Duplicate CV prefix found: {bias_cv.cv.prefix}")
            cv_labels.add(bias_cv.cv.prefix)

        plumed_lines = []
        all_labels = []

        sigmas = []

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

            # Collect sigma values
            if isinstance(bias_cv.sigma, str):
                sigmas.append(bias_cv.sigma)
            else:
                sigmas.append(str(bias_cv.sigma))

        # Determine which OPES method to use
        method_name = "OPES_METAD_EXPLORE" if self.config.explore_mode else "OPES_METAD"

        # Build OPES command
        opes_parts = [
            f"opes: {method_name}",
            f"ARG={','.join(all_labels)}",
            f"PACE={self.config.pace}",
            f"BARRIER={self.config.barrier}",
            f"TEMP={self.config.temp}",
        ]

        # Add SIGMA (required parameter)
        # If all sigmas are the same, use single value; otherwise comma-separated
        if len(set(sigmas)) == 1:
            opes_parts.append(f"SIGMA={sigmas[0]}")
        else:
            opes_parts.append(f"SIGMA={','.join(sigmas)}")

        # Add FILE and COMPRESSION_THRESHOLD
        opes_parts.append(f"FILE={self.config.file}")
        opes_parts.append(f"COMPRESSION_THRESHOLD={self.config.compression_threshold}")

        # Optional parameters
        if self.config.biasfactor is not None:
            opes_parts.append(f"BIASFACTOR={self.config.biasfactor}")
        if self.config.adaptive_sigma_stride is not None:
            opes_parts.append(
                f"ADAPTIVE_SIGMA_STRIDE={self.config.adaptive_sigma_stride}"
            )
        if self.config.sigma_min is not None:
            opes_parts.append(f"SIGMA_MIN={self.config.sigma_min}")
        if self.config.state_wfile is not None:
            opes_parts.append(f"STATE_WFILE={self.config.state_wfile}")
        if self.config.state_rfile is not None:
            opes_parts.append(f"STATE_RFILE={self.config.state_rfile}")
        if self.config.state_wstride is not None:
            opes_parts.append(f"STATE_WSTRIDE={self.config.state_wstride}")
        if self.config.walkers_mpi:
            opes_parts.append("WALKERS_MPI")
        if self.config.calc_work:
            opes_parts.append("CALC_WORK")

        plumed_lines.append(" ".join(opes_parts))

        # Add any additional actions (restraints, walls, print actions, etc.)
        for action in self.actions:
            action_lines = action.to_plumed(atoms)
            plumed_lines.extend(action_lines)

        # Add FLUSH if configured
        if self.config.flush is not None:
            plumed_lines.append(f"FLUSH STRIDE={self.config.flush}")

        return plumed_lines

"""Analysis utilities for metadynamics simulations.

This module provides tools for analyzing metadynamics simulations,
including free energy surface reconstruction and other post-processing tasks.
"""

import os
import shutil
import subprocess
import typing as t
from pathlib import Path


def _validate_multi_cv_params(
    min_bounds: float | list[float] | None = None,
    max_bounds: float | list[float] | None = None,
    bin: int | list[int] | None = None,
    spacing: float | list[float] | None = None,
    sigma: float | list[float] | None = None,
    idw: str | list[str] | None = None,
) -> None:
    """Validate that multi-CV parameters have consistent dimensions.

    Parameters
    ----------
    min_bounds, max_bounds, bin, spacing, sigma, idw
        Parameters from sum_hills that can be lists for multi-CV cases.

    Raises
    ------
    ValueError
        If list parameters have inconsistent lengths.
    """
    # Collect all list parameters and their lengths
    list_params: dict[str, int] = {}

    params_to_check = {
        "min_bounds": min_bounds,
        "max_bounds": max_bounds,
        "bin": bin,
        "spacing": spacing,
        "sigma": sigma,
        "idw": idw,
    }

    for name, value in params_to_check.items():
        if isinstance(value, (list, tuple)):
            list_params[name] = len(value)

    # If no list parameters, nothing to validate (single CV case)
    if not list_params:
        return

    # Check that all list parameters have the same length
    lengths = set(list_params.values())
    if len(lengths) > 1:
        # Build a detailed error message
        param_details = ", ".join(
            f"{name}={length}" for name, length in list_params.items()
        )
        raise ValueError(
            f"Inconsistent number of CVs in parameters. "
            f"All list parameters must have the same length. "
            f"Got: {param_details}"
        )


def sum_hills(
    hills_file: str | Path,
    plumed_bin_path: str | Path | None = None,
    # Boolean flags
    negbias: bool = False,
    nohistory: bool = False,
    mintozero: bool = False,
    # File/histogram options
    histo: str | Path | None = None,
    # Grid parameters
    stride: int | None = None,
    min_bounds: float | list[float] | None = None,
    max_bounds: float | list[float] | None = None,
    bin: int | list[int] | None = None,
    spacing: float | list[float] | None = None,
    # Variable selection
    idw: str | list[str] | None = None,
    # Output options
    outfile: str | Path | None = None,
    outhisto: str | Path | None = None,
    # Integration parameters
    kt: float | None = None,
    sigma: float | list[float] | None = None,
    # Format
    fmt: str | None = None,
    # Additional options
    verbose: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run PLUMED sum_hills to reconstruct free energy surfaces from metadynamics.

    This function wraps the PLUMED ``sum_hills`` command-line tool, which analyzes
    HILLS files from metadynamics simulations to reconstruct the free energy surface.

    Parameters
    ----------
    hills_file : str or Path
        Path to the HILLS file to analyze. This file is generated during
        metadynamics simulations and contains the deposited Gaussian hills.
    plumed_bin_path : str or Path, optional
        Path to the PLUMED installation directory (containing ``bin/`` and ``lib/``
        subdirectories). If None, searches for ``plumed`` in the system PATH.
        When a full installation path is provided, the function will properly set
        LD_LIBRARY_PATH to include the PLUMED libraries.
    negbias : bool, default=False
        Print the negative bias instead of the free energy.
    nohistory : bool, default=False
        To be used with ``stride``: splits the bias/histogram without previous history.
    mintozero : bool, default=False
        Translate all minimum values in bias/histogram to zero.
    histo : str or Path, optional
        Name of the file for histogram (a COLVAR/HILLS file is good).
    stride : int, optional
        Stride for integrating hills file. Default is 0 (never integrate).
    min_bounds : float or list[float], optional
        Lower bounds for the grid. For multi-dimensional CVs, provide a list with
        one value per CV (e.g., ``[-3.14, -3.14]`` for two torsion angles).
    max_bounds : float or list[float], optional
        Upper bounds for the grid. For multi-dimensional CVs, provide a list with
        one value per CV (e.g., ``[3.14, 3.14]`` for two torsion angles).
    bin : int or list[int], optional
        Number of bins for the grid. For multi-dimensional CVs, provide a list with
        one value per CV (e.g., ``[250, 250]`` for two CVs with 250 bins each).
    spacing : float or list[float], optional
        Grid spacing, alternative to the number of bins. For multi-dimensional CVs,
        provide a list with one value per CV.
    idw : str or list[str], optional
        Variables to be used for the free-energy/histogram. For multi-dimensional CVs,
        provide a list with one variable name per CV (e.g., ``['phi', 'psi']``).
    outfile : str or Path, optional
        Output file for sum_hills. Default is ``fes.dat``.
    outhisto : str or Path, optional
        Output file for the histogram.
    kt : float, optional
        Temperature in energy units (kJ/mol) for integrating out variables.
    sigma : float or list[float], optional
        Sigma for binning (only needed when doing histogram). For multi-dimensional CVs,
        provide a list with one value per CV.
    fmt : str, optional
        Output format specification.
    verbose : bool, default=True
        Print command output to stdout/stderr.
    check : bool, default=True
        Raise exception if command fails.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process object from subprocess.run.

    Raises
    ------
    FileNotFoundError
        If the HILLS file, PLUMED executable, or PLUMED installation directory
        cannot be found.
    ValueError
        If list-based parameters (``bin``, ``min_bounds``, ``max_bounds``, etc.)
        have inconsistent lengths when using multiple CVs.
    subprocess.CalledProcessError
        If the PLUMED command fails and ``check=True``.

    Examples
    --------
    Basic usage to reconstruct a 1D free energy surface:

    >>> import hillclimber as hc
    >>> hc.sum_hills("HILLS")

    With custom grid resolution and output file:

    >>> hc.sum_hills(
    ...     "HILLS",
    ...     bin=1000,
    ...     outfile="custom_fes.dat"
    ... )

    For a 2D free energy surface with explicit bounds:

    >>> hc.sum_hills(
    ...     "HILLS",
    ...     bin=[100, 100],
    ...     min_bounds=[0.0, 0.0],
    ...     max_bounds=[10.0, 10.0],
    ...     outfile="fes_2d.dat"
    ... )

    For protein backbone torsion angles (phi and psi):

    >>> hc.sum_hills(
    ...     "HILLS",
    ...     bin=[250, 250],
    ...     min_bounds=[-3.14, -3.14],
    ...     max_bounds=[3.14, 3.14],
    ...     idw=["phi", "psi"],
    ...     outfile="ramachandran.dat"
    ... )

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/sum_hills.html

    Notes
    -----
    The HILLS file is automatically generated during metadynamics simulations
    when using the METAD action. Each line in the file represents a deposited
    Gaussian hill with its position, width (sigma), and height.

    The free energy surface is reconstructed by summing all deposited hills:
    F(s) = -V(s) where V(s) is the bias potential.

    **Multi-CV Consistency:**
    When using multiple collective variables (CVs), all list-based parameters
    must have the same length. For example, if analyzing two CVs (phi and psi),
    then ``bin``, ``min_bounds``, ``max_bounds``, and ``idw`` (if provided as lists)
    must all have exactly 2 elements. The function will raise a ``ValueError``
    if inconsistent list lengths are detected.
    """
    # Convert to Path object
    hills_file = Path(hills_file)

    # Verify HILLS file exists
    if not hills_file.exists():
        raise FileNotFoundError(f"HILLS file not found: {hills_file}")

    # Find PLUMED executable and set up environment
    env = os.environ.copy()

    if plumed_bin_path is None:
        # Try to find plumed in system PATH
        plumed_exec = shutil.which("plumed")
        if plumed_exec is None:
            raise FileNotFoundError(
                "PLUMED executable not found in system PATH. "
                "Please install PLUMED or specify the installation path with plumed_bin_path="
            )
    else:
        # Use provided PLUMED installation path
        plumed_bin_path = Path(plumed_bin_path)
        plumed_exec = plumed_bin_path / "bin" / "plumed"
        lib_path = plumed_bin_path / "lib"

        # Verify paths exist
        if not plumed_exec.exists():
            raise FileNotFoundError(
                f"PLUMED executable not found at: {plumed_exec}\n"
                f"Make sure plumed_bin_path points to the PLUMED installation directory "
                f"containing bin/ and lib/ subdirectories."
            )
        if not lib_path.exists():
            raise FileNotFoundError(f"PLUMED lib directory not found: {lib_path}")

        # Set LD_LIBRARY_PATH for PLUMED libraries
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        if current_ld_path:
            env["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}"
        else:
            env["LD_LIBRARY_PATH"] = str(lib_path)

        plumed_exec = str(plumed_exec)

    # Validate multi-CV parameter consistency
    _validate_multi_cv_params(
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        bin=bin,
        spacing=spacing,
        sigma=sigma,
        idw=idw,
    )

    # Build command
    cmd_parts = [plumed_exec, "sum_hills"]

    # Add hills file
    cmd_parts.extend(["--hills", str(hills_file)])

    # Add boolean flags
    if negbias:
        cmd_parts.append("--negbias")
    if nohistory:
        cmd_parts.append("--nohistory")
    if mintozero:
        cmd_parts.append("--mintozero")

    # Helper function to format list parameters
    def format_param(value: t.Any) -> str:
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    # Add optional parameters
    if histo is not None:
        cmd_parts.extend(["--histo", str(histo)])
    if stride is not None:
        cmd_parts.extend(["--stride", str(stride)])
    if min_bounds is not None:
        cmd_parts.extend(["--min", format_param(min_bounds)])
    if max_bounds is not None:
        cmd_parts.extend(["--max", format_param(max_bounds)])
    if bin is not None:
        cmd_parts.extend(["--bin", format_param(bin)])
    if spacing is not None:
        cmd_parts.extend(["--spacing", format_param(spacing)])
    if idw is not None:
        cmd_parts.extend(["--idw", format_param(idw)])
    if outfile is not None:
        cmd_parts.extend(["--outfile", str(outfile)])
    if outhisto is not None:
        cmd_parts.extend(["--outhisto", str(outhisto)])
    if kt is not None:
        cmd_parts.extend(["--kt", str(kt)])
    if sigma is not None:
        cmd_parts.extend(["--sigma", format_param(sigma)])
    if fmt is not None:
        cmd_parts.extend(["--fmt", str(fmt)])

    # Run command
    if verbose:
        print(f"Running: {' '.join(cmd_parts)}")

    result = subprocess.run(
        cmd_parts,
        env=env,
        capture_output=not verbose,
        text=True,
        check=check,
    )

    if verbose and result.returncode == 0:
        print("sum_hills completed successfully")

    return result

"""Analysis utilities for metadynamics simulations.

This module provides tools for analyzing metadynamics simulations,
including free energy surface reconstruction and other post-processing tasks.
"""

import re
import subprocess
import typing as t
from pathlib import Path

import numpy as np


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
        If the HILLS file cannot be found.
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
    cmd_parts = ["plumed", "sum_hills"]

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
        capture_output=not verbose,
        text=True,
        check=check,
    )

    if verbose and result.returncode == 0:
        print("sum_hills completed successfully")

    return result


def read_colvar(
    colvar_file: str | Path,
) -> dict[str, np.ndarray]:
    """Read a PLUMED COLVAR file and parse its contents.

    This function reads a COLVAR file produced by PLUMED, extracts the field names
    from the header (which starts with ``#! FIELDS``), and returns the data as a
    dictionary mapping field names to numpy arrays.

    Parameters
    ----------
    colvar_file : str or Path
        Path to the COLVAR file to read.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping field names to 1D numpy arrays containing the data.
        Keys correspond to the fields specified in the COLVAR header.

    Raises
    ------
    FileNotFoundError
        If the COLVAR file does not exist.
    ValueError
        If the COLVAR file does not contain a valid ``#! FIELDS`` header.

    Examples
    --------
    >>> import hillclimber as hc
    >>> data = hc.read_colvar("COLVAR")
    >>> print(data.keys())
    dict_keys(['time', 'phi', 'psi'])
    >>> print(data['time'][:5])
    [0. 1. 2. 3. 4.]

    Notes
    -----
    The COLVAR file format from PLUMED starts with a header line:
    ``#! FIELDS time cv1 cv2 ...``

    All subsequent lines starting with ``#`` are treated as comments and ignored.
    Data lines are parsed as whitespace-separated numeric values.

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/colvar.html
    """
    colvar_file = Path(colvar_file)

    if not colvar_file.exists():
        raise FileNotFoundError(f"COLVAR file not found: {colvar_file}")

    # Read the file
    with open(colvar_file, "r") as f:
        lines = f.readlines()

    # Find and parse the header
    field_names: list[str] | None = None
    for line in lines:
        if line.startswith("#! FIELDS"):
            # Extract field names from the header
            # Format: "#! FIELDS time phi psi ..."
            fields_match = re.match(r"#!\s*FIELDS\s+(.+)", line)
            if fields_match:
                field_names = fields_match.group(1).split()
                break

    if field_names is None:
        raise ValueError(
            f"COLVAR file {colvar_file} does not contain a valid '#! FIELDS' header"
        )

    # Parse data lines (skip comments)
    data_lines = []
    for line in lines:
        # Skip comments and empty lines
        if line.startswith("#") or not line.strip():
            continue
        # Parse numeric data
        values = line.split()
        if len(values) == len(field_names):
            data_lines.append([float(v) for v in values])

    # Convert to numpy array
    data_array = np.array(data_lines)

    # Create dictionary mapping field names to columns
    result = {name: data_array[:, i] for i, name in enumerate(field_names)}

    return result


def plot_cv_time_series(
    colvar_file: str | Path,
    cv_names: list[str] | None = None,
    time_unit: str = "ps",
    exclude_patterns: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    kde_width: str = "25%",
    colors: list[str] | None = None,
    alpha: float = 0.5,
    marker: str = "x",
    marker_size: float = 10,
) -> tuple[t.Any, t.Any]:
    """Plot collective variables over time with KDE distributions.

    This function creates a visualization showing CV evolution over time as scatter
    plots, with kernel density estimation (KDE) plots displayed on the right side
    to show the distribution of each CV.

    Parameters
    ----------
    colvar_file : str or Path
        Path to the COLVAR file to plot.
    cv_names : list[str], optional
        List of CV names to plot. If None, automatically detects CVs by excluding
        common non-CV fields like 'time', 'sigma_*', 'height', 'biasf'.
    time_unit : str, default='ps'
        Unit label for the time axis.
    exclude_patterns : list[str], optional
        Additional regex patterns for field names to exclude from auto-detection.
        Default excludes: 'time', 'sigma_.*', 'height', 'biasf'.
    figsize : tuple[float, float], default=(8, 5)
        Figure size in inches (width, height).
    kde_width : str, default='25%'
        Width of the KDE subplot as a percentage of the main plot width.
    colors : list[str], optional
        List of colors to use for each CV. If None, uses default color cycle.
    alpha : float, default=0.5
        Transparency for scatter points.
    marker : str, default='x'
        Marker style for scatter points.
    marker_size : float, default=10
        Size of scatter markers.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : tuple
        Tuple of (main_axis, kde_axis) matplotlib axes objects.

    Raises
    ------
    ImportError
        If matplotlib or seaborn is not installed.
    FileNotFoundError
        If the COLVAR file does not exist.

    Examples
    --------
    Basic usage with auto-detected CVs:

    >>> import hillclimber as hc
    >>> fig, axes = hc.plot_cv_time_series("COLVAR")

    Plot specific CVs:

    >>> fig, axes = hc.plot_cv_time_series("COLVAR", cv_names=["phi", "psi"])

    Customize appearance:

    >>> fig, axes = hc.plot_cv_time_series(
    ...     "COLVAR",
    ...     figsize=(10, 6),
    ...     colors=["blue", "red"],
    ...     alpha=0.7
    ... )

    Notes
    -----
    This function requires matplotlib and seaborn to be installed.

    The function automatically detects CVs by excluding common metadata fields
    such as 'time', 'sigma_*', 'height', and 'biasf'. You can specify additional
    exclusion patterns or explicitly provide the CV names to plot.

    Resources
    ---------
    - https://www.plumed.org/doc-master/user-doc/html/colvar.html
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError as e:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install them with: pip install matplotlib seaborn"
        ) from e

    # Read the COLVAR file
    data = read_colvar(colvar_file)

    # Auto-detect CVs if not specified
    if cv_names is None:
        # Default exclusion patterns
        default_exclude = [
            r"^time$",
            r"^sigma_.*$",
            r"^height$",
            r"^biasf$",
        ]
        if exclude_patterns is not None:
            default_exclude.extend(exclude_patterns)

        # Filter field names
        detected_cvs: list[str] = []
        for field in data.keys():
            # Check if field matches any exclusion pattern
            exclude = False
            for pattern in default_exclude:
                if re.match(pattern, field):
                    exclude = True
                    break
            if not exclude:
                detected_cvs.append(field)

        if not detected_cvs:
            raise ValueError(
                "No CVs detected in COLVAR file. "
                "All fields were excluded by the exclusion patterns."
            )
        cv_names = detected_cvs

    # Verify that all requested CVs exist
    missing_cvs = [cv for cv in cv_names if cv not in data]
    if missing_cvs:
        raise ValueError(
            f"CVs not found in COLVAR file: {missing_cvs}. "
            f"Available fields: {list(data.keys())}"
        )

    # Get time data
    if "time" not in data:
        raise ValueError("COLVAR file must contain a 'time' field")
    time = data["time"]

    # Default colors if not provided
    if colors is None:
        colors = plt.cm.tab10.colors  # type: ignore

    # Set seaborn style
    sns.set(style="whitegrid")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each CV
    for i, cv_name in enumerate(cv_names):
        color = colors[i % len(colors)]
        cv_data = data[cv_name]
        ax.scatter(
            time,
            cv_data,
            c=[color],
            label=cv_name,
            marker=marker,
            s=marker_size,
            alpha=alpha,
        )

    ax.set_xlabel(f"Time / {time_unit}")
    ax.set_ylabel("CV value")
    ax.legend()

    # Create KDE subplot on the right
    divider = make_axes_locatable(ax)
    ax_kde = divider.append_axes("right", size=kde_width, pad=0.1, sharey=ax)

    # Plot KDE for each CV
    for i, cv_name in enumerate(cv_names):
        color = colors[i % len(colors)]
        cv_data = data[cv_name]
        sns.kdeplot(
            y=cv_data,
            ax=ax_kde,
            color=color,
            fill=True,
            alpha=0.3,
            linewidth=1.5,
            label=cv_name,
        )

    # Clean up KDE axis
    ax_kde.set_xlabel("Density")
    ax_kde.yaxis.set_tick_params(labelleft=False)

    plt.tight_layout()

    return fig, (ax, ax_kde)

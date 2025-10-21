# Hillclimber
A Python wrapper around PLUMED.

This package uses numpy style docstrings and type hinting.
Use `t.Literal[]` instead of plain strings for fixed options.
To run tests, use `uv run pytest tests/`.
There must be unit tests added for all new functionality!

The goal of the package is to provide abstractions for collective variables and biases and to interface with ASE and ZnTrack.

The package uses ASE units, e.g. distances are in Angstrom, energies in eV and time in fs.
If available, reference the `https://www.plumed.org/doc-master/user-doc/html/` inside the docstrings to get more information about the underlying PLUMED functionality!

This is a new package, backwards compatibility is not required!
Make design decisions for good code structure and usability, not for backwards compatibility!
Use KISS, DRY, SOLID and YAGNI principles.

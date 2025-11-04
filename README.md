[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zincware/hillclimber)

# hillclimber


**hillclimber** is a Python framework for enhanced sampling with PLUMED. It provides high-level, Pythonic interfaces for configuring metadynamics simulations, making it easy to explore rare events and climb energy barriers in molecular dynamics simulations.

## Installation

### Prerequisites

For `uv add plumed`, you might need
```bash
export CC=gcc
export CXX=g++
```

You'll also need to configure PLUMED per project. Create an `env.yaml` file:

```yaml
global:
  PLUMED_KERNEL: /path/to/plumed2/lib/libplumedKernel.so
```

### Install hillclimber

```bash
uv add hillclimber
```

Or with pip:

```bash
pip install hillclimber
```

## Units

hillclimber uses **ASE units** throughout the package:

- **Distances**: Ångström / Å
- **Energies**: electronvolt / eV
- **Time**: femtoseconds / fs
- **Temperature**: Kelvin / K

## Documentation

Currently, there is no documentation available. Please refer to `/examples` for usage examples.

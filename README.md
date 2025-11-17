[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zincware/hillclimber)

# hillclimber


**hillclimber** is a Python framework for enhanced sampling with PLUMED. It provides high-level, Pythonic interfaces for configuring metadynamics simulations, making it easy to explore rare events and climb energy barriers in molecular dynamics simulations.

## Installation

### Standard Installation (Bundled PLUMED)

Starting from version 0.1.0a6, hillclimber bundles the PLUMED library directly in the wheel:

```bash
pip install hillclimber
```

**That's it!** No separate PLUMED installation or environment variables needed.

### What's Included

The wheel bundles:
- ✅ PLUMED library (`libplumedKernel`)
- ✅ PLUMED executable (for `sum_hills` and other tools)
- ✅ Automatic library loading

Everything works out of the box - no additional installation needed!

### Advanced: Using System PLUMED

If you prefer to use a system-installed PLUMED:

```bash
pip install hillclimber[system-plumed]
export PLUMED_KERNEL=/path/to/libplumedKernel.so
```

See [BUILDING.md](BUILDING.md) for more details on building from source and customization options.

## Units

hillclimber uses **ASE units** throughout the package:

- **Distances**: Ångström / Å
- **Energies**: electronvolt / eV
- **Time**: femtoseconds / fs
- **Temperature**: Kelvin / K

## Documentation

Currently, there is no documentation available. Please refer to `/examples` for usage examples.

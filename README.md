[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zincware/hillclimber)

# hillclimber


**hillclimber** is a Python framework for enhanced sampling with PLUMED. It provides high-level, Pythonic interfaces for configuring metadynamics simulations, making it easy to explore rare events and climb energy barriers in molecular dynamics simulations.

## Installation

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

## Python Collective Variables (PyCV)

hillclimber supports custom Python-based collective variables through PLUMED's `PYCVINTERFACE`. This allows you to define arbitrary CVs in Python with full access to atomic coordinates and gradients.

### Defining a PyCV

Create a subclass of `PyCV` with a `compute` method:

```python
# cv.py
import numpy as np
from ase import Atoms
from hillclimber.pycv import PyCV


class DistanceCV(PyCV):
    """Custom distance CV between two atoms."""

    atoms: list[int]  # Atom indices to use

    def compute(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Compute CV value and gradients.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object with current positions.

        Returns
        -------
        tuple[float, np.ndarray]
            CV value and gradients array of shape (n_atoms, 3).
        """
        positions = atoms.get_positions()
        diff = positions[self.atoms[1]] - positions[self.atoms[0]]
        distance = float(np.linalg.norm(diff))

        # Compute gradients
        grad = np.zeros((len(atoms), 3))
        grad[self.atoms[0]] = -diff / distance
        grad[self.atoms[1]] = diff / distance

        return distance, grad
```

### Using PyCV with ZnTrack

PyCV integrates seamlessly with ZnTrack workflows:

```python
# main.py
import zntrack
import ipsuite as ips
import hillclimber as hc
from cv import DistanceCV  # Import from standalone module


# Define your model (e.g., MACE, LJ, etc.)
class MyModel(zntrack.Node):
    def get_calculator(self, **kwargs):
        # Return your ASE calculator
        ...


project = zntrack.Project()

with project:
    data = ...  # Your data node providing atoms

    # Create the PyCV
    cv = DistanceCV(atoms=[0, 1], prefix="dist")

    # Configure the bias
    bias = hc.MetadBias(
        cv=cv,
        sigma=0.1,
        grid_min=2.0,
        grid_max=6.0,
        grid_bin=100,
    )

    # Configure metadynamics
    config = hc.MetaDynamicsConfig(
        height=0.01,
        pace=10,
        temp=300.0,
    )

    # Create the biased model
    metad = hc.MetaDynamicsModel(
        config=config,
        data=data.frames,
        bias_cvs=[bias],
        model=MyModel(),
    )

    # Run MD with ipsuite
    md = ips.ASEMD(
        data=data.frames,
        data_ids=-1,
        model=metad,
        thermostat=ips.LangevinThermostat(
            temperature=300.0,
            friction=0.01,
            time_step=1.0,
        ),
        steps=1000,
        sampling_rate=10,
    )

project.repro()
```

**Important**: The PyCV class should be defined in a standalone module (not `__main__`) that can be imported by PLUMED's Python interface. The module should only import what's necessary for the CV computation to avoid import issues in the PLUMED subprocess.

## Documentation

Currently, there is no documentation available. Please refer to `/examples` for usage examples.

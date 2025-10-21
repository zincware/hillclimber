# hillclimber

> [!NOTE]
> This README has been AI generated. Once I have the time, I'll update it! 

**hillclimber** is a Python framework for enhanced sampling with PLUMED. It provides high-level, Pythonic interfaces for configuring metadynamics simulations, making it easy to explore rare events and climb energy barriers in molecular dynamics simulations.

## Why hillclimber?

The name reflects the core concept of metadynamics: filling energy valleys with Gaussian "hills" to help systems climb over barriers and explore rare conformational states. hillclimber makes this process simple and intuitive.

## Features

- **Pythonic API**: Configure PLUMED simulations using clean Python dataclasses instead of manual input files
- **Chemistry-Aware Selection**: Use SMILES and SMARTS patterns to select atoms automatically
- **Collective Variables**: Distance, coordination number, torsion, radius of gyration, and more
- **Metadynamics Support**: Full metadynamics with adaptive schemes and well-tempered ensembles
- **Workflow Integration**: Built-in zntrack support for reproducible computational workflows
- **Visualization**: Automatic molecular structure visualization with CV highlighting
- **ASE Integration**: Seamless integration with ASE and machine learning force fields

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

## Quick Start

```python
import hillclimber as hc
import ipsuite as ips
import zntrack

# Create a zntrack project
project = zntrack.Project()

with project:
    # Load molecular structure
    data = ips.AddData("seed.xyz")

    # Select atoms using SMARTS patterns
    oh_group = hc.SMARTSSelector(pattern="[OH]")
    oxygen = hc.SMARTSSelector(pattern="[O]")

    # Define a distance collective variable
    distance_cv = hc.DistanceCV(
        x1=oh_group,
        x2=oxygen,
        prefix="oh_distance",
        multi_group="all_pairs"
    )

    # Configure metadynamics bias
    bias = hc.MetaDBiasCV(
        cv=distance_cv,
        sigma=0.1,
        grid_min=0.0,
        grid_max=5.0,
        grid_bin=200
    )

    # Set up metadynamics configuration
    config = hc.MetaDynamicsConfig(
        height=0.5,        # Height of Gaussian hills (kJ/mol)
        pace=150,          # Frequency of hill deposition
        biasfactor=10.0,   # Well-tempered metadynamics
        temp=300.0         # Temperature (K)
    )

    # Create the metadynamics model (wraps the ML force field with PLUMED bias)
    metad_model = hc.MetaDynamicsModel(
        config=config,
        bias_cvs=[bias],
        data=data.frames,
        model=ips.MACEMPModel(),  # ML force field calculator
        timestep=0.5               # fs
    )

    # Run MD simulation with ipsuite
    md = ips.ASEMD(
        data=data.frames,
        model=metad_model,         # Uses the biased calculator
        thermostat=ips.LangevinThermostat(temperature=300, friction=0.01),
        steps=10_000
    )

# Execute the workflow
project.build()
```

## Collective Variables

hillclimber supports multiple types of collective variables:

### Distance CV

```python
distance = hc.DistanceCV(
    x1=selector1,
    x2=selector2,
    prefix="d",
    group_reduction="com",  # "com", "cog", "first", "all"
    multi_group="first"      # "first", "all_pairs", "corresponding", "first_to_all"
)
```

### Coordination Number CV

```python
coordination = hc.CoordinationNumberCV(
    x1=selector1,
    x2=selector2,
    prefix="cn",
    r_0=2.5,  # Reference distance (Angstroms)
    nn=6,     # Switching function parameters
    mm=12
)
```

### Torsion CV

```python
# Use SMARTS with mapped atoms to define 4-atom groups
torsion = hc.TorsionCV(
    x1=hc.SMARTSSelector(pattern="CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C"),
    prefix="phi"
)
```

### Radius of Gyration CV

```python
gyration = hc.RadiusOfGyrationCV(
    x1=molecule_selector,
    prefix="rg",
    gyration_type="RADIUS"  # "RADIUS", "ASPHERICITY", etc.
)
```

## Atom Selectors

Three types of selectors are available:

### Index Selector

```python
selector = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])
```

### SMILES Selector

```python
selector = hc.SMILESSelector(smiles="CCO")  # Selects ethanol molecules
```

### SMARTS Selector

```python
# Basic pattern
selector = hc.SMARTSSelector(pattern="[OH]")

# Mapped atoms for precise selection
selector = hc.SMARTSSelector(
    pattern="C1[C:1]OC(=[O:1])O1",
    hydrogens="include"  # "exclude", "include", "isolated"
)
```

## Advanced Features

### Multi-Group Strategies

Control how multiple atom groups are combined:

- `"first"`: Only use first groups from each selector
- `"all_pairs"`: All combinations (Cartesian product)
- `"corresponding"`: Match groups by index
- `"first_to_all"`: First group from x1 to all groups in x2

### Group Reduction

Reduce atom groups to single points:

- `"com"`: Center of mass
- `"cog"`: Center of geometry
- `"first"`: Use first atom only
- `"all"`: Use all atoms (when supported)

### Print Actions

Output CV values during simulation:

```python
print_action = hc.PrintCVAction(
    cvs=[distance_cv, coordination_cv],
    stride=100,
    file="COLVAR"
)
```

## Workflow Integration with ipsuite and zntrack

hillclimber is designed to work seamlessly with **ipsuite** and **zntrack** for reproducible, lineage-tracked computational workflows.

### How It Works

The `MetaDynamicsModel` class inherits from `zntrack.Node`, which enables:
- **Automatic dependency tracking**: zntrack records what data, configurations, and models feed into your simulation
- **Reproducibility**: Full lineage tracking means you can always reconstruct how results were generated
- **Change detection**: zntrack knows when to re-run computations based on changed inputs

### The Workflow Pattern

```python
import hillclimber as hc
import ipsuite as ips
import zntrack

# Create a zntrack project for workflow management
project = zntrack.Project()

with project:
    # 1. Data loading (zntrack dependency)
    data = ips.AddData("seed.xyz")

    # 2. Optional: Structure optimization
    geom_opt = ips.GeomOpt(data=data.frames, model=ips.MACEMPModel())

    # 3. Define collective variables and bias
    cv = hc.DistanceCV(
        x1=hc.SMARTSSelector(pattern="[H]O[H]"),  # Water proton
        x2=hc.SMARTSSelector(pattern="CO[C:1]"),  # Carboxyl C
        prefix="d"
    )

    bias = hc.MetaDBiasCV(
        cv=cv,
        sigma=0.1,
        grid_min=0.0,
        grid_max=2.0,
        grid_bin=200
    )

    # 4. Configure metadynamics
    config = hc.MetaDynamicsConfig(
        height=0.25,
        pace=2000,
        biasfactor=10,
        temp=300
    )

    # 5. Create MetaDynamicsModel (this is a zntrack.Node!)
    metad_model = hc.MetaDynamicsModel(
        config=config,               # zntrack.deps()
        bias_cvs=[bias],             # zntrack.deps()
        data=geom_opt.frames,        # zntrack.deps()
        model=ips.MACEMPModel(),     # zntrack.deps() - the base force field
        timestep=0.5,                # zntrack.params() - tunable parameter
        data_idx=-1                  # zntrack.params() - which frame to use
    )

    # 6. Run MD simulation with ipsuite
    md = ips.ASEMD(
        data=geom_opt.frames,
        model=metad_model,           # MetaDynamicsModel wraps MACEMPModel with PLUMED
        thermostat=ips.LangevinThermostat(temperature=300, friction=0.01),
        steps=10_000
    )

# Execute the workflow
project.build()
```

### What Happens Behind the Scenes

1. **Project Context (`with project:`)**: All nodes created within the context manager are registered with zntrack for dependency tracking and workflow management.

2. **Building the Workflow (`project.build()`)**:
   - zntrack analyzes the dependency graph of all nodes
   - Executes nodes in the correct order based on dependencies
   - Skips nodes that are already up-to-date (smart caching)
   - Stores results and tracks lineage

3. **Calculator Wrapping**: When `ips.ASEMD` calls `metad_model.get_calculator()`:
   - PLUMED input files are generated from your CV definitions
   - The base ML calculator (`MACEMPModel`) is wrapped with PLUMED biasing
   - A `NonOverwritingPlumed` calculator is returned that applies bias forces

4. **Dependency Tracking**: zntrack records the entire workflow graph:
   ```
   AddData → GeomOpt → MACEMPModel ↘
                                    → MetaDynamicsModel → ASEMD
   CVs + Config + Bias            ↗
   ```

5. **Output Organization**:
   - CV visualizations are saved to `figures/` directory
   - PLUMED outputs (HILLS, COLVAR) are written to the simulation directory
   - zntrack tracks all inputs and outputs

### Multiple CVs Example

You can bias multiple collective variables simultaneously:

```python
project = zntrack.Project()

with project:
    data = ips.AddData("seed.xyz")

    # Define multiple CVs
    cv1 = hc.DistanceCV(x1=sel1, x2=sel2, prefix="d12")
    cv2 = hc.CoordinationNumberCV(x1=sel3, x2=sel4, prefix="cn", r_0=2.5)
    cv3 = hc.TorsionCV(x1=sel5, prefix="phi")

    # Create biases for each CV
    biases = [
        hc.MetaDBiasCV(cv=cv1, sigma=0.1, grid_min=0.0, grid_max=5.0, grid_bin=100),
        hc.MetaDBiasCV(cv=cv2, sigma=0.05, grid_min=0, grid_max=10, grid_bin=100),
        hc.MetaDBiasCV(cv=cv3, sigma=0.2, grid_min=-3.14, grid_max=3.14, grid_bin=100)
    ]

    # Configure metadynamics
    config = hc.MetaDynamicsConfig(height=0.5, pace=500, biasfactor=10, temp=300)

    # Create model with multiple biases
    metad_model = hc.MetaDynamicsModel(
        config=config,
        bias_cvs=biases,
        data=data.frames,
        model=ips.MACEMPModel()
    )

    md = ips.ASEMD(data=data.frames, model=metad_model, steps=10_000)

project.build()
```

### Adding Print Actions

Monitor CV values during simulation:

```python
project = zntrack.Project()

with project:
    data = ips.AddData("seed.xyz")

    # Define CVs
    cv1 = hc.DistanceCV(...)
    cv2 = hc.CoordinationNumberCV(...)

    # Create print action
    print_action = hc.PrintCVAction(
        cvs=[cv1, cv2],  # CVs to monitor
        stride=100,       # Output frequency
        file="COLVAR"     # Output file
    )

    # Add to model
    metad_model = hc.MetaDynamicsModel(
        config=config,
        bias_cvs=biases,
        actions=[print_action],  # Add monitoring
        data=data.frames,
        model=ips.MACEMPModel()
    )

    md = ips.ASEMD(data=data.frames, model=metad_model, steps=10_000)

project.build()
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

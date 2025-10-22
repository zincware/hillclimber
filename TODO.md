# hillclimber Development TODO

> Comparison of hillclimber features vs PLUMED 2
> Generated: 2025-10-21

## Overview

hillclimber is a high-level Python wrapper around PLUMED 2 focused on metadynamics with ASE and ML force fields. This document tracks missing PLUMED 2 features to guide future development.

---

## 🔴 Phase 1: Critical Features (Highest Priority)

### 1.1 Restraints & Walls ⭐⭐⭐⭐⭐
**Status:** ✅ **IMPLEMENTED** (v0.1.0a1+)
**Priority:** CRITICAL
**Effort:** Low-Medium

**Components:**
- [x] `RestraintBias` - Harmonic restraints (umbrella sampling) ✅
- [x] `UpperWallBias` - Upper wall potentials ✅
- [x] `LowerWallBias` - Lower wall potentials ✅
- [ ] `MovingRestraintBias` - Time-dependent restraints (steered MD)

**Why critical:**
- **Umbrella sampling** is one of the most common enhanced sampling methods
- Required for free energy calculations with WHAM
- Enables steered MD and constrained sampling
- Blocks a major use case for molecular simulations

**Implemented API:**
```python
import hillclimber as hc

# Define a distance CV
distance_cv = hc.DistanceCV(
    x1=hc.SMARTSSelector(pattern="[OH]"),
    x2=hc.SMARTSSelector(pattern="[O]"),
    prefix="d"
)

# Harmonic restraint
restraint = hc.RestraintBias(
    cv=distance_cv,
    kappa=200.0,  # force constant (kJ/mol)
    at=2.5        # target value
)

# Upper wall
upper_wall = hc.UpperWallBias(
    cv=distance_cv,
    at=3.0,
    kappa=100.0,
    exp=2  # exponent (2=harmonic, higher=steeper)
)

# Lower wall
lower_wall = hc.LowerWallBias(
    cv=distance_cv,
    at=1.0,
    kappa=100.0,
    exp=2
)

# Use in MetaDynamicsModel via actions parameter
model = hc.MetaDynamicsModel(
    config=config,
    bias_cvs=[metad_bias],
    actions=[restraint, upper_wall, lower_wall],  # Add biases here!
    data=data.frames,
    model=ips.MACEMPModel()
)
```

**PLUMED equivalent:**
```plumed
# Restraint
RESTRAINT ARG=cv KAPPA=200.0 AT=2.5

# Walls
UPPER_WALLS ARG=cv AT=3.0 KAPPA=100.0 EXP=2
LOWER_WALLS ARG=cv AT=1.0 KAPPA=100.0 EXP=2
```

---

### 1.2 RMSD Collective Variables ⭐⭐⭐⭐⭐
**Status:** Not implemented
**Priority:** CRITICAL
**Effort:** Medium

**Missing components:**
- [ ] `RMSD` - Root mean square deviation from reference
- [ ] `DRMSD` - Distance RMSD
- [ ] `ALPHARMSD` - Alpha helix RMSD
- [ ] `ANTIBETARMSD` - Antiparallel beta sheet RMSD
- [ ] `PARABETARMSD` - Parallel beta sheet RMSD

**Why critical:**
- **Most widely used CV** in biomolecular simulations
- Essential for protein folding studies
- Required for structure-based biasing
- Standard CV for conformational analysis

**Proposed API:**
```python
# Basic RMSD
rmsd_cv = hc.RMSDCV(
    atoms=hc.SMARTSSelector(pattern="[C,N,O]"),  # or specific atoms
    reference="native.pdb",
    alignment_type="optimal",  # or "simple"
    prefix="rmsd"
)

# Distance RMSD
drmsd_cv = hc.DRMSDCV(
    atoms=selector,
    reference="native.pdb",
    lower_cutoff=0.1,  # nm
    upper_cutoff=0.8,  # nm
    prefix="drmsd"
)

# Secondary structure RMSD
alpha_cv = hc.AlphaRMSDCV(
    residues="all",  # or specific residue indices
    prefix="alpha"
)
```

**PLUMED equivalent:**
```plumed
rmsd: RMSD REFERENCE=native.pdb TYPE=OPTIMAL
drmsd: DRMSD REFERENCE=native.pdb LOWER_CUTOFF=0.1 UPPER_CUTOFF=0.8
alpha: ALPHARMSD RESIDUES=all
```

---

### 1.3 ANGLE Collective Variable ⭐⭐⭐⭐
**Status:** Not implemented
**Priority:** HIGH
**Effort:** Low

**Missing components:**
- [ ] `ANGLE` - Angle between three atoms/groups

**Why important:**
- Basic geometric descriptor (alongside distance and torsion)
- Used in almost every biomolecular study
- Simple to implement (similar to existing Distance/Torsion CVs)

**Proposed API:**
```python
angle_cv = hc.AngleCV(
    x1=selector1,  # atom/group 1
    x2=selector2,  # atom/group 2 (vertex)
    x3=selector3,  # atom/group 3
    prefix="angle",
    group_reduction="com",  # "com", "cog", "first"
    multi_group="first"
)
```

**PLUMED equivalent:**
```plumed
angle: ANGLE ATOMS=1,2,3
```

---

### 1.4 Combined/Custom CVs ⭐⭐⭐⭐
**Status:** Not implemented
**Priority:** HIGH
**Effort:** Low-Medium

**Missing components:**
- [ ] `COMBINE` - Linear combinations of CVs
- [ ] `CUSTOM` / `MATHEVAL` - Mathematical functions of CVs

**Why important:**
- Create **complex reaction coordinates** from simple CVs
- Common pattern: difference between two distances
- Enables flexible CV definitions without modifying code

**Proposed API:**
```python
# Linear combination
combined_cv = hc.CombineCV(
    cvs=[cv1, cv2],
    coefficients=[1.0, -1.0],
    powers=[1, 1],  # optional
    periodic=False,
    prefix="combined"
)

# Custom mathematical function
custom_cv = hc.CustomCV(
    cvs=[cv1, cv2],
    function="x**2 + y**2",  # or lambda x,y: x**2 + y**2
    periodic=False,
    prefix="custom"
)
```

**PLUMED equivalent:**
```plumed
# Combine
diff: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO

# Custom
custom: CUSTOM ARG=d1,d2 FUNC=x^2+y^2 PERIODIC=NO
```

---

## 🟠 Phase 2: High-Value Features

### 2.1 OPES (On-the-fly Probability Enhanced Sampling) ⭐⭐⭐⭐
**Status:** Not implemented
**Priority:** HIGH
**Effort:** High

**Missing components:**
- [ ] `OPES_METAD` - Standard OPES metadynamics
- [ ] `OPES_METAD_EXPLORE` - OPES with enhanced exploration
- [ ] `OPES_EXPANDED` - OPES with expanded ensembles
- [ ] `ECV_MULTITHERMAL` - Temperature expansion
- [ ] `ECV_UMBRELLAS_LINE` - Umbrella line for OPES

**Why important:**
- **Modern alternative** to traditional metadynamics
- Better convergence properties
- Automatic variance adaptation
- Lower systematic error
- Becoming preferred method in literature

**Proposed API:**
```python
# OPES config
opes_config = hc.OPESConfig(
    barrier=40.0,  # kJ/mol
    pace=500,
    temp=300.0
)

# OPES model (similar to MetaDynamicsModel)
opes_model = hc.OPESModel(
    config=opes_config,
    cvs=[cv1, cv2],
    data=data.frames,
    model=ips.MACEMPModel(),
    timestep=0.5
)
```

**PLUMED equivalent:**
```plumed
phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17
opes: OPES_METAD ARG=phi,psi PACE=500 BARRIER=40
```

---

### 2.2 Path Collective Variables ⭐⭐⭐
**Status:** Not implemented
**Priority:** MEDIUM-HIGH
**Effort:** Medium

**Missing components:**
- [ ] `PATH` / `PATHMSD` - Progress along reaction path
- [ ] S coordinate (position along path)
- [ ] Z coordinate (distance from path)

**Why important:**
- Essential for **transition path sampling**
- Used in **committor analysis**
- Study complex reaction mechanisms
- Track progress through conformational changes

**Proposed API:**
```python
path_cv = hc.PathCV(
    reference="path.pdb",  # multi-frame PDB with waypoints
    lambda_param=15100.0,
    path_type="optimal",  # or "euclidean"
    prefix="path"
)
# Returns both s (progress) and z (deviation) components
```

**PLUMED equivalent:**
```plumed
path: PATH REFERENCE=path.pdb TYPE=OPTIMAL LAMBDA=15100.
# Outputs path.spath and path.zpath
```

---

### 2.3 Virtual Atoms (Explicit Definition) ⭐⭐⭐
**Status:** Partial (internal only)
**Priority:** MEDIUM
**Effort:** Low

**Missing components:**
- [ ] Expose `COM`/`CENTER` as reusable virtual atoms
- [ ] Allow virtual atoms in subsequent CV definitions

**Current state:**
- hillclimber creates virtual sites internally in CVs
- Cannot define a virtual atom once and reuse it

**Why important:**
- **Hierarchical CV definitions**
- Reduce computational overhead
- Match PLUMED workflow patterns

**Proposed API:**
```python
# Define virtual atoms explicitly
center1 = hc.VirtualAtom(
    atoms=selector1,
    type="com",  # or "cog"
    label="c1"
)

center2 = hc.VirtualAtom(
    atoms=selector2,
    type="com",
    label="c2"
)

# Use in other CVs
distance_cv = hc.DistanceCV(
    x1=center1,
    x2=center2,
    prefix="d_centers"
)
```

**PLUMED equivalent:**
```plumed
c1: COM ATOMS=1-100
c2: COM ATOMS=101-200
d: DISTANCE ATOMS=c1,c2
```

---

### 2.4 Analysis Tools ⭐⭐⭐
**Status:** Not implemented
**Priority:** MEDIUM
**Effort:** Medium-High

**Missing components:**
- [ ] `HISTOGRAM` - Build histograms of CVs
- [ ] `REWEIGHT_BIAS` - Reweight for unbiased ensemble
- [ ] `COLLECT_FRAMES` - Collect trajectory data
- [ ] `COMMITTOR` - Committor analysis

**Why important:**
- Essential for **post-processing biased simulations**
- Analyze free energy surfaces
- Compute unbiased observables
- Quality control for simulations

**Proposed API:**
```python
# Histogram action
histogram = hc.HistogramAction(
    cvs=[cv1, cv2],
    grid_min=[0.0, -3.14],
    grid_max=[5.0, 3.14],
    grid_bins=[100, 100],
    kernel="discrete",
    stride=1000,
    file="histogram.dat"
)

# Reweighting
reweight = hc.ReweightBiasAction(temp=300.0)
```

---

## 🟡 Phase 3: Nice-to-Have Features

### 3.1 Additional Geometric CVs ⭐⭐⭐
**Status:** Partial
**Priority:** MEDIUM
**Effort:** Low-Medium

**Missing components:**
- [ ] `POSITION` - Absolute Cartesian positions
- [ ] `DIHEDRAL_CORRELATION` - Correlations between dihedrals
- [ ] `DIPOLE` - Molecular dipole moment
- [ ] Bond-orientational order parameters (`Q3`, `Q4`, `Q6`)

**Currently implemented:**
- ✅ `DISTANCE`
- ✅ `TORSION`
- ✅ `COORDINATION`
- ✅ `GYRATION`

---

### 3.2 Multi-Replica Features ⭐⭐
**Status:** Not implemented
**Priority:** LOW-MEDIUM
**Effort:** High

**Missing components:**
- [ ] Multiple walker metadynamics (`WALKERS_MPI`)
- [ ] Bias exchange metadynamics
- [ ] Parallel tempering integration
- [ ] `@replicas:` syntax support

**Why important:**
- Efficient **parallel enhanced sampling**
- Improve convergence via information sharing
- Common in HPC environments

---

### 3.3 PCA-based CVs ⭐⭐
**Status:** Not implemented
**Priority:** LOW-MEDIUM
**Effort:** High

**Missing components:**
- [ ] `PCAVARS` - Principal component analysis based CVs
- [ ] Support for different metrics (Euclidean, Optimal)

**Why important:**
- **Dimensionality reduction** in complex systems
- Identify collective motions
- Data-driven CV discovery

---

### 3.4 MOLINFO Integration ⭐⭐
**Status:** Alternative approach
**Priority:** LOW
**Effort:** Medium

**Missing components:**
- [ ] `MOLINFO` shortcuts for atom selection
- [ ] `@phi-2`, `@psi-3` for backbone angles
- [ ] `@water`, `@protein`, `@nonhydrogens` shortcuts

**Current state:**
- hillclimber uses SMARTS/SMILES (chemistry-aware, arguably better)
- No support for MOLINFO shortcuts

**Trade-off:**
- SMARTS is more flexible and chemistry-aware
- MOLINFO shortcuts are convenient for proteins
- Not critical given existing selector capabilities

---

## 📊 Implementation Priority Matrix

| Feature | Usage | Effort | Impact | Priority Score | Status |
|---------|-------|--------|--------|----------------|--------|
| RESTRAINT | ⭐⭐⭐⭐⭐ | Low | 🔥 Critical | **10/10** | ✅ Done |
| WALLS | ⭐⭐⭐⭐⭐ | Low | 🔥 Critical | **10/10** | ✅ Done |
| RMSD | ⭐⭐⭐⭐⭐ | Medium | 🔥 Critical | **9/10** | 🔴 Todo |
| ANGLE | ⭐⭐⭐⭐ | Low | High | **8/10** | 🔴 Todo |
| COMBINE | ⭐⭐⭐⭐ | Low | High | **8/10** | 🔴 Todo |
| OPES | ⭐⭐⭐⭐ | High | High | **7/10** | 🔴 Todo |
| PATH | ⭐⭐⭐ | Medium | Medium | **6/10** | 🔴 Todo |
| CUSTOM | ⭐⭐⭐ | Medium | Medium | **6/10** | 🔴 Todo |
| Virtual Atoms | ⭐⭐⭐ | Low | Medium | **6/10** | 🔴 Todo |
| Analysis Tools | ⭐⭐⭐ | High | Medium | **5/10** | 🔴 Todo |
| PCAVARS | ⭐⭐ | High | Medium | **4/10** | 🔴 Todo |
| Multi-walker | ⭐⭐ | High | Low | **3/10** | 🔴 Todo |

---

## 🎯 Recommended Development Roadmap

### Milestone 1: Core Functionality (v0.2.0)
**Goal:** Enable umbrella sampling and basic restraints
**Status:** 🟡 **IN PROGRESS** (2/5 complete)

- [x] Implement `RESTRAINT`, `UPPER_WALLS`, `LOWER_WALLS` ✅
- [x] Add tests for all new features ✅ (9 tests in test_biases.py)
- [ ] Add `ANGLE` CV
- [ ] Add `COMBINE` CV
- [ ] Update documentation with examples

**Impact:** Enables ~40% more use cases, including umbrella sampling

**Completed in v0.1.0a1+:**
- `RestraintBias` - Harmonic restraints for umbrella sampling
- `UpperWallBias` - Upper wall potentials
- `LowerWallBias` - Lower wall potentials
- `BiasProtocol` - Protocol interface for all bias types
- Comprehensive test coverage (test_biases.py)

---

### Milestone 2: Biomolecular Support (v0.3.0)
**Goal:** Support protein folding and structure studies

- [ ] Implement `RMSD`, `DRMSD`
- [ ] Implement `ALPHARMSD`, `ANTIBETARMSD`, `PARABETARMSD`
- [ ] Add `MOVINGRESTRAINT` for steered MD
- [ ] Improve visualization for RMSD CVs
- [ ] Add protein folding examples

**Impact:** Enables standard biomolecular simulation workflows

---

### Milestone 3: Advanced Sampling (v0.4.0)
**Goal:** Modern enhanced sampling methods

- [ ] Implement OPES (`OPES_METAD`, `OPES_METAD_EXPLORE`)
- [ ] Add `PATH` collective variables
- [ ] Implement `CUSTOM` CV with arbitrary functions
- [ ] Add expanded ensemble support (`ECV_*`)
- [ ] Performance optimization for OPES

**Impact:** State-of-the-art enhanced sampling capabilities

---

### Milestone 4: Analysis & Polish (v0.5.0)
**Goal:** Complete analysis workflow

- [ ] Implement `HISTOGRAM`
- [ ] Implement `REWEIGHT_BIAS`
- [ ] Add explicit virtual atom definitions
- [ ] Improve post-processing tools
- [ ] Add comprehensive examples and tutorials

**Impact:** End-to-end workflow from simulation to analysis

---

## 📝 Notes on Design Philosophy

### What hillclimber does BETTER than PLUMED:
- ✅ **Chemistry-aware selection** (SMARTS/SMILES vs. manual indices)
- ✅ **Pythonic API** (type-safe, self-documenting)
- ✅ **Workflow integration** (zntrack for reproducibility)
- ✅ **Automatic visualization** (molecular structure + CV highlighting)
- ✅ **ML force field integration** (ASE ecosystem)

### Design considerations for new features:
1. **Maintain Pythonic API** - dataclasses, type hints, clear names
2. **Chemistry-aware where possible** - leverage RDKit/SMARTS
3. **zntrack integration** - all new components as Nodes where appropriate
4. **Automatic visualization** - highlight relevant atoms for new CVs
5. **Generate PLUMED** - all features must compile to valid plumed.dat

### Testing requirements:
- Unit tests for each new CV class
- Integration tests with ASE
- PLUMED output validation
- Regression tests against PLUMED reference
- Performance benchmarks

---

## 🔗 References

- PLUMED 2 documentation: https://www.plumed.org/doc-master/
- OPES paper: Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020
- Path CVs: Branduardi et al., J. Chem. Phys. 2007
- hillclimber repository: (local package)

---

## 🤝 Contributing

To work on any of these features:

1. Check if issue exists in GitHub repo
2. Create feature branch: `feature/restraint-cv`
3. Implement following hillclimber design patterns
4. Add tests in `tests/test_*.py`
5. Update documentation and examples
6. Submit PR with description and use cases

---

## 📝 Implementation Notes

### v0.1.0a1+ Updates (2025-10-21)

**Implemented Features:**
- ✅ `RestraintBias` - Harmonic restraints (PLUMED `RESTRAINT`)
- ✅ `UpperWallBias` - Upper wall potentials (PLUMED `UPPER_WALLS`)
- ✅ `LowerWallBias` - Lower wall potentials (PLUMED `LOWER_WALLS`)
- ✅ `BiasProtocol` - Protocol interface for all bias types

**Design Decisions:**
- All bias classes use consistent "Bias" suffix (RestraintBias, UpperWallBias, LowerWallBias, MetadBias)
- Biases implement `PlumedGenerator` protocol and generate PLUMED commands via `to_plumed()`
- Biases work on any `CollectiveVariable` and are added to `MetaDynamicsModel` via the `actions` parameter
- `BiasProtocol` provides common interface: all biases have a `cv: CollectiveVariable` attribute

**Test Coverage:**
- 9 comprehensive tests in `tests/test_biases.py`
- Tests cover basic functionality, custom labels, different CV types, and parameter variations
- All tests passing ✅

**Files Modified:**
- `hillclimber/biases.py` (new) - Bias potential classes
- `hillclimber/interfaces.py` - Added BiasProtocol and MetadynamicsBias protocols
- `hillclimber/metadynamics.py` - Renamed MetaDBiasCV to MetadBias
- `hillclimber/__init__.py` - Added exports for new bias classes
- `tests/test_biases.py` (new) - Test suite

---

**Last Updated:** 2025-10-21
**Review Status:** Phase 1 partially complete (restraints & walls implemented)
**Next Review:** After implementing ANGLE and COMBINE CVs

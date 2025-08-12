import pytest
import rdkit2ase



@pytest.fixture(scope="session")
def ethanol_water():
    ethanol = rdkit2ase.smiles2conformers("CCO", numConfs=100)
    water = rdkit2ase.smiles2conformers("O", numConfs=100)
    box = rdkit2ase.pack(
        [ethanol, water], counts=[16, 16], density=700, packmol="packmol.jl"
    )
    return box.copy()


@pytest.fixture(scope="session")
def small_ethnol_water():
    ethanol = rdkit2ase.smiles2conformers("CCO", numConfs=1)
    water = rdkit2ase.smiles2conformers("O", numConfs=1)
    box = rdkit2ase.pack(
        [ethanol, water], counts=[2, 2], density=700, packmol="packmol.jl"
    )
    return box.copy()


@pytest.fixture(scope="session")
def na_cl_water():
    """Creates a box with 1 Na, 1 Cl, and 10 H2O molecules."""
    na_plus = rdkit2ase.smiles2conformers("[Na+]", numConfs=1)
    cl_minus = rdkit2ase.smiles2conformers("[Cl-]", numConfs=1)
    water = rdkit2ase.smiles2conformers("O", numConfs=1)
    box = rdkit2ase.pack(
        [na_plus, cl_minus, water],
        counts=[1, 1, 10],
        density=1000,  # More realistic for water
        packmol="packmol.jl",
    )
    return box.copy()


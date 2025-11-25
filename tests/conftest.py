import molify
import pytest


@pytest.fixture(scope="session")
def packmol() -> str:
    """Path to packmol executable."""
    import shutil

    packmol_path = shutil.which("packmol")
    if packmol_path is None:
        # we assume either packmol or packmol.jl is installed
        return "packmol.jl"
    return packmol_path


@pytest.fixture(scope="session")
def ethanol_water(packmol):
    ethanol = molify.smiles2conformers("CCO", numConfs=100)
    water = molify.smiles2conformers("O", numConfs=100)
    box = molify.pack([ethanol, water], counts=[16, 16], density=700, packmol=packmol)
    return box.copy()


@pytest.fixture(scope="session")
def small_ethanol_water(packmol):
    ethanol = molify.smiles2conformers("CCO", numConfs=1)
    water = molify.smiles2conformers("O", numConfs=1)
    box = molify.pack([ethanol, water], counts=[2, 2], density=700, packmol=packmol)
    return box.copy()


@pytest.fixture(scope="session")
def na_cl_water(packmol):
    """Creates a box with 1 Na, 1 Cl, and 10 H2O molecules."""
    na_plus = molify.smiles2conformers("[Na+]", numConfs=1)
    cl_minus = molify.smiles2conformers("[Cl-]", numConfs=1)
    water = molify.smiles2conformers("O", numConfs=1)
    box = molify.pack(
        [na_plus, cl_minus, water],
        counts=[1, 1, 10],
        density=1000,  # More realistic for water
        packmol=packmol,
    )
    return box.copy()

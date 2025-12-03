"""Tests for DistanceCV using the new API with VirtualAtom and flatten/pairwise parameters."""

import ase
import molify
import numpy as np
from PIL import Image

import hillclimber as pn


def test_distance_cv_first_strategy(small_ethanol_water):
    """Test first strategy - distance between first ethanol and first water COMs."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),  # First ethanol
        x2=pn.VirtualAtom(x2_selector[0], "com"),  # First water
        prefix="d12",
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethanol_water)

    expected = [
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
    ]
    assert plumed_str == expected
    assert labels == ["d12"]


def test_distance_cv_all_pairs(small_ethanol_water):
    """Test all pairs strategy - all ethanol-water COM combinations."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector, "com"),  # All ethanols
        x2=pn.VirtualAtom(x2_selector, "com"),  # All waters
        prefix="d",
        pairwise="all",
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethanol_water)

    # Should create 2x2=4 distances
    expected = [
        "d_x1_0: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x1_1: COM ATOMS=10,11,12,13,14,15,16,17,18",
        "d_x2_0: COM ATOMS=19,20,21",
        "d_x2_1: COM ATOMS=22,23,24",
        "d_0: DISTANCE ATOMS=d_x1_0,d_x2_0",
        "d_1: DISTANCE ATOMS=d_x1_0,d_x2_1",
        "d_2: DISTANCE ATOMS=d_x1_1,d_x2_0",
        "d_3: DISTANCE ATOMS=d_x1_1,d_x2_1",
    ]
    assert plumed_str == expected
    assert labels == ["d_0", "d_1", "d_2", "d_3"]


def test_distance_cv_corresponding(small_ethanol_water):
    """Test diagonal/corresponding strategy - pair by index."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector, "com"),
        x2=pn.VirtualAtom(x2_selector, "com"),
        prefix="d",
        pairwise="diagonal",
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethanol_water)

    # Should create 2 distances (min of 2 ethanols and 2 waters)
    expected = [
        "d_x1_0: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x1_1: COM ATOMS=10,11,12,13,14,15,16,17,18",
        "d_x2_0: COM ATOMS=19,20,21",
        "d_x2_1: COM ATOMS=22,23,24",
        "d_0: DISTANCE ATOMS=d_x1_0,d_x2_0",
        "d_1: DISTANCE ATOMS=d_x1_1,d_x2_1",
    ]
    assert plumed_str == expected
    assert labels == ["d_0", "d_1"]


def test_distance_cv_first_to_all(small_ethanol_water):
    """Test first-to-all strategy - first ethanol to all waters."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),  # First ethanol only
        x2=pn.VirtualAtom(x2_selector, "com"),  # All waters
        prefix="d",
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethanol_water)

    # Should create 2 distances (1 ethanol to 2 waters)
    expected = [
        "d_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2_0: COM ATOMS=19,20,21",
        "d_x2_1: COM ATOMS=22,23,24",
        "d_0: DISTANCE ATOMS=d_x1,d_x2_0",
        "d_1: DISTANCE ATOMS=d_x1,d_x2_1",
    ]
    assert plumed_str == expected
    assert labels == ["d_0", "d_1"]


def test_distance_cv_single_atoms():
    """Test with single atom selections using flatten."""
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    x1_selector = pn.IndexSelector(indices=[[0]])
    x2_selector = pn.IndexSelector(indices=[[1]])

    distance_cv = pn.DistanceCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="h_h",
        flatten=True,
    )

    labels, plumed_str = distance_cv.to_plumed(atoms)

    expected = ["h_h: DISTANCE ATOMS=1,2"]
    assert plumed_str == expected
    assert labels == ["h_h"]


def test_distance_cv_first_atom_reduction(small_ethanol_water):
    """Test first atom selection using selector indexing."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    # Use selector indexing to get first atom of first group
    distance_cv = pn.DistanceCV(
        x1=x1_selector[0][0],  # First atom of first ethanol
        x2=x2_selector[0][0],  # First atom of first water
        prefix="d",
        flatten=True,
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethanol_water)

    # Should use first atoms directly
    expected = ["d: DISTANCE ATOMS=1,19"]
    assert plumed_str == expected
    assert labels == ["d"]


def test_distance_cv_cog_reduction(small_ethanol_water):
    """Test center of geometry (COG) reduction."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "cog"),  # COG of first ethanol
        x2=pn.VirtualAtom(x2_selector[0], "cog"),  # COG of first water
        prefix="d",
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethanol_water)

    expected = [
        "d_x1: CENTER ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2: CENTER ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_x1,d_x2",
    ]
    assert plumed_str == expected
    assert labels == ["d"]


def test_distance_cv_no_virtual_sites(small_ethanol_water):
    """Test with flatten=True (direct atom lists, no virtual sites)."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=x1_selector[0],
        x2=x2_selector[0],
        prefix="d",
        flatten=True,  # Use atoms directly
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethanol_water)

    # Should use flattened atom lists
    expected = ["d: DISTANCE ATOMS=1,2,3,4,5,6,7,8,9,19,20,21"]
    assert plumed_str == expected
    assert labels == ["d"]


def test_get_img_with_failed_bond_determination():
    """Test that get_img does not fail when molify.ase2rdkit cannot determine bonds.

    This test verifies that get_img handles the case where bonds cannot be determined
    due to jittered positions without connectivity information.
    """
    # Create atoms from SMILES
    atoms = molify.smiles2atoms("CCO")

    # Remove connectivity so molify.ase2rdkit has to infer bonds
    atoms.info.pop("connectivity")

    # Jitter positions enough to make bond determination fail
    np.random.seed(42)
    atoms.positions += np.random.uniform(-5, 5, atoms.positions.shape)

    # Verify that molify.ase2rdkit fails with these jittered positions
    try:
        molify.ase2rdkit(atoms)
        assert False, "Expected molify.ase2rdkit to fail with jittered positions"
    except ValueError:
        pass  # Expected failure

    # Create a DistanceCV with IndexSelector
    x1_selector = pn.IndexSelector(indices=[[0, 1, 2]])
    x2_selector = pn.IndexSelector(indices=[[3, 4, 5]])

    distance_cv = pn.DistanceCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="d",
        flatten=True,
    )

    # This should NOT raise an error
    img = distance_cv.get_img(atoms)

    # Check that we get a PIL Image
    assert isinstance(img, Image.Image)

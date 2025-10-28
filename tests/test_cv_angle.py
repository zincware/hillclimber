"""Tests for AngleCV using the new API with VirtualAtom and flatten/strategy parameters."""

import ase

import hillclimber as pn


def test_angle_cv_first_strategy(small_ethanol_water):
    """Test default FIRST strategy - only first groups."""
    x1_selector = pn.SMARTSSelector(pattern="[C]")  # Carbon atoms
    x2_selector = pn.SMILESSelector(smiles="O")  # Water (oxygen)
    x3_selector = pn.SMARTSSelector(pattern="[C]")  # Carbon atoms

    # Old API: group_reduction="com", multi_group="first"
    # New API: Use VirtualAtom with "com", strategy="first" is default
    angle_cv = pn.AngleCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        x3=pn.VirtualAtom(x3_selector[0], "com"),
        prefix="angle",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethanol_water)

    expected = [
        "angle_x1: COM ATOMS=1,2",
        "angle_x2: COM ATOMS=19,20,21",
        "angle_x3: COM ATOMS=1,2",
        "angle: ANGLE ATOMS=angle_x1,angle_x2,angle_x3",
    ]
    assert plumed_str == expected
    assert labels == ["angle"]


def test_angle_cv_single_atoms():
    """Test with single atom selections."""
    # Create a simple water molecule: H-O-H
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])

    x1_selector = pn.IndexSelector(indices=[[0]])
    x2_selector = pn.IndexSelector(indices=[[1]])  # Oxygen as vertex
    x3_selector = pn.IndexSelector(indices=[[2]])

    angle_cv = pn.AngleCV(
        x1=x1_selector, x2=x2_selector, x3=x3_selector, prefix="hoh_angle"
    )

    labels, plumed_str = angle_cv.to_plumed(atoms)

    expected = ["hoh_angle: ANGLE ATOMS=1,2,3"]
    assert plumed_str == expected
    assert labels == ["hoh_angle"]


def test_angle_cv_first_atom_reduction(small_ethanol_water):
    """Test FIRST_ATOM reduction strategy using selector indexing."""
    x1_selector = pn.SMARTSSelector(pattern="[C]")
    x2_selector = pn.SMILESSelector(smiles="O")
    x3_selector = pn.SMARTSSelector(pattern="[C]")

    # Old API: group_reduction="first", multi_group="first"
    # New API: Use selector[:][0] to get first atom of each group
    angle_cv = pn.AngleCV(
        x1=x1_selector[0][0],  # First atom of first carbon group
        x2=x2_selector[0][0],  # First atom of first water
        x3=x3_selector[0][0],  # First atom of first carbon group
        prefix="angle",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethanol_water)

    expected = ["angle: ANGLE ATOMS=1,19,1"]
    assert plumed_str == expected
    assert labels == ["angle"]


def test_angle_cv_cog_reduction(small_ethanol_water):
    """Test CENTER_OF_GEOMETRY reduction."""
    x1_selector = pn.SMARTSSelector(pattern="[C]")
    x2_selector = pn.SMILESSelector(smiles="O")
    x3_selector = pn.SMARTSSelector(pattern="[C]")

    # Old API: group_reduction="cog", multi_group="first"
    # New API: Use VirtualAtom with "cog"
    angle_cv = pn.AngleCV(
        x1=pn.VirtualAtom(x1_selector[0], "cog"),
        x2=pn.VirtualAtom(x2_selector[0], "cog"),
        x3=pn.VirtualAtom(x3_selector[0], "cog"),
        prefix="angle",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethanol_water)

    expected = [
        "angle_x1: CENTER ATOMS=1,2",
        "angle_x2: CENTER ATOMS=19,20,21",
        "angle_x3: CENTER ATOMS=1,2",
        "angle: ANGLE ATOMS=angle_x1,angle_x2,angle_x3",
    ]
    assert plumed_str == expected
    assert labels == ["angle"]


def test_angle_cv_all_pairs(small_ethanol_water):
    """Test ALL strategy - all combinations."""
    # Use simpler selectors to reduce output
    x1_selector = pn.IndexSelector(indices=[[0], [3]])  # Two single atoms
    x2_selector = pn.IndexSelector(indices=[[6]])  # One vertex atom
    x3_selector = pn.IndexSelector(indices=[[0], [3]])  # Two single atoms

    # Old API: multi_group="all_pairs"
    # New API: strategy="all", flatten=False to preserve groups
    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="a",
        strategy="all",
        flatten=False,
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethanol_water)

    # Smart GROUP creation: single atoms used directly (no GROUP commands)
    expected = [
        "a_0_0_0: ANGLE ATOMS=1,7,1",
        "a_0_0_1: ANGLE ATOMS=1,7,4",
        "a_1_0_0: ANGLE ATOMS=4,7,1",
        "a_1_0_1: ANGLE ATOMS=4,7,4",
    ]
    assert plumed_str == expected
    assert labels == ["a_0_0_0", "a_0_0_1", "a_1_0_0", "a_1_0_1"]


def test_angle_cv_corresponding(small_ethanol_water):
    """Test DIAGONAL strategy - pair by index."""
    x1_selector = pn.IndexSelector(indices=[[0], [3], [6]])
    x2_selector = pn.IndexSelector(indices=[[1], [4], [7]])
    x3_selector = pn.IndexSelector(indices=[[2], [5], [8]])

    # Old API: multi_group="corresponding"
    # New API: strategy="diagonal", flatten=False to preserve groups
    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="a",
        strategy="diagonal",
        flatten=False,
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethanol_water)

    # Smart GROUP creation: single atoms used directly (no GROUP commands)
    expected = [
        "a_0_0_0: ANGLE ATOMS=1,2,3",
        "a_1_1_1: ANGLE ATOMS=4,5,6",
        "a_2_2_2: ANGLE ATOMS=7,8,9",
    ]
    assert plumed_str == expected
    assert labels == ["a_0_0_0", "a_1_1_1", "a_2_2_2"]


def test_angle_cv_first_to_all(small_ethanol_water):
    """Test one-to-many strategy - first of x1 and x2, all of x3."""
    x1_selector = pn.IndexSelector(indices=[[0]])
    x2_selector = pn.IndexSelector(indices=[[6]])
    x3_selector = pn.IndexSelector(indices=[[1], [2], [3]])

    # Use flatten=False to preserve x3 groups, automatically creates one-to-many
    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="a",
        flatten=False,
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethanol_water)

    # Smart GROUP creation: single atoms used directly (no GROUP commands)
    expected = [
        "a_0_0_0: ANGLE ATOMS=1,7,2",
        "a_0_0_1: ANGLE ATOMS=1,7,3",
        "a_0_0_2: ANGLE ATOMS=1,7,4",
    ]
    assert plumed_str == expected
    assert labels == ["a_0_0_0", "a_0_0_1", "a_0_0_2"]


def test_angle_cv_no_virtual_sites(small_ethanol_water):
    """Test with flatten=True (direct atom lists, no virtual sites)."""
    x1_selector = pn.SMARTSSelector(pattern="[C]")
    x2_selector = pn.SMILESSelector(smiles="O")
    x3_selector = pn.SMARTSSelector(pattern="[C]")

    # Old API: create_virtual_sites=False
    # New API: Use selectors directly with flatten=True (default)
    angle_cv = pn.AngleCV(
        x1=x1_selector[0],
        x2=x2_selector[0],
        x3=x3_selector[0],
        prefix="angle",
        flatten=True,
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethanol_water)

    expected = ["angle: ANGLE ATOMS=1,2,19,20,21,1,2"]
    assert plumed_str == expected
    assert labels == ["angle"]


def test_angle_cv_visualization(small_ethanol_water):
    """Test that get_img works (even without _get_atom_highlights implemented)."""
    x1_selector = pn.IndexSelector(indices=[[0]])
    x2_selector = pn.IndexSelector(indices=[[6]])
    x3_selector = pn.IndexSelector(indices=[[18]])

    angle_cv = pn.AngleCV(
        x1=x1_selector, x2=x2_selector, x3=x3_selector, prefix="angle"
    )

    # Test that get_img doesn't raise an error
    img = angle_cv.get_img(small_ethanol_water)
    assert img is not None


def test_angle_cv_empty_selection(small_ethanol_water):
    """Test that empty selection raises appropriate error."""
    x1_selector = pn.SMARTSSelector(pattern="[Cl]")  # No chlorine
    x2_selector = pn.IndexSelector(indices=[[6]])
    x3_selector = pn.IndexSelector(indices=[[18]])

    angle_cv = pn.AngleCV(
        x1=x1_selector, x2=x2_selector, x3=x3_selector, prefix="angle"
    )

    try:
        angle_cv.to_plumed(small_ethanol_water)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Empty selection" in str(e)

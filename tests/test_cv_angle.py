import ase

import hillclimber as pn


def test_angle_cv_first_strategy(small_ethnol_water):
    """Test default FIRST strategy - only first groups."""
    x1_selector = pn.SMARTSSelector(pattern="[C]")  # Carbon atoms
    x2_selector = pn.SMILESSelector(smiles="O")  # Water (oxygen)
    x3_selector = pn.SMARTSSelector(pattern="[C]")  # Carbon atoms

    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="angle",
        multi_group="first",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethnol_water)

    expected = [
        "angle_g1_0_com: COM ATOMS=1,2",
        "angle_g2_0_com: COM ATOMS=19,20,21",
        "angle_g3_0_com: COM ATOMS=1,2",
        "angle: ANGLE ATOMS=angle_g1_0_com,angle_g2_0_com,angle_g3_0_com",
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


def test_angle_cv_first_atom_reduction(small_ethnol_water):
    """Test FIRST_ATOM reduction strategy."""
    x1_selector = pn.SMARTSSelector(pattern="[C]")
    x2_selector = pn.SMILESSelector(smiles="O")
    x3_selector = pn.SMARTSSelector(pattern="[C]")

    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="angle",
        group_reduction="first",
        multi_group="first",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethnol_water)

    # Should use first atom of each group
    expected = ["angle: ANGLE ATOMS=1,19,1"]
    assert plumed_str == expected
    assert labels == ["angle"]


def test_angle_cv_cog_reduction(small_ethnol_water):
    """Test CENTER_OF_GEOMETRY reduction."""
    x1_selector = pn.SMARTSSelector(pattern="[C]")
    x2_selector = pn.SMILESSelector(smiles="O")
    x3_selector = pn.SMARTSSelector(pattern="[C]")

    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="angle",
        group_reduction="cog",
        multi_group="first",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethnol_water)

    # Should use CENTER instead of COM
    expected = [
        "angle_g1_0_cog: CENTER ATOMS=1,2",
        "angle_g2_0_cog: CENTER ATOMS=19,20,21",
        "angle_g3_0_cog: CENTER ATOMS=1,2",
        "angle: ANGLE ATOMS=angle_g1_0_cog,angle_g2_0_cog,angle_g3_0_cog",
    ]

    assert plumed_str == expected
    assert labels == ["angle"]


def test_angle_cv_all_pairs(small_ethnol_water):
    """Test ALL_PAIRS strategy - all combinations."""
    # Use simpler selectors to reduce output
    x1_selector = pn.IndexSelector(indices=[[0], [3]])  # Two single atoms
    x2_selector = pn.IndexSelector(indices=[[6]])  # One vertex atom
    x3_selector = pn.IndexSelector(indices=[[0], [3]])  # Two single atoms

    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="a",
        multi_group="all_pairs",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethnol_water)

    # Should create 2x1x2=4 angles
    expected = [
        "a_0_0_0: ANGLE ATOMS=1,7,1",
        "a_0_0_1: ANGLE ATOMS=1,7,4",
        "a_1_0_0: ANGLE ATOMS=4,7,1",
        "a_1_0_1: ANGLE ATOMS=4,7,4",
    ]

    assert plumed_str == expected
    assert labels == ["a_0_0_0", "a_0_0_1", "a_1_0_0", "a_1_0_1"]


def test_angle_cv_corresponding(small_ethnol_water):
    """Test CORRESPONDING strategy - pair by index."""
    x1_selector = pn.IndexSelector(indices=[[0], [3], [6]])
    x2_selector = pn.IndexSelector(indices=[[1], [4], [7]])
    x3_selector = pn.IndexSelector(indices=[[2], [5], [8]])

    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="a",
        multi_group="corresponding",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethnol_water)

    expected = [
        "a_0_0_0: ANGLE ATOMS=1,2,3",
        "a_1_1_1: ANGLE ATOMS=4,5,6",
        "a_2_2_2: ANGLE ATOMS=7,8,9",
    ]

    assert plumed_str == expected
    assert labels == ["a_0_0_0", "a_1_1_1", "a_2_2_2"]


def test_angle_cv_first_to_all(small_ethnol_water):
    """Test FIRST_TO_ALL strategy - first of x1 and x2, all of x3."""
    x1_selector = pn.IndexSelector(indices=[[0]])
    x2_selector = pn.IndexSelector(indices=[[6]])
    x3_selector = pn.IndexSelector(indices=[[1], [2], [3]])

    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="a",
        multi_group="first_to_all",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethnol_water)

    expected = [
        "a_0_0_0: ANGLE ATOMS=1,7,2",
        "a_0_0_1: ANGLE ATOMS=1,7,3",
        "a_0_0_2: ANGLE ATOMS=1,7,4",
    ]

    assert plumed_str == expected
    assert labels == ["a_0_0_0", "a_0_0_1", "a_0_0_2"]


def test_angle_cv_no_virtual_sites(small_ethnol_water):
    """Test with create_virtual_sites=False (should fail for multi-atom groups)."""
    x1_selector = pn.SMARTSSelector(pattern="[C]")
    x2_selector = pn.SMILESSelector(smiles="O")
    x3_selector = pn.SMARTSSelector(pattern="[C]")

    angle_cv = pn.AngleCV(
        x1=x1_selector,
        x2=x2_selector,
        x3=x3_selector,
        prefix="angle",
        create_virtual_sites=False,
        multi_group="first",
    )

    labels, plumed_str = angle_cv.to_plumed(small_ethnol_water)

    # Without virtual sites, groups are passed directly (not supported by PLUMED ANGLE)
    # This tests the code path, even if the output might not be valid PLUMED
    expected = ["angle: ANGLE ATOMS=1,2,19,20,21,1,2"]
    assert plumed_str == expected
    assert labels == ["angle"]


def test_angle_cv_visualization(small_ethnol_water):
    """Test that visualization works without errors."""
    x1_selector = pn.IndexSelector(indices=[[0]])
    x2_selector = pn.IndexSelector(indices=[[6]])
    x3_selector = pn.IndexSelector(indices=[[18]])

    angle_cv = pn.AngleCV(
        x1=x1_selector, x2=x2_selector, x3=x3_selector, prefix="angle"
    )

    # Test that get_img doesn't raise an error
    img = angle_cv.get_img(small_ethnol_water)
    assert img is not None


def test_angle_cv_empty_selection(small_ethnol_water):
    """Test that empty selection raises appropriate error."""
    x1_selector = pn.SMARTSSelector(pattern="[Cl]")  # No chlorine
    x2_selector = pn.IndexSelector(indices=[[6]])
    x3_selector = pn.IndexSelector(indices=[[18]])

    angle_cv = pn.AngleCV(
        x1=x1_selector, x2=x2_selector, x3=x3_selector, prefix="angle"
    )

    try:
        angle_cv.to_plumed(small_ethnol_water)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Empty selection" in str(e)

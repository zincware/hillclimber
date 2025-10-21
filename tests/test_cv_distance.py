import ase

import plumed_nodes as pn


def test_distance_cv_first_strategy(small_ethnol_water):
    """Test default FIRST strategy - only first groups."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=x1_selector, x2=x2_selector, prefix="d12", multi_group="first"
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethnol_water)

    expected = [
        "d12_g1_0_com: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_g2_0_com: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_g1_0_com,d12_g2_0_com",
    ]

    assert plumed_str == expected
    assert labels == ["d12"]


def test_distance_cv_all_pairs(small_ethnol_water):
    """Test ALL_PAIRS strategy - all combinations."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=x1_selector, x2=x2_selector, prefix="d", multi_group="all_pairs"
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethnol_water)

    # Should create 2x2=4 distances
    expected = [
        "d_g1_0_com: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_g1_1_com: COM ATOMS=10,11,12,13,14,15,16,17,18",
        "d_g2_0_com: COM ATOMS=19,20,21",
        "d_g2_1_com: COM ATOMS=22,23,24",
        "d_0_0: DISTANCE ATOMS=d_g1_0_com,d_g2_0_com",
        "d_0_1: DISTANCE ATOMS=d_g1_0_com,d_g2_1_com",
        "d_1_0: DISTANCE ATOMS=d_g1_1_com,d_g2_0_com",
        "d_1_1: DISTANCE ATOMS=d_g1_1_com,d_g2_1_com",
    ]

    assert plumed_str == expected
    assert labels == ["d_0_0", "d_0_1", "d_1_0", "d_1_1"]


def test_distance_cv_corresponding(small_ethnol_water):
    """Test CORRESPONDING strategy - pair by index."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=x1_selector, x2=x2_selector, prefix="d", multi_group="corresponding"
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethnol_water)

    expected = [
        "d_g1_0_com: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_g1_1_com: COM ATOMS=10,11,12,13,14,15,16,17,18",
        "d_g2_0_com: COM ATOMS=19,20,21",
        "d_g2_1_com: COM ATOMS=22,23,24",
        "d_0_0: DISTANCE ATOMS=d_g1_0_com,d_g2_0_com",
        "d_1_1: DISTANCE ATOMS=d_g1_1_com,d_g2_1_com",
    ]

    assert plumed_str == expected
    assert labels == ["d_0_0", "d_1_1"]


def test_distance_cv_first_to_all(small_ethnol_water):
    """Test FIRST_TO_ALL strategy - first of x1 to all of x2."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=x1_selector, x2=x2_selector, prefix="d", multi_group="first_to_all"
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethnol_water)

    expected = [
        "d_g1_0_com: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_g2_0_com: COM ATOMS=19,20,21",
        "d_g2_1_com: COM ATOMS=22,23,24",
        "d_0_0: DISTANCE ATOMS=d_g1_0_com,d_g2_0_com",
        "d_0_1: DISTANCE ATOMS=d_g1_0_com,d_g2_1_com",
    ]

    assert plumed_str == expected
    assert labels == ["d_0_0", "d_0_1"]


def test_distance_cv_single_atoms():
    """Test with single atom selections."""
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    x1_selector = pn.IndexSelector(indices=[[0]])
    x2_selector = pn.IndexSelector(indices=[[1]])

    distance_cv = pn.DistanceCV(x1=x1_selector, x2=x2_selector, prefix="h_h")

    labels, plumed_str = distance_cv.to_plumed(atoms)

    expected = ["h_h: DISTANCE ATOMS=1,2"]
    assert plumed_str == expected
    assert labels == ["h_h"]


def test_distance_cv_first_atom_reduction(small_ethnol_water):
    """Test FIRST_ATOM reduction strategy."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="d",
        group_reduction="first",
        multi_group="first",
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethnol_water)

    # Should use first atom of each group
    expected = ["d: DISTANCE ATOMS=1,19"]
    assert plumed_str == expected
    assert labels == ["d"]


def test_distance_cv_cog_reduction(small_ethnol_water):
    """Test CENTER_OF_GEOMETRY reduction."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="d",
        group_reduction="cog",
        multi_group="first",
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethnol_water)

    # Should use CENTER instead of COM
    expected = [
        "d_g1_0_cog: CENTER ATOMS=1,2,3,4,5,6,7,8,9",
        "d_g2_0_cog: CENTER ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_g1_0_cog,d_g2_0_cog",
    ]

    assert plumed_str == expected
    assert labels == ["d"]


def test_distance_cv_no_virtual_sites(small_ethnol_water):
    """Test with create_virtual_sites=False."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="d",
        create_virtual_sites=False,
        multi_group="first",
    )

    labels, plumed_str = distance_cv.to_plumed(small_ethnol_water)

    expected = ["d: DISTANCE ATOMS1=1,2,3,4,5,6,7,8,9 ATOMS2=19,20,21"]
    assert plumed_str == expected
    assert labels == ["d"]

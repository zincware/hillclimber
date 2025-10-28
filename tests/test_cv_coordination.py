import hillclimber as pn


def test_coordination_cv_na_water_smiles(na_cl_water):
    """Test coordination of Na+ with water COMs using new API."""
    x1_selector = pn.SMILESSelector(smiles="[Na+]")
    x2_selector = pn.SMILESSelector(smiles="O")

    # Old API: group_reduction_1="all", group_reduction_2="com_per_group"
    # New API: x1 is single atom, x2 is VirtualAtom with "com" for each water
    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector[0],  # Single Na atom
        x2=pn.VirtualAtom(x2_selector, "com"),  # COM for each water
        prefix="cn",
        r_0=0.3,
        d_0=0.0,
    )

    labels, lines = coordination_cv.to_plumed(na_cl_water)

    expected = [
        "cn_x2_0: COM ATOMS=3,4,5",
        "cn_x2_1: COM ATOMS=6,7,8",
        "cn_x2_2: COM ATOMS=9,10,11",
        "cn_x2_3: COM ATOMS=12,13,14",
        "cn_x2_4: COM ATOMS=15,16,17",
        "cn_x2_5: COM ATOMS=18,19,20",
        "cn_x2_6: COM ATOMS=21,22,23",
        "cn_x2_7: COM ATOMS=24,25,26",
        "cn_x2_8: COM ATOMS=27,28,29",
        "cn_x2_9: COM ATOMS=30,31,32",
        "cn_x2_group: GROUP ATOMS=cn_x2_0,cn_x2_1,cn_x2_2,cn_x2_3,cn_x2_4,cn_x2_5,cn_x2_6,cn_x2_7,cn_x2_8,cn_x2_9",
        "cn: COORDINATION GROUPA=1 GROUPB=cn_x2_group R_0=0.3 NN=6 D_0=0.0",
    ]
    assert lines == expected
    assert labels == ["cn"]

    # Verify atom indices are correct (plumed is 1-indexed, ase is 0-indexed)
    assert na_cl_water[0].symbol == "Na"
    assert list(na_cl_water[[2, 3, 4]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[5, 6, 7]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[8, 9, 10]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[11, 12, 13]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[14, 15, 16]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[17, 18, 19]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[20, 21, 22]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[23, 24, 25]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[26, 27, 28]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[29, 30, 31]].symbols) == ["O", "H", "H"]


def test_coordination_cv_na_water_smarts_com_per_group(na_cl_water):
    """Test coordination with SMARTS selector (oxygen only, no hydrogens)."""
    x1_selector = pn.SMILESSelector(smiles="[Na+]")
    x2_selector = pn.SMARTSSelector(pattern="[O]", hydrogens="exclude")

    # Old API: group_reduction_1="all", group_reduction_2="com_per_group"
    # New API: x1 is single atom, x2 is VirtualAtom with "com" for each oxygen
    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector[0],  # Single Na atom
        x2=pn.VirtualAtom(
            x2_selector, "com"
        ),  # COM for each oxygen (single atom groups)
        prefix="cn",
        r_0=0.3,
        d_0=0.0,
    )

    labels, lines = coordination_cv.to_plumed(na_cl_water)

    expected = [
        "cn_x2_0: COM ATOMS=3",
        "cn_x2_1: COM ATOMS=6",
        "cn_x2_2: COM ATOMS=9",
        "cn_x2_3: COM ATOMS=12",
        "cn_x2_4: COM ATOMS=15",
        "cn_x2_5: COM ATOMS=18",
        "cn_x2_6: COM ATOMS=21",
        "cn_x2_7: COM ATOMS=24",
        "cn_x2_8: COM ATOMS=27",
        "cn_x2_9: COM ATOMS=30",
        "cn_x2_group: GROUP ATOMS=cn_x2_0,cn_x2_1,cn_x2_2,cn_x2_3,cn_x2_4,cn_x2_5,cn_x2_6,cn_x2_7,cn_x2_8,cn_x2_9",
        "cn: COORDINATION GROUPA=1 GROUPB=cn_x2_group R_0=0.3 NN=6 D_0=0.0",
    ]
    assert lines == expected
    assert labels == ["cn"]

    # Verify atoms
    assert na_cl_water[0].symbol == "Na"
    assert set(na_cl_water[[2, 5, 8, 11, 14, 17, 20, 23, 26, 29]].symbols) == {"O"}


def test_coordination_cv_na_water_smarts_all(na_cl_water):
    """Test coordination with flattened atom groups (no virtual sites)."""
    x1_selector = pn.SMILESSelector(smiles="[Na+]")
    x2_selector = pn.SMARTSSelector(pattern="[O]", hydrogens="exclude")

    # Old API: group_reduction_1="all", group_reduction_2="all"
    # New API: Use selectors directly with flatten=True (default)
    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector[0],  # Single Na atom
        x2=x2_selector,  # All oxygen atoms, flattened
        prefix="cn",
        r_0=0.3,
        d_0=0.0,
        flatten=True,  # Flatten x2 groups into single group
    )

    labels, lines = coordination_cv.to_plumed(na_cl_water)

    expected = [
        "cn: COORDINATION GROUPA=1 GROUPB=3,6,9,12,15,18,21,24,27,30 R_0=0.3 NN=6 D_0=0.0"
    ]
    assert lines == expected
    assert labels == ["cn"]

    # Verify atoms
    assert na_cl_water[0].symbol == "Na"
    assert set(na_cl_water[[2, 5, 8, 11, 14, 17, 20, 23, 26, 29]].symbols) == {"O"}


def test_coordination_cv_highlights_basic(na_cl_water):
    """Test _get_atom_highlights returns correct atom indices and colors."""
    x1_selector = pn.SMILESSelector(smiles="[Na+]")
    x2_selector = pn.SMARTSSelector(pattern="[O]", hydrogens="exclude")

    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector[0],  # Single Na atom
        x2=x2_selector,  # All oxygen atoms, flattened
        prefix="cn",
        r_0=0.3,
        d_0=0.0,
        flatten=True,
    )

    highlights = coordination_cv._get_atom_highlights(na_cl_water)

    # Should highlight Na (index 0) and all O atoms
    # Na: index 0 (red)
    # O atoms: indices 2, 5, 8, 11, 14, 17, 20, 23, 26, 29 (blue)
    expected_atoms = {0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29}
    assert set(highlights.keys()) == expected_atoms

    # Verify colors
    red = (1.0, 0.2, 0.2)
    blue = (0.2, 0.2, 1.0)

    # Na should be red (from x1)
    assert highlights[0] == red

    # O atoms should be blue (from x2)
    for idx in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]:
        assert highlights[idx] == blue


def test_coordination_cv_highlights_overlap(na_cl_water):
    """Test _get_atom_highlights with overlapping selections."""
    # Select same atoms for both x1 and x2
    na_selector = pn.SMILESSelector(smiles="[Na+]")

    coordination_cv = pn.CoordinationNumberCV(
        x1=na_selector[0],  # Na atom
        x2=na_selector[0],  # Same Na atom
        prefix="cn",
        r_0=0.3,
        d_0=0.0,
    )

    highlights = coordination_cv._get_atom_highlights(na_cl_water)

    # Should highlight only Na (index 0) with purple (overlap color)
    assert set(highlights.keys()) == {0}

    # Overlapping atoms should be purple
    purple = (1.0, 0.2, 1.0)
    assert highlights[0] == purple


def test_coordination_cv_highlights_virtual_atom_skips():
    """Test that _get_atom_highlights returns None for VirtualAtom inputs."""
    import ase

    # Create simple test system
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])

    x1_selector = pn.IndexSelector(indices=[[0]])
    x2_selector = pn.IndexSelector(indices=[[1, 2]])

    # Test with VirtualAtom for x1
    coordination_cv = pn.CoordinationNumberCV(
        x1=pn.VirtualAtom(x1_selector, "com"),
        x2=x2_selector,
        prefix="cn",
        r_0=0.3,
    )

    highlights = coordination_cv._get_atom_highlights(atoms)
    assert highlights is None

    # Test with VirtualAtom for x2
    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector,
        x2=pn.VirtualAtom(x2_selector, "com"),
        prefix="cn",
        r_0=0.3,
    )

    highlights = coordination_cv._get_atom_highlights(atoms)
    assert highlights is None


def test_coordination_cv_highlights_empty_selection(na_cl_water):
    """Test _get_atom_highlights with empty selection returns None."""
    # Use fixture and select something that doesn't exist
    x1_selector = pn.SMARTSSelector(pattern="[Br]")  # No Br in the system
    x2_selector = pn.IndexSelector(indices=[[0]])

    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="cn",
        r_0=0.3,
    )

    highlights = coordination_cv._get_atom_highlights(na_cl_water)
    assert highlights is None


def test_coordination_cv_get_img(na_cl_water):
    """Test that get_img method works without errors."""
    x1_selector = pn.SMILESSelector(smiles="[Na+]")
    x2_selector = pn.SMARTSSelector(pattern="[O]", hydrogens="exclude")

    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector[0],
        x2=x2_selector,
        prefix="cn",
        r_0=0.3,
        flatten=True,
    )

    # This should not raise an error
    img = coordination_cv.get_img(na_cl_water)

    # Check that we get a PIL Image
    from PIL import Image

    assert isinstance(img, Image.Image)

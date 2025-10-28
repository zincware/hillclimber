import pytest

import hillclimber as pn


def test_distance_cv_corresponding_strategy(small_ethanol_water):
    """Test diagonal/corresponding strategy - pair by index."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector, "com"),
        x2=pn.VirtualAtom(x2_selector, "com"),
        prefix="d12",
        pairwise="diagonal",
    )

    biased_distance_cv = pn.MetadBias(
        cv=distance_cv,
        sigma=0.1,
        grid_min=0.0,
        grid_max=10.0,
        grid_bin=100,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=0.5,
        pace=150,
        biasfactor=10.0,
        temp=300.0,
        file="HILLS",
        flush=None,
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_distance_cv],
        actions=[pn.PrintAction(cvs=[distance_cv], stride=100)],
        model=None,  # type: ignore
    )

    assert meta_d_model.to_plumed(small_ethanol_water) == [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1_0: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x1_1: COM ATOMS=10,11,12,13,14,15,16,17,18",
        "d12_x2_0: COM ATOMS=19,20,21",
        "d12_x2_1: COM ATOMS=22,23,24",
        "d12_0: DISTANCE ATOMS=d12_x1_0,d12_x2_0",
        "d12_1: DISTANCE ATOMS=d12_x1_1,d12_x2_1",
        "metad: METAD ARG=d12_0,d12_1 HEIGHT=0.5 PACE=150 TEMP=300.0 FILE=HILLS BIASFACTOR=10.0 SIGMA=0.1,0.1 GRID_MIN=0.0,0.0 GRID_MAX=10.0,10.0 GRID_BIN=100,100",
        "PRINT ARG=d12_0,d12_1 STRIDE=100 FILE=COLVAR",
    ]


def test_distance_cv_first_strategy(small_ethanol_water):
    """Test default FIRST strategy - only first groups."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    biased_distance_cv = pn.MetadBias(
        cv=distance_cv,
        sigma=0.1,
        grid_min=0.0,
        grid_max=10.0,
        grid_bin=100,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=0.5,
        pace=150,
        biasfactor=10.0,
        temp=300.0,
        file="HILLS",
        flush=100,
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_distance_cv],
        actions=[pn.PrintAction(cvs=[distance_cv], stride=100)],
        model=None,  # type: ignore
    )

    assert meta_d_model.to_plumed(small_ethanol_water) == [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "metad: METAD ARG=d12 HEIGHT=0.5 PACE=150 TEMP=300.0 FILE=HILLS BIASFACTOR=10.0 SIGMA=0.1 GRID_MIN=0.0 GRID_MAX=10.0 GRID_BIN=100",
        "PRINT ARG=d12 STRIDE=100 FILE=COLVAR",
        "FLUSH STRIDE=100",
    ]


def test_duplicate_cv_prefix(small_ethanol_water):
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    biased_distance_cv = pn.MetadBias(
        cv=distance_cv,
        sigma=0.1,
        grid_min=0.0,
        grid_max=10.0,
        grid_bin=100,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=0.5,
        pace=150,
        biasfactor=10.0,
        temp=300.0,
        file="HILLS",
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_distance_cv, biased_distance_cv],  # duplicate entry
        actions=[],  # PrintCVAction is automatically added
        model=None,  # type: ignore
    )

    with pytest.raises(ValueError, match="Duplicate CV prefix found: d12"):
        meta_d_model.to_plumed(small_ethanol_water)


def test_cv_used_in_bias_and_wall(small_ethanol_water):
    """Test that a CV used in both bias_cvs and actions (wall) is not duplicated."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    # Create a distance CV
    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d",
    )

    # Use the same CV in both bias_cvs and actions
    biased_cv = pn.MetadBias(
        cv=distance_cv,
        sigma=0.2,
        grid_min=0.0,
        grid_max=5.0,
        grid_bin=300,
    )

    upper_wall = pn.UpperWallBias(
        cv=distance_cv,
        at=4.5,
        kappa=1000.0,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=1.2,
        pace=500,
        biasfactor=10.0,
        temp=300.0,
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_cv],
        actions=[upper_wall],
        model=None,  # type: ignore
    )

    result = meta_d_model.to_plumed(small_ethanol_water)

    # Check that the CV is defined only once
    cv_definitions = [line for line in result if line.startswith("d:")]
    assert len(cv_definitions) == 1, (
        f"Expected 1 CV definition, got {len(cv_definitions)}: {cv_definitions}"
    )

    # Check that both METAD and UPPER_WALLS are present
    assert any("metad: METAD" in line for line in result)
    assert any("UPPER_WALLS" in line for line in result)

    # Expected structure
    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2: COM ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_x1,d_x2",
        "metad: METAD ARG=d HEIGHT=1.2 PACE=500 TEMP=300.0 FILE=HILLS BIASFACTOR=10.0 SIGMA=0.2 GRID_MIN=0.0 GRID_MAX=5.0 GRID_BIN=300",
        "d_uwall: UPPER_WALLS ARG=d AT=4.5 KAPPA=1000.0 EXP=2",
    ]

    assert result == expected


def test_cv_conflict_detection(small_ethanol_water):
    """Test that using the same label with different CV definitions raises an error."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    # Create two different CVs with the same prefix
    distance_cv1 = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d",
    )

    distance_cv2 = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[1], "com"),  # Different group!
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d",  # Same prefix!
    )

    biased_cv = pn.MetadBias(
        cv=distance_cv1,
        sigma=0.2,
        grid_min=0.0,
        grid_max=5.0,
        grid_bin=300,
    )

    # Using a different CV with the same prefix in actions should raise an error
    upper_wall = pn.UpperWallBias(
        cv=distance_cv2,
        at=4.5,
        kappa=1000.0,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=1.2,
        pace=500,
        biasfactor=10.0,
        temp=300.0,
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_cv],
        actions=[upper_wall],
        model=None,  # type: ignore
    )

    # This should raise an error because d_x1 has different definitions
    with pytest.raises(ValueError, match="Conflicting definitions for label 'd_x1'"):
        meta_d_model.to_plumed(small_ethanol_water)


def test_adaptive_geom(small_ethanol_water):
    """Test that adaptive='GEOM' is correctly written to PLUMED output."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d",
    )

    biased_distance_cv = pn.MetadBias(
        cv=distance_cv,
        sigma=0.1,
        grid_min=0.0,
        grid_max=5.0,
        grid_bin=100,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=1.0,
        pace=500,
        biasfactor=10.0,
        temp=300.0,
        adaptive="GEOM",
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_distance_cv],
        model=None,  # type: ignore
    )

    result = meta_d_model.to_plumed(small_ethanol_water)

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2: COM ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_x1,d_x2",
        "metad: METAD ARG=d HEIGHT=1.0 PACE=500 TEMP=300.0 FILE=HILLS ADAPTIVE=GEOM BIASFACTOR=10.0 SIGMA=0.1 GRID_MIN=0.0 GRID_MAX=5.0 GRID_BIN=100",
    ]

    assert result == expected


def test_adaptive_diff(small_ethanol_water):
    """Test that adaptive='DIFF' is correctly written to PLUMED output."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d",
    )

    biased_distance_cv = pn.MetadBias(
        cv=distance_cv,
        sigma=0.1,
        grid_min=0.0,
        grid_max=5.0,
        grid_bin=100,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=1.0,
        pace=500,
        biasfactor=10.0,
        temp=300.0,
        adaptive="DIFF",
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_distance_cv],
        model=None,  # type: ignore
    )

    result = meta_d_model.to_plumed(small_ethanol_water)

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2: COM ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_x1,d_x2",
        "metad: METAD ARG=d HEIGHT=1.0 PACE=500 TEMP=300.0 FILE=HILLS ADAPTIVE=DIFF BIASFACTOR=10.0 SIGMA=0.1 GRID_MIN=0.0 GRID_MAX=5.0 GRID_BIN=100",
    ]

    assert result == expected


def test_adaptive_multiple_cvs_same_sigma(small_ethanol_water):
    """Test that adaptive with multiple CVs and same sigma writes single sigma value."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    # Create two different CVs with the same sigma
    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d",
    )

    torsion_cv = pn.TorsionCV(
        atoms=pn.IndexSelector(indices=[[0, 1, 3, 4]]),
        prefix="t",
    )

    biased_distance = pn.MetadBias(
        cv=distance_cv,
        sigma=0.2,  # Same sigma
        grid_min=0.0,
        grid_max=5.0,
        grid_bin=100,
    )

    biased_torsion = pn.MetadBias(
        cv=torsion_cv,
        sigma=0.2,  # Same sigma
        grid_min="-pi",
        grid_max="pi",
        grid_bin=100,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=1.0,
        pace=500,
        biasfactor=10.0,
        temp=300.0,
        adaptive="DIFF",
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_distance, biased_torsion],
        model=None,  # type: ignore
    )

    result = meta_d_model.to_plumed(small_ethanol_water)

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2: COM ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_x1,d_x2",
        "t: TORSION ATOMS=1,2,4,5",
        "metad: METAD ARG=d,t HEIGHT=1.0 PACE=500 TEMP=300.0 FILE=HILLS ADAPTIVE=DIFF BIASFACTOR=10.0 SIGMA=0.2 GRID_MIN=0.0,-pi GRID_MAX=5.0,pi GRID_BIN=100,100",
    ]

    assert result == expected


def test_adaptive_multiple_cvs_different_sigma_raises_error(small_ethanol_water):
    """Test that adaptive with multiple CVs and different sigmas raises an error."""
    x1_selector = pn.SMILESSelector(smiles="CCO")
    x2_selector = pn.SMILESSelector(smiles="O")

    # Create two different CVs with different sigma values
    distance_cv = pn.DistanceCV(
        x1=pn.VirtualAtom(x1_selector[0], "com"),
        x2=pn.VirtualAtom(x2_selector[0], "com"),
        prefix="d",
    )

    torsion_cv = pn.TorsionCV(
        atoms=pn.IndexSelector(indices=[[0, 1, 3, 4]]),
        prefix="t",
    )

    biased_distance = pn.MetadBias(
        cv=distance_cv,
        sigma=0.2,  # Different sigma
        grid_min=0.0,
        grid_max=5.0,
        grid_bin=100,
    )

    biased_torsion = pn.MetadBias(
        cv=torsion_cv,
        sigma=0.3,  # Different sigma!
        grid_min="-pi",
        grid_max="pi",
        grid_bin=100,
    )

    meta_d_config = pn.MetaDynamicsConfig(
        height=1.0,
        pace=500,
        biasfactor=10.0,
        temp=300.0,
        adaptive="GEOM",
    )

    meta_d_model = pn.MetaDynamicsModel(
        config=meta_d_config,
        data=small_ethanol_water,
        bias_cvs=[biased_distance, biased_torsion],
        model=None,  # type: ignore
    )

    with pytest.raises(
        ValueError,
        match="When using ADAPTIVE=GEOM, all CVs must have the same sigma value",
    ):
        meta_d_model.to_plumed(small_ethanol_water)

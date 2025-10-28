"""Tests for bias potentials (restraints and walls)."""

import hillclimber as hc


def test_restraint_basic(small_ethanol_water):
    """Test basic RESTRAINT bias on a distance CV."""
    # Define a distance CV using new API
    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(hc.SMILESSelector(smiles="CCO")[0], "com"),
        x2=hc.VirtualAtom(hc.SMILESSelector(smiles="O")[0], "com"),
        prefix="d",
    )

    # Apply a harmonic restraint
    restraint = hc.RestraintBias(cv=distance_cv, kappa=200.0, at=2.5)

    # Generate PLUMED commands
    plumed_commands = restraint.to_plumed(small_ethanol_water)

    # Expected output: CV definition + restraint
    expected = [
        "d_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2: COM ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_x1,d_x2",
        "d_restraint: RESTRAINT ARG=d KAPPA=200.0 AT=2.5",
    ]

    assert plumed_commands == expected


def test_restraint_custom_label(small_ethanol_water):
    """Test RESTRAINT with custom label."""
    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(hc.SMILESSelector(smiles="CCO")[0], "com"),
        x2=hc.VirtualAtom(hc.SMILESSelector(smiles="O")[0], "com"),
        prefix="d",
    )

    restraint = hc.RestraintBias(
        cv=distance_cv, kappa=150.0, at=3.0, label="my_restraint"
    )

    plumed_commands = restraint.to_plumed(small_ethanol_water)

    # Check custom label is used
    assert any("my_restraint: RESTRAINT" in cmd for cmd in plumed_commands)
    assert "ARG=d" in plumed_commands[-1]
    assert "KAPPA=150.0" in plumed_commands[-1]
    assert "AT=3.0" in plumed_commands[-1]


def test_upper_wall_basic(small_ethanol_water):
    """Test basic UPPER_WALLS bias on a distance CV."""
    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(hc.SMILESSelector(smiles="CCO")[0], "com"),
        x2=hc.VirtualAtom(hc.SMILESSelector(smiles="O")[0], "com"),
        prefix="d",
    )

    upper_wall = hc.UpperWallBias(cv=distance_cv, at=3.0, kappa=100.0, exp=2)

    plumed_commands = upper_wall.to_plumed(small_ethanol_water)

    # Expected output: CV definition + upper wall
    expected = [
        "d_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2: COM ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_x1,d_x2",
        "d_uwall: UPPER_WALLS ARG=d AT=3.0 KAPPA=100.0 EXP=2",
    ]

    assert plumed_commands == expected


def test_upper_wall_with_eps_offset(small_ethanol_water):
    """Test UPPER_WALLS with eps and offset parameters."""
    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(hc.SMILESSelector(smiles="CCO")[0], "com"),
        x2=hc.VirtualAtom(hc.SMILESSelector(smiles="O")[0], "com"),
        prefix="d",
    )

    upper_wall = hc.UpperWallBias(
        cv=distance_cv, at=3.0, kappa=100.0, exp=4, eps=0.1, offset=0.2
    )

    plumed_commands = upper_wall.to_plumed(small_ethanol_water)

    # Check that eps and offset are included
    wall_cmd = plumed_commands[-1]
    assert "EPS=0.1" in wall_cmd
    assert "OFFSET=0.2" in wall_cmd
    assert "EXP=4" in wall_cmd


def test_lower_wall_basic(small_ethanol_water):
    """Test basic LOWER_WALLS bias on a distance CV."""
    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(hc.SMILESSelector(smiles="CCO")[0], "com"),
        x2=hc.VirtualAtom(hc.SMILESSelector(smiles="O")[0], "com"),
        prefix="d",
    )

    lower_wall = hc.LowerWallBias(cv=distance_cv, at=1.0, kappa=100.0, exp=2)

    plumed_commands = lower_wall.to_plumed(small_ethanol_water)

    # Expected output: CV definition + lower wall
    expected = [
        "d_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d_x2: COM ATOMS=19,20,21",
        "d: DISTANCE ATOMS=d_x1,d_x2",
        "d_lwall: LOWER_WALLS ARG=d AT=1.0 KAPPA=100.0 EXP=2",
    ]

    assert plumed_commands == expected


def test_lower_wall_custom_label(small_ethanol_water):
    """Test LOWER_WALLS with custom label."""
    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(hc.SMILESSelector(smiles="CCO")[0], "com"),
        x2=hc.VirtualAtom(hc.SMILESSelector(smiles="O")[0], "com"),
        prefix="d",
    )

    lower_wall = hc.LowerWallBias(
        cv=distance_cv, at=0.5, kappa=200.0, exp=2, label="min_distance"
    )

    plumed_commands = lower_wall.to_plumed(small_ethanol_water)

    # Check custom label
    assert any("min_distance: LOWER_WALLS" in cmd for cmd in plumed_commands)


def test_restraint_with_gyration_cv(small_ethanol_water):
    """Test RESTRAINT on a gyration CV."""
    # Select ethanol molecules - use indexing to get first group
    gyration_cv = hc.RadiusOfGyrationCV(
        atoms=hc.SMILESSelector(smiles="CCO")[0],
        prefix="rg",
    )

    restraint = hc.RestraintBias(cv=gyration_cv, kappa=100.0, at=2.5)

    plumed_commands = restraint.to_plumed(small_ethanol_water)

    # Should have gyration definition + restraint
    assert any("GYRATION" in cmd for cmd in plumed_commands)
    assert any("RESTRAINT ARG=rg" in cmd for cmd in plumed_commands)
    assert "KAPPA=100.0" in plumed_commands[-1]
    assert "AT=2.5" in plumed_commands[-1]


def test_walls_with_coordination_cv(na_cl_water):
    """Test walls on a coordination number CV."""
    # Define coordination CV for Na+ with water oxygens using new API
    na_selector = hc.SMARTSSelector(pattern="[Na+]")
    water_o_selector = hc.SMARTSSelector(pattern="O")

    coord_cv = hc.CoordinationNumberCV(
        x1=na_selector[0],  # First Na atom
        x2=water_o_selector,  # All oxygen atoms, flattened by default
        prefix="cn",
        r_0=2.5,
        nn=6,
        mm=12,
    )

    # Apply walls to keep coordination between 4 and 8
    lower_wall = hc.LowerWallBias(cv=coord_cv, at=4.0, kappa=50.0, exp=2)
    upper_wall = hc.UpperWallBias(cv=coord_cv, at=8.0, kappa=50.0, exp=2)

    lower_commands = lower_wall.to_plumed(na_cl_water)
    upper_commands = upper_wall.to_plumed(na_cl_water)

    # Check that COORDINATION is defined
    assert any("COORDINATION" in cmd for cmd in lower_commands)
    assert any("LOWER_WALLS ARG=cn" in cmd for cmd in lower_commands)
    assert any("UPPER_WALLS ARG=cn" in cmd for cmd in upper_commands)


def test_multiple_cvs_with_restraints(small_ethanol_water):
    """Test multiple independent CVs with their own restraints."""
    # CV 1: Distance using new API
    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(hc.SMILESSelector(smiles="CCO")[0], "com"),
        x2=hc.VirtualAtom(hc.SMILESSelector(smiles="O")[0], "com"),
        prefix="d",
    )

    # CV 2: Gyration - use indexing to get first group
    gyration_cv = hc.RadiusOfGyrationCV(
        atoms=hc.SMILESSelector(smiles="CCO")[0],
        prefix="rg",
    )

    # Restraints
    restraint1 = hc.RestraintBias(cv=distance_cv, kappa=100.0, at=2.0)
    restraint2 = hc.RestraintBias(cv=gyration_cv, kappa=50.0, at=2.5)

    commands1 = restraint1.to_plumed(small_ethanol_water)
    commands2 = restraint2.to_plumed(small_ethanol_water)

    # Both should have their own CV definitions and restraints
    assert any("DISTANCE" in cmd for cmd in commands1)
    assert any("RESTRAINT ARG=d" in cmd for cmd in commands1)

    assert any("GYRATION" in cmd for cmd in commands2)
    assert any("RESTRAINT ARG=rg" in cmd for cmd in commands2)

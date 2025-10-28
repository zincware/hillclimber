import pytest

import hillclimber as hc


def test_opes_metad_basic_adaptive_sigma(small_ethanol_water):
    """Test basic OPES_METAD with adaptive sigma."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
        explore_mode=False,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "opes: OPES_METAD ARG=d12 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=ADAPTIVE FILE=KERNELS COMPRESSION_THRESHOLD=1.0",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_fixed_sigma(small_ethanol_water):
    """Test OPES_METAD with fixed sigma value."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma=0.2)

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "opes: OPES_METAD ARG=d12 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=0.2 FILE=KERNELS COMPRESSION_THRESHOLD=1.0",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_explore_mode(small_ethanol_water):
    """Test OPES_METAD_EXPLORE variant."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
        explore_mode=True,  # Enable exploration mode
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "opes: OPES_METAD_EXPLORE ARG=d12 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=ADAPTIVE FILE=KERNELS COMPRESSION_THRESHOLD=1.0",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_two_cvs_same_sigma(small_ethanol_water):
    """Test OPES_METAD with two CVs using same sigma."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    torsion_cv = hc.TorsionCV(
        atoms=hc.IndexSelector(indices=[[0, 1, 2, 3]]),
        prefix="phi",
    )

    opes_bias1 = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")
    opes_bias2 = hc.OPESBias(cv=torsion_cv, sigma="ADAPTIVE")

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias1, opes_bias2],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "phi: TORSION ATOMS=1,2,3,4",
        "opes: OPES_METAD ARG=d12,phi PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=ADAPTIVE FILE=KERNELS COMPRESSION_THRESHOLD=1.0",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_two_cvs_different_sigma(small_ethanol_water):
    """Test OPES_METAD with two CVs using different sigma values."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    torsion_cv = hc.TorsionCV(
        atoms=hc.IndexSelector(indices=[[0, 1, 2, 3]]),
        prefix="phi",
    )

    opes_bias1 = hc.OPESBias(cv=distance_cv, sigma=0.1)
    opes_bias2 = hc.OPESBias(cv=torsion_cv, sigma=0.2)

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias1, opes_bias2],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "phi: TORSION ATOMS=1,2,3,4",
        "opes: OPES_METAD ARG=d12,phi PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=0.1,0.2 FILE=KERNELS COMPRESSION_THRESHOLD=1.0",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_with_optional_parameters(small_ethanol_water):
    """Test OPES_METAD with optional parameters."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
        biasfactor=10.0,
        adaptive_sigma_stride=5000,
        sigma_min=0.01,
        compression_threshold=0.5,
        file="MY_KERNELS",
        calc_work=True,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "opes: OPES_METAD ARG=d12 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=ADAPTIVE FILE=MY_KERNELS COMPRESSION_THRESHOLD=0.5 BIASFACTOR=10.0 ADAPTIVE_SIGMA_STRIDE=5000 SIGMA_MIN=0.01 CALC_WORK",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_with_walkers_mpi(small_ethanol_water):
    """Test OPES_METAD with multiple walkers."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
        walkers_mpi=True,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "opes: OPES_METAD ARG=d12 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=ADAPTIVE FILE=KERNELS COMPRESSION_THRESHOLD=1.0 WALKERS_MPI",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_with_state_files(small_ethanol_water):
    """Test OPES_METAD with state file restart."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
        state_wfile="STATE",
        state_wstride=1000,
        state_rfile="OLD_STATE",
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "opes: OPES_METAD ARG=d12 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=ADAPTIVE FILE=KERNELS COMPRESSION_THRESHOLD=1.0 STATE_WFILE=STATE STATE_RFILE=OLD_STATE STATE_WSTRIDE=1000",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_with_actions(small_ethanol_water):
    """Test OPES_METAD with additional actions (restraints, walls, print)."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")

    # Create restraint and wall
    restraint = hc.RestraintBias(cv=distance_cv, kappa=100.0, at=2.5)
    upper_wall = hc.UpperWallBias(cv=distance_cv, at=5.0, kappa=50.0)

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[restraint, upper_wall],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "opes: OPES_METAD ARG=d12 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=ADAPTIVE FILE=KERNELS COMPRESSION_THRESHOLD=1.0",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "d12_restraint: RESTRAINT ARG=d12 KAPPA=100.0 AT=2.5",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "d12_uwall: UPPER_WALLS ARG=d12 AT=5.0 KAPPA=50.0 EXP=2",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_metad_with_flush(small_ethanol_water):
    """Test OPES_METAD with FLUSH parameter."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
        flush=100,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x2: COM ATOMS=19,20,21",
        "d12: DISTANCE ATOMS=d12_x1,d12_x2",
        "opes: OPES_METAD ARG=d12 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=ADAPTIVE FILE=KERNELS COMPRESSION_THRESHOLD=1.0",
        "FLUSH STRIDE=100",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected


def test_opes_duplicate_cv_prefix(small_ethanol_water):
    """Test that duplicate CV prefixes are detected."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector[0], "com"),
        x2=hc.VirtualAtom(x2_selector[0], "com"),
        prefix="d12",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma="ADAPTIVE")

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias, opes_bias],  # duplicate entry
        actions=[],
        model=None,  # type: ignore
    )

    with pytest.raises(ValueError, match="Duplicate CV prefix found: d12"):
        opes_model.to_plumed(small_ethanol_water[0])


def test_opes_metad_diagonal_strategy(small_ethanol_water):
    """Test OPES_METAD with diagonal/corresponding strategy - pair by index."""
    x1_selector = hc.SMILESSelector(smiles="CCO")
    x2_selector = hc.SMILESSelector(smiles="O")

    distance_cv = hc.DistanceCV(
        x1=hc.VirtualAtom(x1_selector, "com"),
        x2=hc.VirtualAtom(x2_selector, "com"),
        prefix="d12",
        pairwise="diagonal",
    )

    opes_bias = hc.OPESBias(cv=distance_cv, sigma=0.15)

    opes_config = hc.OPESConfig(
        barrier=40.0,
        pace=500,
        temp=300.0,
    )

    opes_model = hc.OPESModel(
        config=opes_config,
        data=small_ethanol_water,
        bias_cvs=[opes_bias],
        actions=[],
        model=None,  # type: ignore
    )

    expected = [
        "UNITS LENGTH=A TIME=0.001 ENERGY=96.48533288249877",
        "d12_x1_0: COM ATOMS=1,2,3,4,5,6,7,8,9",
        "d12_x1_1: COM ATOMS=10,11,12,13,14,15,16,17,18",
        "d12_x2_0: COM ATOMS=19,20,21",
        "d12_x2_1: COM ATOMS=22,23,24",
        "d12_0: DISTANCE ATOMS=d12_x1_0,d12_x2_0",
        "d12_1: DISTANCE ATOMS=d12_x1_1,d12_x2_1",
        "opes: OPES_METAD ARG=d12_0,d12_1 PACE=500 BARRIER=40.0 TEMP=300.0 SIGMA=0.15 FILE=KERNELS COMPRESSION_THRESHOLD=1.0",
    ]

    assert opes_model.to_plumed(small_ethanol_water) == expected

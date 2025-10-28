"""Tests for new DistanceCV functionality with flatten and pairwise parameters."""

import ase
import pytest

import hillclimber as hc


@pytest.fixture
def test_atoms():
    """Create test atoms for testing."""
    return ase.Atoms()


@pytest.fixture
def water_sel():
    """Water selector with 3 groups."""
    return hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])


@pytest.fixture
def ethanol_sel():
    """Ethanol selector with 2 groups."""
    return hc.IndexSelector(indices=[[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])


def test_distance_with_virtual_atom_one_to_one(test_atoms, water_sel, ethanol_sel):
    """Test distance between two specific COMs."""
    cv = hc.DistanceCV(
        x1=hc.VirtualAtom(water_sel[0], "com"),
        x2=hc.VirtualAtom(ethanol_sel[0], "com"),
        prefix="d",
    )

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 1 CV
    assert len(labels) == 1
    assert labels[0] == "d"

    # Should have 2 COM commands + 1 DISTANCE command
    assert len(commands) == 3
    assert any("COM ATOMS=1,2,3" in cmd for cmd in commands)
    assert any("COM ATOMS=11,12,13,14,15" in cmd for cmd in commands)
    assert any("DISTANCE ATOMS=" in cmd and "d:" in cmd for cmd in commands)


def test_distance_with_virtual_atom_one_to_many(test_atoms, water_sel, ethanol_sel):
    """Test distance from one COM to multiple COMs."""
    cv = hc.DistanceCV(
        x1=hc.VirtualAtom(ethanol_sel[0], "com"),
        x2=hc.VirtualAtom(water_sel, "com"),  # 3 water COMs
        prefix="d",
    )

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 3 CVs (one ethanol to each of 3 waters)
    assert len(labels) == 3
    assert labels == ["d_0", "d_1", "d_2"]

    # Should have 1 ethanol COM + 3 water COMs + 3 DISTANCE commands = 7 total
    assert len(commands) == 7


def test_distance_with_virtual_atom_pairwise_all(test_atoms, water_sel, ethanol_sel):
    """Test all pairwise distances (explosion!)."""
    cv = hc.DistanceCV(
        x1=hc.VirtualAtom(water_sel, "com"),  # 3 waters
        x2=hc.VirtualAtom(ethanol_sel, "com"),  # 2 ethanols
        prefix="d",
        pairwise="all",
    )

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 3×2 = 6 CVs
    assert len(labels) == 6
    # Check some labels
    assert "d_0" in labels
    assert "d_5" in labels


def test_distance_with_virtual_atom_pairwise_diagonal(
    test_atoms, water_sel, ethanol_sel
):
    """Test diagonal pairing (avoid explosion)."""
    cv = hc.DistanceCV(
        x1=hc.VirtualAtom(water_sel, "com"),  # 3 waters
        x2=hc.VirtualAtom(ethanol_sel, "com"),  # 2 ethanols
        prefix="d",
        pairwise="diagonal",
    )

    labels, commands = cv.to_plumed(test_atoms)

    # Should create min(3,2) = 2 CVs
    assert len(labels) == 2
    assert labels == ["d_0", "d_1"]


def test_distance_with_virtual_atom_pairwise_none_error(
    test_atoms, water_sel, ethanol_sel
):
    """Test that pairwise='none' raises error for many-to-many."""
    cv = hc.DistanceCV(
        x1=hc.VirtualAtom(water_sel, "com"),  # 3 waters
        x2=hc.VirtualAtom(ethanol_sel, "com"),  # 2 ethanols
        prefix="d",
        pairwise="none",
    )

    with pytest.raises(ValueError, match="Both x1 and x2 have multiple groups"):
        cv.to_plumed(test_atoms)


def test_distance_selector_flatten_true(test_atoms, water_sel, ethanol_sel):
    """Test AtomSelector with flatten=True (default)."""
    cv = hc.DistanceCV(x1=water_sel[0], x2=ethanol_sel[0], prefix="d", flatten=True)

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 1 CV
    assert len(labels) == 1
    assert labels[0] == "d"

    # Should have 1 DISTANCE command with flattened atoms
    assert len(commands) == 1
    assert "d: DISTANCE ATOMS=1,2,3,11,12,13,14,15" in commands[0]


def test_distance_selector_flatten_false(test_atoms, water_sel, ethanol_sel):
    """Test AtomSelector with flatten=False (create GROUPs)."""
    cv = hc.DistanceCV(x1=water_sel[0], x2=ethanol_sel[0], prefix="d", flatten=False)

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 1 CV
    assert len(labels) == 1

    # Should have 2 GROUP commands + 1 DISTANCE command
    assert len(commands) == 3
    assert any("d_x1_g0: GROUP ATOMS=1,2,3" in cmd for cmd in commands)
    assert any("d_x2_g0: GROUP ATOMS=11,12,13,14,15" in cmd for cmd in commands)
    assert any("d: DISTANCE ATOMS=d_x1_g0,d_x2_g0" in cmd for cmd in commands)


def test_distance_mixed_virtual_atom_and_selector(test_atoms, water_sel, ethanol_sel):
    """Test mixing VirtualAtom and AtomSelector."""
    cv = hc.DistanceCV(
        x1=hc.VirtualAtom(water_sel[0], "com"),
        x2=ethanol_sel[0],  # Raw selector (will be flattened)
        prefix="d",
        flatten=True,
    )

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 1 CV
    assert len(labels) == 1

    # Should have 1 COM command + 1 DISTANCE command
    assert len(commands) == 2
    assert any("COM ATOMS=1,2,3" in cmd for cmd in commands)
    assert any("DISTANCE ATOMS=" in cmd for cmd in commands)


def test_distance_combined_virtual_atoms(test_atoms):
    """Test using combined VirtualAtoms."""
    sel1 = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])
    sel2 = hc.IndexSelector(indices=[[10, 11, 12]])

    va1 = hc.VirtualAtom(sel1, "com")
    va2 = hc.VirtualAtom(sel2, "com")
    combined = va1 + va2  # 3 COMs total

    cv = hc.DistanceCV(
        x1=hc.VirtualAtom(sel1[0], "com"),  # 1 COM
        x2=combined,  # 3 COMs
        prefix="d",
    )

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 3 CVs (1 to 3)
    assert len(labels) == 3


def test_distance_nested_virtual_atoms(test_atoms):
    """Test using nested VirtualAtoms (COM of COMs)."""
    sel = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # Create COM for each group
    coms = hc.VirtualAtom(sel, "com")
    # Create COM of those COMs
    center = hc.VirtualAtom(coms, "com")

    cv = hc.DistanceCV(x1=hc.VirtualAtom(sel[0], "com"), x2=center, prefix="d")

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 1 CV
    assert len(labels) == 1

    # Commands should include individual COMs and center COM
    assert len([cmd for cmd in commands if "COM" in cmd]) >= 4


def test_distance_selector_indexing(test_atoms):
    """Test that selector indexing works with new DistanceCV."""
    sel = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # Use selector[:][0] to get first atom of each group
    cv = hc.DistanceCV(
        x1=sel[0][0],  # First atom of first group
        x2=sel[1][0],  # First atom of second group
        prefix="d",
    )

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 1 CV
    assert len(labels) == 1

    # Should use flattened atoms
    assert len(commands) == 1
    assert "DISTANCE ATOMS=1,4" in commands[0]


def test_distance_cog_virtual_atoms(test_atoms, water_sel):
    """Test using COG instead of COM."""
    cv = hc.DistanceCV(
        x1=hc.VirtualAtom(water_sel[0], "cog"),
        x2=hc.VirtualAtom(water_sel[1], "cog"),
        prefix="d",
    )

    labels, commands = cv.to_plumed(test_atoms)

    # Should create 1 CV
    assert len(labels) == 1

    # Should have CENTER commands (not COM)
    assert any("CENTER ATOMS=" in cmd for cmd in commands)
    assert not any("COM ATOMS=" in cmd and "CENTER" not in cmd for cmd in commands)

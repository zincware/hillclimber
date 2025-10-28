"""Tests for VirtualAtom functionality including combination and nesting."""

import ase

import hillclimber as hc


def test_virtual_atom_combination():
    """Test combining VirtualAtoms with +."""
    atoms = ase.Atoms()
    sel1 = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])
    sel2 = hc.IndexSelector(indices=[[6, 7, 8], [9, 10, 11]])

    # Create VirtualAtoms
    va1 = hc.VirtualAtom(sel1, "com")
    va2 = hc.VirtualAtom(sel2, "com")

    # Combine them
    combined = va1 + va2
    assert combined.reduction == "com"

    # Check that it creates the right number of virtual sites
    assert combined.count(atoms) == 4  # 2 from va1 + 2 from va2


def test_virtual_atom_combination_indexed():
    """Test combining indexed VirtualAtoms."""
    atoms = ase.Atoms()
    sel1 = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])
    sel2 = hc.IndexSelector(indices=[[6, 7, 8], [9, 10, 11]])

    # Create indexed VirtualAtoms using selector indexing (preferred)
    va1 = hc.VirtualAtom(sel1[0], "com")  # First group only
    va2 = hc.VirtualAtom(sel2[0], "com")  # First group only

    # Combine them
    combined = va1 + va2
    assert combined.count(atoms) == 2  # 1 from va1 + 1 from va2


def test_virtual_atom_combination_different_reductions_error():
    """Test that combining VirtualAtoms with different reductions raises error."""
    atoms = ase.Atoms()
    sel1 = hc.IndexSelector(indices=[[0, 1, 2]])
    sel2 = hc.IndexSelector(indices=[[3, 4, 5]])

    va1 = hc.VirtualAtom(sel1, "com")
    va2 = hc.VirtualAtom(sel2, "cog")

    try:
        combined = va1 + va2
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "different reductions" in str(e).lower()


def test_virtual_atom_nesting():
    """Test nested VirtualAtoms (COM of COMs)."""
    atoms = ase.Atoms()
    sel = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # Create COM for each group
    coms = hc.VirtualAtom(sel, "com")
    assert coms.count(atoms) == 3

    # Create COM of those COMs
    center = hc.VirtualAtom(coms, "com")
    assert center.count(atoms) == 1

    # Check PLUMED generation
    labels, commands = center.to_plumed(atoms)
    assert len(labels) == 1
    # Should have 3 COM commands for inner + 1 COM command for center
    assert len(commands) == 4


def test_virtual_atom_nesting_with_flatten():
    """Test nested VirtualAtoms using flatten."""
    atoms = ase.Atoms()
    sel = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # Create COM for each group
    coms = hc.VirtualAtom(sel, "com")

    # Flatten to create single COM of all COMs
    center = hc.VirtualAtom(coms, "flatten")
    assert center.count(atoms) == 1


def test_virtual_atom_to_plumed_basic():
    """Test basic PLUMED generation."""
    atoms = ase.Atoms()
    sel = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])

    va = hc.VirtualAtom(sel, "com", label="test_com")
    labels, commands = va.to_plumed(atoms)

    assert len(labels) == 2
    assert len(commands) == 2
    assert "test_com_0: COM ATOMS=1,2,3" in commands
    assert "test_com_1: COM ATOMS=4,5,6" in commands


def test_virtual_atom_to_plumed_cog():
    """Test PLUMED generation with COG."""
    atoms = ase.Atoms()
    sel = hc.IndexSelector(indices=[[0, 1, 2]])

    va = hc.VirtualAtom(sel, "cog", label="test_cog")
    labels, commands = va.to_plumed(atoms)

    assert len(labels) == 1
    assert len(commands) == 1
    assert "test_cog: CENTER ATOMS=1,2,3" in commands


def test_virtual_atom_to_plumed_nested():
    """Test PLUMED generation with nested VirtualAtoms."""
    atoms = ase.Atoms()
    sel = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])

    # Inner: COM of each group
    inner = hc.VirtualAtom(sel, "com", label="water_com")
    # Outer: COM of those COMs
    outer = hc.VirtualAtom(inner, "com", label="center")

    labels, commands = outer.to_plumed(atoms)

    # Should have 2 inner COMs + 1 outer COM
    assert len(labels) == 1  # Only the final center label
    assert len(commands) == 3  # 2 inner + 1 outer

    # Check commands
    assert any("water_com_0: COM ATOMS=1,2,3" in cmd for cmd in commands)
    assert any("water_com_1: COM ATOMS=4,5,6" in cmd for cmd in commands)
    assert any("center: COM ATOMS=water_com_0,water_com_1" in cmd for cmd in commands)


def test_virtual_atom_with_first_reduction():
    """Test VirtualAtom with 'first' reduction (no virtual site needed)."""
    atoms = ase.Atoms()
    sel = hc.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])

    va = hc.VirtualAtom(sel, "first")
    labels, commands = va.to_plumed(atoms)

    # Should return atom indices directly, no virtual site commands
    assert len(labels) == 2
    assert len(commands) == 0
    assert labels == [
        "1",
        "4",
    ]  # First atoms (0-indexed in Python, 1-indexed in PLUMED)


def test_virtual_atom_combination_to_plumed():
    """Test PLUMED generation with combined VirtualAtoms."""
    atoms = ase.Atoms()
    sel1 = hc.IndexSelector(indices=[[0, 1, 2]])
    sel2 = hc.IndexSelector(indices=[[3, 4, 5]])

    va1 = hc.VirtualAtom(sel1, "com", label="mol1")
    va2 = hc.VirtualAtom(sel2, "com", label="mol2")

    combined = va1 + va2
    labels, commands = combined.to_plumed(atoms)

    assert len(labels) == 2
    assert len(commands) == 2

import ase

import hillclimber as pn


def test_index_selector(ethanol_water):
    indices = [[0], [1], [2]]
    selector = pn.IndexSelector(indices=indices)
    assert selector.select(ethanol_water) == [[0], [1], [2]]


def test_index_selector_grouped():
    """Test IndexSelector with grouped indices for molecules."""
    # Two molecules: one with atoms [0, 1], another with atoms [2, 3]
    indices = [[0, 1], [2, 3]]
    selector = pn.IndexSelector(indices=indices)
    # Mock atoms object (select doesn't use it for IndexSelector)
    atoms = ase.Atoms()
    assert selector.select(atoms) == [[0, 1], [2, 3]]


def test_smiles_selector(small_ethanol_water):
    # ethanol
    selector = pn.SMILESSelector(smiles="CCO")
    selected_indices = selector.select(small_ethanol_water)
    assert selected_indices == [list(range(9)), list(range(9, 18))]
    # water
    selector = pn.SMILESSelector(smiles="O")
    selected_indices = selector.select(small_ethanol_water)
    assert selected_indices == [[18, 19, 20], [21, 22, 23]]


# --- Tests for two-level indexing ---


def test_selector_group_indexing():
    """Test group-level indexing on selectors."""
    atoms = ase.Atoms()
    # Create a selector with 3 groups
    selector = pn.IndexSelector(indices=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # Test single group indexing
    assert selector[0].select(atoms) == [[0, 1, 2]]
    assert selector[1].select(atoms) == [[3, 4, 5]]
    assert selector[2].select(atoms) == [[6, 7, 8]]

    # Test negative indexing
    assert selector[-1].select(atoms) == [[6, 7, 8]]
    assert selector[-2].select(atoms) == [[3, 4, 5]]

    # Test slice indexing
    assert selector[0:2].select(atoms) == [[0, 1, 2], [3, 4, 5]]
    assert selector[1:].select(atoms) == [[3, 4, 5], [6, 7, 8]]
    assert selector[:2].select(atoms) == [[0, 1, 2], [3, 4, 5]]
    assert selector[::2].select(atoms) == [[0, 1, 2], [6, 7, 8]]

    # Test list indexing
    assert selector[[0, 2]].select(atoms) == [[0, 1, 2], [6, 7, 8]]
    assert selector[[1, 2]].select(atoms) == [[3, 4, 5], [6, 7, 8]]


def test_selector_atom_indexing():
    """Test atom-level indexing (second level)."""
    atoms = ase.Atoms()
    # Create a selector with 3 groups, each with 3 atoms
    selector = pn.IndexSelector(indices=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # Test single atom indexing on first group
    assert selector[0][0].select(atoms) == [[0]]
    assert selector[0][1].select(atoms) == [[1]]
    assert selector[0][2].select(atoms) == [[2]]

    # Test negative atom indexing
    assert selector[0][-1].select(atoms) == [[2]]
    assert selector[0][-2].select(atoms) == [[1]]

    # Test atom slice indexing
    assert selector[0][0:2].select(atoms) == [[0, 1]]
    assert selector[0][1:].select(atoms) == [[1, 2]]

    # Test atom list indexing
    assert selector[0][[0, 2]].select(atoms) == [[0, 2]]

    # Test atom indexing on multiple groups
    assert selector[0:2][0].select(atoms) == [
        [0],
        [3],
    ]  # First atom of first two groups
    assert selector[:][-1].select(atoms) == [[2], [5], [8]]  # Last atom of all groups
    assert selector[:][0:2].select(atoms) == [
        [0, 1],
        [3, 4],
        [6, 7],
    ]  # First two atoms of all groups


def test_selector_three_level_indexing_error():
    """Test that three-level indexing raises an error."""
    atoms = ase.Atoms()
    selector = pn.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])

    # This should raise an error
    try:
        selector[0][0][0].select(atoms)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot index beyond 2 levels" in str(e)


def test_selector_combination():
    """Test combining selectors with +."""
    atoms = ase.Atoms()
    sel1 = pn.IndexSelector(indices=[[0, 1], [2, 3]])
    sel2 = pn.IndexSelector(indices=[[4, 5], [6, 7]])

    # Combine two selectors
    combined = sel1 + sel2
    assert combined.select(atoms) == [[0, 1], [2, 3], [4, 5], [6, 7]]

    # Test that combined selector can be indexed
    assert combined[0].select(atoms) == [[0, 1]]
    assert combined[2].select(atoms) == [[4, 5]]
    assert combined[0:2].select(atoms) == [[0, 1], [2, 3]]


def test_selector_combination_chaining():
    """Test chaining multiple selector combinations."""
    atoms = ase.Atoms()
    sel1 = pn.IndexSelector(indices=[[0, 1]])
    sel2 = pn.IndexSelector(indices=[[2, 3]])
    sel3 = pn.IndexSelector(indices=[[4, 5]])

    # Chain three selectors
    combined = sel1 + sel2 + sel3
    assert combined.select(atoms) == [[0, 1], [2, 3], [4, 5]]


def test_selector_combination_with_indexed():
    """Test combining indexed selectors."""
    atoms = ase.Atoms()
    sel1 = pn.IndexSelector(indices=[[0, 1, 2], [3, 4, 5]])
    sel2 = pn.IndexSelector(indices=[[6, 7, 8], [9, 10, 11]])

    # Combine indexed selectors
    combined = sel1[0] + sel2[0]
    assert combined.select(atoms) == [[0, 1, 2], [6, 7, 8]]

    # Can also combine with atom-level indexing
    combined2 = sel1[0][0] + sel2[0][0]
    assert combined2.select(atoms) == [[0], [6]]


def test_selector_with_smarts(small_ethanol_water):
    """Test indexing with SMARTS selectors."""
    # Use [OH2] pattern to select only water (not ethanol oxygen)
    water_sel = pn.SMARTSSelector("[OH2]", hydrogens="include")

    # Test basic selection
    waters = water_sel.select(small_ethanol_water)
    assert len(waters) == 2  # 2 water molecules

    # Test indexing
    first_water = water_sel[0].select(small_ethanol_water)
    assert len(first_water) == 1  # Only one group
    assert len(first_water[0]) == 3  # O + 2H

    # Test combination
    ethanol_sel = pn.SMARTSSelector("CCO")
    combined = water_sel + ethanol_sel
    result = combined.select(small_ethanol_water)
    assert len(result) == 4  # 2 waters + 2 ethanols

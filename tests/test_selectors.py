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
    import ase
    atoms = ase.Atoms()
    assert selector.select(atoms) == [[0, 1], [2, 3]]


def test_smiles_selector(small_ethnol_water):
    # ethanol
    selector = pn.SMILESSelector(smiles="CCO")
    selected_indices = selector.select(small_ethnol_water)
    assert selected_indices == [list(range(9)), list(range(9, 18))]
    # water
    selector = pn.SMILESSelector(smiles="O")
    selected_indices = selector.select(small_ethnol_water)
    assert selected_indices == [[18, 19, 20], [21, 22, 23]]

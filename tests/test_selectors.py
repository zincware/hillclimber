import plumed_nodes as pn


def test_index_selector(ethanol_water):
    indices = [0, 1, 2]
    selector = pn.IndexSelector(indices=indices)
    assert selector.select(ethanol_water) == [[0], [1], [2]]


def test_smiles_selector(small_ethnol_water):
    # ethanol
    selector = pn.SMILESSelector(smiles="CCO")
    selected_indices = selector.select(small_ethnol_water)
    assert selected_indices == [list(range(9)), list(range(9, 18))]
    # water
    selector = pn.SMILESSelector(smiles="O")
    selected_indices = selector.select(small_ethnol_water)
    assert selected_indices == [[18, 19, 20], [21, 22, 23]]

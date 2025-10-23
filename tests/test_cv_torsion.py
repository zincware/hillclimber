import rdkit2ase
import pytest
import hillclimber as pn


def test_torsion_cv_with_peptide():
    """Test TORSION CV with peptide molecule and SMARTS selections."""
    # Create the peptide molecule
    atoms = rdkit2ase.smiles2atoms("CC(=O)NC(C)C(=O)NC")
    selector1 = pn.SMARTSSelector(pattern="CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C")
    
    torsion_cv1 = pn.TorsionCV(
        atoms=selector1, 
        prefix="phi"
    )
    
    labels1, plumed_str1 = torsion_cv1.to_plumed(atoms)
    
    # Check that we get exactly one torsion
    assert labels1 == ["phi"]
    assert plumed_str1 == [
        "phi: TORSION ATOMS=5,6,7,9"
    ]

def test_torsion_cv_validation_error():
    """Test that TorsionCV raises error for wrong number of atoms."""
    atoms = rdkit2ase.smiles2atoms("CC")

    # Create selector with wrong number of atoms (only 2)
    selector = pn.IndexSelector(indices=[[0, 1]])

    torsion_cv = pn.TorsionCV(
        atoms=selector,
        prefix="bad_tor"
    )

    with pytest.raises(ValueError):
        labels, plumed_str = torsion_cv.to_plumed(atoms)


def test_torsion_cv_strategy_all():
    """Test TorsionCV with strategy='all' to process multiple torsions."""
    # Create a molecule with multiple torsion angles
    atoms = rdkit2ase.smiles2atoms("CC(C)CC")

    # Create selector that returns multiple 4-atom groups
    # Each group represents a different torsion angle
    selector = pn.IndexSelector(indices=[
        [0, 1, 2, 3],  # First torsion
        [1, 2, 3, 4],  # Second torsion
    ])

    torsion_cv = pn.TorsionCV(
        atoms=selector,
        prefix="tor",
        strategy="all"
    )

    labels, plumed_str = torsion_cv.to_plumed(atoms)

    # Should create two torsion CVs
    assert labels == ["tor_0", "tor_1"]
    expected = [
        "tor_0: TORSION ATOMS=1,2,3,4",
        "tor_1: TORSION ATOMS=2,3,4,5"
    ]
    assert plumed_str == expected

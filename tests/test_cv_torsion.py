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

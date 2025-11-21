import molify
import pytest

import hillclimber as pn


def test_torsion_cv_with_peptide():
    """Test TORSION CV with peptide molecule and SMARTS selections."""
    # Create the peptide molecule
    atoms = molify.smiles2atoms("CC(=O)NC(C)C(=O)NC")
    selector1 = pn.SMARTSSelector(pattern="CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C")

    torsion_cv1 = pn.TorsionCV(atoms=selector1, prefix="phi")

    labels1, plumed_str1 = torsion_cv1.to_plumed(atoms)

    # Check that we get exactly one torsion
    assert labels1 == ["phi"]
    assert plumed_str1 == ["phi: TORSION ATOMS=5,6,7,9"]


def test_torsion_cv_validation_error():
    """Test that TorsionCV raises error for wrong number of atoms."""
    atoms = molify.smiles2atoms("CC")

    # Create selector with wrong number of atoms (only 2)
    selector = pn.IndexSelector(indices=[[0, 1]])

    torsion_cv = pn.TorsionCV(atoms=selector, prefix="bad_tor")

    with pytest.raises(ValueError):
        labels, plumed_str = torsion_cv.to_plumed(atoms)


def test_torsion_cv_strategy_all():
    """Test TorsionCV with strategy='all' to process multiple torsions."""
    # Create a molecule with multiple torsion angles
    atoms = molify.smiles2atoms("CC(C)CC")

    # Create selector that returns multiple 4-atom groups
    # Each group represents a different torsion angle
    selector = pn.IndexSelector(
        indices=[
            [0, 1, 2, 3],  # First torsion
            [1, 2, 3, 4],  # Second torsion
        ]
    )

    torsion_cv = pn.TorsionCV(atoms=selector, prefix="tor", strategy="all")

    labels, plumed_str = torsion_cv.to_plumed(atoms)

    # Should create two torsion CVs
    assert labels == ["tor_0", "tor_1"]
    expected = ["tor_0: TORSION ATOMS=1,2,3,4", "tor_1: TORSION ATOMS=2,3,4,5"]
    assert plumed_str == expected


def test_torsion_cv_alanine_dipeptide_phi_psi():
    """
    Integration test for phi and psi torsion angles in alanine dipeptide.

    This tests the classical backbone torsion angles:
    - Phi: C(=O) - N - CA - C(=O) dihedral
    - Psi: N - CA - C(=O) - N dihedral

    Molecule: Ace-Ala-NMe (CC(=O)-NH-CH(CH3)-CO-NH-CH3)
    """
    # Create alanine dipeptide
    atoms = molify.smiles2atoms("CC(=O)NC(C)C(=O)NC")

    # Phi angle: C(carbonyl) - N - CA - C(carbonyl)
    # Correct SMARTS pattern for phi
    phi_selector = pn.SMARTSSelector(pattern="[C:1](=O)[N:2][C:3][C:4](=O)")

    phi_cv = pn.TorsionCV(atoms=phi_selector, prefix="phi")

    phi_labels, phi_plumed = phi_cv.to_plumed(atoms)

    # Phi should select atoms: C(1), N(3), CA(4), C(6)
    assert phi_labels == ["phi"]
    expected_phi = [
        "phi: TORSION ATOMS=2,4,5,7"  # 1-indexed: atoms 1,3,4,6 -> 2,4,5,7
    ]
    assert phi_plumed == expected_phi

    # Psi angle: N - CA - C(carbonyl) - N
    # Correct SMARTS pattern for psi
    psi_selector = pn.SMARTSSelector(pattern="C(=O)[N:1][C:2][C:3](=O)[N:4]")

    psi_cv = pn.TorsionCV(atoms=psi_selector, prefix="psi")

    psi_labels, psi_plumed = psi_cv.to_plumed(atoms)

    # Psi should select atoms: N(3), CA(4), C(6), N(8)
    assert psi_labels == ["psi"]
    expected_psi = [
        "psi: TORSION ATOMS=4,5,7,9"  # 1-indexed: atoms 3,4,6,8 -> 4,5,7,9
    ]
    assert psi_plumed == expected_psi

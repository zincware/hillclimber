import ase
import pytest

import plumed_nodes as pn


def test_gyration_cv_basic(small_ethnol_water):
    """Test basic RadiusOfGyrationCV functionality."""
    # Select ethanol molecules
    ethanol_selector = pn.SMILESSelector(smiles="CCO")
    
    gyration_cv = pn.RadiusOfGyrationCV(
        atoms=ethanol_selector,
        prefix="rg"
    )
    
    labels, plumed_str = gyration_cv.to_plumed(small_ethnol_water)
    
    # Should use first ethanol molecule (atoms 1-9)
    expected = [
        "rg: GYRATION ATOMS=1,2,3,4,5,6,7,8,9"
    ]
    
    assert plumed_str == expected
    assert labels == ["rg"]


def test_gyration_cv_with_type(small_ethnol_water):
    """Test RadiusOfGyrationCV with specific TYPE parameter."""
    ethanol_selector = pn.SMILESSelector(smiles="CCO")
    
    gyration_cv = pn.RadiusOfGyrationCV(
        atoms=ethanol_selector,
        prefix="rg_asp",
        type="ASPHERICITY"
    )
    
    labels, plumed_str = gyration_cv.to_plumed(small_ethnol_water)
    
    expected = [
        "rg_asp: GYRATION ATOMS=1,2,3,4,5,6,7,8,9 TYPE=ASPHERICITY"
    ]
    
    assert plumed_str == expected
    assert labels == ["rg_asp"]


# def test_gyration_cv_all_groups(small_ethnol_water):
#     """Test RadiusOfGyrationCV with all groups (both ethanol molecules)."""
#     ethanol_selector = pn.SMILESSelector(smiles="CCO")
    
#     gyration_cv = pn.RadiusOfGyrationCV(
#         atoms=ethanol_selector,
#         prefix="rg",
#         multi_group="all_pairs"  # Process all groups independently
#     )
    
#     labels, plumed_str = gyration_cv.to_plumed(small_ethnol_water)
    
#     # Should create two separate gyration CVs for each ethanol molecule
#     expected = [
#         "rg_0: GYRATION ATOMS=1,2,3,4,5,6,7,8,9",
#         "rg_1: GYRATION ATOMS=10,11,12,13,14,15,16,17,18"
#     ]
    
#     assert plumed_str == expected
#     assert labels == ["rg_0", "rg_1"]


# def test_gyration_cv_single_molecule():
#     """Test RadiusOfGyrationCV with a simple molecule."""
#     # Create a simple butane molecule  
#     import rdkit2ase
    
#     # Use SMILES to create butane properly
#     butane = rdkit2ase.smiles2conformers("CCCC", numConfs=1)
    
#     # Select all carbon atoms in butane
#     carbon_selector = pn.SMARTSSelector(pattern="[#6]")  # All carbons
    
#     gyration_cv = pn.RadiusOfGyrationCV(
#         atoms=carbon_selector,
#         prefix="butane_rg"
#     )
    
#     labels, plumed_str = gyration_cv.to_plumed(butane)
    
#     # Should select the 4 carbon atoms (1-4 in PLUMED 1-based indexing)
#     expected = [
#         "butane_rg: GYRATION ATOMS=1,2,3,4"
#     ]
    
#     assert plumed_str == expected
#     assert labels == ["butane_rg"]


# def test_gyration_cv_only_heavy_atoms():
#     """Test RadiusOfGyrationCV selecting only heavy atoms."""
#     import rdkit2ase
    
#     # Create ethylene glycol using SMILES
#     ethylene_glycol = rdkit2ase.smiles2conformers("OCCO", numConfs=1)
    
#     # Select only heavy atoms (non-hydrogen)
#     heavy_atoms_selector = pn.SMARTSSelector(pattern="[!H]")  # Not hydrogen
    
#     gyration_cv = pn.RadiusOfGyrationCV(
#         atoms=heavy_atoms_selector,
#         prefix="rg_heavy"
#     )
    
#     labels, plumed_str = gyration_cv.to_plumed(ethylene_glycol)
    
#     # Ethylene glycol heavy atoms: O-C-C-O (4 heavy atoms)
#     expected = [
#         "rg_heavy: GYRATION ATOMS=1,2,3,4"
#     ]
    
#     assert plumed_str == expected
#     assert labels == ["rg_heavy"]


# def test_gyration_cv_with_gtpc_types(small_ethnol_water):
#     """Test RadiusOfGyrationCV with different gyration tensor principal components."""
#     ethanol_selector = pn.SMILESSelector(smiles="CCO")
    
#     # Test different GTPC types
#     for i, gtpc_type in enumerate(["GTPC_1", "GTPC_2", "GTPC_3"], 1):
#         gyration_cv = pn.RadiusOfGyrationCV(
#             atoms=ethanol_selector,
#             prefix=f"rg_gtpc{i}",
#             type=gtpc_type
#         )
        
#         labels, plumed_str = gyration_cv.to_plumed(small_ethnol_water)
        
#         expected = [
#             f"rg_gtpc{i}: GYRATION ATOMS=1,2,3,4,5,6,7,8,9 TYPE={gtpc_type}"
#         ]
        
#         assert plumed_str == expected
#         assert labels == [f"rg_gtpc{i}"]


# def test_gyration_cv_with_chain_molecule():
#     """Test RadiusOfGyrationCV with a longer chain molecule in water."""
#     # This test uses a longer chain molecule to better demonstrate gyration
#     # We'll create a fixture for this specific test
    
#     # Create hexane (C6H14) - a 6-carbon chain
#     import rdkit2ase
    
#     hexane = rdkit2ase.smiles2conformers("CCCCCC", numConfs=1)
#     water = rdkit2ase.smiles2conformers("O", numConfs=1)
    
#     # Pack with some water molecules
#     box = rdkit2ase.pack(
#         [hexane, water], 
#         counts=[1, 5], 
#         density=700, 
#         packmol="packmol.jl"
#     )
    
#     # Select the hexane molecule
#     hexane_selector = pn.SMILESSelector(smiles="CCCCCC")
    
#     gyration_cv = pn.RadiusOfGyrationCV(
#         atoms=hexane_selector,
#         prefix="hexane_rg"
#     )
    
#     labels, plumed_str = gyration_cv.to_plumed(box)
    
#     # Should select the first 20 atoms (hexane molecule)
#     expected_atoms = ",".join(str(i) for i in range(1, 21))  # 1-based indexing
#     expected = [
#         f"hexane_rg: GYRATION ATOMS={expected_atoms}"
#     ]
    
#     assert plumed_str == expected
#     assert labels == ["hexane_rg"]


# def test_gyration_cv_empty_selection():
#     """Test RadiusOfGyrationCV with empty selection raises error."""
#     import rdkit2ase
    
#     # Create a simple water molecule
#     water = rdkit2ase.smiles2conformers("O", numConfs=1)
    
#     # Try to select carbon atoms (which don't exist in water)
#     carbon_selector = pn.SMARTSSelector(pattern="[#6]")  # Carbon
    
#     gyration_cv = pn.RadiusOfGyrationCV(
#         atoms=carbon_selector,
#         prefix="rg_fail"
#     )
    
#     with pytest.raises(ValueError, match="Empty selection"):
#         gyration_cv.to_plumed(water)


# def test_gyration_cv_multiple_types():
#     """Test multiple gyration CVs with different types on the same system."""
#     import rdkit2ase
    
#     # Create methane using SMILES for proper bonding
#     methane = rdkit2ase.smiles2conformers("C", numConfs=1)
    
#     # Select all atoms in methane
#     all_atoms_selector = pn.SMARTSSelector(pattern="*")  # All atoms
    
#     # Test multiple types
#     types_to_test = ["RADIUS", "ASPHERICITY", "ACYLINDRICITY", "KAPPA2"]
    
#     for gyration_type in types_to_test:
#         gyration_cv = pn.RadiusOfGyrationCV(
#             atoms=all_atoms_selector,
#             prefix=f"rg_{gyration_type.lower()}",
#             type=gyration_type
#         )
        
#         labels, plumed_str = gyration_cv.to_plumed(methane)
        
#         if gyration_type == "RADIUS":
#             # Default type doesn't add TYPE parameter
#             expected = [
#                 f"rg_{gyration_type.lower()}: GYRATION ATOMS=1,2,3,4,5"
#             ]
#         else:
#             expected = [
#                 f"rg_{gyration_type.lower()}: GYRATION ATOMS=1,2,3,4,5 TYPE={gyration_type}"
#             ]
        
#         assert plumed_str == expected
#         assert labels == [f"rg_{gyration_type.lower()}"]


# def test_gyration_cv_visualization(small_ethnol_water):
#     """Test that visualization highlights work properly."""
#     ethanol_selector = pn.SMILESSelector(smiles="CCO")
    
#     gyration_cv = pn.RadiusOfGyrationCV(
#         atoms=ethanol_selector,
#         prefix="rg_vis"
#     )
    
#     # Test that _get_atom_highlights returns expected structure
#     highlights = gyration_cv._get_atom_highlights(small_ethnol_water)
    
#     # Should highlight the first ethanol molecule atoms (0-8 in 0-based indexing)
#     expected_highlighted_atoms = set(range(9))  # First 9 atoms (ethanol)
#     actual_highlighted_atoms = set(highlights.keys())
    
#     assert actual_highlighted_atoms == expected_highlighted_atoms
    
#     # All atoms should have the same green color
#     for atom_idx, color in highlights.items():
#         assert color == (0.2, 0.8, 0.2)  # Green color


# def test_gyration_cv_get_img(small_ethnol_water):
#     """Test that get_img method works without errors."""
#     ethanol_selector = pn.SMILESSelector(smiles="CCO")
    
#     gyration_cv = pn.RadiusOfGyrationCV(
#         atoms=ethanol_selector,
#         prefix="rg_img"
#     )
    
#     # This should not raise an error
#     img = gyration_cv.get_img(small_ethnol_water)
    
#     # Check that we get a PIL Image
#     from PIL import Image
#     assert isinstance(img, Image.Image)
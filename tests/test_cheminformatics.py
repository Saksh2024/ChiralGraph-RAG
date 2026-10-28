"""
tests/test_cheminformatics.py
──────────────────────────────
Tests for RDKit stereochemistry and 3D conformer generation engine.
"""

import pytest
from src.cheminformatics import StereochemistryEngine


@pytest.fixture
def chem_engine():
    return StereochemistryEngine(max_strain_threshold_kcal=4.0)


def test_cip_stereocenter_detection(chem_engine):
    # (S)-Thalidomide
    smiles_s = "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"
    mol = chem_engine.parse_smiles(smiles_s)
    centers = chem_engine.get_chiral_centers(mol)
    assert len(centers) == 1
    assert centers[0].cip_code == "S"
    assert centers[0].is_assigned is True

    # (R)-Thalidomide
    smiles_r = "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"
    mol_r = chem_engine.parse_smiles(smiles_r)
    centers_r = chem_engine.get_chiral_centers(mol_r)
    assert len(centers_r) == 1
    assert centers_r[0].cip_code == "R"


def test_stereoisomer_enumeration(chem_engine):
    # Racemic thalidomide (unassigned chiral center)
    racemic_smiles = "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"
    isomers = chem_engine.enumerate_stereoisomers(racemic_smiles)
    assert len(isomers) == 2


def test_3d_conformer_generation(chem_engine):
    smiles = "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"
    profile = chem_engine.analyze_molecule(smiles, generate_3d=True, num_conformers=4)
    assert profile.canonical_smiles is not None
    assert profile.conformer_count > 0
    assert profile.lowest_energy_kcal_mol is not None
    assert profile.asphericity is not None
    assert 0.0 <= profile.asphericity <= 1.0


def test_ketamine_and_ofloxacin_chirality(chem_engine):
    # (S)-Ketamine
    mol_ket_s = chem_engine.parse_smiles("CN[C@]1(c2ccccc2Cl)CCCC1=O")
    centers_ket = chem_engine.get_chiral_centers(mol_ket_s)
    assert len(centers_ket) == 1
    assert centers_ket[0].cip_code == "S"

    # Levofloxacin ((S)-Ofloxacin)
    mol_oflox_s = chem_engine.parse_smiles("C[C@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23")
    centers_oflox = chem_engine.get_chiral_centers(mol_oflox_s)
    assert len(centers_oflox) == 1
    assert centers_oflox[0].cip_code == "S"


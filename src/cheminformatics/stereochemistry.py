"""
src/cheminformatics/stereochemistry.py
──────────────────────────────────────
Core 3D Cheminformatics Engine powered by RDKit.
Handles:
- Cahn-Ingold-Prelog (CIP) stereocenter identification and assignment ((R)/(S))
- Stereoisomer enumeration from racemic/unspecified SMILES
- 3D Conformer generation via ETKDG v3 distance geometry
- Energy minimization and conformational strain scoring via MMFF94 force field
- 3D Shape and conformational volume analysis
"""

import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


class StereocenterInfo(BaseModel):
    atom_index: int
    element: str
    cip_code: Optional[str] = Field(None, description="'R', 'S', or None if unassigned")
    is_assigned: bool


class ConformerSummary(BaseModel):
    conformer_id: int
    energy_kcal_mol: float
    relative_energy_kcal_mol: float
    converged: bool
    is_bioactive_candidate: bool = Field(
        ..., description="True if conformational strain <= 4.0 kcal/mol relative to minimum"
    )


class MolecularProfile(BaseModel):
    canonical_smiles: str
    molecular_formula: str
    molecular_weight: float
    log_p: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    total_stereocenters: int
    chiral_centers: List[StereocenterInfo]
    conformer_count: int
    lowest_energy_kcal_mol: Optional[float] = None
    conformers: List[ConformerSummary] = []
    asphericity: Optional[float] = None
    eccentricity: Optional[float] = None
    inertial_shape_factor: Optional[float] = None


class StereochemistryEngine:
    """High-performance RDKit-backed stereochemistry and 3D conformer analyzer."""

    def __init__(self, max_strain_threshold_kcal: float = 4.0):
        self.max_strain_threshold_kcal = max_strain_threshold_kcal

    def parse_smiles(self, smiles: str) -> Chem.Mol:
        """Parse and sanitize a SMILES string into an RDKit Mol object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string provided: '{smiles}'")
        Chem.SanitizeMol(mol)
        return mol

    def get_chiral_centers(self, mol: Chem.Mol) -> List[StereocenterInfo]:
        """Detect all chiral stereocenters using CIP rules."""
        # Find chiral centers with RDKit
        chiral_list = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        centers: List[StereocenterInfo] = []
        for atom_idx, cip in chiral_list:
            atom = mol.GetAtomWithIdx(atom_idx)
            is_assigned = cip in ("R", "S")
            centers.append(
                StereocenterInfo(
                    atom_index=atom_idx,
                    element=atom.GetSymbol(),
                    cip_code=cip if is_assigned else None,
                    is_assigned=is_assigned,
                )
            )
        return centers

    def enumerate_stereoisomers(self, smiles: str, max_isomers: int = 16) -> List[str]:
        """
        Enumerate all valid chiral stereoisomers for a racemic or unassigned molecule.
        Returns a list of canonical isomeric SMILES.
        """
        mol = self.parse_smiles(smiles)
        opts = StereoEnumerationOptions(unique=True, maxIsomers=max_isomers, onlyUnassigned=False)
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        return [Chem.MolToSmiles(iso, isomericSmiles=True) for iso in isomers]

    def generate_3d_conformers(
        self,
        smiles: str,
        num_conformers: int = 10,
        random_seed: int = 42,
    ) -> tuple[Chem.Mol, List[ConformerSummary]]:
        """
        Generates 3D conformers using ETKDG v3 distance geometry and optimizes
        coordinates using the MMFF94 force field.
        """
        mol = self.parse_smiles(smiles)
        mol_h = Chem.AddHs(mol)

        # Configure ETKDG v3 parameters
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.pruneRmsThresh = 0.5  # Filter near-duplicate conformers
        params.numThreads = 1

        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))
        if not conf_ids:
            # Fallback if strict embedding fails
            conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, randomSeed=random_seed))

        # Optimize geometries with MMFF94
        energies_raw = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500, nonBondedThresh=100.0)

        conformers: List[ConformerSummary] = []
        valid_energies = [e[1] for e in energies_raw if not math.isnan(e[1])]
        min_energy = min(valid_energies) if valid_energies else 0.0

        for idx, (converged_flag, energy) in enumerate(energies_raw):
            if math.isnan(energy):
                continue
            rel_e = energy - min_energy
            conformers.append(
                ConformerSummary(
                    conformer_id=idx,
                    energy_kcal_mol=round(energy, 3),
                    relative_energy_kcal_mol=round(rel_e, 3),
                    converged=(converged_flag == 0),
                    is_bioactive_candidate=(rel_e <= self.max_strain_threshold_kcal),
                )
            )

        conformers.sort(key=lambda c: c.relative_energy_kcal_mol)
        return mol_h, conformers

    def analyze_molecule(self, smiles: str, generate_3d: bool = True, num_conformers: int = 8) -> MolecularProfile:
        """Full characterization: 2D molecular descriptors + stereocenters + 3D conformer ensemble."""
        mol = self.parse_smiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        chiral_centers = self.get_chiral_centers(mol)

        # Basic 2D Descriptors
        mol_wt = round(Descriptors.MolWt(mol), 3)
        log_p = round(Descriptors.MolLogP(mol), 3)
        tpsa = round(Descriptors.TPSA(mol), 2)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)

        conformers: List[ConformerSummary] = []
        min_energy = None
        asphericity = None
        eccentricity = None
        inertial_shape = None

        if generate_3d:
            try:
                mol_3d, conformers = self.generate_3d_conformers(canonical_smiles, num_conformers=num_conformers)
                if conformers:
                    min_energy = conformers[0].energy_kcal_mol
                    # Calculate 3D shape descriptors for lowest energy conformer
                    conf_id = conformers[0].conformer_id
                    asphericity = round(rdMolDescriptors.CalcAsphericity(mol_3d, confId=conf_id), 4)
                    eccentricity = round(rdMolDescriptors.CalcEccentricity(mol_3d, confId=conf_id), 4)
                    inertial_shape = round(rdMolDescriptors.CalcInertialShapeFactor(mol_3d, confId=conf_id), 4)
            except Exception as e:
                # Graceful fallback if 3D embedding fails on highly constrained/exotic structures
                pass

        return MolecularProfile(
            canonical_smiles=canonical_smiles,
            molecular_formula=formula,
            molecular_weight=mol_wt,
            log_p=log_p,
            tpsa=tpsa,
            hbd=hbd,
            hba=hba,
            rotatable_bonds=rot_bonds,
            total_stereocenters=len(chiral_centers),
            chiral_centers=chiral_centers,
            conformer_count=len(conformers),
            lowest_energy_kcal_mol=min_energy,
            conformers=conformers,
            asphericity=asphericity,
            eccentricity=eccentricity,
            inertial_shape_factor=inertial_shape,
        )

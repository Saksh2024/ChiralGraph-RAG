"""
src/graph/schema.py
───────────────────
Graph schema and data transfer models for the stereochemical Knowledge Graph.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ChiralityType(str, Enum):
    R_ENANTIOMER = "R"
    S_ENANTIOMER = "S"
    RACEMIC = "RS"
    MESO = "MESO"
    ACHIRAL = "ACHIRAL"


class LiabilitySeverity(str, Enum):
    CRITICAL = "CRITICAL"   # Teratogenicity, fatal arrhythmia (Torsades de Pointes)
    HIGH = "HIGH"           # Severe hepatotoxicity, irreversible receptor blockade
    MODERATE = "MODERATE"   # Off-target CYP inhibition, moderate QT prolongation
    LOW = "LOW"             # Mild drowsiness, nausea


class DrugNode(BaseModel):
    id: str = Field(..., description="Unique drug identifier, e.g. 'DRUG_THALIDOMIDE'")
    name: str
    generic_name: str
    cas_number: Optional[str] = None
    indication: str
    description: str


class StereoisomerNode(BaseModel):
    id: str = Field(..., description="Unique isomer identifier, e.g. 'ISO_S_THALIDOMIDE'")
    parent_drug_id: str
    name: str
    chirality: ChiralityType
    isomeric_smiles: str
    molecular_formula: str
    molecular_weight: float
    lowest_energy_kcal_mol: Optional[float] = None
    eudismic_role: Optional[str] = Field(None, description="'Eutomer' (active) or 'Distomer' (inactive/toxic)")


class TargetNode(BaseModel):
    id: str = Field(..., description="Target protein ID, e.g. 'TARGET_CRBN'")
    name: str
    gene_symbol: str
    uniprot_id: Optional[str] = None
    target_class: str = Field(..., description="E.g., 'E3 Ubiquitin Ligase', 'Transporter', 'Ion Channel'")
    organism: str = "Homo sapiens"


class BindingPocketNode(BaseModel):
    id: str
    target_id: str
    name: str
    pocket_type: str = Field(..., description="'Orthosteric', 'Allosteric', 'Channel Pore'")
    key_residues: List[str] = []


class ClinicalLiabilityNode(BaseModel):
    id: str
    name: str
    severity: LiabilitySeverity
    mechanism: str
    description: str


class BindingRelation(BaseModel):
    source_isomer_id: str
    target_id: str
    pocket_id: Optional[str] = None
    affinity_value: float = Field(..., description="Numerical value of Kd, Ki, or IC50")
    affinity_type: str = Field(..., description="'Kd', 'Ki', or 'IC50'")
    affinity_unit: str = Field("nM", description="'nM', 'uM', or 'pM'")
    is_primary_therapeutic: bool = True
    steric_clash_observed: bool = False
    evidence_pmid: Optional[str] = None


class LiabilityRelation(BaseModel):
    source_isomer_id: str
    liability_id: str
    severity: LiabilitySeverity
    clinical_evidence: str
    pmid_citation: Optional[str] = None

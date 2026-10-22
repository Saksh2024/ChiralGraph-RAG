"""
src/graph/seed_data.py
──────────────────────
Populates the Stereochemical Knowledge Graph with high-impact, clinically
verified chiral pharmacology benchmark datasets.
"""

from .schema import (
    DrugNode,
    StereoisomerNode,
    TargetNode,
    BindingPocketNode,
    ClinicalLiabilityNode,
    BindingRelation,
    LiabilityRelation,
    ChiralityType,
    LiabilitySeverity,
)
from .store import InMemoryGraphStore


def seed_stereochemical_knowledge_graph(store: InMemoryGraphStore):
    """Seed benchmark stereoisomer relationships into the graph store."""

    # =========================================================================
    # 1. THALIDOMIDE
    # =========================================================================
    thalidomide = DrugNode(
        id="DRUG_THALIDOMIDE",
        name="Thalidomide",
        generic_name="Thalidomide",
        cas_number="50-35-1",
        indication="Multiple myeloma, erythema nodosum leprosum; historically prescribed as anti-emetic/sedative",
        description="Classical chiral tragedy. Racemizes rapidly in vivo via keto-enol tautomerism.",
    )
    store.add_drug(thalidomide)

    iso_r_thal = StereoisomerNode(
        id="ISO_R_THALIDOMIDE",
        parent_drug_id="DRUG_THALIDOMIDE",
        name="(R)-Thalidomide",
        chirality=ChiralityType.R_ENANTIOMER,
        isomeric_smiles="O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
        molecular_formula="C13H10N2O4",
        molecular_weight=258.23,
        lowest_energy_kcal_mol=-43.16,
        eudismic_role="Eutomer (Sedative)",
    )
    iso_s_thal = StereoisomerNode(
        id="ISO_S_THALIDOMIDE",
        parent_drug_id="DRUG_THALIDOMIDE",
        name="(S)-Thalidomide",
        chirality=ChiralityType.S_ENANTIOMER,
        isomeric_smiles="O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
        molecular_formula="C13H10N2O4",
        molecular_weight=258.23,
        lowest_energy_kcal_mol=-43.16,
        eudismic_role="Distomer (Teratogen)",
    )
    store.add_stereoisomer(iso_r_thal)
    store.add_stereoisomer(iso_s_thal)

    crbn_target = TargetNode(
        id="TARGET_CRBN",
        name="Cereblon E3 Ubiquitin Ligase Substrate Receptor",
        gene_symbol="CRBN",
        uniprot_id="Q96SW2",
        target_class="E3 Ubiquitin Ligase",
    )
    store.add_target(crbn_target)

    crbn_triad_pocket = BindingPocketNode(
        id="POCKET_CRBN_TRIAD",
        target_id="TARGET_CRBN",
        name="Tri-tryptophan hydrophobic binding pocket",
        pocket_type="Orthosteric",
        key_residues=["Trp380", "Trp386", "Trp400", "His378"],
    )
    store.add_binding_pocket(crbn_triad_pocket)

    teratogenicity_liability = ClinicalLiabilityNode(
        id="LIAB_TERATOGENICITY",
        name="Phocomelia & Severe Teratogenesis",
        severity=LiabilitySeverity.CRITICAL,
        mechanism="SALL4 and p63 transcription factor polyubiquitination and proteasomal degradation",
        description="Failure of embryonic limb bud outgrowth, amelia, phocomelia, and internal organ malformations.",
    )
    store.add_liability(teratogenicity_liability)

    # (S)-Thalidomide binds CRBN with high affinity; (R)-isomer is 10-fold weaker
    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_S_THALIDOMIDE",
            target_id="TARGET_CRBN",
            pocket_id="POCKET_CRBN_TRIAD",
            affinity_value=25.0,
            affinity_type="Kd",
            affinity_unit="nM",
            is_primary_therapeutic=False,
            steric_clash_observed=False,
            evidence_pmid="20203043",
        )
    )
    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_R_THALIDOMIDE",
            target_id="TARGET_CRBN",
            pocket_id="POCKET_CRBN_TRIAD",
            affinity_value=250.0,
            affinity_type="Kd",
            affinity_unit="nM",
            is_primary_therapeutic=False,
            steric_clash_observed=True,
            evidence_pmid="29440439",
        )
    )
    store.add_liability_relation(
        LiabilityRelation(
            source_isomer_id="ISO_S_THALIDOMIDE",
            liability_id="LIAB_TERATOGENICITY",
            severity=LiabilitySeverity.CRITICAL,
            clinical_evidence="Direct crystallographic and cellular proof of SALL4 recruitment leading to selective limb agenesis.",
            pmid_citation="29440439",
        )
    )

    # =========================================================================
    # 2. CITALOPRAM / ESCITALOPRAM
    # =========================================================================
    citalopram = DrugNode(
        id="DRUG_CITALOPRAM",
        name="Citalopram",
        generic_name="Citalopram",
        cas_number="59729-33-8",
        indication="Major depressive disorder, generalized anxiety disorder",
        description="Selective serotonin reuptake inhibitor (SSRI) marketed as racemic mixture and as purified (S)-enantiomer (Escitalopram).",
    )
    store.add_drug(citalopram)

    iso_s_cit = StereoisomerNode(
        id="ISO_S_CITALOPRAM",
        parent_drug_id="DRUG_CITALOPRAM",
        name="Escitalopram ((S)-Citalopram)",
        chirality=ChiralityType.S_ENANTIOMER,
        isomeric_smiles="CN(C)CCC[C@@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
        molecular_formula="C20H21FN2O",
        molecular_weight=324.39,
        eudismic_role="Eutomer (Active SSRI)",
    )
    iso_r_cit = StereoisomerNode(
        id="ISO_R_CITALOPRAM",
        parent_drug_id="DRUG_CITALOPRAM",
        name="(R)-Citalopram",
        chirality=ChiralityType.R_ENANTIOMER,
        isomeric_smiles="CN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
        molecular_formula="C20H21FN2O",
        molecular_weight=324.39,
        eudismic_role="Distomer (Allosteric Antagonist & hERG Blocker)",
    )
    store.add_stereoisomer(iso_s_cit)
    store.add_stereoisomer(iso_r_cit)

    sert_target = TargetNode(
        id="TARGET_SERT",
        name="Sodium-dependent Serotonin Transporter",
        gene_symbol="SLC6A4",
        uniprot_id="P31645",
        target_class="Monoamine Transporter",
    )
    herg_target = TargetNode(
        id="TARGET_HERG",
        name="Potassium Voltage-Gated Channel Subfamily H Member 2",
        gene_symbol="KCNH2",
        uniprot_id="Q12809",
        target_class="Voltage-Gated Potassium Channel",
    )
    store.add_target(sert_target)
    store.add_target(herg_target)

    herg_cardiac_liability = ClinicalLiabilityNode(
        id="LIAB_QT_PROLONGATION",
        name="hERG Blockade & QT Interval Prolongation",
        severity=LiabilitySeverity.HIGH,
        mechanism="Selective blockade of cardiac IKr rapid delayed rectifier potassium current",
        description="Can precipitate fatal ventricular tachyarrhythmia (Torsades de Pointes). FDA black box warning.",
    )
    store.add_liability(herg_cardiac_liability)

    # (S)-Citalopram binds primary SERT site with sub-nanomolar affinity; (R)-isomer is 160x weaker
    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_S_CITALOPRAM",
            target_id="TARGET_SERT",
            affinity_value=1.1,
            affinity_type="Ki",
            affinity_unit="nM",
            is_primary_therapeutic=True,
            evidence_pmid="12049832",
        )
    )
    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_R_CITALOPRAM",
            target_id="TARGET_SERT",
            affinity_value=170.0,
            affinity_type="Ki",
            affinity_unit="nM",
            is_primary_therapeutic=False,
            evidence_pmid="12049832",
        )
    )
    # (R)-isomer blocks hERG with higher potency, driving cardiac liability in racemic form
    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_R_CITALOPRAM",
            target_id="TARGET_HERG",
            affinity_value=3900.0,
            affinity_type="IC50",
            affinity_unit="nM",
            is_primary_therapeutic=False,
            evidence_pmid="22814541",
        )
    )
    store.add_liability_relation(
        LiabilityRelation(
            source_isomer_id="ISO_R_CITALOPRAM",
            liability_id="LIAB_QT_PROLONGATION",
            severity=LiabilitySeverity.HIGH,
            clinical_evidence="Clinical dose-dependent QT interval prolongation significantly driven by (R)-enantiomer off-target pore binding.",
            pmid_citation="22814541",
        )
    )

    # =========================================================================
    # 3. KETAMINE / ESKETAMINE
    # =========================================================================
    ketamine = DrugNode(
        id="DRUG_KETAMINE",
        name="Ketamine",
        generic_name="Ketamine",
        cas_number="6740-88-1",
        indication="General anesthesia, treatment-resistant depression",
        description="Dissociative anesthetic acting as non-competitive NMDA receptor pore blocker.",
    )
    store.add_drug(ketamine)

    iso_s_ket = StereoisomerNode(
        id="ISO_S_KETAMINE",
        parent_drug_id="DRUG_KETAMINE",
        name="Esketamine ((S)-Ketamine)",
        chirality=ChiralityType.S_ENANTIOMER,
        isomeric_smiles="CNC1(c2ccccc2Cl)CCCC1=O",
        molecular_formula="C13H16ClNO",
        molecular_weight=237.72,
        eudismic_role="Eutomer (Rapid Antidepressant)",
    )
    iso_r_ket = StereoisomerNode(
        id="ISO_R_KETAMINE",
        parent_drug_id="DRUG_KETAMINE",
        name="Arketamine ((R)-Ketamine)",
        chirality=ChiralityType.R_ENANTIOMER,
        isomeric_smiles="CNC1(c2ccccc2Cl)CCCC1=O",
        molecular_formula="C13H16ClNO",
        molecular_weight=237.72,
        eudismic_role="Distomer (Reduced Anesthetic Potency)",
    )
    store.add_stereoisomer(iso_s_ket)
    store.add_stereoisomer(iso_r_ket)

    nmda_target = TargetNode(
        id="TARGET_NMDAR",
        name="Glutamate Ionotropic Receptor NMDA Type Subunit 1",
        gene_symbol="GRIN1",
        uniprot_id="Q05586",
        target_class="Ligand-Gated Ion Channel",
    )
    store.add_target(nmda_target)

    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_S_KETAMINE",
            target_id="TARGET_NMDAR",
            affinity_value=300.0,
            affinity_type="Ki",
            affinity_unit="nM",
            is_primary_therapeutic=True,
            evidence_pmid="11152058",
        )
    )
    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_R_KETAMINE",
            target_id="TARGET_NMDAR",
            affinity_value=1400.0,
            affinity_type="Ki",
            affinity_unit="nM",
            is_primary_therapeutic=True,
            evidence_pmid="11152058",
        )
    )

    # =========================================================================
    # 4. OFLOXACIN / LEVOFLOXACIN
    # =========================================================================
    ofloxacin = DrugNode(
        id="DRUG_OFLOXACIN",
        name="Ofloxacin",
        generic_name="Ofloxacin",
        cas_number="82419-36-1",
        indication="Broad-spectrum fluoroquinolone antibiotic",
        description="Inhibitor of bacterial DNA gyrase and topoisomerase IV.",
    )
    store.add_drug(ofloxacin)

    iso_s_oflox = StereoisomerNode(
        id="ISO_S_OFLOXACIN",
        parent_drug_id="DRUG_OFLOXACIN",
        name="Levofloxacin ((S)-Ofloxacin)",
        chirality=ChiralityType.S_ENANTIOMER,
        isomeric_smiles="CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
        molecular_formula="C18H20FN3O4",
        molecular_weight=361.37,
        eudismic_role="Eutomer (Active Antibacterial)",
    )
    iso_r_oflox = StereoisomerNode(
        id="ISO_R_OFLOXACIN",
        parent_drug_id="DRUG_OFLOXACIN",
        name="(R)-Ofloxacin",
        chirality=ChiralityType.R_ENANTIOMER,
        isomeric_smiles="CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
        molecular_formula="C18H20FN3O4",
        molecular_weight=361.37,
        eudismic_role="Distomer (Inactive / Renal Load)",
    )
    store.add_stereoisomer(iso_s_oflox)
    store.add_stereoisomer(iso_r_oflox)

    dna_gyrase_target = TargetNode(
        id="TARGET_GYRA",
        name="Bacterial DNA Gyrase Subunit A",
        gene_symbol="gyrA",
        target_class="Type II Topoisomerase",
        organism="Escherichia coli",
    )
    store.add_target(dna_gyrase_target)

    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_S_OFLOXACIN",
            target_id="TARGET_GYRA",
            affinity_value=80.0,
            affinity_type="IC50",
            affinity_unit="nM",
            is_primary_therapeutic=True,
            evidence_pmid="8438781",
        )
    )
    store.add_binding_relation(
        BindingRelation(
            source_isomer_id="ISO_R_OFLOXACIN",
            target_id="TARGET_GYRA",
            affinity_value=10500.0,
            affinity_type="IC50",
            affinity_unit="nM",
            is_primary_therapeutic=False,
            steric_clash_observed=True,
            evidence_pmid="8438781",
        )
    )

    return store

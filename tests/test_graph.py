"""
tests/test_graph.py
───────────────────
Tests for the stereochemical Knowledge Graph store.
"""

import pytest
from src.graph import InMemoryGraphStore, seed_stereochemical_knowledge_graph


@pytest.fixture
def graph_store():
    store = InMemoryGraphStore()
    seed_stereochemical_knowledge_graph(store)
    return store


def test_seed_graph_stats(graph_store):
    stats = graph_store.stats()
    assert stats["drugs"] >= 4
    assert stats["stereoisomers"] >= 8
    assert stats["targets"] >= 5
    assert stats["clinical_liabilities"] >= 2


def test_thalidomide_disparity(graph_store):
    disparity = graph_store.find_chiral_disparity("Thalidomide")
    assert disparity["isomer_count"] == 2
    profiles = disparity["isomeric_profiles"]
    
    # Check that (S)-isomer has teratogenicity liability
    s_prof = next(p for p in profiles if p["isomer"]["chirality"] == "S")
    assert any(l["liability"]["id"] == "LIAB_TERATOGENICITY" for l in s_prof["clinical_liabilities"])

    # Check that (R)-isomer does NOT trigger the critical teratogenicity liability
    r_prof = next(p for p in profiles if p["isomer"]["chirality"] == "R")
    assert len(r_prof["clinical_liabilities"]) == 0


def test_citalopram_disparity(graph_store):
    disparity = graph_store.find_chiral_disparity("Citalopram")
    assert disparity["isomer_count"] == 2
    profiles = disparity["isomeric_profiles"]

    # (S)-citalopram has high affinity for SERT
    s_prof = next(p for p in profiles if p["isomer"]["chirality"] == "S")
    sert_binding = next(b for b in s_prof["target_bindings"] if b["target"]["gene_symbol"] == "SLC6A4")
    assert sert_binding["affinity"]["affinity_value"] == 1.1

    # (R)-citalopram has lower affinity for SERT (170 nM) and blocks hERG
    r_prof = next(p for p in profiles if p["isomer"]["chirality"] == "R")
    herg_binding = next(b for b in r_prof["target_bindings"] if b["target"]["gene_symbol"] == "KCNH2")
    assert herg_binding["affinity"]["affinity_value"] == 3900.0


def test_ketamine_disparity(graph_store):
    disparity = graph_store.find_chiral_disparity("Ketamine")
    assert disparity["isomer_count"] == 2
    profiles = disparity["isomeric_profiles"]

    s_prof = next(p for p in profiles if p["isomer"]["chirality"] == "S")
    r_prof = next(p for p in profiles if p["isomer"]["chirality"] == "R")

    # (S)-ketamine has higher affinity (300 nM) than (R)-ketamine (1400 nM)
    s_aff = s_prof["target_bindings"][0]["affinity"]["affinity_value"]
    r_aff = r_prof["target_bindings"][0]["affinity"]["affinity_value"]
    assert s_aff < r_aff
    assert s_aff == 300.0
    assert r_aff == 1400.0


def test_ofloxacin_disparity(graph_store):
    disparity = graph_store.find_chiral_disparity("Ofloxacin")
    assert disparity["isomer_count"] == 2
    profiles = disparity["isomeric_profiles"]

    s_prof = next(p for p in profiles if p["isomer"]["chirality"] == "S")
    r_prof = next(p for p in profiles if p["isomer"]["chirality"] == "R")

    # Levofloxacin ((S)) is ~130x more potent against DNA Gyrase
    s_ic50 = s_prof["target_bindings"][0]["affinity"]["affinity_value"]
    r_ic50 = r_prof["target_bindings"][0]["affinity"]["affinity_value"]
    assert s_ic50 == 80.0
    assert r_ic50 == 10500.0
    assert r_prof["target_bindings"][0]["affinity"]["steric_clash_observed"] is True


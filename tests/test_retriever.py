"""
tests/test_retriever.py
───────────────────────
Tests for the Think-on-Graph hybrid retriever and pharmacological reasoner.
"""

import pytest
from src.engine import ThinkOnGraphRetriever, PharmacologicalReasoner


@pytest.fixture
def retriever():
    return ThinkOnGraphRetriever()


@pytest.fixture
def reasoner():
    return PharmacologicalReasoner(llm_provider="mock")


def test_retriever_thalidomide_dossier(retriever):
    dossier = retriever.retrieve(query="What is the teratogenic risk of (S)-thalidomide?")
    assert dossier.resolved_compound == "Thalidomide"
    assert dossier.target_chirality == "S"
    assert len(dossier.literature_evidence) > 0
    assert "VERIFIED" in dossier.stereochemical_guardrail_status


def test_reasoner_sar_report_generation(retriever, reasoner):
    dossier = retriever.retrieve(query="Evaluate citalopram enantiomer binding profiles and hERG cardiac liabilities.")
    report = reasoner.generate_dossier(dossier)
    assert report.compound_name == "Citalopram"
    assert len(report.target_affinity_breakdown) > 0
    assert report.verification_guardrail_passed is True
    assert len(report.medicinal_chemistry_recommendations) >= 2

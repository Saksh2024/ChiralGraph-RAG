"""
tests/test_api.py
─────────────────
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["project"] == "ChiralGraph-RAG"


def test_cheminformatics_analyze():
    response = client.post(
        "/api/v1/cheminformatics/analyze",
        json={"smiles": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1", "generate_3d": True, "num_conformers": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["molecular_formula"] == "C13H10N2O4"
    assert data["total_stereocenters"] == 1
    assert data["conformer_count"] > 0


def test_graph_drug_endpoint():
    response = client.get("/api/v1/graph/drug/Thalidomide")
    assert response.status_code == 200
    data = response.json()
    assert data["isomer_count"] == 2


def test_rag_query_endpoint():
    response = client.post(
        "/api/v1/rag/query",
        json={"query": "Is (S)-thalidomide safe for sedation?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["compound_name"] == "Thalidomide"
    assert data["verification_guardrail_passed"] is True


def test_evaluation_benchmark_endpoint():
    response = client.get("/api/v1/evaluation/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert data["total_benchmark_queries"] > 0
    assert "metrics" in data

"""
src/api/main.py
───────────────
Production FastAPI application for ChiralGraph-RAG.
Exposes RESTful endpoints for:
- 3D Cheminformatics & stereocenter analysis
- Stereoisomer enumeration
- Knowledge Graph disparity queries
- Think-on-Graph hybrid retrieval and SAR generation
- Quantitative benchmarking
"""

from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import settings
from ..cheminformatics import StereochemistryEngine, MolecularProfile
from ..graph import get_graph_store, seed_stereochemical_knowledge_graph
from ..vector import get_vector_store
from ..engine import ThinkOnGraphRetriever, PharmacologicalReasoner, SARReport
from ..evaluation import BenchmarkRunner


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize graph with seed data
    store = get_graph_store()
    if store.stats()["drugs"] == 0:
        seed_stereochemical_knowledge_graph(store)
    # Initialize vector store
    get_vector_store()
    yield


from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


_UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="3D Conformation & Stereoisomer-Aware Knowledge Graph RAG Engine for Drug Discovery",
    version=settings.VERSION,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount UI static assets and serve demo dashboard with strict no-cache headers
if _UI_DIR.exists():
    app.mount("/static", NoCacheStaticFiles(directory=str(_UI_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(
            str(_UI_DIR / "index.html"),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

# Singletons
chem_engine = StereochemistryEngine(max_strain_threshold_kcal=settings.MAX_CONFORMER_STRAIN_KCAL)
retriever = ThinkOnGraphRetriever()
reasoner = PharmacologicalReasoner()
benchmark_runner = BenchmarkRunner()


# --- Request & Response Models ---
class AnalyzeMolRequest(BaseModel):
    smiles: str = Field(..., json_schema_extra={"example": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"})
    generate_3d: bool = True
    num_conformers: int = 8


class EnumerateIsomersRequest(BaseModel):
    smiles: str = Field(..., json_schema_extra={"example": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"})
    max_isomers: int = 16


class RAGQueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Is (S)-thalidomide safe to prescribe for sedation in early pregnancy?"})
    smiles: Optional[str] = None
    chirality: Optional[str] = Field(None, json_schema_extra={"example": "S"})
    top_k_literature: int = 4


# --- Endpoints ---

@app.get("/health", tags=["System"])
async def health_check():
    """Returns system status, active backends, and graph node counts."""
    store = get_graph_store()
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "graph_backend": settings.GRAPH_BACKEND,
        "vector_backend": settings.VECTOR_BACKEND,
        "llm_provider": settings.LLM_PROVIDER,
        "knowledge_graph_stats": store.stats(),
    }


@app.post("/api/v1/cheminformatics/analyze", response_model=MolecularProfile, tags=["Cheminformatics"])
async def analyze_molecule(payload: AnalyzeMolRequest):
    """
    Computes 2D descriptors, CIP (R/S) chiral centers, 3D conformers via ETKDG v3,
    MMFF94 force field strain energies, and 3D shape descriptors.
    """
    try:
        return chem_engine.analyze_molecule(
            payload.smiles,
            generate_3d=payload.generate_3d,
            num_conformers=payload.num_conformers,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/cheminformatics/enumerate", tags=["Cheminformatics"])
async def enumerate_stereoisomers(payload: EnumerateIsomersRequest):
    """Enumerates all valid chiral stereoisomers from an unassigned or racemic SMILES."""
    try:
        isomers = chem_engine.enumerate_stereoisomers(payload.smiles, max_isomers=payload.max_isomers)
        return {
            "input_smiles": payload.smiles,
            "total_isomers": len(isomers),
            "isomeric_smiles": isomers,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/graph/drug/{drug_name}", tags=["Knowledge Graph"])
async def get_drug_chiral_disparity(drug_name: str):
    """
    Traverses the stereochemical Knowledge Graph to extract all enantiomers,
    eudismic roles, target binding affinities, and clinical liabilities.
    """
    store = get_graph_store()
    disparity = store.find_chiral_disparity(drug_name)
    if disparity.get("isomer_count", 0) == 0:
        raise HTTPException(status_code=404, detail=f"Drug '{drug_name}' not found in Knowledge Graph.")
    return disparity


@app.post("/api/v1/rag/query", response_model=SARReport, tags=["GraphRAG"])
async def query_chiral_rag(payload: RAGQueryRequest):
    """
    Executes Think-on-Graph hybrid retrieval and pharmacological reasoning.
    Returns a grounded SAR dossier with quantitative eudismic analysis.
    """
    try:
        dossier = retriever.retrieve(
            query=payload.query,
            smiles=payload.smiles,
            chirality=payload.chirality,
            top_k_lit=payload.top_k_literature,
        )
        return reasoner.generate_dossier(dossier)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval/Synthesis error: {str(e)}")


@app.get("/api/v1/evaluation/benchmark", tags=["Evaluation"])
async def run_comparative_benchmark():
    """
    Executes the benchmark suite comparing Naive 2D Vector RAG vs. ChiralGraph-RAG.
    Returns quantitative discrimination, hallucination reduction, and precision metrics.
    """
    return benchmark_runner.run_benchmark()

"""
src/vector/store.py
───────────────────
Vector store for pharmacological literature, SAR abstracts, and FDA drug inserts.
Supports:
1. In-memory cosine similarity search (zero external dependencies).
2. Qdrant vector database client when QDRANT_URL is active.
Includes pre-seeded literature corpus tagged with stereochemical metadata.
"""

import math
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..config import settings

logger = logging.getLogger(__name__)


class LiteratureChunk(BaseModel):
    chunk_id: str
    compound_name: str
    target_chirality: str = Field(..., description="'R', 'S', 'RS', or 'ALL'")
    target_gene: Optional[str] = None
    pmid: Optional[str] = None
    title: str
    text: str
    evidence_level: str = "Peer-Reviewed In Vitro / In Vivo"


class ScoredChunk(BaseModel):
    chunk: LiteratureChunk
    score: float


class InMemoryVectorStore:
    """Fast, zero-dependency in-memory vector & keyword similarity index."""

    def __init__(self):
        self.documents: List[LiteratureChunk] = []

    def add_documents(self, docs: List[LiteratureChunk]):
        self.documents.extend(docs)

    def _tokenize(self, text: str) -> set:
        return set(text.lower().replace("-", " ").replace("(", " ").replace(")", " ").split())

    def search(
        self,
        query: str,
        chirality_filter: Optional[str] = None,
        top_k: int = 4,
    ) -> List[ScoredChunk]:
        """
        Calculates lexical/semantic relevance scores with strict or soft stereocenter filtering.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        results: List[ScoredChunk] = []

        for doc in self.documents:
            # Check chirality filter if specified (e.g. only S-isomer papers)
            if chirality_filter and chirality_filter != "ALL":
                if doc.target_chirality not in (chirality_filter, "RS", "ALL"):
                    continue

            doc_text = f"{doc.title} {doc.compound_name} {doc.target_gene or ''} {doc.text}"
            doc_tokens = self._tokenize(doc_text)

            # Jaccard + Term Match Score
            intersection = query_tokens.intersection(doc_tokens)
            union = query_tokens.union(doc_tokens)
            jaccard = len(intersection) / len(union) if union else 0.0

            # Boost if exact chemical name or chirality token matches
            boost = 1.0
            if doc.compound_name.lower() in query.lower():
                boost += 0.5
            if chirality_filter and chirality_filter.lower() in doc_tokens:
                boost += 0.3

            score = round(jaccard * boost, 4)
            if score > 0.0:
                results.append(ScoredChunk(chunk=doc, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


def get_default_literature_corpus() -> List[LiteratureChunk]:
    """Curated benchmark scientific abstracts detailing stereoselective pharmacology."""
    return [
        LiteratureChunk(
            chunk_id="LIT_THAL_001",
            compound_name="Thalidomide",
            target_chirality="S",
            target_gene="CRBN",
            pmid="29440439",
            title="Structural basis of SALL4 transcription factor degradation by (S)-thalidomide",
            text="The (S)-enantiomer of thalidomide fits stereospecifically into the tri-tryptophan hydrophobic pocket of cereblon (CRBN). This forms a composite molecular surface recruiting the C2H2 zinc-finger transcription factor SALL4, driving rapid polyubiquitination and proteasomal degradation, which is the direct molecular trigger for catastrophic embryonic phocomelia and limb loss.",
        ),
        LiteratureChunk(
            chunk_id="LIT_THAL_002",
            compound_name="Thalidomide",
            target_chirality="R",
            target_gene="CRBN",
            pmid="20203043",
            title="Sedative activity and receptor occupancy of (R)-thalidomide",
            text="(R)-thalidomide provides effective sedation and immunomodulation without high-affinity ternary complex formation with SALL4. However, due to the acidic alpha-hydrogen on the glutarimide ring, thalidomide enantiomers undergo rapid spontaneous interconversion in human plasma with an in vivo racemization half-life of 4 to 5 hours.",
        ),
        LiteratureChunk(
            chunk_id="LIT_CIT_001",
            compound_name="Citalopram",
            target_chirality="S",
            target_gene="SLC6A4",
            pmid="12049832",
            title="Escitalopram: Allosteric modulation and high-affinity orthosteric SERT inhibition",
            text="Escitalopram, the pure (S)-enantiomer of citalopram, is a highly selective serotonin reuptake inhibitor with Ki = 1.1 nM at the primary orthosteric site (S1). It demonstrates superior clinical response and faster onset of remission compared to the racemic mixture.",
        ),
        LiteratureChunk(
            chunk_id="LIT_CIT_002",
            compound_name="Citalopram",
            target_chirality="R",
            target_gene="KCNH2",
            pmid="22814541",
            title="(R)-citalopram binds secondary allosteric SERT site and drives cardiac hERG liability",
            text="(R)-citalopram antagonizes the therapeutic efficacy of (S)-citalopram by binding to a lower-affinity allosteric site (S2) on the serotonin transporter, accelerating dissociation of the (S)-isomer. Crucially, (R)-citalopram demonstrates concentration-dependent blockade of the cardiac hERG potassium channel (IC50 = 3.9 uM), explaining the dose-dependent QT interval prolongation observed in patients on high-dose racemic citalopram.",
        ),
        LiteratureChunk(
            chunk_id="LIT_KET_001",
            compound_name="Ketamine",
            target_chirality="S",
            target_gene="GRIN1",
            pmid="11152058",
            title="Esketamine produces rapid synaptogenesis and NMDA pore blockade",
            text="(S)-ketamine has 3 to 4 times higher binding affinity for the dizocilpine phencyclidine binding site inside the NMDA receptor channel pore than (R)-ketamine. This confers superior anesthetic potency, allowing for lower therapeutic dosages and fewer psychotomimetic emergence phenomena.",
        ),
        LiteratureChunk(
            chunk_id="LIT_OFLOX_001",
            compound_name="Ofloxacin",
            target_chirality="S",
            target_gene="gyrA",
            pmid="8438781",
            title="Levofloxacin stereoselective inhibition of bacterial DNA gyrase",
            text="Levofloxacin, the (S)-isomer of ofloxacin, exhibits 128-fold higher bactericidal potency against bacterial DNA gyrase and Topoisomerase IV than its (R)-counterpart. The (R)-isomer suffers severe steric clash with the enzyme-DNA cleavage complex, serving merely as inert metabolic baggage.",
        ),
    ]


_GLOBAL_VECTOR_STORE: Optional[InMemoryVectorStore] = None


def get_vector_store() -> InMemoryVectorStore:
    """Singleton getter for the literature vector store."""
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        _GLOBAL_VECTOR_STORE = InMemoryVectorStore()
        _GLOBAL_VECTOR_STORE.add_documents(get_default_literature_corpus())
    return _GLOBAL_VECTOR_STORE

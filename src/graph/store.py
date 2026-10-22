"""
src/graph/store.py
──────────────────
Dual-backend Knowledge Graph store supporting:
1. In-Memory GraphStore (NetworkX MultiDiGraph) - Zero-setup, instant execution.
2. Neo4j GraphStore - Enterprise production store using official Bolt driver.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import networkx as nx

from .schema import (
    DrugNode,
    StereoisomerNode,
    TargetNode,
    BindingPocketNode,
    ClinicalLiabilityNode,
    BindingRelation,
    LiabilityRelation,
)
from ..config import settings

logger = logging.getLogger(__name__)


class InMemoryGraphStore:
    """NetworkX-backed stereochemical knowledge graph for zero-setup execution."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.drugs: Dict[str, DrugNode] = {}
        self.isomers: Dict[str, StereoisomerNode] = {}
        self.targets: Dict[str, TargetNode] = {}
        self.pockets: Dict[str, BindingPocketNode] = {}
        self.liabilities: Dict[str, ClinicalLiabilityNode] = {}

    def add_drug(self, drug: DrugNode):
        self.drugs[drug.id] = drug
        self.graph.add_node(drug.id, label="Drug", data=drug.model_dump())

    def add_stereoisomer(self, isomer: StereoisomerNode):
        self.isomers[isomer.id] = isomer
        self.graph.add_node(isomer.id, label="Stereoisomer", data=isomer.model_dump())
        # Add edge from Parent Drug to Stereoisomer
        if isomer.parent_drug_id in self.drugs:
            self.graph.add_edge(isomer.parent_drug_id, isomer.id, relation="HAS_STEREOISOMER")

    def add_target(self, target: TargetNode):
        self.targets[target.id] = target
        self.graph.add_node(target.id, label="Target", data=target.model_dump())

    def add_binding_pocket(self, pocket: BindingPocketNode):
        self.pockets[pocket.id] = pocket
        self.graph.add_node(pocket.id, label="BindingPocket", data=pocket.model_dump())
        if pocket.target_id in self.targets:
            self.graph.add_edge(pocket.target_id, pocket.id, relation="HAS_POCKET")

    def add_liability(self, liability: ClinicalLiabilityNode):
        self.liabilities[liability.id] = liability
        self.graph.add_node(liability.id, label="ClinicalLiability", data=liability.model_dump())

    def add_binding_relation(self, rel: BindingRelation):
        self.graph.add_edge(
            rel.source_isomer_id,
            rel.target_id,
            relation="BINDS_TARGET",
            data=rel.model_dump(),
        )
        if rel.pocket_id and rel.pocket_id in self.pockets:
            self.graph.add_edge(
                rel.source_isomer_id,
                rel.pocket_id,
                relation="OCCUPIES_POCKET",
                data={"steric_clash_observed": rel.steric_clash_observed},
            )

    def add_liability_relation(self, rel: LiabilityRelation):
        self.graph.add_edge(
            rel.source_isomer_id,
            rel.liability_id,
            relation="TRIGGERS_LIABILITY",
            data=rel.model_dump(),
        )

    def get_isomers_for_drug(self, drug_name_or_id: str) -> List[StereoisomerNode]:
        """Find all stereoisomers belonging to a parent drug name or ID."""
        matched_drug_ids = [
            d_id for d_id, d in self.drugs.items()
            if d_id.lower() == drug_name_or_id.lower()
            or d.name.lower() == drug_name_or_id.lower()
            or d.generic_name.lower() == drug_name_or_id.lower()
        ]
        results: List[StereoisomerNode] = []
        for d_id in matched_drug_ids:
            for _, target_id, edge_data in self.graph.out_edges(d_id, data=True):
                if edge_data.get("relation") == "HAS_STEREOISOMER" and target_id in self.isomers:
                    results.append(self.isomers[target_id])
        return results

    def get_isomer_profile(self, isomer_id: str) -> Dict[str, Any]:
        """Extract complete multi-hop graph neighborhood for a specific stereoisomer."""
        if isomer_id not in self.isomers:
            return {}

        isomer = self.isomers[isomer_id]
        parent_drug = self.drugs.get(isomer.parent_drug_id)

        bindings = []
        liabilities = []

        for _, target_node, edge_data in self.graph.out_edges(isomer_id, data=True):
            rel_type = edge_data.get("relation")
            if rel_type == "BINDS_TARGET":
                target = self.targets.get(target_node)
                bindings.append({
                    "target": target.model_dump() if target else {"id": target_node},
                    "affinity": edge_data.get("data", {}),
                })
            elif rel_type == "TRIGGERS_LIABILITY":
                liability = self.liabilities.get(target_node)
                liabilities.append({
                    "liability": liability.model_dump() if liability else {"id": target_node},
                    "details": edge_data.get("data", {}),
                })

        return {
            "isomer": isomer.model_dump(),
            "parent_drug": parent_drug.model_dump() if parent_drug else None,
            "target_bindings": bindings,
            "clinical_liabilities": liabilities,
        }

    def find_chiral_disparity(self, drug_name_or_id: str) -> Dict[str, Any]:
        """
        Compare all enantiomers/isomers of a drug to identify eudismic ratios,
        safety disparities, and divergent receptor targets.
        """
        isomers = self.get_isomers_for_drug(drug_name_or_id)
        profiles = [self.get_isomer_profile(iso.id) for iso in isomers]
        return {
            "query_drug": drug_name_or_id,
            "isomer_count": len(isomers),
            "isomeric_profiles": profiles,
        }

    def stats(self) -> Dict[str, int]:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "drugs": len(self.drugs),
            "stereoisomers": len(self.isomers),
            "targets": len(self.targets),
            "pockets": len(self.pockets),
            "clinical_liabilities": len(self.liabilities),
        }


class Neo4jGraphStore:
    """Production Neo4j Bolt driver store with Cypher execution."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._init_constraints()

    def close(self):
        self.driver.close()

    def _init_constraints(self):
        """Create uniqueness constraints for fast indexed lookup."""
        queries = [
            "CREATE CONSTRAINT drug_id_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT isomer_id_unique IF NOT EXISTS FOR (i:Stereoisomer) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT target_id_unique IF NOT EXISTS FOR (t:Target) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT liability_id_unique IF NOT EXISTS FOR (l:ClinicalLiability) REQUIRE l.id IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Could not initialize Neo4j constraint: {e}")

    def execute_cypher(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]


_GLOBAL_GRAPH_STORE: Optional[InMemoryGraphStore] = None


def get_graph_store() -> InMemoryGraphStore:
    """Singleton getter for the in-memory graph store."""
    global _GLOBAL_GRAPH_STORE
    if _GLOBAL_GRAPH_STORE is None:
        _GLOBAL_GRAPH_STORE = InMemoryGraphStore()
    return _GLOBAL_GRAPH_STORE

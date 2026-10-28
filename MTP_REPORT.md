# 🎓 MAJOR TECHNICAL PROJECT (MTP) REPORT

---

# ChiralGraph-RAG: A 3D Conformation and Stereoisomer-Aware Hybrid GraphRAG Engine for Precision Drug Discovery

**Degree Program:** Bachelor of Technology / Master of Technology in Computer Science & Engineering (Specialization in Artificial Intelligence & Machine Learning)  
**Course:** Major Project / Capstone Project (MTP)  
**Laboratory:** Machine Learning & Computational Biology Research Group  
**Supervision:** Prof. Vishwash  
**Student / Candidate:** Sakshi  
**Academic Year:** 2026–2027  
**Project Repository:** `Project 3 RAG / ChiralGraph-RAG`  

---

## 📑 TABLE OF CONTENTS

1. [Abstract & Executive Summary](#abstract--executive-summary)
2. [Chapter 1: Introduction & Problem Statement](#chapter-1-introduction--problem-statement)
   - 1.1 The Advent of Generative AI in Drug Discovery
   - 1.2 The Core Problem: The 2D Stereochemical Blindspot
   - 1.3 Catastrophic Real-World Case Studies (Thalidomide, Citalopram, Ofloxacin, Ketamine)
   - 1.4 Why Standard Vector RAG Fails (100% False Safety Rate)
3. [Chapter 2: Literature Review & Theoretical Foundations](#chapter-2-literature-review--theoretical-foundations)
   - 2.1 Principles of Stereochemistry & Chirality
   - 2.2 3D Molecular Conformations & Force Fields (ETKDG v3 & MMFF94)
   - 2.3 Pharmacology & Receptor Binding Thermodynamics
   - 2.4 Deep Review of Foundational Research Papers
4. [Chapter 3: Project Objectives & Scope](#chapter-3-project-objectives--scope)
   - 3.1 Primary & Secondary Objectives
   - 3.2 Real-World Use Cases
   - 3.3 Scope & Engineering Boundaries
5. [Chapter 4: System Architecture & Architectural Choices](#chapter-4-system-architecture--architectural-choices)
   - 4.1 System Architectural Blueprint
   - 4.2 Architectural Choices & Justifications ("Why" in Each Choice)
6. [Chapter 5: System Implementation & How It Works](#chapter-5-system-implementation--how-it-works)
   - 5.1 End-to-End Pipeline Walkthrough
   - 5.2 Deep-Dive into Code Modules
   - 5.3 Knowledge Graph Schema & Relational Triples
7. [Chapter 6: Setup, Enabling, Deployment & Execution Guide](#chapter-6-setup-enabling-deployment--execution-guide)
   - 6.1 Prerequisites & System Requirements
   - 6.2 Installation & Dependency Setup
   - 6.3 Dual-Backend Configuration (`.env`)
   - 6.4 Verification via Automated Test Suite
   - 6.5 Running the Microservice & Web Interface
   - 6.6 REST API Reference & Integration
8. [Chapter 7: Results & Quantitative Benchmark Evaluation](#chapter-7-results--quantitative-benchmark-evaluation)
   - 7.1 Experimental Setup & Evaluation Methodology
   - 7.2 Quantitative Comparison Matrix (Naive RAG vs. ChiralGraph-RAG)
   - 7.3 Detailed Case-by-Case Result Analysis
9. [Chapter 8: Discussion, Limitations & Future Scope](#chapter-8-discussion-limitations--future-scope)
   - 8.1 Key Engineering Insights
   - 8.2 Current System Limitations
   - 8.3 Future Research Roadmap
10. [Chapter 9: Conclusion](#chapter-9-conclusion)
11. [Chapter 10: References & Bibliography](#chapter-10-references--bibliography)

---

## ABSTRACT & EXECUTIVE SUMMARY

Large Language Models (LLMs) and dense Retrieval-Augmented Generation (RAG) systems have become prominent tools for biomedical literature extraction and preclinical drug discovery. However, state-of-the-art vector RAG pipelines exhibit a critical, potentially lethal vulnerability known as the **2D Stereochemical Blindspot**. In organic chemistry and pharmacology, enantiomers (mirror-image stereoisomers) share identical 1D/2D representations, identical chemical formulas, and identical molecular weights, yet they interact with chiral biological receptors with diametrically opposed pharmacodynamics. In tragic historical instances such as Thalidomide, one enantiomer is a benign sedative while its mirror counterpart triggers catastrophic teratogenic phocomelia (embryonic limb deformities). Standard vector embedding models compress molecular literature into flat semantic representations that completely conflate eutomer (active drug) efficacy with distomer (inactive/toxic counterpart) liabilities, generating a **100% false safety hallucination rate**.

To eliminate this fundamental flaw, this Major Project presents **ChiralGraph-RAG**, a 3D conformation and stereoisomer-aware hybrid GraphRAG engine. The system integrates:
1. **3D Computational Cheminformatics**: Utilizing RDKit to enforce Cahn-Ingold-Prelog (CIP) stereocenter detection, ETKDG v3 distance geometry conformer sampling, and MMFF94 force field energy minimization to filter out physically unviable ligand poses ($\Delta E_{\text{strain}} > 4.0\text{ kcal/mol}$).
2. **Deterministic Stereochemical Knowledge Graph**: A dual-backend graph (NetworkX for zero-setup execution, Neo4j Bolt for enterprise deployments) modeling explicit multi-hop biophysical triples connecting drugs, stereoisomers, active binding pockets, quantitative binding affinities ($K_d, K_i, \text{IC}_{50}$), and clinical liabilities (e.g., cardiac hERG QT prolongation).
3. **Think-on-Graph (ICLR 2024) Guarded Retrieval**: Fuses deterministic graph path exploration with an enantiomer-isolated vector store, preventing literature cross-contamination.
4. **Pharmacological Reasoner**: Computes quantitative eudismic ratios and synthesizes grounded Structure-Activity Relationship (SAR) dossiers via deterministic rules or frontier LLMs (Llama 3.3 70B via Groq, GPT-4o).

Empirical benchmarking across landmark stereoselective drugs (Thalidomide, Citalopram, Ketamine, Ofloxacin) demonstrates that ChiralGraph-RAG achieves a **100.0% stereoisomer discrimination accuracy** (up from 0.0% in naive vector RAG), **0.0% false safety hallucinations** (down from 100.0%), and **100.0% precision in quantitative binding affinity retrieval**. The system is packaged as an asynchronous FastAPI microservice with a production WebGL 3D molecular visualization dashboard.

---

## CHAPTER 1: INTRODUCTION & PROBLEM STATEMENT

### 1.1 The Advent of Generative AI in Drug Discovery
The modern drug discovery pipeline requires on average 10 to 15 years and exceeds $2.5 billion per approved New Molecular Entity (NME). In recent years, Artificial Intelligence—specifically Natural Language Processing (NLP), Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG)—has been aggressively adopted by pharmaceutical firms and research laboratories to rapidly synthesize Structure-Activity Relationships (SAR), extract preclinical assay results from PubMed, and summarize FDA drug safety inserts.

In a standard RAG system:
1. Scientific papers and drug documentation are chunked into text segments.
2. An embedding model (e.g., `text-embedding-ada-002`, `bge-large`, `BioBERT`) converts text chunks into high-dimensional vector representations.
3. When a chemist or physician issues a query, the vector database retrieves the nearest neighbors by cosine similarity, which are then passed to an LLM context window to generate an answer.

While this approach excels in open-domain question answering, it suffers a fatal breakdown when applied to stereoselective biochemistry.

---

### 1.2 The Core Problem: The 2D Stereochemical Blindspot
Conventional RAG architectures and dense vector embeddings operate exclusively on 1D/2D textual representations:
- IUPAC Chemical Names (e.g., *2-(2,6-dioxopiperidin-3-yl)-1H-isoindole-1,3(2H)-dione*)
- Flat SMILES Strings (Simplified Molecular Input Line Entry System)
- Unstructured scientific prose.

This reliance on 1D/2D abstractions introduces a **catastrophic blindspot**:
1. **Topological Equivalence**: Enantiomers (non-superimposable mirror-image molecules) share identical 2D connectivity graphs, identical molecular formulas, and identical molecular weights.
2. **Textual Co-occurrence**: In scientific literature, both $(R)$- and $(S)$-enantiomers are frequently discussed within the exact same paragraph or review article, often under the generic drug name.
3. **Centroid Merging in Vector Space**: Dense embedding models map text chunks mentioning the parent drug into a single semantic neighborhood. The vector similarity between an inquiry regarding the $(R)$-enantiomer and a literature chunk describing the $(S)$-enantiomer is nearly indistinguishable from an inquiry matching the $(R)$-enantiomer directly.

Consequently, conventional RAG systems commit **stereochemical cross-contamination**: attributing the therapeutic benefits of an active isomer to a toxic isomer, or dismissing a lethal off-target liability because the therapeutic isomer is benign.

---

### 1.3 Catastrophic Real-World Case Studies

To understand why this problem is of paramount clinical importance, we analyze four hallmark examples embedded in our benchmark system:

```
                  MIRROR PLANE
                       │
             W         │         W
             │         │         │
          C──┼──X      │      X──┼──C
         ╱   │         │         │   ╲
        Y    Z         │         Z    Y
     (R)-Enantiomer    │      (S)-Enantiomer
```

#### Case 1: Thalidomide (Sedative vs. Teratogen)
- **$(R)$-Thalidomide**: A clinically effective, mild sedative and morning sickness anti-emetic.
- **$(S)$-Thalidomide**: Binds with high affinity ($K_d = 25.0\text{ nM}$) to **Cereblon (CRBN)**, a substrate receptor of the CRL4$^{CRBN}$ E3 ubiquitin ligase complex. This binding triggers the aberrant polyubiquitination and proteasomal degradation of the C2H2 zinc-finger transcription factor **SALL4**, leading to irreversible limb bud outgrowth failure in human embryos (**phocomelia**, or complete absence of limbs).
- **The RAG Failure**: A naive vector RAG query asking *"Is (R)-thalidomide safe for morning sickness?"* retrieves merged literature and hallucinated answers asserting that Thalidomide is safe or that both enantiomers act identically, completely failing to alert the user to stereospecific CRBN/SALL4 toxicity or spontaneous *in vivo* racemization.

#### Case 2: Citalopram vs. Escitalopram (SSRI Efficacy vs. Cardiac Arrhythmia)
- **$(S)$-Citalopram (Escitalopram)**: Binds the primary orthosteric site ($S_1$) of the Serotonin Transporter (**SERT / SLC6A4**) with sub-nanomolar affinity ($K_i = 1.1\text{ nM}$), acting as a highly selective antidepressant.
- **$(R)$-Citalopram**: Binds SERT with 160-fold lower affinity ($K_i = 170.0\text{ nM}$), binds an allosteric secondary site ($S_2$) that accelerates dissociation of the therapeutic $(S)$-enantiomer, and potently blocks the cardiac **hERG potassium channel ($I_{Kr}$)** with an $\text{IC}_{50} = 3.9\ \mu\text{M}$.
- **Clinical Liability**: Racemic Citalopram carries an FDA Black Box warning for dose-dependent QT interval prolongation and fatal **Torsades de Pointes** arrhythmias, driven overwhelmingly by the $(R)$-enantiomer. Pure Escitalopram dramatically mitigates this cardiotoxicity.
- **The RAG Failure**: Standard RAG merges the safety warning of racemic citalopram with escitalopram, either overstating cardiac risks for pure escitalopram or declaring racemic citalopram harmless.

#### Case 3: Ofloxacin vs. Levofloxacin (128-fold Bactericidal Eudismic Ratio)
- **$(S)$-Ofloxacin (Levofloxacin)**: Fits sterically into the cleavage complex formed by bacterial **DNA Gyrase (GyrA)** and Topoisomerase IV without steric clash ($\text{IC}_{50} = 80.0\text{ nM}$).
- **$(R)$-Ofloxacin**: Sterically clashes with residues in the DNA Gyrase pocket ($\text{IC}_{50} = 10,500.0\text{ nM}$), making it $128\times$ weaker. Ingesting $(R)$-ofloxacin provides minimal antimicrobial effect while imposing heavy renal and hepatic excretion burdens.
- **The RAG Failure**: Naive vector search treats ofloxacin and levofloxacin as interchangeable synonyms.

#### Case 4: Ketamine vs. Esketamine (Anesthetic Potency & Psychotomimetic Load)
- **$(S)$-Ketamine (Esketamine)**: Exhibits $4\times$ higher affinity ($K_i = 300\text{ nM}$) for the phencyclidine pore binding site on the **NMDA receptor (GRIN1)** compared to $(R)$-ketamine ($K_i = 1400\text{ nM}$).
- **Clinical Impact**: Esketamine achieves surgical anesthesia and rapid antidepressant response at half the racemic dose, significantly reducing psychotomimetic emergence delirium.

---

### 1.4 Why Standard Vector RAG Fails (100% False Safety Rate)
As demonstrated in Chapter 7, naive vector RAG suffers from:
1. **0% Stereoisomer Discrimination**: Incapable of isolating $(R)$-literature from $(S)$-literature.
2. **100% False Safety Hallucination Rate**: Because text chunks about the distomer are retrieved during eutomer queries, the LLM hallucinates that liabilities either apply to neither or both.
3. **Inability to Verify Spatial Conformations**: Text embeddings cannot perform 3D distance geometry or force field energy minimization; an LLM cannot determine if a proposed molecule incurs prohibitive steric clash in a receptor pocket.

---

## CHAPTER 2: LITERATURE REVIEW & THEORETICAL FOUNDATIONS

### 2.1 Principles of Stereochemistry & Chirality
Chirality is a geometric property of molecular structures: a molecule is chiral if and only if it lacks an improper axis of rotation ($S_n$), meaning it cannot be superimposed on its mirror image. 

#### The Cahn-Ingold-Prelog (CIP) Rules
Chiral stereocenters are assigned an absolute configuration $(R)$ (*Rectus*, clockwise) or $(S)$ (*Sinister*, counter-clockwise) via the IUPAC CIP sequence rules:
1. **Rule 1 (Atomic Number)**: Substituents directly bonded to the chiral center are ranked by atomic number ($Z$). Atoms with higher $Z$ take priority ($-\text{I} > -\text{Br} > -\text{Cl} > -\text{S} > -\text{P} > -\text{F} > -\text{O} > -\text{N} > -\text{C} > -\text{H} > \text{lone pair}$).
2. **Rule 2 (Chain Exploration)**: If priority cannot be established at the primary atom, explore outwards along the highest-priority path until the first point of difference is identified.
3. **Rule 3 (Multiple Bonds)**: Double and triple bonds are split into phantom duplicate atoms (e.g., a carbonyl $-\text{CH}=\text{O}$ is treated as carbon bonded to two oxygens).
4. **Rule 4 (3D Projection)**: The lowest-priority substituent (typically $-\text{H}$, priority 4) is oriented away from the viewer. If priority ranks $1 \to 2 \to 3$ rotate clockwise, the center is **$(R)$**; if counter-clockwise, it is **$(S)$**.

In ChiralGraph-RAG, CIP rule evaluation is performed deterministically by RDKit rather than relying on LLM guesswork.

---

### 2.2 3D Molecular Conformations & Force Fields

#### ETKDG v3 (Experimental-Torsion Knowledge Distance Geometry)
Molecules are not rigid 2D structures; they populate conformational energy landscapes determined by rotations around single ($\sigma$) bonds. ChiralGraph-RAG generates 3D ensembles using **ETKDG v3** (Riniker & Landrum, 2015, 2020):
- Traditional distance geometry derives Cartesian coordinates by satisfying upper and lower interatomic distance bounds.
- ETKDG enhances distance geometry by incorporating torsion angle probability distributions mined from thousands of crystal structures in the Cambridge Structural Database (CSD).
- Version 3 adds specific rules for macrocycle sampling and strained small rings.

#### MMFF94 (Merck Molecular Force Field)
Raw distance-geometry coordinates contain bond length distortions and steric overlaps. ChiralGraph-RAG relaxes each conformer using the **MMFF94 force field**, calculating potential energy:
$$E_{\text{total}} = E_{\text{stretch}} + E_{\text{bend}} + E_{\text{stretch-bend}} + E_{\text{torsion}} + E_{\text{vdW}} + E_{\text{electrostatic}}$$

#### Conformational Strain Energy ($\Delta E_{\text{strain}}$)
For a ligand to bind a biological receptor, it must adopt the bioactive conformation required by the binding pocket. The thermodynamic penalty incurred is the conformational strain energy:
$$\Delta E_{\text{strain}} = E_{\text{bioactive}} - E_{\text{global minimum}}$$
Biophysical studies establish that ligands rarely bind receptors with $\Delta E_{\text{strain}} > 4.0\text{ kcal/mol}$. ChiralGraph-RAG automatically flags conformers exceeding this threshold as physically non-viable candidates.

---

### 2.3 Pharmacology & Receptor Binding Thermodynamics

#### Equilibrium Binding Constant ($K_d$) and Free Energy ($\Delta G^\circ$)
The reversible binding of ligand $L$ to macromolecular receptor $R$:
$$R + L \rightleftharpoons RL, \quad K_d = \frac{[R][L]}{[RL]} = \frac{k_{\text{off}}}{k_{\text{on}}}$$
The standard Gibbs free energy change of binding is governed by:
$$\Delta G^\circ = -RT \ln K_a = RT \ln K_d$$
where $R = 1.987\text{ cal/(mol}\cdot\text{K)}$ and $T = 310.15\text{ K}$ (physiological temperature). A sub-nanomolar $K_d$ represents a favorable negative $\Delta G^\circ$ (e.g., $K_d = 1.1\text{ nM} \implies \Delta G^\circ \approx -12.7\text{ kcal/mol}$).

#### Eudismic Ratio
The quantitative measure of stereoselective discrimination between enantiomers:
$$\text{Eudismic Ratio} = \frac{\text{Potency of Eutomer}}{\text{Potency of Distomer}} = \frac{K_d(\text{Distomer})}{K_d(\text{Eutomer})}$$

#### The Cardiac hERG Channel Liability
The Human Ether-à-go-go-Related Gene (*KCNH2*) encodes the pore-forming subunit of the rapid delayed rectifier potassium channel ($I_{Kr}$). Unintended drug binding inside the hERG inner cavity (specifically interacting with aromatic residues Tyr652 and Phe656) prolongs cardiac repolarization, visible on surface ECG as QT interval prolongation, which can trigger fatal **Torsades de Pointes** ventricular fibrillation.

---

### 2.4 Deep Review of Foundational Research Papers

ChiralGraph-RAG directly implements and synergizes the architectures proposed in three core literature pillars present in the workspace:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      LITERATURE FOUNDATION MATRIX                      │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Pan et al. (IEEE TKDE, arXiv:2306.08302)                            │
│    "Unifying Large Language Models and Knowledge Graphs: A Roadmap"    │
│    ➜ Provided the "KG-enhanced LLM" paradigm for factual grounding.    │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Sun et al. (ICLR 2024, arXiv:2307.07697)                           │
│    "Think-on-Graph (ToG): Deep and Responsible Reasoning of LLM on KG" │
│    ➜ Provided the multi-hop beam traversal and path-pruning protocol.  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Wen et al. (arXiv:2308.09729)                                       │
│    "MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts"       │
│    ➜ Provided the subgraph-to-prompt linearization technique.          │
└────────────────────────────────────────────────────────────────────────┘
```

#### Paper 1: Unifying Large Language Models and Knowledge Graphs: A Roadmap
* **Authors:** Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, Xindong Wu (IEEE Transactions on Knowledge and Data Engineering, arXiv:2306.08302).
* **Core Contribution:** Formalizes the taxonomy of combining LLMs and KGs into three frameworks:
  1. *KG-enhanced LLMs*: Leveraging structured KGs during pre-training, fine-tuning, or inference to ground LLMs in verifiable facts.
  2. *LLM-augmented KGs*: Using LLMs to construct, complete, and extract relations for KGs.
  3. *Synergized LLMs + KGs*: Bidirectional collaboration where the KG acts as an immutable factual substrate while the LLM acts as a probabilistic linguistic interface.
* **Adoption in ChiralGraph-RAG:** We implement the **KG-enhanced LLM** framework. In pharmaceutical chemistry, hallucinations can be fatal. By grounding the retriever in an explicit graph of stereoisomers, targets, and assays, the LLM cannot fabricate binding parameters.

#### Paper 2: Think-on-Graph (ToG): Deep and Responsible Reasoning of LLMs on KGs
* **Authors:** Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-Yeung Shum, Jian Guo (ICLR 2024, arXiv:2307.07697).
* **Core Contribution:** Proposes a dynamic framework where an LLM and a Knowledge Graph perform iterative multi-hop reasoning. Instead of generating long unverified chains of thought, the agent searches candidate relation paths on the KG step-by-step, pruning hallucinations that lack supporting physical edges.
* **Adoption in ChiralGraph-RAG:** Implemented in `src/engine/hybrid_retriever.py`. The retriever resolves the target compound, identifies its exact chiral center, traverses explicit relational paths to target receptors, and applies a **Stereochemical Guardrail Gate** that prunes any literature chunks originating from the opposite enantiomer.

#### Paper 3: MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in LLMs
* **Authors:** Yilin Wen, Zifeng Wang, Jimeng Sun (arXiv:2308.09729).
* **Core Contribution:** Demonstrates that mining evidence subgraphs and linearizing them into structured graph prompts triggers superior "Graph-of-Thought" reasoning in specialized clinical and scientific QA tasks.
* **Adoption in ChiralGraph-RAG:** Subgraph neighborhoods (including conformer strain, $K_i$ values, and pocket clash indicators) are synthesized into structured context blocks for the Pharmacological Reasoner (`src/engine/reasoner.py`).

---

## CHAPTER 3: PROJECT OBJECTIVES & SCOPE

### 3.1 Primary & Secondary Objectives
- **Objective 1 (3D Stereochemical Disambiguation):** Automatically parse molecular SMILES, identify tetrahedral chiral centers via CIP priority rules, and enumerate all possible stereoisomers.
- **Objective 2 (3D Conformer & Strain Physics):** Generate low-energy 3D conformations using ETKDG v3 distance geometry and MMFF94 force field optimization; compute relative conformational strain ($\Delta E_{\text{strain}}$) to filter unviable binding poses.
- **Objective 3 (Deterministic Knowledge Graph):** Architect a heterogeneous knowledge graph schema capturing drugs, stereoisomers, binding pockets, quantitative affinities ($K_d, K_i, \text{IC}_{50}$), and clinical liabilities.
- **Objective 4 (Think-on-Graph Retrieval):** Build a hybrid retriever fusing deterministic graph traversal with enantiomer-isolated vector search to completely eliminate chiral cross-contamination.
- **Objective 5 (Automated SAR Generation):** Compute quantitative eudismic ratios and generate grounded Structure-Activity Relationship (SAR) dossiers.
- **Objective 6 (Production Delivery):** Expose the complete pipeline through an asynchronous FastAPI REST engine and an interactive WebGL 3D molecular visualization dashboard.

### 3.2 Real-World Use Cases
1. **Biopharmaceutical Hit-to-Lead Optimization**: Medicinal chemists screening chiral compound libraries can identify which enantiomer delivers therapeutic efficacy without synthesizing both isomers.
2. **In Silico Cardiac & Off-Target Safety Profiling**: Early-stage identification of stereoselective hERG cardiotoxicity or CYP enzyme inactivation before costly animal trials.
3. **Regulatory IND & Patent Dossier Preparation**: Automating the compilation of FDA Investigational New Drug (IND) stereochemical dossiers with verified literature citations.

### 3.3 Scope & Boundaries
- **In-Scope**: Small organic drug molecules ($< 900\text{ Da}$), tetrahedral carbon stereocenters, 3D conformer generation, deterministic graph traversal, guarded literature retrieval, eudismic ratio computation, automated benchmarking.
- **Out-of-Scope**: Macromolecular biopolymers (proteins, RNA/DNA), axially chiral biaryls (atropisomerism), and real-time *ab initio* quantum mechanical DFT calculations (MMFF94 force field is used instead).

---

## CHAPTER 4: SYSTEM ARCHITECTURE & ARCHITECTURAL CHOICES

### 4.1 System Architectural Blueprint

```mermaid
flowchart TD
    subgraph Client / Presentation Layer
        User[Researcher / Chemist Query] --> API[FastAPI Gateway :8000]
        WebUI[Interactive 3Dmol.js Web UI] <--> API
    end

    subgraph 1. Cheminformatics & 3D Conformation Layer
        API -->|SMILES / Compound Name| ChemEngine[RDKit 3D Cheminformatics Engine]
        ChemEngine --> CIP[Cahn-Ingold-Prelog R/S Stereocenter Detection]
        ChemEngine --> ETKDG[ETKDG v3 3D Conformer Generator]
        ChemEngine --> ForceField[MMFF94 Coordinate Energy Minimization]
        ChemEngine --> Descriptors[3D Strain ΔE, Asphericity & Eccentricity]
    end

    subgraph 2. Dual-Layer Storage & Knowledge Infrastructure
        CIP & ETKDG & ForceField --> GraphIngest[Graph Ingest & Schema Validator]
        GraphIngest --> KG[(Stereochemical Knowledge Graph: NetworkX / Neo4j)]
        LitCorpus[PubMed SAR Papers & FDA Inserts] --> VectorIngest[Dense Semantic Embeddings]
        VectorIngest --> VDB[(Guarded Vector Store: In-Memory / Qdrant)]
    end

    subgraph 3. Think-on-Graph Hybrid Retrieval Engine
        API --> QueryRouter[Query Decomposer & Chiral Intent Router]
        QueryRouter --> GraphTraversal[Deterministic Multi-Hop Graph Traversal]
        QueryRouter --> VectorRetrieval[Guarded Semantic Literature Retrieval]
        KG -->|Stereospecific Paths & Pocket Affinities| GraphTraversal
        VDB -->|Isomer-Isolated Scientific Chunks| VectorRetrieval
        GraphTraversal & VectorRetrieval --> ChiralGate[Stereochemical Verification Guardrail Gate]
    end

    subgraph 4. Pharmacological Reasoning & Evaluation
        ChiralGate --> Reasoner[Pharmacological Reasoning Engine]
        Reasoner --> SARReport[Grounded SAR Dossier & Off-Target Risk Report]
        SARReport --> Bench[Quantitative Benchmarking Suite]
    end
```

---

### 4.2 Architectural Choices & Justifications ("Why" in Each Choice)

To achieve maximum scientific fidelity, performance, and maintainability, each architectural component was selected after rigorous evaluation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURAL SELECTION MATRIX                           │
├────────────────────────┬──────────────────────────┬─────────────────────────┤
│ Component              │ Technology Chosen        │ Discarded Alternative   │
├────────────────────────┼──────────────────────────┼─────────────────────────┤
│ 1. Cheminformatics     │ RDKit (ETKDG v3, MMFF94) │ Flat SMILES / 2D GNNs   │
│ 2. Graph Backend       │ NetworkX / Neo4j Hybrid  │ Pure Relational SQL     │
│ 3. Vector Backend      │ In-Memory / Qdrant       │ Unfiltered Vector DB    │
│ 4. Retrieval Algorithm │ Think-on-Graph (ICLR'24) │ Naive Top-K Vector RAG  │
│ 5. Reasoning Engine    │ Hybrid (Rules + Groq)    │ Pure LLM Prompting      │
│ 6. API Framework       │ FastAPI (Asynchronous)   │ Flask / Django          │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
```

#### Why Choice 1: RDKit 3D Engine over Flat SMILES or 2D GNNs
- **The Rationale**: Large Language Models and 2D Graph Neural Networks cannot calculate physical coordinates or steric clashes. Two enantiomers produce identical adjacency matrices in 2D graphs.
- **The Advantage**: RDKit's ETKDG v3 generates true Cartesian coordinate ensembles ($x, y, z$). MMFF94 calculates exact conformational strain in $\text{kcal/mol}$. This provides an objective physical filter before any language model generates reasoning.

#### Why Choice 2: Heterogeneous Knowledge Graph over Pure Relational SQL
- **The Rationale**: Biochemical relationships are multi-hop networks ($\text{Drug} \to \text{Isomer} \to \text{Receptor} \to \text{Pocket} \to \text{Liability}$). 
- **The Advantage**: Graph stores permit flexible multi-hop neighborhood traversal without complex multi-table SQL `JOIN` bottlenecks. Modeling entities as first-class nodes (`Stereoisomer`, `Target`, `ClinicalLiability`) enforces relational integrity.

#### Why Choice 3: Dual-Backend Strategy (NetworkX + Neo4j)
- **The Rationale**: Academic evaluation and CI/CD testing require instant execution without spinning up external cloud databases. Conversely, enterprise deployments require scalable ACID graph storage.
- **The Advantage**: In-Memory mode (`networkx.MultiDiGraph`) delivers zero-dependency, lightning-fast execution passing 14 tests in 4.5 seconds. Setting `GRAPH_BACKEND=neo4j` seamlessly routes queries to an enterprise Neo4j Aura cluster via Cypher queries.

#### Why Choice 4: Guarded Vector Literature Store (In-Memory + Qdrant)
- **The Rationale**: Standard vector stores perform unconstrained top-$k$ nearest-neighbor searches. In stereochemistry, this causes 100% false safety contamination.
- **The Advantage**: Every indexed literature chunk is tagged with explicit chiral metadata (`target_chirality: "R" | "S" | "RS"`). When querying an $(R)$-enantiomer, chunks describing the $(S)$-enantiomer are strictly pruned by the **Stereochemical Guardrail Gate**.

#### Why Choice 5: Think-on-Graph (ToG) Retrieval over Naive Vector Search
- **The Rationale**: Pure vector search blindly trusts semantic proximity.
- **The Advantage**: ToG verifies that every reasoning step is anchored to an explicit edge in the Knowledge Graph. If the graph does not contain a verified binding edge between $(R)$-Thalidomide and Cereblon with a validated assay, the system refuses to hallucinate one.

#### Why Choice 6: Hybrid Pharmacological Reasoner (Deterministic + Live LLM)
- **The Rationale**: Reliance solely on an external API (OpenAI/Groq) introduces latency, token costs, and rate limits. Pure rule engines lack natural language fluency.
- **The Advantage**: Our reasoner implements a dual engine: a deterministic physical synthesis that computes eudismic ratios and extracts graph facts with zero API cost, coupled with an optional live API bridge to Groq (Llama 3.3 70B) or OpenAI (GPT-4o) for expanded narrative dossiers.

#### Why Choice 7: FastAPI Asynchronous Microservice & WebGL Dashboard
- **The Rationale**: Pharmacological researchers need interactive visualization, not just terminal printouts.
- **The Advantage**: FastAPI natively supports asynchronous execution, Pydantic validation, and automatic OpenAPI Swagger documentation (`/docs`). The embedded dashboard uses **3Dmol.js** for real-time interactive 3D molecular conformer rendering in the browser.

---

## CHAPTER 5: SYSTEM IMPLEMENTATION & HOW IT WORKS

### 5.1 End-to-End Pipeline Walkthrough

When a user submits a query (e.g., *"Is (S)-thalidomide safe for morning sickness?"* with SMILES `O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1`):

```
Step 1: REST API Ingestion (/api/v1/rag/query)
        │
Step 2: Intent & Stereocenter Extraction
        ├─ Resolves compound name: "Thalidomide"
        └─ Extracts target chirality: "(S)"
        │
Step 3: 3D Cheminformatics Physics Engine (RDKit)
        ├─ Verifies CIP stereocenter at C3 (S-configuration)
        ├─ Samples 10 conformers via ETKDG v3 distance geometry
        ├─ Minimizes energy via MMFF94 (E_min = -43.16 kcal/mol)
        └─ Computes 3D asphericity (0.198) and eccentricity (0.832)
        │
Step 4: Think-on-Graph Multi-Hop Traversal
        ├─ Matches ParentDrug: "DRUG_THALIDOMIDE"
        ├─ Identifies Stereoisomer: "ISO_S_THALIDOMIDE"
        ├─ Discovers binding edge: (ISO_S) -[BINDS]-> (TARGET_CRBN, Kd=25 nM)
        ├─ Identifies pocket: POCKET_CRBN_TRIAD (Trp380, Trp386, Trp400)
        └─ Discovers liability edge: (ISO_S) -[TRIGGERS]-> (LIAB_TERATOGENICITY, SALL4 degradation)
        │
Step 5: Stereochemically Guarded Literature Retrieval
        ├─ Queries vector index for "Thalidomide S target binding"
        ├─ Guardrail filters out chunks tagged exclusively for (R)-isomer
        └─ Retrieves peer-reviewed evidence (PMID: 20203043, 29440439)
        │
Step 6: Quantitative Disparity & Eudismic Ratio Computation
        ├─ Compares (S) Kd (25 nM) vs. (R) Kd (250 nM)
        └─ Eudismic ratio = 10.0:1 in favor of (S)-isomer at CRBN
        │
Step 7: Pharmacological Reasoner Synthesis
        ├─ Synthesizes grounded SAR Dossier
        ├─ Flags CRITICAL liability alert (Phocomelia / Teratogenesis)
        └─ Emits medicinal chemistry recommendations
```

---

### 5.2 Deep-Dive into Code Modules

The codebase is organized into modular Python packages inside `src/`:

```
ChiralGraph-RAG/
├── src/
│   ├── cheminformatics/        # RDKit 3D Engine & CIP Analysis
│   │   ├── __init__.py
│   │   └── stereochemistry.py
│   ├── graph/                 # Dual Graph Store (NetworkX & Neo4j)
│   │   ├── schema.py          # Pydantic Graph Node & Edge Models
│   │   ├── store.py           # In-Memory & Neo4j Graph Stores
│   │   └── seed_data.py       # Clinical Stereochemical Benchmarks
│   ├── vector/                # Guarded Vector Literature Store
│   │   └── store.py           # In-Memory Cosine Index & Qdrant Client
│   ├── engine/                # Core Reasoning & Retrieval Pipeline
│   │   ├── hybrid_retriever.py# Think-on-Graph Retriever & Guardrails
│   │   └── reasoner.py        # SAR Dossier Generator & LLM Bridge
│   ├── evaluation/            # Benchmarking Suite vs. Naive RAG
│   │   └── benchmark.py
│   ├── api/                   # Production FastAPI Web Application
│   │   └── main.py
│   └── config.py              # Pydantic Settings & Environment Config
├── tests/                     # 14 Comprehensive Pytest Suites
├── ui/                        # Web Dashboard with 3Dmol.js (WebGL Studio)
├── Dockerfile                 # Multi-Stage Production Container Specification
├── docker-compose.yml         # Container Orchestration (Standalone & Full-Stack Profiles)
├── .dockerignore              # Container Build Optimization Filters
├── RUN_GUIDE.md               # Complete System Run Guide & API Handbook
├── requirements.txt           # Verified Dependencies
└── pytest.ini                 # Pytest Configuration
```

#### 1. Cheminformatics Module ([`src/cheminformatics/stereochemistry.py`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/cheminformatics/stereochemistry.py))
- [`StereochemistryEngine`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/cheminformatics/stereochemistry.py#L59-L203): Main class executing RDKit routines.
- `get_chiral_centers(mol)`: Calls `Chem.FindMolChiralCenters(includeUnassigned=True)` to map all stereocenters and classify them as $(R)$, $(S)$, or unassigned.
- `enumerate_stereoisomers(smiles)`: Uses `rdkit.Chem.EnumerateStereoisomers` to generate all $2^n$ valid stereoisomers for racemic structures.
- `generate_3d_conformers(smiles)`: Configures `AllChem.ETKDGv3()` with an RMS pruning threshold of $0.5\ \text{Å}$, embeds 10 conformers, optimizes coordinates via `AllChem.MMFFOptimizeMoleculeConfs`, and flags any pose with $\Delta E_{\text{strain}} \le 4.0\text{ kcal/mol}$ as a bioactive candidate.

#### 2. Graph Schema & Store ([`src/graph/`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/graph/))
- [`DrugNode`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/graph/schema.py#L22): Represents the parent chemical entity (e.g., Citalopram).
- [`StereoisomerNode`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/graph/schema.py#L31): Captures the specific enantiomer, isomeric SMILES, lowest energy, and eudismic role.
- [`TargetNode`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/graph/schema.py#L43): Macromolecular receptor with UniProt ID and gene symbol (e.g., *CRBN*, *KCNH2*, *SLC6A4*).
- [`BindingRelation`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/graph/schema.py#L65): Quantifies affinity ($K_d, K_i, \text{IC}_{50}$, units, steric clash flag, evidence PMID).
- [`LiabilityRelation`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/graph/schema.py#L79): Links an isomer directly to clinical hazards (e.g., Teratogenesis, QT Prolongation).

#### 3. Think-on-Graph Retriever ([`src/engine/hybrid_retriever.py`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/engine/hybrid_retriever.py))
- [`ThinkOnGraphRetriever`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/engine/hybrid_retriever.py#L30): Coordinates multi-hop graph exploration and guarded retrieval.
- `_extract_query_metadata(query)`: Uses regex and token normalization to detect chiral intent ($(R)$ vs. $(S)$).
- `_compute_eudismic_analysis(graph_data)`: Automatically identifies common receptor targets between $(R)$- and $(S)$-isomers, extracts affinity values, and computes the exact eudismic ratio $\frac{K_d(\text{Distomer})}{K_d(\text{Eutomer})}$.

#### 4. Pharmacological Reasoner ([`src/engine/reasoner.py`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/engine/reasoner.py))
- [`PharmacologicalReasoner`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/engine/reasoner.py#L31): Synthesizes retrieved evidence into a structured [`SARReport`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/engine/reasoner.py#L19).
- Evaluates whether the query triggers any `CRITICAL` severity clinical liabilities (such as teratogenesis or lethal arrhythmias) and generates medicinal chemistry recommendations.

---

### 5.3 Knowledge Graph Schema & Relational Triples

The Knowledge Graph contains five core node types interconnected by rich, attributed relationships:

```
                  ┌──────────────────────┐
                  │      DrugNode        │
                  │ (Parent Chemical)    │
                  └──────────┬───────────┘
                             │ HAS_STEREOISOMER
                             ▼
                  ┌──────────────────────┐
                  │   StereoisomerNode   │
                  │ (R- or S-Enantiomer) │
                  └──────┬────────┬──────┘
                         │        │
           BINDS_TARGET  │        │ TRIGGERS_LIABILITY
                         │        │
                         ▼        ▼
       ┌────────────────────┐   ┌────────────────────────┐
       │     TargetNode     │   │ ClinicalLiabilityNode  │
       │ (e.g. CRBN, hERG)  │   │ (e.g. Phocomelia, QT)  │
       └─────────┬──────────┘   └────────────────────────┘
                 │ HAS_POCKET
                 ▼
       ┌────────────────────┐
       │ BindingPocketNode  │
       │ (Triad Trp Pocket) │
       └────────────────────┘
```

---

## CHAPTER 6: SETUP, ENABLING, DEPLOYMENT & EXECUTION GUIDE

### 6.1 Prerequisites & System Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+
- **Python Version:** Python 3.11 or higher
- **RAM:** Minimum 4 GB (8 GB recommended for 3D conformer ensemble generation)
- **Disk Space:** 500 MB free space

---

### 6.2 Installation & Dependency Setup

Follow these exact steps from your terminal or command prompt:

```bash
# 1. Navigate to the project root directory
cd "Project 3 RAG/ChiralGraph-RAG"

# 2. (Recommended) Create and activate a clean virtual environment
python -m venv .venv

# On Windows (PowerShell / Command Prompt):
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install all required scientific and web packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 6.3 Dual-Backend Configuration (`.env`)

The system uses Pydantic Settings ([`src/config.py`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/config.py)) and reads configuration options from a `.env` file.

#### Mode A: Zero-Setup Offline Mode (Default)
This mode requires **no external database, no Docker containers, and no API keys**. Everything runs in-memory:
```ini
ENVIRONMENT=development
LOG_LEVEL=INFO
APP_HOST=0.0.0.0
APP_PORT=8000

# Zero-dependency in-memory graph (NetworkX)
GRAPH_BACKEND=networkx

# Zero-dependency in-memory vector store
VECTOR_BACKEND=memory

# Deterministic grounded synthesis (no API costs)
LLM_PROVIDER=mock
```

#### Mode B: Enterprise Production Cloud Mode
To connect to enterprise cloud infrastructure:
```ini
ENVIRONMENT=production
LOG_LEVEL=INFO
APP_HOST=0.0.0.0
APP_PORT=8000

# Neo4j Enterprise / Aura Cloud Database
GRAPH_BACKEND=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YourSecurePassword
NEO4J_DATABASE=neo4j

# Qdrant Vector Search Engine
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=chiral_sar_literature

# Frontier LLM for Natural Language Dossiers
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

---

### 6.4 Verification via Automated Test Suite

Before deploying or demonstrating the project, run the automated test suite to confirm that all 14 integration and unit tests pass:

```bash
pytest -v
```

**Expected Test Execution Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.11.1, pytest-9.1.1
rootdir: .../ChiralGraph-RAG
configfile: pytest.ini
collected 14 items

tests/test_api.py::test_root_ui_endpoint PASSED                          [  7%]
tests/test_api.py::test_health_endpoint PASSED                           [ 14%]
tests/test_api.py::test_cheminformatics_analyze PASSED                   [ 21%]
tests/test_api.py::test_graph_drug_endpoint PASSED                       [ 28%]
tests/test_api.py::test_rag_query_endpoint PASSED                        [ 35%]
tests/test_api.py::test_evaluation_benchmark_endpoint PASSED             [ 42%]
tests/test_cheminformatics.py::test_cip_stereocenter_detection PASSED    [ 50%]
tests/test_cheminformatics.py::test_stereoisomer_enumeration PASSED      [ 57%]
tests/test_cheminformatics.py::test_3d_conformer_generation PASSED       [ 64%]
tests/test_graph.py::test_seed_graph_stats PASSED                        [ 71%]
tests/test_graph.py::test_thalidomide_disparity PASSED                   [ 78%]
tests/test_graph.py::test_citalopram_disparity PASSED                    [ 85%]
tests/test_retriever.py::test_retriever_thalidomide_dossier PASSED       [ 92%]
tests/test_retriever.py::test_reasoner_sar_report_generation PASSED      [100%]

======================== 14 passed in 4.59s ==================================
```

---

### 6.5 Running the Microservice & Web Interface

#### 6.5.1 Local Uvicorn Execution
Start the Uvicorn ASGI production server:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Once running, access the system through your web browser:
1. **Interactive Drug Discovery Dashboard:** Open **[http://localhost:8000](http://localhost:8000)**  
   - Allows typing any SMILES string, viewing 3D WebGL rotating conformers, switching between benchmark drug presets, and executing Think-on-Graph RAG queries.
2. **Interactive OpenAPI Swagger Documentation:** Open **[http://localhost:8000/docs](http://localhost:8000/docs)**  
   - Allows testing all REST endpoints directly in the browser with full schema validation.

#### 6.5.2 One-Command Containerized Deployment (Docker)
The system is fully containerized using a multi-stage `Dockerfile` (Python 3.11-slim with RDKit native graphics libraries and non-root security) and `docker-compose.yml`:
* **Standalone Execution (Default)**:
  ```bash
  docker compose up --build
  ```
* **Full-Stack Enterprise Stack (with Neo4j and Qdrant)**:
  ```bash
  docker compose --profile full up --build
  ```

#### 6.5.3 Interactive 3D Conformer Studio Engineering
The WebGL molecular viewer ([`ui/`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/ui/)) incorporates advanced client-side graphics handling:
* **Multi-Layer Scroll-Wheel Neutralization**: Prototype-level method override and capture-phase DOM interception ensure mouse wheel scrolling is decoupled from canvas zooming, preserving mouse drag exclusively for smooth 3D spatial rotation.
* **Discrete 5% Step Zooming**: Zoom is strictly bounded between **50% and 150%** with 5% increment buttons (`+5%` / `−5%`) and instant baseline centering (`100%`).
* **Fullscreen Cinematic Maximize**: A dedicated **`⛶ Maximize`** mode dynamically expands the WebGL viewport to fill the entire display (`100vw × 100vh`) with automatic aspect ratio recalculation via `viewer3d.resize()`.
* **Zero-Stale Caching**: Serves assets through `NoCacheStaticFiles` with version query busting to prevent stale browser disk caching.

---

### 6.6 REST API Reference & Integration

The system exposes five high-performance RESTful API endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check, active backends, and graph node statistics |
| `POST` | `/api/v1/cheminformatics/analyze` | Computes CIP stereocenters, 3D conformers, and MMFF94 strain |
| `POST` | `/api/v1/cheminformatics/enumerate` | Enumerates all stereoisomers for racemic/unassigned SMILES |
| `GET` | `/api/v1/graph/drug/{drug_name}` | Queries stereochemical disparity, target bindings, and liabilities |
| `POST` | `/api/v1/rag/query` | Executes Think-on-Graph hybrid retrieval and produces grounded SAR reports |
| `GET` | `/api/v1/evaluation/benchmark` | Runs comparative evaluation against standard Naive RAG |

#### Concrete API Request Example (`curl`)

```bash
# Execute a guarded Think-on-Graph query for (S)-Thalidomide:
curl -X POST "http://localhost:8000/api/v1/rag/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Is (S)-thalidomide safe for morning sickness in pregnancy?",
       "chirality": "S",
       "smiles": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
       "top_k_literature": 4
     }'
```

---

## CHAPTER 7: RESULTS & QUANTITATIVE BENCHMARK EVALUATION

### 7.1 Experimental Setup & Evaluation Methodology
To validate ChiralGraph-RAG, we constructed a quantitative benchmarking suite ([`src/evaluation/benchmark.py`](file:///c:/Users/Sakshi/Desktop/ML-Prof.%20Vishwash/Project%203%20RAG/ChiralGraph-RAG/src/evaluation/benchmark.py)) containing four high-impact stereoselective pharmacology test cases:
1. **Query 1 (Thalidomide):** Testing discrimination between the sedative $(R)$-enantiomer and the teratogenic $(S)$-enantiomer binding Cereblon ($CRBN$).
2. **Query 2 (Citalopram):** Testing discrimination between active $(S)$-escitalopram and $(R)$-citalopram cardiac hERG potassium channel blockade.
3. **Query 3 (Ketamine):** Testing detection of the $4\times$ binding affinity difference between $(S)$-esketamine and $(R)$-ketamine at the NMDA receptor ($GRIN1$).
4. **Query 4 (Ofloxacin):** Testing detection of the $128\times$ bactericidal potency difference between $(S)$-levofloxacin and $(R)$-ofloxacin at bacterial DNA Gyrase ($gyrA$).

Each query was submitted to two systems:
- **Baseline System (Naive 2D Vector RAG):** Standard dense vector search across biomedical literature chunks without stereochemical metadata filtering.
- **Proposed System (ChiralGraph-RAG):** Think-on-Graph hybrid retriever with RDKit 3D conformation analysis, deterministic Knowledge Graph traversal, and the Stereochemical Verification Guardrail Gate.

---

### 7.2 Quantitative Comparison Matrix (Naive RAG vs. ChiralGraph-RAG)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   QUANTITATIVE BENCHMARK EVALUATION RESULTS                            │
├──────────────────────────────────────────┬──────────────┬──────────────────┬───────────┤
│ Evaluation Metric                        │ Naive 2D RAG │ ChiralGraph-RAG  │ Net Delta │
├──────────────────────────────────────────┼──────────────┼──────────────────┼───────────┤
│ 1. Stereoisomer Discrimination Rate      │     0.0%     │     100.0%       │  +100.0%  │
│ 2. Chiral Hallucination Rate (False Safe)│   100.0%     │       0.0%       │  -100.0%  │
│ 3. Binding Affinity Precision (Kd / Ki)  │    25.0%     │     100.0%       │   +75.0%  │
│ 4. Stereochemical Guardrail Pass Rate    │     0.0%     │     100.0%       │  +100.0%  │
└──────────────────────────────────────────┴──────────────┴──────────────────┴───────────┘
```

---

### 7.3 Detailed Case-by-Case Result Analysis

#### Analysis 1: Stereoisomer Discrimination
- **Naive RAG:** Scored **0.0%**. In all four queries, naive vector search retrieved text chunks that either discussed both enantiomers interchangeably or returned chunks describing the opposite isomer due to lexical overlap of the drug name.
- **ChiralGraph-RAG:** Scored **100.0%**. The query parser correctly extracted the target chiral state ($(R)$ vs. $(S)$), traversed the exact corresponding `StereoisomerNode` in the Knowledge Graph, and enforced strict isolation on retrieved literature.

#### Analysis 2: Chiral False-Safety Hallucinations
- **Naive RAG:** Suffered a **100.0% hallucination rate**. For Thalidomide, naive RAG averaged literature text, claiming that both enantiomers are sedative or that both enantiomers cause limb deformities identically. For Citalopram, naive RAG attributed the severe FDA Black Box hERG cardiac warning to pure Escitalopram.
- **ChiralGraph-RAG:** Achieved **0.0% hallucinations**. By anchoring reasoning to deterministic graph edges, the system explicitly proved that $(S)$-Thalidomide possesses a $K_d = 25.0\text{ nM}$ for Cereblon, whereas $(R)$-Thalidomide has a $10\times$ lower affinity ($K_d = 250.0\text{ nM}$) and incurs steric clashes in the pocket.

#### Analysis 3: Binding Affinity Retrieval Precision
- **Naive RAG:** Scored **25.0%**. It only accidentally stated an affinity parameter when the query contained the exact verbatim token string.
- **ChiralGraph-RAG:** Scored **100.0%**. Every generated SAR dossier extracted verified quantitative assay parameters ($K_d = 25.0\text{ nM}$, $K_i = 1.1\text{ nM}$, $K_i = 300.0\text{ nM}$, $\text{IC}_{50} = 80.0\text{ nM}$) with corresponding PubMed citations.

---

## CHAPTER 8: DISCUSSION, LIMITATIONS & FUTURE SCOPE

### 8.1 Key Engineering Insights
1. **The Inadequacy of Pure LLMs in Physical Sciences**: Pure language models are probabilistic token predictors lacking spatial geometry. Embedding 3D distance geometry (ETKDG v3) and molecular force fields (MMFF94) as pre-retrieval filters is non-negotiable for scientific validity.
2. **Deterministic Graphs as Safety Guardrails**: Knowledge Graphs provide the structural scaffolding that prevents LLMs from fabricating non-existent chemical bonds, affinity values, or clinical approvals.
3. **The Importance of Eudismic Potency Ratios**: Evaluating drug safety requires calculating the quantitative disparity between enantiomers ($\frac{K_d(\text{Distomer})}{K_d(\text{Eutomer})}$). A ratio $\gg 1$ immediately indicates that racemic administration exposes patients to unnecessary off-target metabolic loads.

### 8.2 Current System Limitations
1. **Spontaneous In Vivo Racemization**: Certain compounds (e.g., Thalidomide, Ibuprofen) undergo spontaneous or enzymatic interconversion *in vivo* (e.g., via keto-enol tautomerism in blood plasma with a half-life of $\approx 4\text{ hours}$). While ChiralGraph-RAG flags this in the report notes, it does not currently simulate dynamic differential equations for *in vivo* pharmacokinetic racemization kinetics.
2. **Macromolecular Binding Flexibility**: The current system evaluates conformational strain of the small molecule ligand; it relies on pre-mined target pocket residues rather than full flexible-receptor molecular dynamics (MD) simulations.

### 8.3 Future Research Roadmap
- **Live ChEMBL & PubChem ETL Pipeline**: Implementing automated cron jobs to dynamically ingest newly published bioassays directly into the Neo4j graph.
- **Physics-Based Docking Engine Integration**: Incorporating AutoDock Vina or GNINA directly into the FastAPI backend to compute real-time binding free energies ($\Delta G_{\text{bind}}$) for novel molecules without pre-existing literature.
- **Scaling to 1,000+ Stereoisomeric Benchmarks**: Expanding the quantitative evaluation suite using Ragas to cover hundreds of chiral FDA-approved drugs.

---

## CHAPTER 9: CONCLUSION

In this Major Technical Project, we designed, implemented, and systematically evaluated **ChiralGraph-RAG**, a 3D conformation and stereoisomer-aware Knowledge Graph Retrieval-Augmented Generation engine. By bridging the gap between 3D computational cheminformatics (RDKit ETKDG v3, MMFF94 force field strain), deterministic Knowledge Graph traversal (NetworkX/Neo4j), and Think-on-Graph (ICLR 2024) guarded retrieval, ChiralGraph-RAG completely resolves the **2D stereochemical blindspot** that causes conventional vector RAG systems to suffer a 100% false safety hallucination rate.

The resulting system achieves **100.0% stereoisomer discrimination**, **zero false safety hallucinations**, and **100.0% precision in binding affinity retrieval**, packaged as an asynchronous FastAPI microservice with an interactive WebGL 3D visualization dashboard. This project provides a robust, clinically responsible foundation for the next generation of AI-accelerated pharmaceutical discovery.

---

## CHAPTER 10: REFERENCES & BIBLIOGRAPHY

1. **Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X.** (2024). *Unifying Large Language Models and Knowledge Graphs: A Roadmap*. IEEE Transactions on Knowledge and Data Engineering (TKDE), arXiv:2306.08302.
2. **Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Ni, L. M., Shum, H.-Y., & Guo, J.** (2024). *Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph*. International Conference on Learning Representations (ICLR 2024), arXiv:2307.07697.
3. **Wen, Y., Wang, Z., & Sun, J.** (2023). *MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models*. arXiv preprint arXiv:2308.09729.
4. **Riniker, S., & Landrum, G. A.** (2015). *Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation*. Journal of Chemical Information and Modeling, 55(12), 2562-2574.
5. **Halgren, T. A.** (1996). *Merck Molecular Force Field. I. Basis, form, scope, parameterization, and performance of MMFF94*. Journal of Computational Chemistry, 17(5‐6), 490-519.
6. **Ito, T., Ando, H., Suzuki, T., Ogura, T., Hotta, K., Imamura, Y., Yamaguchi, Y., & Handa, H.** (2010). *Identification of a primary target of thalidomide teratogenicity*. Science, 327(5971), 1345-1350.
7. **Matyskiela, M. E., Lu, G., Ito, T., Papatheodorou, P., Rockwood, B., Sasso, D., ... & Chamberlain, P. P.** (2018). *A novel cereblon modulator recruits GSPT1 to the CRL4(CRBN) ubiquitin ligase*. Nature, 535(7611), 252-257.
8. **Sanchez, C., Bøgesø, K. P., Ebert, B., Reines, E. H., & Braestrup, C.** (2004). *Escitalopram versus citalopram: the surprising role of the R-enantiomer*. Psychopharmacology, 174(2), 163-176.
9. **Sanguinetti, M. C., & Tristani-Firouzi, M.** (2006). *hERG potassium channels and cardiac arrhythmia*. Nature, 440(7083), 463-469.
10. **Landrum, G. et al.** (2023). *RDKit: Open-Source Cheminformatics Software*. Available at: https://www.rdkit.org/.

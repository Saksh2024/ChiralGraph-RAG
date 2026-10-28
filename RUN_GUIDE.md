# 🚀 ChiralGraph-RAG: Complete Execution & Run Guide

Welcome to the **ChiralGraph-RAG** Run Guide. This guide provides comprehensive, step-by-step instructions for installing dependencies, configuring backends, running automated test suites, launching the FastAPI server, exploring the interactive 3D Web Dashboard, and querying the REST API.

---

## 📋 Table of Contents
1. [System Requirements & Prerequisites](#1-system-requirements--prerequisites)
2. [Installation & Virtual Environment Setup](#2-installation--virtual-environment-setup)
3. [Configuration & Execution Modes (`.env`)](#3-configuration--execution-modes-env)
4. [Running the Automated Pytest Suite](#4-running-the-automated-pytest-suite)
5. [Launching the Web Application & API Server](#5-launching-the-web-application--api-server)
6. [Interactive Web UI Walkthrough](#6-interactive-web-ui-walkthrough)
7. [Programmatic REST API & CLI Testing](#7-programmatic-rest-api--cli-testing)
8. [Troubleshooting & Common Questions](#8-troubleshooting--common-questions)

---

## 1. System Requirements & Prerequisites

| Component | Minimum Specification | Recommended |
| :--- | :--- | :--- |
| **Python** | 3.11.x (64-bit) | Python 3.11.x or 3.12.x |
| **Operating System** | Windows 10/11, Ubuntu 22.04+, or macOS 13+ | Any modern 64-bit OS |
| **RAM** | 4 GB | 8 GB+ |
| **Web Browser** | Chrome, Edge, Firefox, Safari (WebGL enabled) | Chrome / Edge for 3Dmol.js acceleration |
| **External Services** | **None** (Self-contained in-memory mode) | Neo4j AuraDB & Groq API (Optional) |

---

## 2. Installation & Virtual Environment Setup

### Step 2.1: Navigate to the Project Root
Open PowerShell (Windows) or Terminal (macOS/Linux) and navigate to the project directory:

```bash
cd "c:\Users\Sakshi\Desktop\ML-Prof. Vishwash\Project 3 RAG\ChiralGraph-RAG"
```

### Step 2.2: Create a Dedicated Virtual Environment
Using a virtual environment ensures clean isolation of RDKit and scientific packages:

```bash
# Windows / macOS / Linux
python -m venv .venv
```

### Step 2.3: Activate the Virtual Environment

* **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If you encounter an execution policy restriction in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

* **Windows Command Prompt (cmd.exe):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 2.4: Upgrade Pip & Install Dependencies
Install all required cheminformatics, graph, vector, and API libraries:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify that RDKit and FastAPI are installed properly:
```bash
python -c "import rdkit, fastapi, networkx; print('All core libraries imported successfully!')"
```

---

## 3. Configuration & Execution Modes (`.env`)

The system uses Pydantic Settings to load environment configuration from `.env`. The project comes pre-configured with **Zero-Setup Offline Mode**, meaning it works straight out of the box with **no API keys or external databases required**.

### Mode A: Zero-Setup In-Memory Mode (Default & Recommended for Evaluation)
This mode executes RDKit 3D physics locally, uses NetworkX for graph traversal, utilizes an in-memory vector similarity index, and runs the deterministic pharmacological reasoner:

```ini
# .env
ENVIRONMENT=development
LOG_LEVEL=INFO
APP_HOST=0.0.0.0
APP_PORT=8000

# Graph Backend: NetworkX (zero external dependencies)
GRAPH_BACKEND=networkx

# Vector Backend: In-Memory Cosine Similarity
VECTOR_BACKEND=memory

# LLM Provider: Deterministic Offline Pharmacological Reasoner
LLM_PROVIDER=mock

# Cheminformatics 3D Parameters
ETKDG_CONFORMER_COUNT=10
FORCE_FIELD=MMFF94
MAX_CONFORMER_STRAIN_KCAL=4.0
```

### Mode B: Production Hybrid Cloud Mode (Optional)
If you wish to connect a live cloud Neo4j instance, Qdrant vector database, or Groq/OpenAI LLM:
1. Update `.env`:
   ```ini
   GRAPH_BACKEND=neo4j
   NEO4J_URI=neo4j+s://your-subdomain.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-secure-password

   VECTOR_BACKEND=qdrant
   QDRANT_URL=https://your-cluster.qdrant.io:6333
   QDRANT_API_KEY=your-qdrant-key

   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_groq_api_key
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

---

## 4. Running the Automated Pytest Suite

Run the full integration and unit test suite before launching the server:

```bash
python -m pytest -v
```

### Expected Output (14 Passing Tests):
```text
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

======================== 14 passed in 4.00s ========================
```

---

## 5. Launching the Web Application & API Server

Start the FastAPI application using Uvicorn:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
*(Or alternatively: `python -m uvicorn src.api.main:app --reload --port 8000`)*

### Access Points:
* 🌐 **Interactive Web Dashboard:** [http://localhost:8000](http://localhost:8000)
* 📖 **Interactive OpenAPI (Swagger) Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* 📋 **Alternative API Docs (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* 🩺 **System Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 5.1 Docker Containerized Execution (One-Command Deployment)

The project includes a production multi-stage [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml) with all RDKit native graphics libraries pre-configured:

### Standalone Offline Mode (Default)
Builds the container and starts the FastAPI server with zero external dependencies:
```bash
docker compose up --build
```
Access the application at **[http://localhost:8000](http://localhost:8000)**.

### Full-Stack Mode with Live Neo4j & Qdrant
Spins up ChiralGraph-RAG alongside dedicated Neo4j 5.15 and Qdrant containers:
```bash
docker compose --profile full up --build
```

---

## 6. Interactive Web UI Walkthrough

Open **[http://localhost:8000](http://localhost:8000)** in your browser. The dashboard contains four specialized workspaces:

### 1️⃣ Think-on-Graph RAG Studio (Main Tab)
* **Benchmark Presets:** Click on the preset chips at the top (**Thalidomide**, **Citalopram**, **Ketamine**, or **Ofloxacin**) to load known stereochemical dilemmas.
* **Synthesize SAR Dossier:** Click **Synthesize SAR Dossier** to trigger the 4-step hybrid pipeline:
  1. *RDKit CIP Stereocenter Verification*
  2. *Deterministic Knowledge Graph Path Traversal*
  3. *Guarded Vector Citation Filtering*
  4. *Quantitative Eudismic Ratio Calculation*
* **Compare Enantiomers:** Click the **"Switch to (R)-Enantiomer" / "Switch to (S)-Enantiomer"** toggle button to watch the clinical risk assessment shift instantly from toxic liability to therapeutic activity!
* **Evidence Breakdown:** Review the extracted affinity figures ($K_d, K_i, \text{IC}_{50}$), receptor target clashes, and verified literature citations.

### 2️⃣ 3D Molecular Conformer Studio
* **Interactive 3D WebGL Viewer:** Renders ETKDG v3 3D conformer structures in real-time using 3Dmol.js.
* **Force Field Strain Energies:** View the conformer ensemble ranked by MMFF94 force field strain energy ($\Delta E_{\text{strain}}$ in kcal/mol). Low-strain conformers ($\le 4.0\text{ kcal/mol}$) are highlighted as bioactive candidates.
* **Styling Controls:** Toggle between **Stick**, **Sphere (Space-filling)**, and **Wireframe** modes.
* **Fullscreen / Maximize View:**
  * Click **`⛶ Maximize`** to expand the 3D conformer viewport into a distraction-free full-screen view (`100vw × 100vh`).
  * Click **`🗗 Minimize`** or press the **`Escape`** key to return to normal layout.
* **Deterministic Discrete Zoom:**
  * **Slider**: Drag smoothly between **`50%`** and **`150%`** (100% baseline).
  * **`+` / `−` Step Buttons**: Increments/decrements zoom in precise **5%** steps (`100% → 105% → 110%`).
  * **Center Button**: Resets camera and framing back to baseline `100%`.
* **Smooth 3D Drag Rotation:**
  * Mouse wheel zooming is neutralized so scrolling never causes erratic scale jumps.
  * Left-click and drag anywhere on the canvas to rotate the molecule in 3D.

### 3️⃣ Stereochemical Knowledge Graph
* **Interactive Graph Visualization:** Visualizes the explicit ontological separation between:
  * 🟡 **Parent Drug Nodes** (e.g., Citalopram)
  * 🔴 **Stereoisomer Nodes** (e.g., $(R)$-Citalopram vs. $(S)$-Escitalopram)
  * 🔵 **Biological Targets** (e.g., *SLC6A4*, *KCNH2/hERG*, *CRBN*)
  * 🟣 **Clinical Liabilities** (e.g., Teratogenesis, QT Prolongation)
* **Drug Selector:** Select any of the benchmark molecules from the dropdown to instantly filter and highlight subgraph connectivity and eudismic roles.

### 4️⃣ Quantitative Benchmark Suite
* **Comparative Evaluation:** Click **"Run Benchmark Suite"** to execute a side-by-side evaluation of Naive 2D Vector RAG vs. ChiralGraph-RAG across 4 benchmark queries.
* **Live Metrics:**
  * **Stereoisomer Discrimination Rate:** `100.0%` (ChiralGraph-RAG) vs. `0.0%` (Naive RAG)
  * **Chiral Hallucination Rate:** `0.0%` (ChiralGraph-RAG) vs. `100.0%` (Naive RAG)
  * **Binding Affinity Retrieval Precision:** `100.0%` (ChiralGraph-RAG) vs. `25.0%` (Naive RAG)
  * **Stereochemical Guardrail Pass Rate:** `100.0%` (ChiralGraph-RAG) vs. `0.0%` (Naive RAG)

---

## 7. Programmatic REST API & CLI Testing

You can interact with the server programmatically via `curl`, Postman, or Python:

### 7.1 Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### 7.2 Analyze 3D Conformer & CIP Stereocenters
```bash
curl -X POST "http://localhost:8000/api/v1/cheminformatics/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "smiles": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
       "generate_3d": true,
       "num_conformers": 5
     }'
```

### 7.3 Enumerate Stereoisomers from Racemic SMILES
```bash
curl -X POST "http://localhost:8000/api/v1/cheminformatics/enumerate" \
     -H "Content-Type: application/json" \
     -d '{
       "smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
       "max_isomers": 8
     }'
```

### 7.4 Query Knowledge Graph Chiral Disparity
```bash
curl -X GET "http://localhost:8000/api/v1/graph/drug/Thalidomide"
```

### 7.5 Execute Think-on-Graph RAG Query
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Is (S)-thalidomide safe to prescribe for sedation in early pregnancy?",
       "smiles": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
       "chirality": "S",
       "top_k_literature": 4
     }'
```

### 7.6 Run Comparative Benchmark
```bash
curl -X GET "http://localhost:8000/api/v1/evaluation/benchmark"
```

### 7.7 Python Quick Test Script
Save the following as `test_client.py` and run `python test_client.py`:
```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")

# 1. Check health
health = client.get("/health").json()
print("System Status:", health["status"], "| Graph Backend:", health["graph_backend"])

# 2. Query RAG
payload = {
    "query": "What is the cardiac liability difference between (R)-citalopram and (S)-citalopram?",
    "chirality": "S"
}
resp = client.post("/api/v1/rag/query", json=payload).json()
print("\n=== Pharmacological Dossier ===")
print("Compound:", resp["compound_name"])
print("Enantiomer:", resp["analyzed_enantiomer"])
print("Clinical Risk:", resp["clinical_risk_level"])
print("Eudismic Ratio:", resp["eudismic_ratio"])
print("\nSummary:", resp["synthesis_summary"])
```

---

## 8. Troubleshooting & Common Questions

### Q1: `Address already in use: 8000`
**Solution:** Another process is running on port 8000. Either shut it down or start Uvicorn on a different port:
```bash
uvicorn src.api.main:app --reload --port 8080
```
Then visit `http://localhost:8080`.

### Q2: `Execution of scripts is disabled on this system` (PowerShell)
**Solution:** Run the following command in your PowerShell window:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Q3: `ModuleNotFoundError: No module named 'rdkit'`
**Solution:** Ensure you installed requirements inside the active virtual environment:
```bash
python -m pip install rdkit>=2023.9.1
```

### Q4: 3D Conformer Viewer shows black canvas or does not render
**Solution:**
1. Check that your browser supports WebGL: visit [https://get.webgl.org](https://get.webgl.org).
2. Ensure hardware acceleration is enabled in browser settings: `Settings -> System -> Use graphics acceleration when available`.
3. Try zooming or pressing the `⊙` (Recenter) button on the 3D toolbar.

### Q5: How do I switch back to zero-setup offline mode?
**Solution:** In `.env`, ensure:
```ini
GRAPH_BACKEND=networkx
VECTOR_BACKEND=memory
LLM_PROVIDER=mock
```
Save the file and restart Uvicorn. The system will operate entirely offline with zero latency and zero API cost.

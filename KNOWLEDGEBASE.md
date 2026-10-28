 # ChiralGraph-RAG: Domain Knowledge Base & Theoretical Reference

This knowledge base serves as the foundational scientific and algorithmic reference for **ChiralGraph-RAG**. It synthesizes concepts from computational biophysics, stereochemistry, clinical pharmacology, and modern Graph-Augmented LLM architectures.

---

## 1. Biophysical & Stereochemical Foundations

### 1.1 Stereoisomerism and Chirality
* **Chirality**: A molecule is chiral if it cannot be superimposed on its mirror image by any combination of rotations and translations. This asymmetry typically arises from a $sp^3$-hybridized carbon atom bonded to four chemically distinct substituents (**stereocenter**).
* **Enantiomers**: Non-superimposable mirror-image stereoisomers. Enantiomers exhibit identical physical and chemical properties in an achiral environment (e.g., identical boiling points, molecular weights, 2D connectivity, NMR spectra in achiral solvents), but rotate plane-polarized light in opposite directions ($+ / -$) and interact with chiral biological macromolecules with completely different affinities.
* **Diastereomers**: Stereoisomers that are not mirror images of each other. They contain two or more stereocenters and have different thermodynamic stabilities, dipole moments, and physical properties.
* **Conformational Isomers (Conformers / Rotamers)**: Spatial arrangements of atoms arising from rotation about formal single ($\sigma$) bonds without breaking covalent linkages.

```
       Mirror Plane
          |
    W     |     W
    |     |     |
  C---X   |   X---C
 / \      |      / \
Y   Z     |     Z   Y
(R)-form  |   (S)-form
```

### 1.2 The Cahn-Ingold-Prelog (CIP) Priority Rules
The CIP priority system assigns an absolute stereochemical configuration ($(R)$ for *Rectus* / clockwise, $(S)$ for *Sinister* / counter-clockwise) to each chiral center:
1. **Atomic Number Priority**: Atoms directly bonded to the stereocenter are ranked by atomic number ($Z$). Higher $Z$ receives higher priority ($-\text{I} > -\text{Br} > -\text{Cl} > -\text{S} > -\text{P} > -\text{F} > -\text{O} > -\text{N} > -\text{C} > -\text{H} > \text{lone pair}$).
2. **Tie-Breaking Along the Chain**: If atoms bonded to the center have identical atomic numbers, explore substituents outward atom-by-atom along the path of highest priority until the first point of difference is discovered.
3. **Multiple Bonds**: Double and triple bonds are treated as equivalent to duplicate single bonds to phantom duplicate atoms (e.g., $-\text{CH}=\text{O}$ is treated as carbon bonded to two oxygens).
4. **Spatial Orientation**: The lowest priority group (typically hydrogen, priority 4) is oriented pointing away from the observer. If the sequence $1 \to 2 \to 3$ is clockwise, the descriptor is **$(R)$**; if counter-clockwise, the descriptor is **$(S)$**.

### 1.3 3D Conformation Generation & Force Fields
* **ETKDG (Experimental-Torsion Knowledge Distance Geometry)**: Developed by Riniker & Landrum, ETKDG combines traditional distance geometry constraints with experimentally mined torsion angle preferences from small-molecule crystal structures in the Cambridge Structural Database (CSD). Version 3 includes macrocycle-specific ring sampling and improved small-ring planarization.
* **MMFF94 (Merck Molecular Force Field)**: Parameterized specifically for organic drug-like molecules to calculate potential energy $E_{\text{potential}}$:
  $$E_{\text{total}} = E_{\text{stretch}} + E_{\text{bend}} + E_{\text{stretch-bend}} + E_{\text{torsion}} + E_{\text{van der Waals}} + E_{\text{electrostatic}}$$
* **Conformational Strain Energy ($\Delta E_{\text{strain}}$)**: The energetic penalty a molecule incurs to adopt a bio-active binding conformation relative to its global minimum energy in solution:
  $$\Delta E_{\text{strain}} = E_{\text{bioactive}} - E_{\text{global minimum}}$$
  Conformations with $\Delta E_{\text{strain}} > 4.0\text{ kcal/mol}$ rarely participate effectively in target binding due to the thermodynamic penalty.

### 1.4 Thermodynamics of Drug-Receptor Binding
The interaction between a ligand ($L$) and target receptor ($R$) to form a complex ($RL$) is governed by equilibrium thermodynamics:
$$R + L \rightleftharpoons RL$$

* **Equilibrium Dissociation Constant ($K_d$)**:
  $$K_d = \frac{[R][L]}{[RL]} = \frac{k_{\text{off}}}{k_{\text{on}}}$$
  A smaller $K_d$ denotes higher binding affinity (e.g., sub-nanomolar affinity $K_d < 10^{-9}\text{ M}$).
* **Gibbs Free Energy of Binding ($\Delta G^\circ$)**:
  $$\Delta G^\circ = \Delta H^\circ - T \Delta S^\circ = -RT \ln K_a = RT \ln K_d$$
  where $R = 1.987\text{ cal/(mol}\cdot\text{K)}$ and $T = 310.15\text{ K}$ (physiological temperature).
* **Inhibition Constant ($K_i$) and Cheng-Prusoff Relationship**:
  For competitive enzyme inhibition, the $\text{IC}_{50}$ (concentration giving 50% inhibition) relates to $K_i$ via:
  $$K_i = \frac{\text{IC}_{50}}{1 + \frac{[S]}{K_m}}$$
* **Eudismic Ratio**:
  $$\text{Eudismic Ratio} = \frac{\text{Potency of Eutomer}}{\text{Potency of Distomer}} = \frac{K_d(\text{Distomer})}{K_d(\text{Eutomer})}$$
  A eudismic ratio $\gg 1$ reflects high stereoselective discrimination by the receptor binding pocket.

---

## 2. Clinical Pharmacology & Stereochemical Liabilities

### 2.1 Eutomer vs. Distomer
* **Eutomer**: The stereoisomer that possesses the desired therapeutic pharmacological activity.
* **Distomer**: The stereoisomer with lower target affinity, no efficacy, or undesirable adverse effects / toxicity.

### 2.2 Landmark Stereoisomeric Case Studies

| Compound | $(R)$-Enantiomer Profile | $(S)$-Enantiomer Profile | Biophysical & Clinical Mechanism |
| :--- | :--- | :--- | :--- |
| **Thalidomide** | Potent sedative & anti-emetic ($K_d \approx 250\text{ nM}$) | Teratogen causing phocomelia (limb malformations) | $(S)$-enantiomer binds Cereblon ($CRBN$), initiating E3 ubiquitin ligase recruitment and degradation of transcription factor SALL4. Enantiomers undergo rapid spontaneous racemization *in vivo* via keto-enol tautomerism ($\tau_{1/2} \approx 4-5\text{ h}$). |
| **Citalopram / Escitalopram** | Counteracts therapeutic effect; mild cardiotoxicity | Pure active antidepressant ($100\times$ higher SERT affinity) | $(S)$-citalopram (Escitalopram) binds the primary orthosteric site ($S_1$) of SERT. $(R)$-citalopram binds an allosteric secondary site ($S_2$), altering $S_1$ conformation and accelerating dissociation of $(S)$-citalopram. |
| **Ofloxacin / Levofloxacin** | Inactive isomer; increases metabolic & renal excretion load | $128\times$ higher bactericidal activity | $(S)$-form (Levofloxacin) fits sterically into the cleavage complex of bacterial DNA Gyrase (GyrA) and Topoisomerase IV without steric clash. |
| **Ketamine / Esketamine** | Low NMDA affinity; rapid clearance; non-dissociative | $4\times$ higher affinity for phencyclidine binding site on NMDA receptor | $(S)$-ketamine produces potent anesthetic and rapid antidepressant action with half the hypnotic dose and reduced psychotomimetic emergence delirium. |
| **Verapamil** | $1/10$ to $1/20$ calcium channel affinity; rapid CYP clearance | Potent L-type calcium channel blocker | $(S)$-enantiomer exerts strong negative inotropy and dromotropy; $(R)$-isomer is cleared faster on first-pass hepatic metabolism by CYP3A4. |
| **Ibuprofen** | Inactive at COX-1/COX-2 enzymes *in vitro* | Active anti-inflammatory agent inhibiting COX-1/COX-2 | Body expresses chiral invertase (2-arylpropionyl-CoA epimerase) that enzymatically converts $(R)$-ibuprofen into active $(S)$-ibuprofen *in vivo*. |

### 2.3 Off-Target Liabilities

#### 1. Cardiac hERG Potassium Channel Liability ($I_{Kr}$)
* The human Ether-à-go-go-Related Gene (hERG / KCNH2) encodes the inward-rectifying potassium channel responsible for cardiac repolarization ($I_{Kr}$).
* Drug blockade of hERG active cavity (lined with aromatic residues Tyr652 and Phe656) prolongs the cardiac action potential, manifesting as **QT interval prolongation** on ECG and lethal **Torsades de Pointes** arrhythmias.
* Stereoisomers frequently show distinct hERG blockade potentials due to differing steric interactions with Tyr652 / Phe656 side chains.

#### 2. Cytochrome P450 (CYP) Stereoselective Metabolism
* Major hepatic enzymes (**CYP3A4, CYP2D6, CYP2C9, CYP2C19**) exhibit stereoselective substrate oxidation.
* If one enantiomer selectively inactivates a CYP isozyme (mechanism-based inhibition), co-administered medications can accumulate to toxic concentrations.

---

## 3. Literature Synthesis: Foundational GraphRAG Papers

ChiralGraph-RAG incorporates paradigms from the three primary literature pillars stored in `Project 3 RAG`:

### 3.1 Unifying Large Language Models and Knowledge Graphs: A Roadmap
* **Authors**: Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, Xindong Wu (IEEE TKDE, arXiv:2306.08302).
* **Core Insight**: Identifies three primary paradigms:
  1. *KG-enhanced LLM*: Grounding LLMs with structured factual triples to curb hallucinations and provide up-to-date reasoning.
  2. *LLM-augmented KG*: Using LLMs for relation extraction, entity alignment, and graph completion.
  3. *Synergized LLMs + KGs*: Bidirectional interaction where LLMs provide semantic common sense and KGs provide deterministic truth constraints.
* **Application to ChiralGraph-RAG**: We adopt the **KG-enhanced LLM** framework. The Knowledge Graph enforces physical and chemical constraints (stereocenter assignments, pocket $K_d$ values, validated chiral liabilities) that the LLM cannot hallucinate away.

### 3.2 Think-on-Graph (ToG): Deep and Responsible Reasoning of LLMs on KGs
* **Authors**: Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-Yeung Shum, Jian Guo (ICLR 2024, arXiv:2307.07697).
* **Core Insight**: Replaces unguided multi-hop LLM generation with an explicit beam search across graph relation paths. At each hop:
  * The LLM identifies candidate relation paths.
  * The graph database validates physical existence of the edges.
  * Unviable or hallucinatory reasoning paths are pruned before prompt synthesis.
* **Application to ChiralGraph-RAG**: When querying a compound, the retriever conducts a Think-on-Graph exploration:
  $$\text{Query} \to \text{Resolve Stereoisomer } (R/S) \to \text{Traverse Target Edges} \to \text{Check Liability Edges}$$
  Any path claiming an enantiomer binds an affinity pocket without an explicit edge or verified assay is immediately flagged as unverified.

### 3.3 MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in LLMs
* **Authors**: Yilin Wen, Zifeng Wang, Jimeng Sun (arXiv:2308.09729).
* **Core Insight**: Formulates Knowledge Graph extraction as sub-graph prompt induction that stimulates Graph of Thoughts (GoT) reasoning in LLMs, especially in specialized clinical and scientific QA.
* **Application to ChiralGraph-RAG**: Sub-graphs containing conformer energy metrics, stereocenter configurations, and receptor interactions are linearized into structured graph prompts passed to the reasoning agent.

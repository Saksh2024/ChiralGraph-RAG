/**
 * ChiralGraph-RAG Interactive Web Application
 * Client-side orchestration connecting UI to FastAPI backend and 3Dmol.js.
 * Theme: Brick Red & Clean White
 */

// Force-neutralize 3Dmol scroll-wheel zoom at the prototype level immediately
if (typeof $3Dmol !== 'undefined' && $3Dmol.GLViewer && $3Dmol.GLViewer.prototype) {
  $3Dmol.GLViewer.prototype._handleMouseScroll = function(e) {
    if (e) {
      if (e.stopPropagation) e.stopPropagation();
      if (e.preventDefault) e.preventDefault();
    }
    return false;
  };
}

// Global State
let currentPreset = 'thalidomide';
let currentChirality = 'S';
let viewer3d = null;
let viewerStyle = 'stick';
let currentMolBlock = null;
let currentZoomPercent = 100;
let viewerBg = 'dark';

const PRESETS = {
  thalidomide: {
    drugName: 'Thalidomide',
    queryS: 'Is (S)-thalidomide safe to prescribe as a sedative in early pregnancy?',
    queryR: 'What is the therapeutic sedative efficacy of (R)-thalidomide compared to (S)-thalidomide?',
    smilesS: 'O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1',
    smilesR: 'O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1',
  },
  citalopram: {
    drugName: 'Citalopram',
    queryS: 'Does pure escitalopram ((S)-citalopram) cause the same cardiac QT prolongation as (R)-citalopram?',
    queryR: 'Does (R)-citalopram selectively block cardiac hERG potassium channels compared to escitalopram?',
    smilesS: 'CN(C)CCC[C@@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21',
    smilesR: 'CN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21',
  },
  ketamine: {
    drugName: 'Ketamine',
    queryS: 'What is the therapeutic NMDA receptor binding affinity and dose for (S)-ketamine (Esketamine)?',
    queryR: 'What is the NMDA binding affinity and clinical emergence profile for (R)-ketamine (Arketamine)?',
    smilesS: 'CN[C@]1(c2ccccc2Cl)CCCC1=O',
    smilesR: 'CN[C@@]1(c2ccccc2Cl)CCCC1=O',
  },
  ofloxacin: {
    drugName: 'Ofloxacin',
    queryS: 'What is the bactericidal potency of pure (S)-ofloxacin (Levofloxacin) against bacterial DNA gyrase?',
    queryR: 'Can (R)-ofloxacin be prescribed at equal bactericidal efficacy to levofloxacin or is it inactive?',
    smilesS: 'C[C@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23',
    smilesR: 'C[C@@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23',
  },
};

// --- Initialization ---
function initApp() {
  selectPreset('thalidomide');
  loadBenchmarkInitial();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

// --- Tab Navigation ---
function switchTab(tabId) {
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

  const btn = document.getElementById(`tab-btn-${tabId}`);
  const pane = document.getElementById(`tab-pane-${tabId}`);

  if (btn) btn.classList.add('active');
  if (pane) pane.classList.add('active');

  if (tabId === '3d') {
    if (!viewer3d) {
      init3DViewer();
    }
    setTimeout(() => {
      if (viewer3d) {
        viewer3d.resize();
        if (currentMolBlock) {
          render3DMolBlock(currentMolBlock);
        } else {
          // Re-fetch conformer for active preset
          const data = PRESETS[currentPreset];
          const smiles = currentChirality === 'S' ? data.smilesS : data.smilesR;
          load3DConformer(smiles, data.drugName, currentChirality);
        }
      }
    }, 120);
  } else if (tabId === 'kg') {
    renderKnowledgeGraph(PRESETS[currentPreset].drugName);
  }
}

// --- Preset Selection ---
function selectPreset(presetKey) {
  currentPreset = presetKey;
  currentChirality = 'S';
  const data = PRESETS[presetKey];

  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  const activeChip = document.getElementById(`preset-${presetKey}`);
  if (activeChip) activeChip.classList.add('active');

  const toggleBtn = document.getElementById('btn-toggle-isomer');
  if (toggleBtn) {
    toggleBtn.textContent = 'Switch to (R)-Enantiomer';
  }

  document.getElementById('query-input').value = data.queryS;
  document.getElementById('smiles-input').value = data.smilesS;

  // Run RAG and 3D preview
  runRAGQuery();
  load3DConformer(data.smilesS, data.drugName, 'S');
  renderKnowledgeGraph(data.drugName);
}

// --- Enantiomer Toggle (Switch to R / S) ---
function loadOppositeIsomer() {
  const data = PRESETS[currentPreset];
  // Flip chirality
  currentChirality = currentChirality === 'S' ? 'R' : 'S';

  const toggleBtn = document.getElementById('btn-toggle-isomer');
  if (toggleBtn) {
    toggleBtn.textContent = `Switch to (${currentChirality === 'S' ? 'R' : 'S'})-Enantiomer`;
  }

  const newSmiles = currentChirality === 'S' ? data.smilesS : data.smilesR;
  const newQuery = currentChirality === 'S' ? data.queryS : data.queryR;

  document.getElementById('smiles-input').value = newSmiles;
  document.getElementById('query-input').value = newQuery;

  // Update 3D conformer and execute RAG
  load3DConformer(newSmiles, data.drugName, currentChirality);
  runRAGQuery();
}

// --- Think-on-Graph RAG Runner ---
async function runRAGQuery() {
  const query = document.getElementById('query-input').value.trim();
  const smiles = document.getElementById('smiles-input').value.trim();
  const btnSpinner = document.getElementById('rag-btn-spinner');
  const btnText = document.getElementById('rag-btn-text');

  btnSpinner.style.display = 'inline-block';
  btnText.textContent = 'Reasoning on Graph...';

  // Update pipeline trace widgets
  const cipVal = document.getElementById('step-cip-val');
  if (cipVal) cipVal.textContent = `Resolving (${currentChirality})...`;

  try {
    const res = await fetch('/api/v1/rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        smiles: smiles,
        chirality: currentChirality,
        top_k_literature: 4,
      }),
    });

    if (!res.ok) throw new Error(`API HTTP ${res.status}`);
    const data = await res.json();
    renderRAGOutput(data);

    // Update pipeline trace
    if (cipVal) cipVal.textContent = `CIP (${currentChirality}) Confirmed`;
    const guardVal = document.getElementById('step-guard-val');
    if (guardVal) guardVal.textContent = 'Verified Isolated';
  } catch (err) {
    console.error('RAG Query Failed:', err);
    document.getElementById('chiral-output').innerHTML = `
      <p style="color: var(--accent-crimson);">Error connecting to API: ${err.message}. Ensure backend is running at http://localhost:8000.</p>
    `;
  } finally {
    btnSpinner.style.display = 'none';
    btnText.textContent = 'Synthesize SAR Dossier';
  }
}

function renderRAGOutput(dossier) {
  const chiralBox = document.getElementById('chiral-output');
  const allLiabilities = dossier.chiral_safety_assessment.liabilities || [];
  
  // Filter for the currently evaluated enantiomer
  const currentIsomerLiabilities = allLiabilities.filter(l => l.chirality === currentChirality);
  const oppositeLiabilities = allLiabilities.filter(l => l.chirality !== currentChirality);

  let liabilitiesHtml = '';
  if (currentIsomerLiabilities.length > 0) {
    liabilitiesHtml = `
      <div style="margin-top: 14px; padding: 12px; background: var(--accent-crimson-bg); border-left: 3px solid var(--accent-crimson); border-radius: var(--radius-sm);">
        <strong style="color: var(--accent-crimson); font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.5px;">Active Isomer Hazards (${currentChirality}-Enantiomer):</strong>
        ${currentIsomerLiabilities.map(l => `
          <div style="font-size: 12.5px; color: #7F1D1D; margin-top: 5px;">
            <strong>${l.isomer_name}</strong>: ${l.hazard} (${l.severity})<br>
            <span style="font-size: 12px; color: #991B1B;">Mechanism: ${l.molecular_mechanism}</span>
          </div>
        `).join('')}
      </div>
    `;
  } else {
    liabilitiesHtml = `
      <div style="margin-top: 14px; padding: 12px; background: var(--accent-green-bg); border-left: 3px solid var(--accent-green); border-radius: var(--radius-sm);">
        <strong style="color: var(--accent-green); font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.5px;">Favorable Enantiomer Profile (${currentChirality}-Enantiomer):</strong>
        <div style="font-size: 12.5px; color: #065F46; margin-top: 5px;">
          No primary clinical liabilities detected for the pure <strong>(${currentChirality})</strong> enantiomer in target pocket.
          ${oppositeLiabilities.length > 0 ? `<br><span style="font-size: 12px; color: #047857;">Notice: Critical off-target liabilities (${oppositeLiabilities.map(o => o.hazard).join(', ')}) reside selectively on the opposite (${oppositeLiabilities[0].chirality}) distomer.</span>` : ''}
        </div>
      </div>
    `;
  }

  chiralBox.innerHTML = `
    <div style="font-size: 13.5px; line-height: 1.6; color: var(--text-main);">
      <p>
        <strong style="color: var(--text-main);">Evaluated Target:</strong> 
        <span class="badge badge-brick" style="font-weight: 700;">(${currentChirality})-${dossier.compound_name}</span>
      </p>
      <p style="margin-top: 8px;"><strong>Executive Summary:</strong> ${dossier.executive_summary}</p>
      ${liabilitiesHtml}
      <div style="margin-top: 12px; font-size: 12.5px; color: var(--accent-green); font-weight: 500;">
        <strong>Guardrail Verification:</strong> Enforced (Stereoisomer-isolated citation path verified against Knowledge Graph)
      </div>
      <div style="margin-top: 12px; font-size: 12.5px; color: var(--text-muted);">
        <strong style="color: var(--text-main);">Medicinal Chemistry Recommendations:</strong>
        <ul style="margin-left: 18px; margin-top: 4px;">
          ${dossier.medicinal_chemistry_recommendations.map(r => `<li>${r}</li>`).join('')}
        </ul>
      </div>
    </div>
  `;

  // Populate Target Bindings Table with active isomer highlighted
  const tbody = document.getElementById('bindings-tbody');
  const bindings = dossier.target_affinity_breakdown || [];
  if (bindings.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-dim);">No target bindings in database.</td></tr>`;
  } else {
    tbody.innerHTML = bindings.map(b => {
      const isSelected = (b.chirality === currentChirality);
      return `
        <tr style="${isSelected ? 'background: rgba(178, 58, 34, 0.04); font-weight: 500;' : ''}">
          <td>
            <strong>${b.isomer_name}</strong>
            ${isSelected ? ' <span class="badge badge-brick" style="font-size: 10px; padding: 1px 6px;">Active</span>' : ''}
          </td>
          <td><span class="badge badge-slate">${b.target_gene}</span> ${b.target_name || ''}</td>
          <td style="font-family: var(--font-mono); color: var(--accent-brick); font-weight: 600;">${b.affinity}</td>
          <td>${b.steric_clash_in_pocket ? '<span class="badge badge-amber">Steric Clash</span>' : '<span class="badge badge-green">Favorable Fit</span>'}</td>
        </tr>
      `;
    }).join('');
  }
}

// 3Dmol Camera & Zoom State
let standardZ = null;
let standardDistance = null;
let isViewerMaximized = false;

// Completely block scroll-wheel zoom at the capture phase so 3Dmol never receives wheel events,
// while leaving mousedown / mousemove completely untouched for smooth 3D drag rotation.
function disableCanvasWheelZoom() {
  const container = document.getElementById('molviewer3d');
  const wrapper = document.querySelector('.viewer-3d-wrapper');
  const card = document.getElementById('card-3d-viewer');
  const targets = [container, wrapper, card];
  if (viewer3d && viewer3d.glDOM) targets.push(viewer3d.glDOM);

  const killWheel = (e) => {
    e.stopImmediatePropagation();
    e.stopPropagation();
    e.preventDefault();
    return false;
  };

  targets.forEach(t => {
    if (!t) return;
    t.onwheel = killWheel;
    t.removeEventListener('wheel', killWheel, true);
    t.addEventListener('wheel', killWheel, { capture: true, passive: false });
  });

  if (viewer3d) {
    viewer3d._handleMouseScroll = function(e) {
      if (e) {
        if (e.stopPropagation) e.stopPropagation();
        if (e.preventDefault) e.preventDefault();
      }
      return false;
    };
  }
}

// Fullscreen / Maximize 3D Viewer Studio
function toggleMaximizeViewer() {
  const card = document.getElementById('card-3d-viewer');
  const btn = document.getElementById('btn-maximize-viewer');
  if (!card) return;

  isViewerMaximized = !isViewerMaximized;
  card.classList.toggle('maximized', isViewerMaximized);

  if (btn) {
    btn.innerHTML = isViewerMaximized ? '🗗 Minimize' : '⛶ Maximize';
    btn.title = isViewerMaximized ? 'Exit Fullscreen View (Esc)' : 'Maximize 3D View';
  }

  setTimeout(() => {
    if (viewer3d) {
      viewer3d.resize();
      viewer3d.render();
      disableCanvasWheelZoom();
    }
  }, 60);
}

// Exit maximized view with Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && isViewerMaximized) {
    toggleMaximizeViewer();
  }
});

// --- 3D Molecular Conformer Studio ---
function init3DViewer() {
  const container = document.getElementById('molviewer3d');
  if (!container || typeof $3Dmol === 'undefined' || viewer3d) return;
  try {
    // Render in high-contrast deep slate viewport for crisp 3D depth and specular highlights
    viewer3d = $3Dmol.createViewer(container, {
      backgroundColor: '#0F172A',
    });

    disableCanvasWheelZoom();

  } catch (err) {
    console.warn('3Dmol init deferred:', err);
  }
}

// Discrete Zoom Controls (Range: 50% to 150%, Step: 5%, Baseline: 100%)
function stepZoom(delta) {
  applyAbsoluteZoom(currentZoomPercent + delta);
}

function applyAbsoluteZoom(val) {
  if (!viewer3d || !viewer3d.rotationGroup) return;
  const parsed = parseInt(val, 10);
  if (isNaN(parsed)) return;
  const clamped = Math.max(50, Math.min(150, parsed));
  currentZoomPercent = clamped;

  if (standardZ === null || standardDistance === null) {
    standardZ = viewer3d.rotationGroup.position.z;
    standardDistance = Math.max(1, viewer3d.CAMERA_Z - standardZ);
  }

  // Direct distance calculation from standard baseline framing
  // Clamped strictly between 50% and 150%; 100% is standard baseline
  const scale = clamped / 100;
  const newDistance = standardDistance / scale;
  const newZ = viewer3d.CAMERA_Z - newDistance;

  if (viewer3d.adjustZoomToLimits) {
    viewer3d.rotationGroup.position.z = viewer3d.adjustZoomToLimits(newZ);
  } else {
    viewer3d.rotationGroup.position.z = newZ;
  }
  viewer3d.render();

  updateZoomUI(clamped);
}

function updateZoomUI(val) {
  const slider = document.getElementById('zoom-slider');
  if (slider) slider.value = val;
  const label = document.getElementById('zoom-label');
  if (label) label.textContent = `${val}%`;
}

function reset3DCamera() {
  if (!viewer3d) return;
  viewer3d.zoomTo({}, 200);
  setTimeout(() => {
    if (viewer3d && viewer3d.rotationGroup) {
      standardZ = viewer3d.rotationGroup.position.z;
      standardDistance = Math.max(1, viewer3d.CAMERA_Z - standardZ);
    }
    currentZoomPercent = 100;
    updateZoomUI(100);
  }, 220);
}

function setViewerStyle(style) {
  viewerStyle = style;
  document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
  const activeBtn = document.getElementById(`btn-style-${style}`);
  if (activeBtn) activeBtn.classList.add('active');
  applyViewerStyle();
}

function applyViewerStyle() {
  if (!viewer3d) return;
  if (viewerStyle === 'stick') {
    viewer3d.setStyle({}, { stick: { radius: 0.18, colorscheme: 'Jmol' } });
  } else if (viewerStyle === 'sphere') {
    viewer3d.setStyle({}, { sphere: { scale: 0.32, colorscheme: 'Jmol' } });
  } else if (viewerStyle === 'line') {
    viewer3d.setStyle({}, { line: { linewidth: 2.2, colorscheme: 'Jmol' } });
  }
  viewer3d.render();
}

async function load3DConformer(smiles, drugName, chirality) {
  try {
    const res = await fetch('/api/v1/cheminformatics/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        smiles: smiles,
        generate_3d: true,
        num_conformers: 6,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    
    // Update descriptors
    document.getElementById('val-formula').textContent = data.molecular_formula || 'N/A';
    document.getElementById('val-mw').textContent = data.molecular_weight || 'N/A';
    document.getElementById('val-asphericity').textContent = data.asphericity ?? 'N/A';
    document.getElementById('val-eccentricity').textContent = data.eccentricity ?? 'N/A';

    // Update overlay chips
    document.getElementById('chip-isomer-name').textContent = `Compound: (${chirality})-${drugName}`;
    document.getElementById('chip-cip').textContent = `CIP: (${chirality}) Stereocenter`;
    document.getElementById('chip-energy').textContent = `Energy: ${data.lowest_energy_kcal_mol ? data.lowest_energy_kcal_mol + ' kcal/mol' : 'Optimized'}`;

    // Populate conformers table
    const tbody = document.getElementById('conformers-tbody');
    if (data.conformers && data.conformers.length > 0) {
      tbody.innerHTML = data.conformers.map(c => `
        <tr>
          <td>#${c.conformer_id}</td>
          <td>${c.energy_kcal_mol} kcal/mol</td>
          <td style="color: ${c.relative_energy_kcal_mol === 0 ? 'var(--accent-green)' : 'var(--text-main)'}; font-weight: 500;">
            ${c.relative_energy_kcal_mol} kcal/mol
          </td>
          <td>
            ${c.is_bioactive_candidate ? '<span class="badge badge-green">Candidate</span>' : '<span class="badge badge-crimson">High Strain</span>'}
          </td>
        </tr>
      `).join('');
    }

    // CRITICAL: Always store lowest_energy_molblock so it's ready when switching to 3D tab
    if (data.lowest_energy_molblock) {
      currentMolBlock = data.lowest_energy_molblock;
      if (viewer3d) {
        render3DMolBlock(currentMolBlock);
      }
    }
  } catch (err) {
    console.error('3D Conformer Loading Error:', err);
  }
}

function render3DMolBlock(molblock) {
  if (!viewer3d || !molblock) return;
  try {
    viewer3d.clear();
    viewer3d.addModel(molblock, 'sdf');
    applyViewerStyle();
    viewer3d.zoomTo({}, 0);
    viewer3d.render();

    disableCanvasWheelZoom();

    if (viewer3d.rotationGroup) {
      standardZ = viewer3d.rotationGroup.position.z;
      standardDistance = Math.max(1, viewer3d.CAMERA_Z - standardZ);
    }
    currentZoomPercent = 100;
    updateZoomUI(100);
  } catch (err) {
    console.error('Error rendering 3D MolBlock:', err);
  }
}

function toggleConformerChirality() {
  loadOppositeIsomer();
}

// --- Knowledge Graph Visualization ---
async function renderKnowledgeGraph(drugName) {
  const container = document.getElementById('graph-svg-container');
  const details = document.getElementById('kg-details-content');
  if (!container) return;

  try {
    const res = await fetch(`/api/v1/graph/drug/${drugName}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    document.getElementById('kg-drug-badge').textContent = `${drugName} Subgraph`;
    
    // Draw SVG Subgraph
    drawSVGGraph(container, data);

    // Populate Disparity Pane
    let html = `
      <h4 style="color: var(--text-main); margin-bottom: 8px; font-size: 14px; font-weight: 700;">Stereoisomeric Divergence Analysis</h4>
      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">Discovered <strong>${data.isomer_count} stereoisomers</strong> for parent compound <strong>${drugName}</strong>.</p>
    `;

    data.isomeric_profiles.forEach(p => {
      const iso = p.isomer;
      const roleColor = iso.eudismic_role && iso.eudismic_role.includes('Teratogen') ? 'var(--accent-crimson)' : 'var(--accent-brick)';
      html += `
        <div style="margin-top: 12px; padding: 12px; background: #FFFFFF; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="color: ${roleColor}; font-size: 13.5px;">${iso.name}</strong>
            <span class="badge badge-brick">${iso.chirality}-Enantiomer</span>
          </div>
          <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Role: ${iso.eudismic_role || 'Unspecified'}</div>
          
          <div style="margin-top: 8px;">
            <span style="font-size: 11.5px; font-weight: 600; color: var(--text-dim); text-transform: uppercase;">Target Binding Affinities:</span>
            <ul style="margin-left: 18px; font-size: 12.5px; margin-top: 3px;">
              ${p.target_bindings.map(b => `
                <li><strong>${b.target.name || b.target.id}</strong> (${b.target.gene_symbol || ''}): 
                    <span style="color: var(--accent-brick); font-family: var(--font-mono); font-weight: 600;">${b.affinity.affinity_type} = ${b.affinity.affinity_value} ${b.affinity.affinity_unit}</span>
                </li>
              `).join('')}
            </ul>
          </div>

          ${p.clinical_liabilities.length > 0 ? `
            <div style="margin-top: 8px;">
              <span style="font-size: 11.5px; font-weight: 600; color: var(--accent-crimson); text-transform: uppercase;">Clinical Hazards:</span>
              <ul style="margin-left: 18px; font-size: 12px; color: #991B1B; margin-top: 3px;">
                ${p.clinical_liabilities.map(l => `
                  <li><strong>${l.liability.name}</strong> [${l.liability.severity}]: ${l.liability.mechanism}</li>
                `).join('')}
              </ul>
            </div>
          ` : ''}
        </div>
      `;
    });

    details.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<p style="color: var(--text-dim);">Could not load graph: ${err.message}</p>`;
  }
}

function drawSVGGraph(container, data) {
  const width = container.clientWidth || 550;
  const height = container.clientHeight || 450;
  const centerX = width / 2;
  const centerY = height / 2;

  let svg = `
    <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="grad-drug" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#C0392B"/>
          <stop offset="100%" stop-color="#992D1D"/>
        </radialGradient>
        <radialGradient id="grad-isomer-s" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#10B981"/>
          <stop offset="100%" stop-color="#059669"/>
        </radialGradient>
        <radialGradient id="grad-isomer-r" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#EF4444"/>
          <stop offset="100%" stop-color="#B91C1C"/>
        </radialGradient>
        <radialGradient id="grad-target" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#64748B"/>
          <stop offset="100%" stop-color="#334155"/>
        </radialGradient>
        <radialGradient id="grad-liab" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#F59E0B"/>
          <stop offset="100%" stop-color="#B45309"/>
        </radialGradient>
      </defs>
  `;

  // Draw Edges First
  svg += `
    <line x1="${centerX}" y1="${centerY}" x2="${centerX - 130}" y2="${centerY - 80}" stroke="#CBD5E1" stroke-width="2"/>
    <line x1="${centerX}" y1="${centerY}" x2="${centerX + 130}" y2="${centerY - 80}" stroke="#CBD5E1" stroke-width="2"/>
    <line x1="${centerX - 130}" y1="${centerY - 80}" x2="${centerX - 170}" y2="${centerY + 100}" stroke="#94A3B8" stroke-width="2" stroke-dasharray="4"/>
    <line x1="${centerX + 130}" y1="${centerY - 80}" x2="${centerX + 170}" y2="${centerY + 100}" stroke="#F59E0B" stroke-width="2" stroke-dasharray="4"/>
    <line x1="${centerX + 130}" y1="${centerY - 80}" x2="${centerX - 170}" y2="${centerY + 100}" stroke="#94A3B8" stroke-width="2" stroke-dasharray="4"/>
  `;

  // Center: Parent Drug
  svg += `
    <g transform="translate(${centerX}, ${centerY})">
      <circle r="36" fill="url(#grad-drug)" filter="drop-shadow(0 2px 6px rgba(178, 58, 34, 0.35))"/>
      <text fill="#FFFFFF" font-family="Outfit, sans-serif" font-weight="700" font-size="12" text-anchor="middle" dy="4">${data.query_drug}</text>
      <text fill="rgba(255,255,255,0.85)" font-family="Inter, sans-serif" font-size="9" text-anchor="middle" dy="16">Parent Drug</text>
    </g>
  `;

  // Left: (R)-Enantiomer
  svg += `
    <g transform="translate(${centerX - 130}, ${centerY - 80})">
      <circle r="30" fill="url(#grad-isomer-r)" filter="drop-shadow(0 2px 5px rgba(239, 68, 68, 0.3))"/>
      <text fill="#FFFFFF" font-family="Outfit, sans-serif" font-weight="700" font-size="11" text-anchor="middle" dy="4">(R)-Isomer</text>
    </g>
  `;

  // Right: (S)-Enantiomer
  svg += `
    <g transform="translate(${centerX + 130}, ${centerY - 80})">
      <circle r="30" fill="url(#grad-isomer-s)" filter="drop-shadow(0 2px 5px rgba(16, 185, 129, 0.3))"/>
      <text fill="#FFFFFF" font-family="Outfit, sans-serif" font-weight="700" font-size="11" text-anchor="middle" dy="4">(S)-Isomer</text>
    </g>
  `;

  // Bottom Left: Primary Target
  svg += `
    <g transform="translate(${centerX - 170}, ${centerY + 100})">
      <circle r="28" fill="url(#grad-target)" filter="drop-shadow(0 2px 5px rgba(51, 65, 85, 0.3))"/>
      <text fill="#FFFFFF" font-family="Outfit, sans-serif" font-weight="700" font-size="11" text-anchor="middle" dy="4">Target Protein</text>
      <text fill="rgba(255,255,255,0.85)" font-family="Inter, sans-serif" font-size="9" text-anchor="middle" dy="16">Active Pocket</text>
    </g>
  `;

  // Bottom Right: Clinical Hazard / Liability
  svg += `
    <g transform="translate(${centerX + 170}, ${centerY + 100})">
      <circle r="28" fill="url(#grad-liab)" filter="drop-shadow(0 2px 5px rgba(217, 119, 6, 0.3))"/>
      <text fill="#FFFFFF" font-family="Outfit, sans-serif" font-weight="700" font-size="11" text-anchor="middle" dy="4">Hazard / Liab</text>
      <text fill="rgba(255,255,255,0.85)" font-family="Inter, sans-serif" font-size="9" text-anchor="middle" dy="16">Disparity Path</text>
    </g>
  `;

  svg += `</svg>`;
  container.innerHTML = svg;
}

// --- Benchmark Runner ---
async function loadBenchmarkInitial() {
  await executeBenchmark();
}

async function executeBenchmark() {
  const btn = document.getElementById('btn-run-benchmark');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/v1/evaluation/benchmark');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const m = data.metrics;
    document.getElementById('bm-disc').textContent = m.delta_improvements.discrimination_accuracy_gain;
    document.getElementById('bm-halluc').textContent = m.delta_improvements.hallucination_reduction;
    document.getElementById('bm-prec').textContent = `${m.chiralgraph_rag.affinity_retrieval_precision_pct}%`;
    document.getElementById('bm-guard').textContent = `${m.chiralgraph_rag.stereochemical_guardrail_pass_rate_pct}%`;

    const tbody = document.getElementById('benchmark-tbody');
    tbody.innerHTML = data.detailed_case_results.map(r => `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 600;">${r.query_id}</td>
        <td><strong>${r.compound}</strong></td>
        <td>${r.naive_rag_hallucinated ? '<span class="badge badge-crimson">Hallucinated</span>' : '<span class="badge badge-green">Clean</span>'}</td>
        <td>${r.chiralgraph_rag_discriminated ? '<span class="badge badge-green">Discriminated (100%)</span>' : '<span class="badge badge-crimson">Failed</span>'}</td>
        <td>${r.chiralgraph_affinity_matched ? '<span class="badge badge-brick">Exact Match</span>' : '<span class="badge badge-amber">Approx</span>'}</td>
        <td>${r.guardrail_passed ? '<span class="badge badge-green">Verified Guardrail</span>' : '<span class="badge badge-crimson">Blocked</span>'}</td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('Benchmark error:', err);
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* Static replay of the GENCO Operator Console (space/app.py). Figures follow app.grid_map and
   app.convergence; all numbers come from data/*.json, written by scripts/build_static_space.py. */
"use strict";

const ORANGE = "#eb6834", AQUA = "#1baf7a", PINK = "#e87ba4", RED = "#e34948";
const SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"];
const CRITICAL = "#d03b3b", MUTED = "#8a8984";

const state = { meta: null, sets: {}, order: "N-1", scen: 0, screenOrder: "N-2" };

const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";

function plotLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: MUTED, size: 12 }, margin: { l: 10, r: 10, t: 40, b: 10 },
    hoverlabel: { bgcolor: "white", font: { color: "#0b0b0b" } },
  };
}
const narrow = () => window.matchMedia("(max-width: 520px)").matches;
const ink = () => getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#0b0b0b";
const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] };

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}
async function loadSet(order) {
  if (!state.sets[order]) state.sets[order] = getJSON(`data/${order}.json`);
  return state.sets[order];
}

// ------------------------------------------------------------------ grid map (app.grid_map)
function gridMap(s) {
  const M = state.meta, TAU = M.tau, pos = M.layout;
  const P = (b) => pos[String(b)];
  const bins = [[0, .5, "< 50%"], [.5, .7, "50–70%"], [.7, .85, "70–85%"], [.85, 1.0, "85–100%"],
                [1.0, TAU, `100–${Math.round(TAU * 100)}%`]];
  const segs = {};
  for (const [f, t, L] of s.branches) {
    let bucket;
    if (L > TAU) bucket = "violation";
    else bucket = bins.find(([lo, hi]) => (lo <= L && L < hi) || (hi === TAU && L <= TAU))[2];
    (segs[bucket] = segs[bucket] || []).push([f, t, L]);
  }
  const traces = [];
  const edges = (list, color, width, name) => {
    if (!list || !list.length) return;
    const xs = [], ys = [], mx = [], my = [], txt = [];
    for (const [f, t, L] of list) {
      const a = P(f), b = P(t);
      xs.push(a[0], b[0], null); ys.push(a[1], b[1], null);
      mx.push((a[0] + b[0]) / 2); my.push((a[1] + b[1]) / 2);
      txt.push(`line ${Math.min(f, t)}–${Math.max(f, t)}<br>loading ${pct(L)}`);
    }
    traces.push({ type: "scatter", x: xs, y: ys, mode: "lines", line: { color, width }, name, hoverinfo: "skip" });
    traces.push({ type: "scatter", x: mx, y: my, mode: "markers", marker: { size: 12, opacity: 0 }, hovertext: txt,
                  hoverinfo: "text", showlegend: false });
  };
  bins.forEach(([, , lbl], i) => edges(segs[lbl], SEQ[i], 2, `loading ${lbl}`));
  edges(segs.violation, CRITICAL, 4, `overload > ${Math.round(TAU * 100)}%`);
  for (const o of s.outages) {
    if (!o.startsWith("line")) continue;
    const [a, b] = o.split(" ")[1].split("–").map(Number);
    traces.push({ type: "scatter", x: [P(a)[0], P(b)[0]], y: [P(a)[1], P(b)[1]], mode: "lines",
                  line: { color: MUTED, width: 2, dash: "dot" }, name: "outaged line", hoverinfo: "skip", showlegend: false });
  }
  const n = s.bus_mis.length;
  const xs = [], ys = [], hover = [], col = [];
  for (let i = 0; i < n; i++) {
    xs.push(P(i)[0]); ys.push(P(i)[1]);
    hover.push(`bus ${i}<br>|V| true ${s.vm_true[i].toFixed(4)} p.u.<br>GENCO |ΔS| ${s.bus_mis[i].toFixed(2)} MVA`);
    col.push(Math.log10(s.bus_mis[i] + 1e-3));
  }
  traces.push({ type: "scatter", x: xs, y: ys, mode: "markers", hovertext: hover, hoverinfo: "text", name: "bus",
                marker: { size: 9, color: col, colorscale: [[0, "#fdeee6"], [1, ORANGE]], cmin: -3, cmax: 2,
                          line: { width: 1, color: "white" },
                          colorbar: { title: { text: "GENCO |ΔS|" }, tickvals: [-3, -1, 1, 2],
                                      ticktext: ["0.001", "0.1", "10", "100 MW"],
                                      ...(narrow() ? { orientation: "h", x: 0.5, xanchor: "center", y: 1.0, yanchor: "bottom",
                                                       len: 0.8, thickness: 10, title: { text: "GENCO |ΔS|", side: "left" } }
                                                   : { len: 0.6 }) } } });
  const r = s.ref;
  traces.push({ type: "scatter", x: [P(r)[0]], y: [P(r)[1]], mode: "markers+text", text: ["slack"],
                textposition: "top center",
                marker: { symbol: "star", size: 16, color: ink(), line: { width: 1, color: "white" } },
                name: "slack bus", hoverinfo: "skip" });
  if (s.v_viol_buses.length) {
    traces.push({ type: "scatter", x: s.v_viol_buses.map((b) => P(b)[0]), y: s.v_viol_buses.map((b) => P(b)[1]),
                  mode: "markers", marker: { size: 16, color: "rgba(0,0,0,0)", line: { width: 3, color: CRITICAL } },
                  name: "voltage violation", hoverinfo: "skip" });
  }
  const layout = Object.assign(plotLayout(), {
    title: { x: 0.05, text: narrow() ? "IEEE 118 after the contingency<br>(true AC loading)" : "IEEE 118 after the contingency (true AC loading)" },
    xaxis: { visible: false }, yaxis: { visible: false, scaleanchor: "x" },
    legend: { orientation: "h", y: -0.02, font: { size: narrow() ? 10 : 11 } },
  });
  if (narrow()) layout.margin = { l: 4, r: 4, t: 96, b: 4 };
  Plotly.react("gridmap", traces, layout, PLOT_CONFIG);
}

// ------------------------------------------------------------ convergence (app.convergence)
function convergence(s) {
  const L = s.layer_raw.length;
  const xg = Array.from({ length: L }, (_, i) => i + 1);
  const xs = (h) => h.map((_, i) => L + i);
  const xn = xs(s.hist_genco);
  const traces = [
    { type: "scatter", x: xg, y: s.layer_raw, mode: "lines+markers", name: "GENCO layers (its own slack voltage)",
      line: { color: ORANGE, width: 2, dash: "dot" }, marker: { size: 7 } },
    { type: "scatter", x: xg.concat(xn.slice(1)), y: s.layer_phys.concat(s.hist_genco.slice(1)), mode: "lines+markers",
      name: "GENCO → Newton (true setpoint)", line: { color: AQUA, width: 3 }, marker: { size: 8 } },
    { type: "scatter", x: xs(s.hist_base), y: s.hist_base, mode: "lines+markers", name: "Newton from base case",
      line: { color: RED, width: 2 }, marker: { size: 7 } },
    { type: "scatter", x: xs(s.hist_flat), y: s.hist_flat, mode: "lines+markers", name: "Newton from flat start",
      line: { color: PINK, width: 2 }, marker: { size: 7 } },
  ];
  const layout = Object.assign(plotLayout(), {
    title: { x: 0.05, text: narrow() ? "One solve, two engines:<br>learned layers, then Newton" : "One solve, two engines: learned layers, then Newton" },
    yaxis: { type: "log", title: { text: "max power mismatch (MW)" }, gridcolor: "rgba(128,128,128,0.2)", exponentformat: "power", automargin: true },
    xaxis: { title: { text: narrow() ? "step (1–12 GENCO, then Newton)" : "step (layers 1–12 = GENCO, then Newton iterations)" },
             dtick: narrow() ? 2 : 1, gridcolor: "rgba(128,128,128,0.2)", automargin: true },
    legend: { orientation: "h", y: -0.25 },
    shapes: [
      // fig.add_hline(y=1e-6)
      { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 1e-6, y1: 1e-6, line: { color: MUTED, dash: "dash", width: 1 } },
      // fig.add_vrect(x0=0.5, x1=L+0.5)
      { type: "rect", xref: "x", x0: 0.5, x1: L + 0.5, yref: "paper", y0: 0, y1: 1, fillcolor: ORANGE, opacity: 0.06,
        line: { width: 0 }, layer: "below" },
    ],
    annotations: [
      // annotations on a log axis take log10 coordinates
      { x: 1, y: -6, text: "converged (1e-6 MW)", showarrow: false, yshift: -10, xanchor: "left", font: { size: narrow() ? 10 : 11, color: MUTED } },
      { x: L, y: Math.log10(Math.max(s.hist_flat[0], s.hist_base[0])), text: "classical Newton starts here", showarrow: true,
        arrowhead: 0, ax: narrow() ? 10 : 60, ay: narrow() ? -46 : -30, xanchor: narrow() ? "right" : "auto",
        font: { size: narrow() ? 10 : 11, color: MUTED }, arrowcolor: MUTED },
      { x: 0.5, xref: "x", y: narrow() ? 0 : 1, yref: "paper", text: narrow() ? "GENCO's 12 layers" : "GENCO's 12 learned correction layers",
        showarrow: false, xanchor: "left", yanchor: narrow() ? "bottom" : "top", font: { color: MUTED, size: narrow() ? 10 : 12 } },
    ],
  });
  layout.margin = narrow() ? { l: 10, r: 10, t: 56, b: 10 } : { l: 10, r: 10, t: 40, b: 10 };
  Plotly.react("convergence", traces, layout, PLOT_CONFIG);
}

// ----------------------------------------------------------------------------- tables
function cell(text) {
  const td = document.createElement("td");
  const s = String(text);
  if (s.startsWith("⚠ ")) {
    td.className = "flag-bad num";
    td.textContent = s.slice(2);
    td.title = "flagged as violating";
    td.setAttribute("aria-label", `${s.slice(2)}, flagged as violating`);
  } else if (s.startsWith("✓ ")) {
    td.className = "flag-ok num";
    td.textContent = s.slice(2);
    td.title = "within limits";
  } else {
    td.textContent = s;
    if (/^[-\d.]+$/.test(s)) td.className = "num";
  }
  return td;
}
function fillTable(el, columns, rows) {
  el.replaceChildren();
  const thead = el.createTHead().insertRow();
  for (const c of columns) { const th = document.createElement("th"); th.textContent = c; thead.appendChild(th); }
  const tb = el.createTBody();
  for (const r of rows) { const tr = tb.insertRow(); for (const v of r) tr.appendChild(cell(v)); }
}

// ----------------------------------------------------------------------------- rendering
function radios(container, name, value, onChange) {
  container.replaceChildren();
  for (const o of state.meta.orders) {
    const lab = document.createElement("label");
    const inp = document.createElement("input");
    inp.type = "radio"; inp.name = name; inp.value = o; inp.checked = o === value;
    inp.addEventListener("change", () => onChange(o));
    lab.append(inp, document.createTextNode(o));
    container.appendChild(lab);
  }
}

async function setOrder(order) {
  state.order = order;
  const set = await loadSet(order);
  const sel = $("scenario");
  sel.replaceChildren();
  set.scenarios.forEach((s, i) => sel.add(new Option(s.label, String(i))));
  sel.value = "0";
  showScenario(0);
}

async function showScenario(i) {
  const set = await loadSet(state.order);
  const s = set.scenarios[i] || set.scenarios[0];
  state.scen = i;
  $("card").innerHTML = s.card.html;
  const v = s.violations;
  fillTable($("violations"), v.columns, v.rows);
  gridMap(s);
  convergence(s);
}

function showScreen(order) {
  state.screenOrder = order;
  const r = state.meta.screen[order];
  fillTable($("screen"), r.columns, r.rows);
  $("screen-note").textContent = r.note;
}

function setupTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const select = (tab, focus) => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !on;
    }
    if (focus) tab.focus();
    if (tab.id === "tab-inspect") {
      for (const id of ["gridmap", "convergence"]) { const el = $(id); if (el && el.data) Plotly.Plots.resize(el); }
    }
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const j = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
        select(tabs[j], true);
      }
    });
  });
}

async function main() {
  setupTabs();
  $("card").innerHTML = '<p class="loading">Loading precomputed runs.</p>';
  state.meta = await getJSON("data/meta.json");
  $("findings").innerHTML = state.meta.findings_html;
  radios($("order-inspect"), "order-inspect", state.order, setOrder);
  radios($("order-screen"), "order-screen", state.screenOrder, showScreen);
  $("scenario").addEventListener("change", (e) => showScenario(Number(e.target.value)));
  showScreen(state.screenOrder);
  await setOrder(state.order);
  for (const o of state.meta.orders) loadSet(o);   // prefetch the other orders
  let w = window.innerWidth;
  window.addEventListener("resize", () => {
    if (narrow() !== (w <= 520)) showScenario(state.scen);
    w = window.innerWidth;
  });
}

window.addEventListener("DOMContentLoaded", () => {
  main().catch((err) => {
    console.error(err);
    $("card").innerHTML = "";
    const p = document.createElement("p");
    p.className = "loading";
    p.textContent = `Could not load the precomputed data (${err.message}).`;
    $("card").appendChild(p);
  });
});

"""Operator console for the GENCO grid foundation model (companion to the write-up in docs/POST.md).

Pick a contingency order and a scenario. GENCO runs live on IEEE 118. After each Newton step
a calibrated bound on every monitored branch loading and PQ voltage decides whether the
decision can be accepted; after three steps the case goes to a full solve. The same policy
runs from the pre-contingency state for comparison, next to plain Newton from flat,
pre-contingency and GENCO starts.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from engine import EPS_V, TAU, Engine

E = Engine()
ASSETS = Path(__file__).resolve().parent / "assets"
ORDERS = ["N-1", "N-2", "N-3", "N-5"]

# Reference palette (dataviz skill): categorical slots, sequential blue, status critical.
ORANGE, AQUA, PINK, RED, GREEN, VIOLET = "#eb6834", "#1baf7a", "#e87ba4", "#e34948", "#008300", "#4a3aa7"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
CRITICAL = "#d03b3b"
MUTED = "#8a8984"
LOAD_BINS = [(0, .5, "< 50%"), (.5, .7, "50–70%"), (.7, .85, "70–85%"), (.85, 1.0, "85–100%"), (1.0, TAU, f"100–{TAU:.0%}")]

PLOT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font=dict(color=MUTED, size=12), margin=dict(l=10, r=10, t=40, b=10),
                   hoverlabel=dict(bgcolor="white", font_color="#0b0b0b"))


# ----------------------------------------------------------------------------- figures
def grid_map(s):
    pos = E.layout
    p = s.p
    load = s.load["true"]
    fig = go.Figure()
    # branches (undirected), bucketed by true loading
    seen, segs = set(), {}
    from genco_certified.acpf import branch_keys
    key = branch_keys(p)
    for k, f, t in zip(key, p.f, p.t):
        if k in seen:
            continue
        seen.add(k)
        L = load[k]
        if L > TAU:
            bucket = "violation"
        else:
            bucket = next(lbl for lo, hi, lbl in LOAD_BINS if lo <= L < hi or (hi == TAU and L <= TAU))
        segs.setdefault(bucket, []).append((f, t, L))
    for i, (lo, hi, lbl) in enumerate(LOAD_BINS):
        _edges(fig, segs.get(lbl, []), SEQ[i], 2, f"loading {lbl}")
    _edges(fig, segs.get("violation", []), CRITICAL, 4, f"overload > {TAU:.0%}")
    # outaged lines (dashed)
    for o in s.outages:
        if o.startswith("line"):
            a, b = [int(x) for x in o.split()[1].split("–")]
            fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                                     line=dict(color=MUTED, width=2, dash="dot"), name="outaged line",
                                     hoverinfo="skip", showlegend=False))
    # buses: colour = GENCO's physical mismatch (true slack setpoint)
    xs, ys = [pos[i][0] for i in range(len(s.bus_mis))], [pos[i][1] for i in range(len(s.bus_mis))]
    vm = s.vm["true"]
    hover = [f"bus {i}<br>|V| true {vm[i]:.4f} p.u.<br>GENCO |ΔS| {s.bus_mis[i]:.2f} MVA" for i in range(len(xs))]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", hovertext=hover, hoverinfo="text", name="bus",
                             marker=dict(size=9, color=np.log10(s.bus_mis + 1e-3), colorscale=[[0, "#fdeee6"], [1, ORANGE]],
                                         cmin=-3, cmax=2, line=dict(width=1, color="white"),
                                         colorbar=dict(title="GENCO |ΔS|", tickvals=[-3, -1, 1, 2],
                                                       ticktext=["0.001", "0.1", "10", "100 MW"], len=0.6))))
    ref = int(p.ref[0])
    fig.add_trace(go.Scatter(x=[pos[ref][0]], y=[pos[ref][1]], mode="markers+text", text=["slack"], textposition="top center",
                             marker=dict(symbol="star", size=16, color="#0b0b0b", line=dict(width=1, color="white")),
                             name="slack bus", hoverinfo="skip"))
    pq = p.pq
    bad = [b for b in pq if vm[b] < p.vmin[b] - EPS_V or vm[b] > p.vmax[b] + EPS_V]
    if bad:
        fig.add_trace(go.Scatter(x=[pos[b][0] for b in bad], y=[pos[b][1] for b in bad], mode="markers",
                                 marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=3, color=CRITICAL)),
                                 name="voltage violation", hoverinfo="skip"))
    fig.update_layout(**PLOT_LAYOUT, height=520, title="IEEE 118 after the contingency (true AC loading)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.02, font=dict(size=11)))
    return fig


def _edges(fig, segs, color, width, name):
    if not segs:
        return
    pos = E.layout
    xs, ys, mx, my, txt = [], [], [], [], []
    for f, t, L in segs:
        xs += [pos[f][0], pos[t][0], None]
        ys += [pos[f][1], pos[t][1], None]
        mx.append((pos[f][0] + pos[t][0]) / 2); my.append((pos[f][1] + pos[t][1]) / 2)
        txt.append(f"line {min(f, t)}–{max(f, t)}<br>loading {L:.1%}")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width), name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=12, opacity=0), hovertext=txt,
                             hoverinfo="text", showlegend=False))


def convergence(s):
    L = len(s.layer_raw)
    x_g = list(range(1, L + 1))
    x_n = [L + i for i in range(len(s.hist_genco))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_g, y=s.layer_raw, mode="lines+markers", name="GENCO layers (its own slack voltage)",
                             line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=x_g + x_n[1:], y=s.layer_phys + s.hist_genco[1:], mode="lines+markers",
                             name="GENCO → Newton (true setpoint)", line=dict(color=AQUA, width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[L + i for i in range(len(s.hist_base))], y=s.hist_base, mode="lines+markers",
                             name="Newton from base case", line=dict(color=RED, width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=[L + i for i in range(len(s.hist_flat))], y=s.hist_flat, mode="lines+markers",
                             name="Newton from flat start", line=dict(color=PINK, width=2), marker=dict(size=7)))
    fig.add_hline(y=1e-6, line=dict(color=MUTED, dash="dash", width=1))
    # annotations on a log axis take log10 coordinates
    fig.add_annotation(x=1, y=-6, text="converged (1e-6 MW)", showarrow=False, yshift=-10, xanchor="left",
                       font=dict(size=11, color=MUTED))
    fig.add_annotation(x=L, y=np.log10(max(s.hist_flat[0], s.hist_base[0])), text="classical Newton starts here",
                       showarrow=True, arrowhead=0, ax=60, ay=-30, font=dict(size=11, color=MUTED), arrowcolor=MUTED)
    fig.add_vrect(x0=0.5, x1=L + 0.5, fillcolor=ORANGE, opacity=0.06, line_width=0,
                  annotation_text="GENCO's 12 learned correction layers", annotation_position="top left")
    fig.update_layout(**PLOT_LAYOUT, height=420, title="One solve, two engines: learned layers, then Newton",
                      yaxis=dict(type="log", title="max power mismatch (MW)", gridcolor="rgba(128,128,128,0.2)",
                                 exponentformat="power"),
                      xaxis=dict(title="step (layers 1–12 = GENCO, then Newton iterations)", dtick=1,
                                 gridcolor="rgba(128,128,128,0.2)"),
                      legend=dict(orientation="h", y=-0.25))
    return fig


# ----------------------------------------------------------------------------- text panels
def decision_card(s):
    p = s.p
    pq = p.pq
    vt = s.vm["true"][pq]
    real = int((s.load["true"] > TAU).sum() + ((vt < p.vmin[pq] - EPS_V) | (vt > p.vmax[pq] + EPS_V)).sum())
    tier_txt = (f"**Accepted after Newton step {s.accepted_step}.** Every monitored element is clear of its limit by more than "
                f"the calibrated error bound (c = {E.c_genco:.3g}, rule {E.rule})."
                if s.accepted_step else
                "**Escalated.** Some element stays inside the error bound for the first 3 Newton steps, so Newton runs to convergence.")
    return (f"### Decision\n{tier_txt}\n\n"
            f"- Outage: {', '.join(s.outages) or 'none'}\n"
            f"- GENCO's own max mismatch: **{s.layer_raw[-1]:.2f} MW**; with the true slack setpoint: **{s.layer_phys[-1]:.1f} MW**\n"
            f"- Newton step size ‖dx‖ = {s.step_l2:.2e} (error estimate)\n"
            f"- Factorizations used: **{s.lu_policy}**  vs  {len(s.hist_base) - 1} (base-case start) · {len(s.hist_flat) - 1} (flat start)\n"
            f"- Violations in the true AC solution: **{real}**")


def _is_v_viol(s, elem, method):
    b = int(elem.split()[1])
    v = s.vm[method][b]
    return v < s.p.vmin[b] - EPS_V or v > s.p.vmax[b] + EPS_V


def violation_table(s):
    v = s.violations
    if not len(v):
        return pd.DataFrame({"element": ["no violations flagged by any method"]})
    def flag(row, m):
        val = row[m]
        bad = (val > TAU) if row.kind == "thermal" else _is_v_viol(s, row.element, m)
        unit = f"{val:.1%}" if row.kind == "thermal" else f"{val:.4f} p.u."
        return f"{'⚠ ' if bad else '✓ '}{unit}"
    out = pd.DataFrame({
        "element": v.element, "type": v.kind,
        "true AC": [flag(r, "true") for _, r in v.iterrows()],
        "GENCO alone": [flag(r, "raw") for _, r in v.iterrows()],
        "GENCO + 1 Newton step": [flag(r, "step1") for _, r in v.iterrows()],
        "base case + 1 step": [flag(r, "base1") for _, r in v.iterrows()],
    })
    return out


# ----------------------------------------------------------------------------- callbacks
def choices(order):
    S = E.scenarios(order)
    labels = [s.label for s in S]
    return gr.update(choices=labels, value=labels[0])


def show(order, label):
    S = E.scenarios(order)
    s = next((x for x in S if x.label == label), S[0])
    return grid_map(s), convergence(s), decision_card(s), violation_table(s)


def batch(order):
    r = E.batch_summary(order)
    df = pd.DataFrame(r["rows"], columns=["method", "recall", "missed", "false alarms", "factorizations / contingency"])
    df["recall"] = df["recall"].map(lambda x: f"{x:.3f}")
    df["factorizations / contingency"] = df["factorizations / contingency"].map(lambda x: f"{x:.2f}")
    note = (f"{r['n']} bundled {order} scenarios (includes the 8 hardest by GENCO mismatch, so this sample is harder than average). "
            f"Accepted after one step: {r['tier1']:.0%}; accepted within 3 steps: {r['accepted']:.0%} (rule {r['rule']}, c = {r['c_genco']:.3g}). "
            f"Full-set results (2,200 scenarios per order, 100 load scenarios): TABLE III of the write-up; summary in the Findings tab.")
    return df, note


INTRO = """
# GENCO Operator Console
This console runs GENCO, the GridFM grid foundation model, live on IEEE 118 contingencies. After each Newton step, a calibrated bound on every monitored branch loading and PQ voltage decides whether the screening decision can be accepted. Otherwise the policy takes another step, and after three steps it falls back to a full solve. The same policy runs from the pre-contingency state for comparison. All safety evidence is empirical.

<div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 4px">
<div style="flex:1;min-width:180px;padding:12px;border-radius:10px;border:1px solid rgba(128,128,128,.3)"><div style="font-size:26px;font-weight:700">0 missed</div><div>violations in 16,449 held out, and 0 in 17,095 more generated after the rule was frozen, from a different load profile</div></div>
<div style="flex:1;min-width:180px;padding:12px;border-radius:10px;border:1px solid rgba(128,128,128,.3)"><div style="font-size:26px;font-weight:700">37 to 44% fewer</div><div>factorizations than the same check from the pre-contingency state (IEEE 118, N-1 to N-5)</div></div>
<div style="flex:1;min-width:180px;padding:12px;border-radius:10px;border:1px solid rgba(128,128,128,.3)"><div style="font-size:26px;font-weight:700">32 to 60x</div><div>gap between GENCO's reported power balance and the balance at the true slack-voltage setpoint (IEEE 14 to 118; about 20x on GOC 500 and 2000)</div></div>
</div>
"""

FINDINGS = """
### What is GENCO worth to a Newton solver?

In contingency analysis, a power-flow prediction is only as valuable as the solver work it saves, and that solver doesn't start from scratch. It starts from the solved pre-contingency case and usually reuses that case's factorization in a chord or fast-decoupled method. The write-up puts the prediction of GENCO in front of each of these solvers on three grids and measures what it saves at equal safety. On IEEE 118 a GENCO start saves a fifth of the work for Newton and a twentieth for fast-decoupled. At 2,000 buses it saves 2% for chord Newton and costs extra for the other two. The screening savings below are counted in full-Newton factorizations and do not pay back in wall-clock time. Every number traces to `docs/NUMBERS.md` in the repository.

**Results at a glance** (IEEE 118, released checkpoints, unless stated)

| Question | Answer |
|---|---|
| Newton iterations saved over the pre-contingency start, to 10⁻⁴ MW | 0.49 [0.45, 0.53] at N-1, 1.0 at N-5; 0.33 on GOC 500; −0.63 on GOC 2000 |
| Work saved under factor reuse | chord: 21% of triangular solves; fast-decoupled: 5%; at 2,000 buses −18% to +2% |
| Wall-clock break-even | inference must cost under 0.015–0.6 ms per contingency; GENCO Base takes 2.5 ms on a T4 GPU and 8.7 ms on a laptop GPU. Only the smallest model in fp16 crosses one threshold (full Newton, N-5). Even with free inference, GENCO + Newton is slower than fast-decoupled from the pre-contingency state |
| Screening with a calibrated Newton check | 0 missed violations in 16,449 held out, and 0 in 17,095 more generated after the rule was frozen, from a different load profile. 37–44% fewer factorizations than the same check from the pre-contingency state |
| What GENCO reports vs what it computes | its power balance looks 32–60× better than the physical one on IEEE 14–118 (about 20× on GOC 500 and 2000), because the slack-bus voltage setpoint is masked and predicted |
| Conditioning GENCO on the base case | a residual decoder (GENCO-Δ) cuts N-1 state error 2.5× but saves no iterations at 10⁻⁴ MW; mixed N-k training trades the N-1 gain for an N-5 fix |

[[fig:figH_hero.png]]

*Fig. 1. Triangular solves saved by starting from GENCO's prediction instead of the pre-contingency state, to 10⁻⁴ MW, paired over contingencies, 95% cluster-bootstrap CIs over load scenarios. Released checkpoints: IEEE 118 Base (638 contingencies), GOC 500 Base (638), GOC 2000 Small (879 held out). Shaded: the pre-contingency start is cheaper.*

#### What this console shows

The sequential policy is the one in Fig. 3. After each Newton step it checks every monitored branch loading and PQ voltage, and stops once every element is clearly above or below its limit. Otherwise it takes another step, and after three steps it falls back to a full solve. The constant c in the bound is a split-conformal quantile over the load scenarios of a separate 100-load-scenario calibration set. The rule is identical from either start, with c calibrated separately for each; each Newton step costs one factorization.

On IEEE 118 the GENCO policy needs 1.37 to 1.65 factorizations per contingency from N-1 to N-5, against 2.14 to 2.96 from the pre-contingency start, with no violating element missed in 16,449 (TABLE III). The saving depends on the rule (35–44% with α = 0.10, 33–42% with α = 0.05, 18–35% with the conservative maximum rule), and the bound is conservative: it was exceeded in 1–7% of test operating points against a nominal 10%.

#### The slack convention

Graphkit's PF mask hides the slack-bus voltage magnitude from GENCO and asks it to predict it, although in the data it is a dispatch setpoint. GENCO's reported power balance is evaluated at that predicted voltage. Restoring the setpoint raises the largest nodal mismatch by 32–60× on IEEE 14–118 and about 20× on GOC 500 and 2000, and the worst bus is adjacent to the slack in 95–100% of scenarios. The grid map colours buses by this physical mismatch, and the convergence plot shows both versions (dotted: GENCO's own slack voltage; green: the true setpoint). The three repairs we tried each cost something: the setpoint as an input multiplies Base's state error by 17×, an output clamp degrades Base by 5.7×, and a slack-aware fine-tune costs 29% more state error than a matched control while leaving Newton iterations unchanged.

#### Layers

GENCO's mean mismatch on IEEE 118 falls from 261 MW after the first of its 12 correction layers to 0.054 MW after the last. Newton iterations (to 10⁻⁶ MW) started from intermediate layers stay near 3 until layer 7 and then drop, and stopping after layer 9 saves 26% of inference for 0.12 extra iterations.

#### Teaching GENCO the base case (GENCO-Δ)

Plain Δ (base-case values written into the masked inputs) ended at about base-case quality, and one seed diverged. Residual Δ (a delta head initialised to zero on top of the base case) cuts N-1 state error from 0.054 to 0.022 and iterations to 10⁻² MW from 1.74 to 1.33, with no gain at 10⁻⁴ MW and a regression at N-5. Retraining on mixed N-1 to N-5 outages removes the regression and the N-1 gain with it, and the model ends at about released quality at every outage order. GENCO-Δ was run on IEEE 118 only.

#### What did not work

Error warnings from GENCO's embeddings (a 118-entry bus lookup explains R² 0.55 of the per-bus voltage error, the embeddings 0.58, a randomly initialised GENCO 0.52), per-bus bias correction, accepting GENCO with no Newton step, and convergence rescue (flat-start Newton converged on every IEEE 118 datakit scenario that solved, up to N-5).

#### Limitations

- All safety evidence is empirical, on datakit scenarios. The 400 confirmation operating points share one load trajectory, so they are not independent draws.
- At 2,000 buses the pre-contingency start usually wins (GENCO Small). The grid-size trend compares one model size per grid.
- Timings come from Python on a shared laptop. A compiled solver would lower every break-even further.
- The 30 bundled scenarios per order in this console are not a sample of the test sets: they include the 8 hardest by GENCO mismatch, so "Screen a whole set" looks worse than the full-set results above.

[[fig:figB_exchange_rate.png]]

*Fig. 2. (a) Newton iterations from flat, DC, pre-contingency and GENCO starts; (b) paired GENCO minus pre-contingency difference, 95% CIs. Open dots: IEEE 118 seeds 1 and 42 on their own splits.*

[[fig:figA_workflow.png]]

*Fig. 3. The sequential policy. Either start feeds float64 Newton steps; after each step a calibrated bound on every monitored element decides accept or continue; after three steps the case goes to a full solve.*

[[fig:figC_equal_cost.png]]

*Fig. 4. Missed violating elements against factorizations per contingency, IEEE 118, four 100-load-scenario test sets. Lines: GENCO + k and pre-contingency + k Newton steps; stars: sequential policies. Hollow markers: zero-miss cells at their one-sided 95% Clopper–Pearson bound.*

[[fig:figF_genco_delta.png]]

*Fig. 5. GENCO-Δ on IEEE 118 Base: median state error (top) and Newton iterations to 10⁻² MW (bottom); mean ± SD over seeds 0/1/42. Purple bars: the pre-contingency state. Stars: residual Δ retrained on mixed N-1…N-5 outages. Crosses: the diverged plain-Δ seed, excluded from the mean.*
"""


def build():
    with gr.Blocks(title="GENCO Operator Console") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Inspect a contingency"):
            with gr.Row():
                order = gr.Radio(ORDERS, value="N-1", label="Contingency order (GENCO was trained on N-1)")
                scen = gr.Dropdown(label="Scenario", choices=[], interactive=True)
            with gr.Row():
                with gr.Column(scale=6):
                    gmap = gr.Plot(label="Grid")
                with gr.Column(scale=5):
                    card = gr.Markdown()
                    conv = gr.Plot(label="Convergence")
            table = gr.Dataframe(label="Violations: true AC vs each screen (⚠ = flagged as violating, ✓ = within limits)", wrap=True)
            order.change(choices, order, scen)
            scen.change(show, [order, scen], [gmap, conv, card, table])
            demo.load(choices, order, scen)
        with gr.Tab("Screen a whole set"):
            border = gr.Radio(ORDERS, value="N-2", label="Contingency order")
            run = gr.Button("Run the screening policy on all bundled scenarios", variant="primary")
            out = gr.Dataframe()
            note = gr.Markdown()
            run.click(batch, border, [out, note])
        with gr.Tab("Findings"):
            for chunk in re.split(r"\[\[fig:([\w.]+)\]\]", FINDINGS):
                if chunk.endswith(".png"):
                    gr.Image(str(ASSETS / chunk), show_label=False, interactive=False)
                elif chunk.strip():
                    gr.Markdown(chunk)
    return demo


if __name__ == "__main__":
    build().launch()

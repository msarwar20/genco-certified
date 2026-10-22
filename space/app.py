"""GENCO, Certified: an operator console for the GENCO grid foundation model.

Pick a contingency order and a scenario. GENCO runs live on IEEE 118; one Newton step
turns its answer into an exact AC power flow and certifies (or escalates) every
thermal and voltage decision. Classical baselines run alongside.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from engine import EPS_V, TAU, Engine

E = Engine()
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
    tier_txt = (f"**Accepted after Newton step {s.accepted_step}.** Every decision clears the calibrated error band "
                f"(c = {E.c_genco:.3g}, rule {E.rule})."
                if s.accepted_step else
                "**Escalated.** A decision stays inside the error band for the first 3 Newton steps, so Newton runs to convergence.")
    return (f"### Decision\n{tier_txt}\n\n"
            f"- Outage: {', '.join(s.outages) or 'none'}\n"
            f"- GENCO's own max mismatch: **{s.layer_raw[-1]:.2f} MW**; with the true slack setpoint: **{s.layer_phys[-1]:.1f} MW**\n"
            f"- Newton step size ‖dx‖ = {s.step_l2:.2e} (error certificate)\n"
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
            f"Accepted after one step: {r['tier1']:.0%}; accepted within 3 steps: {r['accepted']:.0%} (rule {r['rule']}, c = {r['c_genco']:.3g}). Full-set results (~660 scenarios per order): see the Findings tab.")
    return df, note


INTRO = """
# GENCO, Certified
This console runs GENCO, the GridFM grid foundation model, live on IEEE 118 contingencies. A single Newton step then checks every thermal and voltage decision against the AC power-flow equations. Decisions that clear a calibrated error band are accepted; the rest are solved to convergence.

<div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 4px">
<div style="flex:1;min-width:180px;padding:12px;border-radius:10px;border:1px solid rgba(128,128,128,.3)"><div style="font-size:26px;font-weight:700">0 missed</div><div>violations on N-1 to N-5 (about 2,600 contingencies), with the rule calibrated on N-1 only</div></div>
<div style="flex:1;min-width:180px;padding:12px;border-radius:10px;border:1px solid rgba(128,128,128,.3)"><div style="font-size:26px;font-weight:700">40 to 51% fewer</div><div>factorizations than Newton warm-started from the pre-contingency state (IEEE 118)</div></div>
<div style="flex:1;min-width:180px;padding:12px;border-radius:10px;border:1px solid rgba(128,128,128,.3)"><div style="font-size:26px;font-weight:700">26 to 60x</div><div>gap between GENCO's reported power balance and the balance at the true slack-voltage setpoint</div></div>
</div>
"""

FINDINGS = """
### Summary of findings

1. **GENCO behaves like a learned iterative solver.** Its own power mismatch shrinks by about half per layer. Layers 10 to 12 add little once a Newton step follows, and dropping them saves about a quarter of the inference time.

2. **The PF task masks a known setpoint.** gridfm-graphkit asks GENCO to predict the slack-bus voltage magnitude, which in the data is a dispatch setpoint.
   - GENCO's states are accurate almost everywhere, but they are inconsistent with that setpoint around the slack bus.
   - As a result, the reported power balance is 26 to 60 times better than the physical one.

3. **One Newton step fixes and measures the error.** It reduces GENCO's state error about 900-fold on IEEE 118, and its size ranks the true error exactly.

4. **Screening.** Calibrated on N-1 and applied unchanged to N-1, N-2, N-3 and N-5, the two-tier rule found every violation, with no false alarms.
   - It needs 1.4 to 2.0 factorizations per contingency.
   - Newton from the base case needs 2.9 to 3.3, and Newton from a flat start needs 4.
   - DC screening misses about half of the violations; base case plus one Newton step misses 5 to 8%.

5. **The embeddings know where the voltage errors are.** A linear probe on the final-layer embeddings predicts each bus's voltage-magnitude error with R² of 0.63. The local power mismatch alone gives 0.24.

6. **Where it stops working.**
   - At 2,000 buses, with the released Small checkpoint, the pre-contingency state is a better starting point than GENCO (2.95 vs 3.23 Newton iterations). Picking whichever of the two has the smaller mismatch costs nothing and is never worse.
   - On a laptop, a lean sparse Newton solver is faster in wall-clock time than GENCO inference, so all costs here are counted in factorizations.
   - A slack-aware fine-tune removed the reported-balance gap, but it lost accuracy at the budget we could afford.

The bundled scenarios include the hardest cases on purpose, so the "Screen a whole set" numbers are worse than the full-set results quoted above.
"""


def build():
    with gr.Blocks(title="GENCO, Certified") as demo:
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
            table = gr.Dataframe(label="Violations: true AC vs each screen (⚠ = flagged as violating)", wrap=True)
            order.change(choices, order, scen)
            scen.change(show, [order, scen], [gmap, conv, card, table])
            demo.load(choices, order, scen)
        with gr.Tab("Screen a whole set"):
            border = gr.Radio(ORDERS, value="N-2", label="Contingency order")
            run = gr.Button("Run GENCO + certificate on all bundled scenarios", variant="primary")
            out = gr.Dataframe()
            note = gr.Markdown()
            run.click(batch, border, [out, note])
        with gr.Tab("Findings"):
            gr.Markdown(FINDINGS)
            for f in ["fig4_stress.png", "fig3_screening.png", "fig2_hidden_slack.png", "fig1_layer_dynamics.png"]:
                gr.Image(f"assets/{f}", show_label=False, interactive=False)
    return demo


if __name__ == "__main__":
    build().launch()

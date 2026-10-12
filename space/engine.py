"""Computation behind the Space: run GENCO live on bundled scenarios and certify it.

Everything here reuses the project package `genco_certified` (vendored next to this
file in the Space): graphkit's own data pipeline + the released GENCO checkpoint, the
float64 Newton-Raphson on the exact network GENCO sees, and the calibrated escalation
rule from the experiments (constants fitted on N-1 only).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
os.environ.setdefault("GENCO_CERTIFIED_ROOT", str(HERE))

from genco_certified import acpf  # noqa: E402
from genco_certified.loading import load_genco  # noqa: E402
from genco_certified.tracing import trace_layers  # noqa: E402
from gridfm_graphkit.datasets.masking import RemovePFMask  # noqa: E402

BUNDLE = HERE / "bundle"
GRID = "case118_ieee"
TAU = 1.01        # branch overload threshold on |S|/rate
EPS_V = 1e-3      # PQ-bus voltage tolerance (p.u.)
C1 = 0.261        # tier-1 bound constant, calibrated on stress N-1 (results/policy_with_basecase.log)


@dataclass
class Scenario:
    idx: int
    label: str
    outages: list
    p: acpf.PFProblem
    layer_raw: list           # max |mismatch| per GENCO layer, GENCO's own slack Vm (MW)
    layer_phys: list          # same with the true slack setpoint (MW)
    hist_genco: list          # NR history after GENCO (iteration 0 = GENCO output)
    hist_flat: list
    hist_base: list
    load: dict                # branch loadings per method
    vm: dict                  # bus |V| per method
    bus_mis: np.ndarray       # |ΔS| per bus at GENCO output, true setpoint (MW)
    step_l2: float
    tier: int
    lu_policy: int
    violations: pd.DataFrame = field(default=None)


def _outages(raw_branch, raw_gen):
    out = [f"line {int(r.from_bus)}–{int(r.to_bus)}" for r in raw_branch[raw_branch.br_status == 0].itertuples()]
    out += [f"generator at bus {int(r.bus)}" for r in raw_gen[raw_gen.in_service == 0].itertuples()]
    return out


def _violations(p, loads, vms):
    """Elements violating in truth or flagged by any method (PQ voltages only)."""
    key = acpf.branch_keys(p)
    ends = {}
    for k, f, t in zip(key, p.f, p.t):
        ends.setdefault(k, (min(f, t), max(f, t)))
    rows = []
    for b in range(len(loads["true"])):
        if max(loads[m][b] for m in loads) > TAU:
            rows.append({"element": f"line {ends[b][0]}–{ends[b][1]}", "kind": "thermal",
                         **{m: loads[m][b] for m in ["true", "raw", "step1", "base1"]}})
    pq = p.pq
    lo, hi = p.vmin[pq] - EPS_V, p.vmax[pq] + EPS_V
    for j, b in enumerate(pq):
        vals = {m: vms[m][b] for m in vms}
        if any(v < lo[j] or v > hi[j] for v in vals.values()):
            rows.append({"element": f"bus {b}", "kind": "voltage",
                         **{m: vals[m] for m in ["true", "raw", "step1", "base1"]}})
    return pd.DataFrame(rows)


def _tier(p, load1, vm1, step_l2):
    """Tier 1 if every decision after one Newton step clears the calibrated bound."""
    band = C1 * step_l2 ** 2
    ok_b = np.all(np.abs(TAU - load1) > band)
    pq = p.pq
    m = np.minimum(vm1[pq] - (p.vmin[pq] - EPS_V), (p.vmax[pq] + EPS_V) - vm1[pq])
    return 1 if ok_b and np.all(np.abs(m) > band) else 2


class Engine:
    def __init__(self):
        self.manifest = json.loads((BUNDLE / "manifest.json").read_text())
        self.layout = {int(k): v for k, v in json.loads((BUNDLE / "layout_case118.json").read_text()).items()}
        self.cache: dict[str, list[Scenario]] = {}

    def scenarios(self, order: str) -> list[Scenario]:
        if order not in self.cache:
            self.cache[order] = self._run(order)
        return self.cache[order]

    def _run(self, order: str) -> list[Scenario]:
        root = BUNDLE / order
        Lg = load_genco("pf", GRID, "base", batch_size=64, data_root=root, split_dir=root / GRID / "splits_all")
        raw = {t: pd.read_parquet(root / GRID / "raw" / f"{t}_data.parquet") for t in ["bus", "branch", "gen"]}
        base = dict(np.load(root / "base_cases.npz"))
        loads_of = self.manifest[order]["load_scenarios"]
        out, sid = [], 0
        for batch in Lg.test_loader():
            with torch.no_grad(), trace_layers(Lg.task.model) as tr:
                Lg.task.model(batch)
            states = [s.double().numpy() for s in tr["state"]]
            Lg.normalizer.inverse_transform(batch)
            RemovePFMask()(batch)
            ptr = batch["bus"].ptr.numpy()
            for g, p in enumerate(acpf.extract_problems(batch)):
                sl = slice(ptr[g], ptr[g + 1])
                st = [s[sl] for s in states]
                raw_V = [s[:, 0] * np.exp(1j * s[:, 1]) for s in st]
                phys_V = [p.V_from(s[:, 0], s[:, 1]) for s in st]
                Vg = phys_V[-1]
                V1, dx, _ = acpf.newton_step(p, Vg)
                b = base.get(str(loads_of[sid]))
                Vb = p.V_from(b[0], b[1]) if b is not None else p.flat_start()
                Vb1, _, _ = acpf.newton_step(p, Vb)
                V = {"true": p.V_true, "raw": raw_V[-1], "step1": V1, "base1": Vb1}
                loads = {k: acpf.branch_loading(p, acpf.branch_flows(p, v)) for k, v in V.items()}
                vms = {k: np.abs(v) for k, v in V.items()}
                step_l2 = float(np.linalg.norm(dx))
                tier = _tier(p, loads["step1"], vms["step1"], step_l2)
                nr_g, nr_f, nr_b = acpf.newton_pf(p, Vg), acpf.newton_pf(p, p.flat_start()), acpf.newton_pf(p, Vb)
                lu = 1 if tier == 1 else 1 + acpf.newton_pf(p, V1)["iterations"]
                mis = np.abs(Vg * np.conj(p.Ybus @ Vg) - p.Sbus)
                mis[p.ref] = 0.0
                rb = raw["branch"][raw["branch"].scenario == sid]
                rg = raw["gen"][raw["gen"].scenario == sid]
                outs = _outages(rb, rg)
                sc = Scenario(
                    idx=sid, label="", outages=outs, p=p,
                    layer_raw=[float(np.abs(acpf.mismatch(p, v)).max()) for v in raw_V],
                    layer_phys=[float(np.abs(acpf.mismatch(p, v)).max()) for v in phys_V],
                    hist_genco=nr_g["history"], hist_flat=nr_f["history"], hist_base=nr_b["history"],
                    load=loads, vm=vms, bus_mis=mis, step_l2=step_l2, tier=tier, lu_policy=lu,
                )
                sc.violations = _violations(p, loads, vms)
                n_true = int((sc.violations["true"] > 0).sum()) if len(sc.violations) else 0
                sc.label = (f"#{sid:02d} · out: {', '.join(outs) or 'none'} · "
                            f"GENCO mismatch {sc.layer_raw[-1]:.2f} MW · tier {tier}")
                out.append(sc)
                sid += 1
        return out

    def batch_summary(self, order: str) -> dict:
        S = self.scenarios(order)
        def recall(method):
            tp = fn = fp = 0
            for s in S:
                p = s.p
                tb = s.load["true"] > TAU
                pq = p.pq
                lo, hi = p.vmin[pq] - EPS_V, p.vmax[pq] + EPS_V
                tv = (s.vm["true"][pq] < lo) | (s.vm["true"][pq] > hi)
                if method == "policy":
                    lb = s.load["step1"] if s.tier == 1 else s.load["true"]
                    vm = s.vm["step1"] if s.tier == 1 else s.vm["true"]
                else:
                    lb, vm = s.load[method], s.vm[method]
                pb = lb > TAU
                pv = (vm[pq] < lo) | (vm[pq] > hi)
                tp += (pb & tb).sum() + (pv & tv).sum()
                fn += (~pb & tb).sum() + (~pv & tv).sum()
                fp += (pb & ~tb).sum() + (pv & ~tv).sum()
            return tp / max(tp + fn, 1), int(fn), int(fp)
        lu = lambda h: np.mean([len(x) - 1 for x in h])
        rows = [
            ("GENCO alone", *recall("raw"), 0.0),
            ("GENCO + 1 Newton step", *recall("step1"), 1.0),
            ("Base case + 1 Newton step", *recall("base1"), 1.0),
            ("Certified tiered policy", *recall("policy"), float(np.mean([s.lu_policy for s in S]))),
            ("Newton, GENCO warm start", 1.0, 0, 0, lu([s.hist_genco for s in S])),
            ("Newton, base-case start", 1.0, 0, 0, lu([s.hist_base for s in S])),
            ("Newton, flat start", 1.0, 0, 0, lu([s.hist_flat for s in S])),
        ]
        return {"rows": rows, "n": len(S), "tier1": float(np.mean([s.tier == 1 for s in S]))}

"""Computation behind the Space: run GENCO live on bundled scenarios and certify it.

Everything here reuses the project package `genco_certified` (vendored next to this
file in the Space): graphkit's own data pipeline + the released GENCO checkpoint, the
float64 Newton-Raphson on the exact network GENCO sees, and the sequential verification
policy from the experiments. The calibrated constants are read from bundle/calibration.json
(copied from results/exp02/big118_policies.csv; calibrated on data/calib/n1, IEEE 118 N-1).
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
BUNDLE = HERE / "bundle"
os.environ.setdefault("GENCO_CERTIFIED_ROOT", str(HERE))

from genco_certified import acpf  # noqa: E402
from genco_certified.loading import load_genco  # noqa: E402
from genco_certified.tracing import trace_layers  # noqa: E402
from gridfm_graphkit.datasets.masking import RemovePFMask  # noqa: E402

GRID = "case118_ieee"
CAL = json.loads((BUNDLE / "calibration.json").read_text())
TAU = float(CAL["tau"])      # branch overload threshold on |S|/rate
EPS_V = float(CAL["eps_v"])  # PQ-bus voltage tolerance (p.u.)
MAX_STEPS = int(CAL["max_steps"])   # sequential policy: verify after Newton step 1..MAX_STEPS, then escalate


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
    step_l2: float            # ||dx|| of the first Newton step from GENCO
    tier: int                 # 1 = accepted after the first step, 2 = accepted after step 2..MAX_STEPS, 0 = escalated to NR
    lu_policy: int            # factorizations charged to the sequential policy (GENCO prior)
    accepted_step: int = 0    # Newton step at which the policy accepted (0 = escalated)
    lu_policy_base: int = 0   # same policy started from the base case
    accepted_step_base: int = 0
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


def _margin(p, load, vm):
    """Smallest distance to a violation boundary over all branches and PQ buses (as in analyze_exp02.Screen.mmin)."""
    pq = p.pq
    m_v = np.minimum(vm[pq] - (p.vmin[pq] - EPS_V), (p.vmax[pq] + EPS_V) - vm[pq])
    return float(min(np.abs(TAU - load).min(), np.abs(m_v).min()))


def _sequential(p, V0, c):
    """Sequential verification: after Newton step k = 1..MAX_STEPS accept if margin_k > c * ||dx_k||^2.

    Returns (accepted step or 0, iterates [V_1..V_K], step norms, Newton-Raphson iterations from V0 to 1e-6 MW).
    Cost follows scripts/analyze_exp02.py::lu_per_scenario: accepted scenarios pay the steps taken, escalated
    ones pay their own NR iteration count but never less than MAX_STEPS.
    """
    V, Vs, norms, acc = V0, [], [], 0
    for k in range(1, MAX_STEPS + 1):
        V, dx, _ = acpf.newton_step(p, V)
        Vs.append(V)
        norms.append(float(np.linalg.norm(dx)))
        load, vm = acpf.branch_loading(p, acpf.branch_flows(p, V)), np.abs(V)
        if acc == 0 and _margin(p, load, vm) > c * norms[-1] ** 2:
            acc = k
            break
    nri = acpf.newton_pf(p, V0)["iterations"]
    return acc, Vs, norms, (acc if acc else max(nri, MAX_STEPS))


class Engine:
    def __init__(self, rule: str | None = None):
        self.rule = rule or CAL["default_rule"]
        self.c_genco = float(CAL["c"]["genco"][self.rule])
        self.c_base = float(CAL["c"]["base_case"][self.rule])
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
                b = base.get(str(loads_of[sid]))
                Vb = p.V_from(b[0], b[1]) if b is not None else p.flat_start()
                acc, Vs, norms, lu = _sequential(p, Vg, self.c_genco)
                acc_b, Vbs, _, lu_b = _sequential(p, Vb, self.c_base)
                V1, Vb1 = Vs[0], Vbs[0]
                V = {"true": p.V_true, "raw": raw_V[-1], "step1": V1, "base1": Vb1,
                     # decisions the policy acts on: the accepted iterate, or the converged solution if it escalated
                     "policy": Vs[acc - 1] if acc else p.V_true,
                     "policy_base": Vbs[acc_b - 1] if acc_b else p.V_true}
                loads = {k: acpf.branch_loading(p, acpf.branch_flows(p, v)) for k, v in V.items()}
                vms = {k: np.abs(v) for k, v in V.items()}
                step_l2 = norms[0]
                tier = 0 if acc == 0 else (1 if acc == 1 else 2)
                nr_g, nr_f, nr_b = acpf.newton_pf(p, Vg), acpf.newton_pf(p, p.flat_start()), acpf.newton_pf(p, Vb)
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
                    accepted_step=acc, lu_policy_base=lu_b, accepted_step_base=acc_b,
                )
                sc.violations = _violations(p, loads, vms)
                n_true = int((sc.violations["true"] > 0).sum()) if len(sc.violations) else 0
                sc.label = (f"#{sid:02d} · out: {', '.join(outs) or 'none'} · "
                            f"GENCO mismatch {sc.layer_raw[-1]:.2f} MW · "
                            f"{'accepted after step ' + str(acc) if acc else 'escalated'}")
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
            ("Sequential policy, GENCO prior", *recall("policy"), float(np.mean([s.lu_policy for s in S]))),
            ("Sequential policy, base-case prior", *recall("policy_base"), float(np.mean([s.lu_policy_base for s in S]))),
            ("Newton, GENCO warm start", 1.0, 0, 0, lu([s.hist_genco for s in S])),
            ("Newton, base-case start", 1.0, 0, 0, lu([s.hist_base for s in S])),
            ("Newton, flat start", 1.0, 0, 0, lu([s.hist_flat for s in S])),
        ]
        return {"rows": rows, "n": len(S), "tier1": float(np.mean([s.tier == 1 for s in S])),
                "accepted": float(np.mean([s.accepted_step > 0 for s in S])), "rule": self.rule,
                "c_genco": self.c_genco, "c_base": self.c_base}

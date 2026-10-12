"""Pre-contingency ("base-case") voltage store for GENCO-Delta.

At contingency time the operator already holds the solved pre-contingency state. For every
load scenario of a datakit data root this module reconstructs that state
(`acpf.base_case_voltages`: intact topology = the majority status of each element across the
load scenario's topology variants, then a converged Newton-Raphson) and keeps

    Vm  [p.u.]      voltage magnitude, bus order 0..n-1
    Va  [radians]   voltage angle (REF bus at its datakit reference angle)

for every *scenario id* of that load scenario. Scenario ids are the keys graphkit stores on
each sample (`data["scenario_id"]`), so a per-sample transform can look the base case up.

Memory layout: one dense row per load scenario, plus an int map scenario_id -> row, so a
2000-bus grid with a few hundred load scenarios costs a few MB.

The raw parquet tables are read one Hive partition at a time (a load scenario's variants
can straddle a partition boundary, so each load scenario is solved after its last partition
has been read).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from genco_certified import acpf

STORE_VERSION = 1
BUS_COLS = ["scenario", "load_scenario_idx", "bus", "Pd", "Qd", "Qg", "Vm", "Va", "PQ", "PV", "REF",
            "min_vm_pu", "max_vm_pu", "GS", "BS"]
BRANCH_COLS = ["scenario", "load_scenario_idx", "idx", "from_bus", "to_bus", "br_status",
               "Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i", "rate_a"]
GEN_COLS = ["scenario", "load_scenario_idx", "idx", "bus", "in_service", "p_mw", "q_mvar"]
TABLE_COLS = {"bus": BUS_COLS, "branch": BRANCH_COLS, "gen": GEN_COLS}


@dataclass
class BaseCaseStore:
    """scenario_id -> (Vm, Va) of the pre-contingency state of the scenario's load scenario."""

    row_of: torch.Tensor          # [max_sid + 1] int64, -1 where the scenario id is not covered
    load_of: torch.Tensor         # [max_sid + 1] int64 load_scenario_idx, -1 where not covered
    load_ids: torch.Tensor        # [L] int64 load_scenario_idx of each row
    Vm: torch.Tensor              # [L, n_bus] float32, p.u.
    Va: torch.Tensor              # [L, n_bus] float32, radians
    converged: torch.Tensor       # [L] bool
    iterations: torch.Tensor      # [L] int32
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ queries
    @property
    def n_bus(self) -> int:
        return int(self.Vm.shape[1])

    @property
    def n_load_scenarios(self) -> int:
        return int(self.Vm.shape[0])

    def covers(self, scenario_ids) -> bool:
        return len(self.missing(scenario_ids)) == 0

    def missing(self, scenario_ids) -> list[int]:
        ids = np.asarray(list(scenario_ids), dtype=np.int64)
        ok = (ids >= 0) & (ids < len(self.row_of))
        ok[ok] = self.row_of.numpy()[ids[ok]] >= 0
        return ids[~ok].tolist()

    def row(self, scenario_id: int) -> int:
        sid = int(scenario_id)
        r = int(self.row_of[sid]) if 0 <= sid < len(self.row_of) else -1
        if r < 0:
            raise KeyError(f"scenario_id {sid} is not in the base-case store (covers {self.meta.get('n_scenarios', '?')} "
                           f"scenarios of {self.n_load_scenarios} load scenarios); rebuild it with scripts/build_basecase_store.py")
        return r

    def lookup(self, scenario_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(Vm [p.u.], Va [rad]) float32 tensors of length n_bus. Views into the store: do not modify."""
        r = self.row(scenario_id)
        return self.Vm[r], self.Va[r]

    def load_scenario(self, scenario_id: int) -> int:
        self.row(scenario_id)
        return int(self.load_of[int(scenario_id)])

    def load_scenarios_of(self, scenario_ids) -> np.ndarray:
        return np.array([self.load_scenario(s) for s in scenario_ids], dtype=np.int64)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": STORE_VERSION, "row_of": self.row_of, "load_of": self.load_of,
                    "load_ids": self.load_ids, "Vm": self.Vm, "Va": self.Va, "converged": self.converged,
                    "iterations": self.iterations, "meta_json": json.dumps(self.meta, sort_keys=True)}, path)
        return path

    @classmethod
    def load(cls, path) -> "BaseCaseStore":
        d = torch.load(str(path), map_location="cpu", weights_only=True)
        if d.get("version") != STORE_VERSION:
            raise ValueError(f"{path}: unsupported base-case store version {d.get('version')}")
        return cls(d["row_of"], d["load_of"], d["load_ids"], d["Vm"], d["Va"], d["converged"], d["iterations"],
                   json.loads(d["meta_json"]))


_CACHE: dict[tuple[str, float], BaseCaseStore] = {}


def load_store_cached(path) -> BaseCaseStore:
    """Load a store once per process (dataset transforms are re-created per datamodule)."""
    p = Path(path).expanduser().resolve()
    key = (str(p), p.stat().st_mtime)
    if key not in _CACHE:
        _CACHE[key] = BaseCaseStore.load(p)
    return _CACHE[key]


# ---------------------------------------------------------------------------- base-case solve
START_NAMES = {0: "flat", 1: "variant", 2: "dc"}


@contextmanager
def _capture_problem():
    """Capture the PFProblem that `acpf.base_case_voltages` builds (keeps one source of truth for the
    majority-status topology logic) so a failed flat-start solve can be retried from another start."""
    box, orig = {}, acpf.newton_pf

    def spy(p, V0, *a, **k):
        box.setdefault("p", p)
        return orig(p, V0, *a, **k)

    acpf.newton_pf = spy
    try:
        yield box
    finally:
        acpf.newton_pf = orig


def solve_base_case(bus_ls, branch_ls, gen_ls, base_mva: float = 100.0, max_variant_starts: int = 3) -> dict:
    """`acpf.base_case_voltages` (flat-start Newton), retried when it fails to converge.

    On GOC 2000 about 6% of load scenarios diverge from a flat start although the intact network is
    solvable (each variant is). The operator holds the solved pre-contingency state, so the retry starts
    Newton from the stored solution of a few topology variants (a near-solution), then from a DC start.
    Returns {"V", "converged", "iterations", "variants", "start"} with start in START_NAMES.
    """
    with _capture_problem() as box:
        bc = acpf.base_case_voltages(bus_ls, branch_ls, gen_ls, base_mva)
    if bc["converged"]:
        return {**bc, "start": 0}
    p = box["p"]
    best = {**bc, "start": 0}
    scen = np.sort(bus_ls.scenario.unique())
    pick = scen[np.unique(np.linspace(0, len(scen) - 1, max_variant_starts).round().astype(int))]
    starts = []
    for sc in pick:
        b = bus_ls[bus_ls.scenario == sc].sort_values("bus")
        starts.append((1, p.V_from(b.Vm.to_numpy(), np.deg2rad(b.Va.to_numpy()))))
    try:
        starts.append((2, p.dc_start()))
    except Exception:  # singular DC system
        pass
    for code, V0 in starts:
        r = acpf.newton_pf(p, V0)
        if r["converged"]:
            return {"V": r["V"], "converged": True, "iterations": r["iterations"], "variants": bc["variants"], "start": code}
    return best


# ---------------------------------------------------------------------------- raw data access
def _table_parts(raw_dir: Path, table: str) -> list[Path]:
    """Hive partition directories of `<table>_data.parquet` (sorted by partition id), or the flat file."""
    p = raw_dir / f"{table}_data.parquet"
    if p.is_dir():
        parts = sorted((d for d in p.iterdir() if d.is_dir() and d.name.startswith("scenario_partition=")),
                       key=lambda d: int(d.name.split("=")[1]))
        if parts:
            return parts
    if not p.exists():
        raise FileNotFoundError(p)
    return [p]


def _read(path: Path, table: str) -> pd.DataFrame:
    cols = TABLE_COLS[table]
    df = pd.read_parquet(path, columns=cols)
    df = df.assign(scenario=df.scenario.astype(np.int64), load_scenario_idx=df.load_scenario_idx.astype(np.int64))
    return df


def build_store(raw_dir, scenario_ids=None, base_mva: float = 100.0, log=print) -> BaseCaseStore:
    """Build the base-case store of a datakit data root (`<root>/<grid>/raw`).

    scenario_ids: restrict to the load scenarios that contain these scenario ids (default: all).
    The store always covers *all* variants (scenario ids) of every load scenario it contains.
    """
    raw_dir = Path(raw_dir)
    t0 = time.time()
    parts = {t: _table_parts(raw_dir, t) for t in TABLE_COLS}
    nparts = len(parts["bus"])
    assert all(len(v) == nparts for v in parts.values()), "tables have different partition counts"

    # pass 1: scenario -> load scenario, and the last partition each load scenario appears in
    sc_rows, last_part = [], {}
    for i, pb in enumerate(parts["bus"]):
        m = pd.read_parquet(pb, columns=["scenario", "load_scenario_idx"]).drop_duplicates("scenario")
        m = m.astype(np.int64)
        sc_rows.append(m)
        for ls in m.load_scenario_idx.unique():
            last_part[int(ls)] = i
    sc_map = pd.concat(sc_rows, ignore_index=True).drop_duplicates("scenario").set_index("scenario").load_scenario_idx
    if scenario_ids is not None:
        want = set(int(s) for s in sc_map.reindex(list(scenario_ids)).dropna().astype(int).unique())
    else:
        want = set(last_part)
    log(f"  {raw_dir}: {len(sc_map)} scenarios, {len(last_part)} load scenarios, {nparts} partition(s); solving {len(want)}")

    # pass 2: partition by partition, solving a load scenario once all its rows are in memory
    pending: dict[str, pd.DataFrame] = {t: pd.DataFrame(columns=TABLE_COLS[t]) for t in TABLE_COLS}
    solved: dict[int, dict] = {}
    for i in range(nparts):
        new = {t: _read(parts[t][i], t) for t in TABLE_COLS}
        new = {t: d[d.load_scenario_idx.isin(want)] for t, d in new.items()}
        for t in TABLE_COLS:
            frames = [f for f in (pending[t], new[t]) if len(f)]
            pending[t] = pd.concat(frames, ignore_index=True) if frames else new[t]
        ready = {ls for ls in want if last_part.get(ls) == i}
        if not ready:
            continue
        sel = {t: pending[t][pending[t].load_scenario_idx.isin(ready)] for t in TABLE_COLS}
        pending = {t: pending[t][~pending[t].load_scenario_idx.isin(ready)] for t in TABLE_COLS}
        groups = {t: {k: g for k, g in sel[t].groupby("load_scenario_idx")} for t in TABLE_COLS}
        for ls in sorted(ready):
            bc = solve_base_case(groups["bus"][ls], groups["branch"][ls], groups["gen"][ls], base_mva)
            solved[ls] = {"V": bc["V"], "converged": bool(bc["converged"]), "iterations": int(bc["iterations"]),
                          "variants": int(bc["variants"]), "start": int(bc["start"])}
        log(f"    partition {i + 1}/{nparts}: {len(solved)}/{len(want)} load scenarios solved ({time.time() - t0:.0f}s)")
    assert not pending["bus"].shape[0], "unsolved load scenarios remain (scenario/partition bookkeeping bug)"

    load_ids = np.array(sorted(solved), dtype=np.int64)
    n_bus = len(next(iter(solved.values()))["V"])
    Vm = np.empty((len(load_ids), n_bus), dtype=np.float64)
    Va = np.empty_like(Vm)
    for r, ls in enumerate(load_ids):
        s = solved[int(ls)]
        if s["converged"]:
            Vm[r], Va[r] = np.abs(s["V"]), np.angle(s["V"])
        else:  # fall back to the flat start the control arm would otherwise have to guess
            Vm[r], Va[r] = 1.0, 0.0
    sids = sc_map[sc_map.isin(load_ids)]
    max_sid = int(sc_map.index.max())
    row_of = torch.full((max_sid + 1,), -1, dtype=torch.long)
    load_of = torch.full((max_sid + 1,), -1, dtype=torch.long)
    row_of[torch.as_tensor(sids.index.to_numpy(dtype=np.int64))] = torch.as_tensor(
        np.searchsorted(load_ids, sids.to_numpy(dtype=np.int64)))
    load_of[torch.as_tensor(sids.index.to_numpy(dtype=np.int64))] = torch.as_tensor(sids.to_numpy(dtype=np.int64))
    conv = np.array([solved[int(ls)]["converged"] for ls in load_ids])
    meta = {
        "raw_dir": str(raw_dir), "n_bus": int(n_bus), "base_mva": base_mva, "n_scenarios": int(len(sids)),
        "n_load_scenarios": int(len(load_ids)), "n_nonconverged": int((~conv).sum()),
        "n_retried_from_variant_or_dc": int(sum(solved[int(ls)]["start"] != 0 for ls in load_ids)),
        "solver": "acpf.base_case_voltages (flat start) -> retry from variant solutions -> DC start",
        "variants_per_load_min": int(min(s["variants"] for s in solved.values())),
        "variants_per_load_max": int(max(s["variants"] for s in solved.values())),
        "build_seconds": round(time.time() - t0, 1), "store_version": STORE_VERSION,
        "units": {"Vm": "p.u.", "Va": "radians"},
    }
    store = BaseCaseStore(row_of, load_of, torch.as_tensor(load_ids), torch.as_tensor(Vm, dtype=torch.float32),
                          torch.as_tensor(Va, dtype=torch.float32), torch.as_tensor(conv),
                          torch.as_tensor([solved[int(ls)]["iterations"] for ls in load_ids], dtype=torch.int32), meta)
    log(f"  built: {meta['n_scenarios']} scenarios / {meta['n_load_scenarios']} load scenarios, "
        f"{meta['n_nonconverged']} non-converged ({meta['n_retried_from_variant_or_dc']} needed a retry), {meta['build_seconds']}s")
    return store


def split_ids(root: Path) -> dict[str, list[int]]:
    """{<split dir>/<name>: ids} for every `splits*/{train,val,test}.pt` under a grid folder."""
    out = {}
    for d in sorted(root.glob("splits*")):
        for name in ("train", "val", "test"):
            f = d / f"{name}.pt"
            if f.is_file():
                try:
                    out[f"{d.name}/{name}"] = [int(i) for i in torch.load(f, weights_only=True).tolist()]
                except Exception:  # a split being written by another process
                    continue
    return out

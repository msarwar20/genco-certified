"""Verify the result artifacts of this repository.

Checks
  (a) SHA256SUMS: every listed file exists and has the recorded digest; files that make_checksums.py
      would list but SHA256SUMS does not are reported (an error with --strict).
  (b) every results/**/*.json parses, and every number in it is finite. Fields that are NaN on purpose
      (a standard deviation over one seed, a GPU timing that was not measured) are listed in NAN_OK.
  (c) with --checkpoints: the released checkpoints in checkpoints/genco-pf-datakit/ match the digests
      recorded in results/genco_delta/runs/*.json (init_weights_sha256, data.normalizer_stats_sha256).
      Runs whose checkpoint is not on disk are skipped and counted.

Only the standard library is needed for (a) and (b), so this runs in CI without any download.

Usage:
    python scripts/verify_artifacts.py [--strict] [--checkpoints] [--also PATH ...]
Exit status is 0 when nothing failed.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_checksums import ROOT, artifact_files, sha256  # noqa: E402

# file -> regex (full match on the JSON path) of fields that may legitimately be NaN:
# a standard deviation over a single seed, a GPU timing that was not measured, a precision with no predicted positives
NAN_OK = {
    "results/exp04/probe_v2_summary_loo_seed0.json": r".*_sd",
    "results/exp07/cost_breakeven.json": r".*/model_(solve|total)_ms",
    "results/exp07/cost_breakeven_stress_n5_base.json": r".*/model_(solve|total)_ms",
    "results/exp07/cost_breakeven_goc2000.json": r".*/model_(solve|total)_ms",
    "results/genco_delta/residual_smoke/eval/smoke_residual_init_s0__paper.json": r"/g0_el_precision/(value|ci95\[\d\])",
}


def check_sums(strict: bool = False) -> list[str]:
    errs = []
    sums = ROOT / "SHA256SUMS"
    if not sums.exists():
        return ["SHA256SUMS is missing (run scripts/make_checksums.py)"]
    listed = {}
    for ln, line in enumerate(sums.read_text().splitlines(), 1):
        m = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", line)
        if not m:
            errs.append(f"SHA256SUMS:{ln}: malformed line")
            continue
        listed[m.group(2)] = m.group(1)
    for name, digest in listed.items():
        p = ROOT / name
        if not p.is_file():
            errs.append(f"missing: {name}")
        elif sha256(p) != digest:
            errs.append(f"digest mismatch: {name}")
    extra = sorted(set(artifact_files()) - set(listed))
    if extra:
        msg = f"{len(extra)} artifact file(s) not in SHA256SUMS, e.g. {extra[0]}"
        if strict:
            errs.append(msg)
        else:
            print("warning:", msg)
    print(f"[sums] {len(listed)} listed, {len(errs)} problem(s)")
    return errs


def _nonfinite(obj, key="", path=""):
    if isinstance(obj, float) and not math.isfinite(obj):
        yield path, key
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _nonfinite(v, str(k), f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _nonfinite(v, key, f"{path}[{i}]")


def check_json(also=()) -> list[str]:
    errs, n, tolerated = [], 0, 0
    for name in artifact_files(also):
        if not name.endswith(".json"):
            continue
        n += 1
        try:
            obj = json.loads((ROOT / name).read_text())  # accepts NaN/Infinity literals; checked below
        except (ValueError, OSError) as e:
            errs.append(f"unparsable JSON: {name}: {e}")
            continue
        allow = re.compile(NAN_OK[name]) if name in NAN_OK else None
        for path, key in _nonfinite(obj):
            if allow and allow.fullmatch(path):
                tolerated += 1
            else:
                errs.append(f"non-finite number: {name}{path}")
    print(f"[json] {n} files parsed, {tolerated} tolerated NaN field(s), {len(errs)} problem(s)")
    return errs


def check_checkpoints() -> list[str]:
    errs, ok, skipped = [], 0, 0
    cache: dict[Path, str] = {}
    for run in sorted((ROOT / "results/genco_delta/runs").glob("*.json")):
        d = json.loads(run.read_text())
        a = d.get("args", {})
        base = ROOT / "checkpoints/genco-pf-datakit" / a.get("grid", "?") / a.get("size", "?") / f"seed{a.get('seed')}"
        pairs = [(base / "best_model_state_dict.pt", d.get("init_weights_sha256")),
                 (base / "normalizer_stats.pt", d.get("data", {}).get("normalizer_stats_sha256"))]
        for path, want in pairs:
            if want is None:
                continue
            if not path.is_file():
                skipped += 1
                continue
            cache.setdefault(path, sha256(path))
            if cache[path] == want:
                ok += 1
            else:
                errs.append(f"{run.name}: {path.name} digest differs from the recorded one")
    print(f"[checkpoints] {ok} digest(s) match, {skipped} skipped (file not on disk), {len(errs)} problem(s)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strict", action="store_true", help="fail when result files are missing from SHA256SUMS")
    ap.add_argument("--also", nargs="*", default=[], metavar="PATH", help="also parse these untracked JSON files")
    ap.add_argument("--checkpoints", action="store_true", help="also verify released checkpoint digests (needs checkpoints/)")
    args = ap.parse_args()
    errs = check_sums(args.strict) + check_json(args.also)
    if args.checkpoints:
        errs += check_checkpoints()
    for e in errs:
        print("FAIL:", e)
    print("OK" if not errs else f"{len(errs)} failure(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

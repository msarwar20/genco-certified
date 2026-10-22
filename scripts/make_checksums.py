"""Write SHA256SUMS for the result artifacts that belong in the repository.

Covers results/**/*.{json,csv,md,log} that git tracks (`git ls-files`). Format is the one `sha256sum -c` reads:
    <hex digest>  <path relative to the repo root>

Usage:
    python scripts/make_checksums.py                       # writes SHA256SUMS
    python scripts/make_checksums.py --also results/x.json # include a file that is not yet tracked
    python scripts/make_checksums.py --check               # check an existing SHA256SUMS (checksums only)

Re-run after anything under results/ changes, and after `git add` of new result files; logs of running jobs
change while the job runs.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = (".json", ".csv", ".md", ".log")
OUT = ROOT / "SHA256SUMS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_files(also=()) -> list[str]:
    """Repo-relative paths under results/ with a checked extension that git tracks (plus any in `also`)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "results"],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    names = sorted({n for n in [*out.split("\0"), *also] if n.endswith(EXT)})
    return [n for n in names if (ROOT / n).is_file()]  # a tracked file deleted from the work tree is skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify an existing SHA256SUMS instead of writing it")
    ap.add_argument("--also", nargs="*", default=[], metavar="PATH", help="extra untracked files to include")
    args = ap.parse_args()
    if args.check:
        from verify_artifacts import check_sums  # noqa: PLC0415
        return 1 if check_sums() else 0
    files = artifact_files(args.also)
    OUT.write_text("".join(f"{sha256(ROOT / n)}  {n}\n" for n in files))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())

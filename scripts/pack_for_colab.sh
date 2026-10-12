#!/usr/bin/env bash
# Pack our code and the tiny split files into one tarball for upload to Google Colab.
#
# Our GitHub repository is private, so Colab cannot clone it. The notebooks in notebooks/ ask for
# this tarball (file picker or Google Drive) and unpack it to /content/genco.
#
# Contents: src/  scripts/  configs/  tests/  pyproject.toml  BUNDLE_INFO.txt
#           and the split files (train/val/test.pt, a few KB each) of the grids in $GRIDS:
#           data/pf/<grid>/splits_*   and   data/*/*/<grid>/splits_all
# Not included: raw data, processed data, checkpoints, results, .venv, external/, caches.
#
# Usage:   ./scripts/pack_for_colab.sh [output.tar.gz]        (default dist/genco_colab_bundle.tar.gz)
# Options (environment variables):
#   GRIDS="case118_ieee case500_goc case2000_goc"   grids whose split files are packed
#   WITH_TESTS=0                                    leave tests/ out
#
# dist/ is not git-ignored at the time of writing: do not commit the tarball.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-dist/genco_colab_bundle.tar.gz}"
GRIDS="${GRIDS:-case118_ieee case500_goc case2000_goc}"
WITH_TESTS="${WITH_TESTS:-1}"

for need in src/genco_certified/__init__.py scripts/finetune.py pyproject.toml; do
  [ -e "$need" ] || { echo "error: $need not found; run from a checkout of the repository" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

copy() {  # copy a file or directory (relative path) into the staging area, keeping its path
  mkdir -p "$STAGE/$(dirname "$1")"
  cp -R "$1" "$STAGE/$1"
}

copy src
copy scripts
[ -d configs ] && copy configs
[ "$WITH_TESTS" = "1" ] && [ -d tests ] && copy tests
copy pyproject.toml

n_splits=0
for grid in $GRIDS; do
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    copy "$d"; n_splits=$((n_splits + 1))
  done < <(
    { [ -d "data/pf/$grid" ] && find "data/pf/$grid" -maxdepth 1 -type d -name 'splits_*'
      [ -d data ] && find data -mindepth 4 -maxdepth 4 -type d -path "data/*/*/$grid/splits_all"
    } | sort
  )
done

# drop caches, editor and macOS leftovers
find "$STAGE" -type d \( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache \) -prune -exec rm -rf {} +
find "$STAGE" -type f \( -name '*.pyc' -o -name '.DS_Store' -o -name '._*' \) -delete

# refuse to ship anything big by accident (raw data, checkpoints, ...)
big="$(find "$STAGE" -type f -size +2048k | head -5 || true)"
if [ -n "$big" ]; then
  echo "error: unexpectedly large files in the bundle (>2 MB):" >&2; echo "$big" >&2; exit 1
fi

{
  echo "genco_colab_bundle"
  echo "created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
    echo "git_rev: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty_files: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  fi
  echo "grids_with_splits: $GRIDS"
  echo "split_dirs: $n_splits"
  echo "files: $(find "$STAGE" -type f | wc -l | tr -d ' ')"
} > "$STAGE/BUNDLE_INFO.txt"

mkdir -p "$(dirname "$OUT")"
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
rm -f "$OUT_ABS"
TAR_OPTS=()
if tar --no-xattrs -cf /dev/null /dev/null 2>/dev/null; then TAR_OPTS+=(--no-xattrs); fi  # no ._* entries on macOS
# shellcheck disable=SC2046
COPYFILE_DISABLE=1 tar "${TAR_OPTS[@]+"${TAR_OPTS[@]}"}" -czf "$OUT_ABS" -C "$STAGE" $(cd "$STAGE" && ls)

# sanity checks on the finished archive
LIST="$(tar -tzf "$OUT_ABS")"
for need in src/genco_certified/graphkit_plugin.py src/genco_certified/loading.py src/genco_certified/acpf.py \
            scripts/finetune.py scripts/bench_runtime.py pyproject.toml BUNDLE_INFO.txt; do
  grep -qx "$need" <<<"$LIST" || { echo "error: $need missing from the bundle" >&2; exit 1; }
done
if grep -q '/\._\|^\._' <<<"$LIST"; then echo "error: AppleDouble (._*) entries in the bundle" >&2; exit 1; fi

echo "wrote $OUT_ABS"
echo "  size:   $(du -h "$OUT_ABS" | cut -f1) ($(wc -c < "$OUT_ABS" | tr -d ' ') bytes), $(grep -vc '/$' <<<"$LIST") files, $n_splits split folders"
if command -v shasum >/dev/null 2>&1; then echo "  sha256: $(shasum -a 256 "$OUT_ABS" | cut -d' ' -f1)"; \
elif command -v sha256sum >/dev/null 2>&1; then echo "  sha256: $(sha256sum "$OUT_ABS" | cut -d' ' -f1)"; fi
echo "  top level:"
grep -v '/$' <<<"$LIST" | cut -d/ -f1 | sort | uniq -c | sed 's/^/    /'
echo "Upload this file in the notebook's upload cell (or copy it to Google Drive)."
echo "Note: dist/ is not in .gitignore; keep the tarball out of commits."

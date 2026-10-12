#!/usr/bin/env bash
# Replace machine-specific absolute paths in tracked text files (logs, JSON) before committing.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_ABS="$(pwd)"
git ls-files -z | xargs -0 grep -lI -e "${ROOT_ABS}" -e "<tmp>" 2>/dev/null | while read -r f; do
  sed -i '' -e "s#${ROOT_ABS}#<repo>#g" -e "s#<tmp> \"']*#<tmp>#g" "$f"
done

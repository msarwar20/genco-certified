#!/usr/bin/env bash
# Reproducible environment for the GENCO challenge project.
# Clones the OpenGridFM toolkits at pinned commits and builds a Python 3.12 venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GRAPHKIT_COMMIT=8a8defbc9ab41caa65b07c75e8e1ee23636877cd   # gridfm-graphkit main, 2026-10-01
DATAKIT_COMMIT=d8a092dcfdcc05f58f12595e9f7a9fd165d3c2f0    # gridfm-datakit main, 2026-10-01
TORCH_SPEC="torch>=2.10,<2.13"                              # torch-scatter has macOS wheels up to 2.12

clone_pinned() {  # url dir commit
  if [ ! -d "$2/.git" ]; then
    git clone --filter=blob:none "$1" "$2"
  fi
  git -C "$2" fetch --quiet origin "$3" || true
  git -C "$2" checkout --quiet "$3"
}

mkdir -p external
clone_pinned https://github.com/gridfm/gridfm-graphkit external/gridfm-graphkit "$GRAPHKIT_COMMIT"
clone_pinned https://github.com/gridfm/gridfm-datakit  external/gridfm-datakit  "$DATAKIT_COMMIT"

UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
[ -d .venv ] || "$UV" venv --python 3.12 .venv
source .venv/bin/activate

"$UV" pip install "$TORCH_SPEC" torchvision torchaudio
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
case "$(uname -s)" in
  Darwin) PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER%.*}.0+cpu.html" ;;
  *)      PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER%.*}.0+$(python -c 'import torch; print("cu"+torch.version.cuda.replace(".","") if torch.version.cuda else "cpu")').html" ;;
esac
"$UV" pip install torch-scatter -f "$PYG_URL"

# datakit first (editable) so graphkit's pinned gridfm-datakit dependency resolves to our checkout
"$UV" pip install -e external/gridfm-datakit
"$UV" pip install -e external/gridfm-graphkit
# pandas 3 breaks graphkit's OPF DC baseline (drop(columns=..., axis=1) in tasks/opf_ac_dc_baseline.py)
"$UV" pip install huggingface_hub jupyterlab ipykernel matplotlib seaborn "pandas>=2.3,<3" pyarrow scikit-learn

python - <<'PY'
import torch, torch_geometric, gridfm_graphkit, gridfm_datakit
print("torch", torch.__version__, "| mps:", torch.backends.mps.is_available(), "| cuda:", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("graphkit", getattr(gridfm_graphkit, "__version__", "?"), "| datakit", getattr(gridfm_datakit, "__version__", "?"))
PY
echo "Env ready. Activate with: source .venv/bin/activate && export MLFLOW_ALLOW_FILE_STORE=true"

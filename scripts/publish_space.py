"""Publish space/ to a Hugging Face Space (outward-facing: run only with the owner's OK).

Usage:
    huggingface-cli login            # or export HF_TOKEN=...
    python scripts/publish_space.py --repo <hf-username>/genco-certified [--private]
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    # refresh vendored package and figures
    shutil.rmtree(SPACE / "genco_certified", ignore_errors=True)
    shutil.copytree(ROOT / "src" / "genco_certified", SPACE / "genco_certified",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (SPACE / "assets").mkdir(exist_ok=True)
    for f in (ROOT / "results" / "figures" / "post").glob("fig[ABCFH]_*.png"):  # the post figures the Findings tab shows
        shutil.copy(f, SPACE / "assets" / f.name)
    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=SPACE, repo_id=args.repo, repo_type="space",
                      ignore_patterns=["checkpoints/*", "results/*", "**/processed/*", "**/__pycache__/*", "*.pyc"])
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()

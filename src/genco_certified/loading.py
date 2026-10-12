"""Build a GENCO task + test dataloader in-process, mirroring `gridfm_graphkit evaluate`."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from gridfm_graphkit.cli import _load_config, _normalize_loaded_state_dict_keys
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import get_task
from huggingface_hub import hf_hub_download

import genco_certified.graphkit_plugin  # noqa: F401  registers PowerFlowSlackAware

# Project root (checkpoints/, data/, results/). Override with GENCO_CERTIFIED_ROOT, e.g. in the HF Space.
ROOT = Path(os.environ.get("GENCO_CERTIFIED_ROOT", Path(__file__).resolve().parents[2]))
WEIGHTS = {"pf": "best_model_state_dict.pt", "opf": "last.pt"}


@dataclass
class Loaded:
    task: torch.nn.Module  # Lightning task; `.model` is GNS_heterogeneous (GENCO)
    datamodule: LitGridHeteroDataModule
    normalizer: object
    config: object

    def test_loader(self):
        return self.datamodule.test_dataloader()[0]


def load_genco(
    task: str,
    grid: str,
    size: str = "base",
    seed: int = 0,
    split_dir: Path | None = None,
    batch_size: int = 32,
    overrides: dict | None = None,
    weights: str | Path | None = None,
    data_root: str | Path | None = None,
) -> Loaded:
    """Load a released GENCO checkpoint and the paper-test split for `grid`.

    `split_dir` defaults to the split built by scripts/fetch_data.py.
    `overrides` is deep-merged into the released YAML (e.g. {"task": {"task_name": ...}}).
    `weights` replaces the released state dict (e.g. a fine-tuned checkpoint); the
    released normalizer stats are always used.
    `data_root` holds `<grid>/raw/...` datakit output (default: data/<task>).
    """
    data_root = Path(data_root) if data_root else ROOT / "data" / task
    repo = f"gridfm/genco-{task}-datakit"
    local = ROOT / "checkpoints" / f"genco-{task}-datakit"
    sub = f"{grid}/{size}/seed{seed}"
    get = lambda name: hf_hub_download(repo, f"{sub}/{name}", local_dir=local)
    stats, cfg_src = get("normalizer_stats.pt"), get(f"{grid}_{size}_seed{seed}.yaml")
    weights = str(weights) if weights else get(WEIGHTS[task])

    cfg = yaml.safe_load(Path(cfg_src).read_text())
    _deep_update(cfg, overrides or {})
    cfg["data"]["split_by_load_scenario_idx"] = False
    cfg["data"]["split_from_existing_files"] = str(split_dir or data_root / grid / "splits_paper_test")
    cfg["data"]["workers"] = 0
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["accelerator"] = "cpu"
    cfg_path = ROOT / "results" / "configs" / f"{cfg['task']['task_name']}_{grid}_{size}_seed{seed}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    config = _load_config(argparse.Namespace(config=str(cfg_path), config_conversion="auto_inline"))
    dm = LitGridHeteroDataModule(config, str(data_root), normalizer_stats_path=stats)
    lit_task = get_task(config, dm.data_normalizers)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:  # Lightning checkpoint
        state = state["state_dict"]
    state = _normalize_loaded_state_dict_keys(state)
    lit_task.load_state_dict(state)
    lit_task.eval()
    dm.setup("test")
    return Loaded(task=lit_task, datamodule=dm, normalizer=dm.data_normalizers[0], config=config)


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v

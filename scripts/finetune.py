"""Fine-tune a released GENCO PF checkpoint, keeping the checkpoint's own normalizer.

`gridfm_graphkit finetune` refits the normalizer on the fine-tuning data, which would
silently rescale the inputs the pretrained weights expect. This runner reuses the
released `normalizer_stats.pt` and a train/val/test scenario split on disk.

Examples:
    # slack-aware GENCO (REF-bus Vm becomes a known input)
    python scripts/finetune.py --grid case118_ieee --size base --task_name PowerFlowSlackAware --epochs 10

    # GENCO-Delta: masked inputs are filled with the pre-contingency state (needs a base-case store,
    # built with scripts/build_basecase_store.py; default <data_root>/<grid>/basecase_store.pt)
    python scripts/finetune.py --task_name PowerFlowContingencyAware --delta_inputs vm_va_flows \\
        --seed 0 --seed_train 0 --epochs 12 --accelerator mps

    # its matched control: same data, epochs, seeds, original PowerFlow task
    python scripts/finetune.py --task_name PowerFlow --seed 0 --seed_train 0 --epochs 12 --accelerator mps

Outputs: checkpoints/finetuned/<tag>/{best.ckpt,best_model_state_dict.pt} (best = lowest validation loss; the
test split is never used) and a run record results/genco_delta/runs/<tag>.json (data/split/store hashes, best
epoch, validation curve, versions, wall time).
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import genco_certified.graphkit_plugin  # noqa: F401  registers PowerFlowSlackAware / PowerFlowContingencyAware
from genco_certified.loading import ROOT, load_genco

DELTA = "PowerFlowContingencyAware"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def raw_manifest(grid_dir: Path) -> dict:
    """sha256 over (relative path, size, content hash) of every raw parquet file of the data root."""
    files = sorted(p for p in (grid_dir / "raw").rglob("*.parquet") if p.is_file() and ".cache" not in p.parts)
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(grid_dir / 'raw')}|{p.stat().st_size}|{sha256_file(p)}\n".encode())
    return {"n_files": len(files), "bytes": sum(p.stat().st_size for p in files), "sha256": h.hexdigest()}


def split_info(split_dir: Path) -> dict:
    h, out = hashlib.sha256(), {}
    for name in ("train", "val", "test"):
        ids = torch.load(split_dir / f"{name}.pt", weights_only=True).to(torch.int64)
        h.update(name.encode() + ids.numpy().tobytes())
        out[f"n_{name}"] = int(len(ids))
    out["sha256"] = h.hexdigest()
    return out


def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


class ValHistory(Callback):
    """Per-epoch validation loss (the model-selection signal)."""

    def __init__(self):
        self.rows = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get("Validation loss")
        if v is not None:
            self.rows.append({"epoch": trainer.current_epoch + 1, "val_loss": float(v)})


def default_tag(a) -> str:
    tag = f"{a.task_name}"
    if a.task_name == DELTA:
        tag += f"-{a.delta_inputs}"
    tag += f"_{a.grid}_{a.size}_seed{a.seed}"
    if a.seed_train is not None:
        tag += f"_s{a.seed_train}"
    return tag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--seed", type=int, default=0, help="seed of the RELEASED checkpoint used as initialisation")
    ap.add_argument("--seed_train", type=int, default=None, help="training seed (batch order etc.); default: unseeded")
    ap.add_argument("--task_name", default="PowerFlowSlackAware",
                    help="PowerFlow (control) | PowerFlowSlackAware | PowerFlowContingencyAware (GENCO-Delta)")
    ap.add_argument("--data_root", default=None, help="folder holding <grid>/raw and the split folders (default data/pf)")
    ap.add_argument("--split", default="splits_paper_test", help="split folder under <data_root>/<grid>/ (train/val/test.pt)")
    ap.add_argument("--basecase_store", default=None,
                    help="PowerFlowContingencyAware only; default <data_root>/<grid>/basecase_store.pt")
    ap.add_argument("--delta_inputs", default="vm_va_flows", choices=["vm_va", "vm_va_flows"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accelerator", default="cpu", help="cpu | mps | gpu | auto")
    ap.add_argument("--limit_train_batches", type=float, default=1.0)
    ap.add_argument("--max_time", default=None, help="stop training after DD:HH:MM:SS (the best checkpoint so far is kept)")
    ap.add_argument("--tag", default=None, help="output name (default derived from task/grid/size/seeds)")
    ap.add_argument("--run_dir", default=None, help="where to write the run record (default results/genco_delta/runs)")
    ap.add_argument("--log_dir", default=None,
                    help="CSV logger root; graphkit also writes <log_dir>/stats/{normalizer_stats.pt,...} there, which overwrites "
                         "the files of earlier runs when shared (default results/finetune, as before)")
    args = ap.parse_args()

    if args.seed_train is not None:
        L.seed_everything(args.seed_train, workers=True)
    data_root = Path(args.data_root) if args.data_root else Path("data/pf")
    data_root = data_root if data_root.is_absolute() else ROOT / data_root
    grid_dir = data_root / args.grid
    split_dir = grid_dir / args.split

    data_over = {}
    store = None
    if args.task_name == DELTA:
        store = Path(args.basecase_store) if args.basecase_store else grid_dir / "basecase_store.pt"
        store = store if store.is_absolute() else ROOT / store
        if not store.is_file():
            raise SystemExit(f"{store} not found: run scripts/build_basecase_store.py --data_root {args.data_root or 'data/pf'} --grid {args.grid}")
        from genco_certified.basecase_store import BaseCaseStore
        s = BaseCaseStore.load(store)
        for name in ("train", "val", "test"):
            miss = s.missing(torch.load(split_dir / f"{name}.pt", weights_only=True).tolist())
            if miss:
                raise SystemExit(f"base-case store misses {len(miss)} {name} scenario ids (e.g. {miss[:5]}): rebuild it")
        data_over = {"basecase_store": str(store), "delta_inputs": args.delta_inputs}

    Lg = load_genco("pf", args.grid, args.size, args.seed, batch_size=args.batch_size, split_dir=split_dir,
                    data_root=data_root,
                    overrides={"task": {"task_name": args.task_name}, "data": data_over,
                               "optimizer": {"learning_rate": args.lr}, "training": {"epochs": args.epochs}})
    tag = args.tag or default_tag(args)
    out = ROOT / "checkpoints" / "finetuned" / tag
    ckpt = ModelCheckpoint(dirpath=out, filename="best", monitor="Validation loss", mode="min", save_top_k=1)
    hist = ValHistory()
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
        logger=CSVLogger(Path(args.log_dir) if args.log_dir else ROOT / "results" / "finetune", name=tag),
        callbacks=[ckpt, hist], limit_train_batches=args.limit_train_batches, max_time=args.max_time,
        log_every_n_steps=20, enable_progress_bar=False,
    )
    t0 = time.time()
    trainer.fit(Lg.task, datamodule=Lg.datamodule)
    wall = time.time() - t0
    print(f"fit done in {wall:.0f}s; best val {ckpt.best_model_score}")
    best_ckpt = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)
    torch.save(best_ckpt["state_dict"], out / "best_model_state_dict.pt")
    print("saved", out / "best_model_state_dict.pt")

    # run record
    stats = ROOT / "checkpoints" / "genco-pf-datakit" / args.grid / args.size / f"seed{args.seed}" / "normalizer_stats.pt"
    rows = hist.rows
    best = min(rows, key=lambda r: r["val_loss"]) if rows else None
    record = {
        "tag": tag, "args": vars(args), "git": git_rev(), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "lightning": L.__version__,
                     "platform": platform.platform()},
        "data": {"root": str(data_root), "grid": args.grid, "split_dir": str(split_dir), "split": split_info(split_dir),
                 "raw": raw_manifest(grid_dir), "basecase_store": str(store) if store else None,
                 "basecase_store_sha256": sha256_file(store) if store else None,
                 "normalizer_stats_sha256": sha256_file(stats) if stats.exists() else None},
        "init_weights_sha256": sha256_file(ROOT / "checkpoints" / "genco-pf-datakit" / args.grid / args.size /
                                           f"seed{args.seed}" / "best_model_state_dict.pt"),
        "training": {"epochs_requested": args.epochs, "epochs_run": int(trainer.current_epoch),
                     "global_steps": int(trainer.global_step), "limit_train_batches": args.limit_train_batches,
                     "lr": args.lr, "batch_size": args.batch_size, "accelerator": args.accelerator,
                     "seed_everything": args.seed_train, "wall_seconds": round(wall, 1)},
        "selection": {"metric": "Validation loss (val split only; test split untouched)",
                      "best_epoch": best["epoch"] if best else None, "best_val_loss": best["val_loss"] if best else None,
                      "checkpoint_epoch_field": int(best_ckpt.get("epoch", -1)) + 1,
                      "best_model_score": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
                      "val_curve": rows},
        "weights": str(out / "best_model_state_dict.pt"),
        "config": Lg.config.to_dict() if hasattr(Lg.config, "to_dict") else None,
    }
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "results" / "genco_delta" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{tag}.json").write_text(json.dumps(record, indent=1, default=str))
    print("run record", run_dir / f"{tag}.json")


if __name__ == "__main__":
    main()

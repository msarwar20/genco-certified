"""Fine-tune a released GENCO PF checkpoint, keeping the checkpoint's own normalizer.

`gridfm_graphkit finetune` refits the normalizer on the fine-tuning data, which would
silently rescale the inputs the pretrained weights expect. This runner reuses the
released `normalizer_stats.pt` and the paper's train/val/test scenario split.

Example (slack-aware GENCO; REF-bus Vm becomes a known input):
    python scripts/finetune.py --grid case118_ieee --size base --task_name PowerFlowSlackAware --epochs 10
"""

import argparse
import time
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import genco_certified.graphkit_plugin  # noqa: F401  registers PowerFlowSlackAware
from genco_certified.loading import ROOT, load_genco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="case118_ieee")
    ap.add_argument("--size", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task_name", default="PowerFlowSlackAware")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--limit_train_batches", type=float, default=1.0)
    args = ap.parse_args()

    Lg = load_genco("pf", args.grid, args.size, args.seed, batch_size=args.batch_size,
                    overrides={"task": {"task_name": args.task_name},
                               "optimizer": {"learning_rate": args.lr},
                               "training": {"epochs": args.epochs}})
    tag = f"{args.task_name}_{args.grid}_{args.size}_seed{args.seed}"
    out = ROOT / "checkpoints" / "finetuned" / tag
    ckpt = ModelCheckpoint(dirpath=out, filename="best", monitor="Validation loss", mode="min", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
        logger=CSVLogger(ROOT / "results" / "finetune", name=tag),
        callbacks=[ckpt], limit_train_batches=args.limit_train_batches,
        log_every_n_steps=20, enable_progress_bar=False,
    )
    t0 = time.time()
    trainer.fit(Lg.task, datamodule=Lg.datamodule)
    print(f"fit done in {time.time() - t0:.0f}s; best val {ckpt.best_model_score}")
    best = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
    torch.save(best, out / "best_model_state_dict.pt")
    print("saved", out / "best_model_state_dict.pt")


if __name__ == "__main__":
    main()

# Colab notebooks

Two notebooks are meant to be run by a human on Google Colab (GPU runtime). Our GitHub repository is private, so Colab cannot clone it; instead both notebooks ask you to upload one small tarball that `scripts/pack_for_colab.sh` builds.

| Notebook | Question | Runtime (T4) | Result |
|---|---|---|---|
| `colab_gpu_timing.ipynb` | How many ms per scenario does GENCO inference take on a GPU (batch 1 to 1024, fp32/bf16, 12 and 9 layers), and how does that compare with the Newton break-even thresholds? | 5 to 10 min setup, 20 to 45 min timing | `genco_gpu_timing_<gpu>.zip` |
| `colab_genco_delta.ipynb` | Does GENCO-Δ (inputs filled with the pre-contingency state, fine-tuned) beat the base case as a Newton start on GOC 2000 (Small)? | 1.5 to 3 h | `genco_delta_goc2000_results.zip` |

Do the timing notebook first. It is shorter, and it exercises the same upload and install steps.

## 0. Build the bundle (on your Mac, once per code change)

```bash
cd /Users/msarwar/Desktop/webDev/GridFMChallenge
./scripts/pack_for_colab.sh            # writes dist/genco_colab_bundle.tar.gz, prints size and sha256
```

The bundle holds `src/`, `scripts/`, `configs/`, `tests/`, `pyproject.toml`, the tiny split files (`data/pf/<grid>/splits_*`, `data/*/*/<grid>/splits_all`) and a `BUNDLE_INFO.txt` with the git revision. It has no raw data, checkpoints, results or `.venv` (about 0.2 MB). Options: `GRIDS="case118_ieee case2000_goc" ./scripts/pack_for_colab.sh` limits the split files to some grids, `WITH_TESTS=0` leaves `tests/` out, and a path as the first argument changes the output file. `dist/` is not in `.gitignore`: do not commit the tarball.

Commit or at least save your work first, because `BUNDLE_INFO.txt` records the revision and a "dirty" count, and the notebooks print it, so the results can be traced to a code state.

## 1. GPU timing (`colab_gpu_timing.ipynb`)

1. Open Colab (https://colab.research.google.com), *File > Upload notebook*, pick `notebooks/colab_gpu_timing.ipynb`.
2. *Runtime > Change runtime type*: hardware accelerator **T4 GPU** (free tier). If you have Colab Pro and an A100 or L4 is offered, run the notebook once on each: the comparison between GPUs is useful. The notebook stops with an error if there is no GPU.
3. Optional sanity run: set `DRY_RUN = True` in the parameters cell (first code cell) and run all cells: after the install it takes about a minute (one model, tiny batches, one data partition). Set it back to `False` and re-run the parameters cell afterwards. Its numbers mean nothing; its output goes to `results/gpu_timing_dry`.
4. *Runtime > Run all*, or run cell by cell:
   - parameters cell: nothing to change for the default study (IEEE 118 Base/Small/Tiny, GOC 500 Base/Small/Tiny, GOC 2000 Small/Tiny; batch sizes 1, 16, 64, 256, 1024);
   - GPU check cell: prints the GPU name;
   - upload cell: opens a file picker, choose `genco_colab_bundle.tar.gz`. It prints `BUNDLE_INFO.txt`;
   - install cell: takes a few minutes. If it prints `RESTART REQUIRED`, use *Runtime > Restart session*, run the parameters cell again, skip the upload and install cells and continue with the environment cell;
   - environment cell: prints torch, CUDA, torch_geometric, graphkit versions and which `torch_scatter` backend is used. Check that it says `torch_scatter 2.1.2` (compiled). If it says "pure-torch stand-in", note it, the timings are then slightly pessimistic;
   - data cell: downloads partitions of the public `gridfm/pf_small_*` datasets from Hugging Face (a few hundred MB);
   - timing cell: the long one. It prints one line per configuration (`batch 256 bf16 12 layers: x ms/scenario`). Out-of-memory configurations are recorded and skipped. Each finished model is saved under `results/gpu_timing/partial`, so after a disconnect re-run the cells down to this one and it continues;
   - Newton cell (optional) and report cell: the report is displayed as markdown;
   - last cell: zips the results and starts the download. If the browser blocks it, open the Files panel (left) and download `/content/genco_gpu_timing_<gpu>.zip` by hand.
5. Repeat on another GPU type if you can (a new runtime resets everything; upload the bundle again).

## 2. GENCO-Δ on GOC 2000 (`colab_genco_delta.ipynb`)

1. Upload the notebook and select a GPU runtime as above.
2. In the parameters cell set `DRY_RUN = True` first and run all cells: a 10 to 15 minute end-to-end check (1 epoch on 3% of the batches). Then set it back to `False`, run the parameters cell again and run the real thing. Dry-run outputs carry a `dry_` prefix, so they do not interfere with the real run. Finished stages are skipped when you re-run; `FORCE = True` redoes them.
3. Upload cell: pick `genco_colab_bundle.tar.gz` (the same file as for the timing notebook).
4. Install cell: may print `RESTART REQUIRED`. Restart the session, run the parameters cell and continue from "3. Environment check". The first import of graphkit also imports `gridfm-datakit` (Julia bindings), which can be slow once.
5. Defaults are 6 epochs per arm, batch size 8, a 100 minute cap per arm. Free-tier sessions can be cut off after a few hours or when idle: keep the tab open, or set `SAVE_TO_DRIVE = True` (it asks to mount your Drive and copies results after each stage; after a disconnect copy `genco_delta_goc2000/results` back to `/content/genco/results/genco_delta`).
6. The last cell zips everything for download (`genco_delta_goc2000_results.zip`, about 80 MB with the three fine-tuned weights; set `INCLUDE_WEIGHTS = False` to leave them out).

## 3. Putting the results back into the repository

Do not unzip into the repository root. Tags clash: the Colab run records and weights are called `control_s0`, `delta_vm_va_s0`, ... just like the IEEE 118 matrix of `scripts/run_genco_delta.sh` (`results/genco_delta/runs/control_s0.json`, `checkpoints/finetuned/control_s0/`), and unzipping there would overwrite them.

```bash
cd /Users/msarwar/Desktop/webDev/GridFMChallenge

# GPU timing: the zip has one folder named after the GPU
mkdir -p results/gpu_timing
unzip ~/Downloads/genco_gpu_timing_*.zip -d results/gpu_timing/
#   -> results/gpu_timing/<gpu>/{genco_gpu_timing.csv, genco_precision_check.csv, report.md, env.json, genco_gpu_timing.png, partial/}

# GENCO-Δ on GOC 2000: paths inside the zip are relative to the Colab repository root
mkdir -p results/colab/genco_delta_goc2000
unzip ~/Downloads/genco_delta_goc2000_results.zip -d results/colab/genco_delta_goc2000/
#   -> .../results/genco_delta/{eval,runs,logs,summary_*}, .../data/pf/case2000_goc/{splits_delta,basecase_store.pt}, .../checkpoints/finetuned/*
mkdir -p checkpoints/colab && mv results/colab/genco_delta_goc2000/checkpoints/finetuned checkpoints/colab/   # weights: keep them out of git (checkpoints/ is ignored)
```

Then tell the coordinating session (or add to `RESEARCH_LOG.md`): the GPU name from `env.json`, the scatter backend, the headline rows of `report.md` and, for the Δ run, `summary_by_arm.md` plus the "model − base" rows of the results table.

## Troubleshooting

- *"No GPU found" / "CUDA is not available"*: *Runtime > Change runtime type > T4 GPU*, reconnect, run the cells again.
- *The picker does not open or the upload is slow*: put the tarball on Google Drive, mount Drive in a new first cell (`from google.colab import drive; drive.mount("/content/drive")`) and set `TARBALL = "/content/drive/MyDrive/genco_colab_bundle.tar.gz"` in the parameters cell.
- *Import errors from `gridfm_graphkit` after the install*: the pinned commit (`8a8defb`) is written for torch 2.10 to 2.13. If Colab ships something else, run `!pip install "torch>=2.10,<2.14" torchvision torchaudio` in a new cell, restart the session and start again from the parameters cell. The delta notebook does this automatically; the timing notebook keeps Colab's torch by default on purpose, to measure what a normal Colab user gets.
- *Out of memory in the delta notebook*: lower `BATCH_SIZE` to 4 and re-run section 7 with `FORCE = True`.
- *The bundle is stale* (a notebook complains about a missing task or function): rebuild with `./scripts/pack_for_colab.sh` and upload it again; delete `/content/genco` in the Colab file panel first, because the upload cell skips unpacking when the code is already there.

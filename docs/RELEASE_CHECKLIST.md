# Release checklist

Steps for the author to take the work public before posting. Nothing here has been run: no commit, push, upload, repo creation or visibility change was made while preparing it. Commands assume the repo root, `source .venv/bin/activate`, and zsh.

Deadline: 25 Oct AoE is 12:00 UTC on 26 Oct. Budget about 2.5 hours without the GPU timing (it is the only step that can run in parallel), and 3.5 hours with it.

Totals to upload:

| Target | Content | Size |
|---|---|---|
| HF model repo | 12 state dicts (`best_model_state_dict.pt`, 80,445,203 bytes each) + 12 `run.json` + model card | 0.97 GB |
| HF dataset repo | raw parquet of 12 IEEE 118 sets + 4 stress30 sets (no `processed/`), configs, card | 0.51 GB (0.47 GB without stress30) |
| HF Space | `app.py`, `engine.py`, `bundle/`, `assets/`, vendored package, README, requirements (checkpoint is fetched at run time from `gridfm/genco-pf-datakit`, about 80 MB) | about 6.5 MB |
| GitHub | the existing private repo, flipped to public | no change |

## 0. Decide names (5 min)

- HF account or org: `HF_USER=<name>`. Suggested repos: `$HF_USER/genco-ieee118-finetunes` (model), `$HF_USER/genco-contingency-ieee118` (dataset), `$HF_USER/genco-certified` (Space; the title is now "GENCO Operator Console", so rename the slug if you prefer, e.g. `genco-console`, and use it in `publish_space.py` below).
- GitHub stays `msarwar20/genco-certified` (already in `CITATION.cff`).
- Affiliation text for `{{AFFIL}}`. (Reference [1] in the post now has the GENCO authors written out; check the author list against arXiv:2608.09921 once.)
- `export HF_USER=<name>`; log in once: `hf auth login` (the older `huggingface-cli login` also works).

## 1. GPU timing for {{GPU}} (40 to 60 min, can run while you do steps 2 to 5; optional but the post has two placeholders for it)

1. `./scripts/pack_for_colab.sh` (writes `dist/genco_colab_bundle.tar.gz`; do not commit it).
2. Colab, T4 runtime, `notebooks/colab_gpu_timing.ipynb`; follow `notebooks/README.md` section 1. Unzip `genco_gpu_timing_<gpu>.zip` into `results/colab/`.
3. `python scripts/make_post_figures.py --only figD_factor_reuse_breakeven` if you want that figure, and note the GPU name, ms per contingency at the best batch size, and whether any break-even is crossed.
4. In `docs/POST.md`: replace `{{GPU}}` in Table 0 (row "Wall-clock break-even") with the number, e.g. "and X ms on a T4", and replace the whole `{{GPU: ...}}` paragraph at the end of Section III. If you skip this step, delete both placeholders instead and keep the 8.7 ms laptop figure. Update the same Table 0 row in `space/app.py` (`FINDINGS`) and `space/README.md` if you add a number.
5. Skip `colab_genco_delta.ipynb` (GOC 2000 GENCO-Δ, 1.5 to 3 h). The post states GENCO-Δ was run on IEEE 118 only.

## 2. Pre-flight checks on the repo (10 min)

```bash
python -m pytest -m "not data" -q
python scripts/verify_artifacts.py
```

Untracked files that belong in the release (see `git status`): `hf/`, `docs/VIDEO_SCRIPT.md`, `docs/RELEASE_CHECKLIST.md`, `configs/datakit/case118_confirm_n*.yaml`, `results/exp02/confirm_n*_base_scenarios.parquet` (check whether `.gitignore` should cover it; the confirmation CSVs and `results/confirm/` are the tracked evidence), the five `space/assets/fig*.png` and the deleted `space/assets/fig1..4_*.png`. `docs/video_script.md` (old storyboard) was replaced by `docs/VIDEO_SCRIPT.md`; they are the same name on a case-insensitive Mac, so `git status` shows one rename or delete and one add.

`scripts/publish_space.py` copies every `results/figures/*.png` into `space/assets/`. That re-adds the four old figures (`fig1_layer_dynamics.png` and others, built before the post) as unused files, about 0.7 MB. Remove them from `space/assets/` after publishing, or delete the copy loop in the script.

## 3. Create the Hugging Face repos, private first (5 min)

```bash
hf repos create $HF_USER/genco-ieee118-finetunes --type model --private
hf repos create $HF_USER/genco-contingency-ieee118 --type dataset --private
hf repos create $HF_USER/genco-certified --type space --space-sdk gradio --private
```

## 4. Upload the model repo (15 to 30 min, depends on upload speed)

```bash
STAGE=$HOME/hf_stage/model; rm -rf $STAGE; mkdir -p $STAGE
cp hf/model_card.md $STAGE/README.md
for t in residual_vm_va_flows_lr3e-4 residual_nk_vm_va_flows_lr3e-4 slack_aware control; do
  for s in 0 1 42; do
    mkdir -p $STAGE/${t}_s$s
    cp checkpoints/finetuned/${t}_s$s/best_model_state_dict.pt $STAGE/${t}_s$s/
    cp results/genco_delta/runs/${t}_s$s.json $STAGE/${t}_s$s/run.json
  done
done
du -sh $STAGE            # expect about 921 MiB; 24 files plus README
```

Fill `[GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console)` and `sarwar21255` in `$STAGE/README.md` first (step 6 has the commands; run them on the staged copy or on `hf/model_card.md` before copying). Then:

```bash
hf upload $HF_USER/genco-ieee118-finetunes $STAGE . --type model --commit-message "Fine-tuned GENCO checkpoints for IEEE 118"
```

Do not upload the `best.ckpt` Lightning files (241 MB each, 2.9 GB in total). The loader accepts state dicts.

## 5. Upload the dataset repo (10 to 20 min)

```bash
STAGE=$HOME/hf_stage/data; rm -rf $STAGE; mkdir -p $STAGE/configs
for s in calib/n1 stress100/n1 stress100/n2 stress100/n3 stress100/n5 confirm/n1 confirm/n2 confirm/n3 confirm/n5 \
         exhaustive/n1 exhaustive/n1_gen train_nk stress/n1 stress/n2 stress/n3 stress/n5; do
  mkdir -p $STAGE/$s/case118_ieee
  rsync -a data/$s/case118_ieee/raw $STAGE/$s/case118_ieee/
done
cp configs/datakit/case118_*.yaml $STAGE/configs/
cp data/exhaustive/run_exhaustive_gen.py $STAGE/configs/
cp hf/dataset_card.md $STAGE/README.md
du -sh $STAGE            # expect about 510 MB
grep -rIl '/Users[/]' $STAGE | head    # must print nothing
hf upload $HF_USER/genco-contingency-ieee118 $STAGE . --type dataset --commit-message "IEEE 118 contingency sets (datakit raw)"
```

`stress/n*` (30 load scenarios, 41 MB) are the sets the GENCO-Δ evaluation and the Space bundle use, so keep them. To save 41 MB, drop the last four entries of the loop and the matching row in the dataset card. `processed/` caches and `splits_all/` are not uploaded: `scripts/make_splits.py` and graphkit rebuild them. The card sets `viewer: false`, because the partitioned parquet layout is not something the HF dataset viewer handles.

## 6. Fill the placeholders (15 min)

Placeholders and where they are:

| Placeholder | Files |
|---|---|
| `[GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console)` | `docs/POST.md` (author line and Section IX), `README.md`, `space/README.md`, `hf/model_card.md` (twice, in prose and BibTeX), `hf/dataset_card.md`, `docs/VIDEO_SCRIPT.md` |
| `sarwar21255` | `hf/model_card.md`, `hf/dataset_card.md` |
| `{{AFFIL}}` | `docs/POST.md` |
| `{{GPU}}`, `{{GPU: ...}}` | `docs/POST.md` (step 1 above) |

Suggested `[GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console)` text for the post (one line, Markdown): `[GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/$HF_USER/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/$HF_USER/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/$HF_USER/genco-certified)`. In `hf/model_card.md` BibTeX and `CITATION.cff` use the GitHub URL alone.

```bash
export HF_USER=<name>
LINKS='[GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/'$HF_USER'/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/'$HF_USER'/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/'$HF_USER'/genco-certified)'
sed -i '' "s|[GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console)|$LINKS|g; s|sarwar21255|$HF_USER|g" docs/POST.md README.md space/README.md hf/dataset_card.md docs/VIDEO_SCRIPT.md
sed -i '' "s|sarwar21255|$HF_USER|g; s|url    = {https://github.com/msarwar20/genco-certified}|url    = {https://github.com/msarwar20/genco-certified}|; s|[GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console)|$LINKS|g" hf/model_card.md
```

Edit `{{AFFIL}}` and the GPU text by hand. The `|` in the sed expression must not appear in the replacement (it does not here). Then delete the draft HTML comment at the top of `docs/POST.md` (it lists the placeholders and says "do not post"), and in `docs/POST.md` check that Table 0 and `space/app.py` Table 0 are still identical after the GPU edit.

If you stage the model card or dataset card before this step, re-copy them to the staging directories (`cp hf/model_card.md $HOME/hf_stage/model/README.md`, same for data) and run `hf upload` again for the README only.

## 7. Publish the Space (10 min upload, 5 to 10 min build)

```bash
python scripts/publish_space.py --repo $HF_USER/genco-operator-console --private
```

It refreshes the vendored `space/genco_certified/` from `src/` and the figures, then uploads `space/` without `checkpoints/`, `results/` and caches. Wait for the build on CPU basic (the install pulls torch CPU, torch-scatter, mlflow and graphkit from GitHub, so the first build is the slow one). Then, in the Space:
- Inspect a contingency: pick N-1, N-2, N-3, N-5, load one scenario of each. The first load downloads the 80 MB checkpoint; allow about 30 s.
- Screen a whole set: run it for N-2. Expect 0 missed and 0 false alarms for the sequential policy, and 1.5 to 1.7 factorizations against 2.3 to 2.9.
- Findings: Table 0 and five figures render.

Local headless smoke test of the same app (all four orders, callbacks, HTTP 200, five images) passed on 2026-10-25.

## 8. Upload the figures to the discussion (10 min)

Files, from `results/figures/post/`: `figH_hero.png`, `figB_exchange_rate.png`, `figA_workflow.png`, `figC_equal_cost.png`, `figF_genco_delta.png`. Drag each into the Community-tab reply box of the challenge Space and copy the resulting `![...](https://...)` link. In the post text replace the bare filenames (`![Fig. 1](figH_hero.png)` and so on) with those links. Preview before submitting.

## 9. Regenerate the checksums (5 min)

```bash
git add results docs hf space configs
python scripts/make_checksums.py
python scripts/verify_artifacts.py --strict
```

`make_checksums.py` covers tracked `results/**/*.{json,csv,md,log}` files, so `git add` new result files first, and rerun it after anything under `results/` changes. Placeholders in `docs/` and `hf/` do not affect the sums.

## 10. Final greps (5 min)

```bash
grep -rn '{{' README.md docs/POST.md docs/VIDEO_SCRIPT.md space/README.md space/app.py hf/ CITATION.cff   # no output
grep -rn '/Users[/]' README.md docs hf space/app.py space/engine.py space/README.md scripts src tests configs notebooks   # no output
git ls-files -z | xargs -0 grep -lI '/Users[/]'                                                              # no output
```

Known and fine: `.github/workflows/ci.yml` uses `${{ ... }}` expressions, and `docs/HANDOFF.md`, `docs/PLAN.md` and `results/notes/review_post_*.md` mention placeholder names in prose. They are internal notes; clean them or leave them, they are not part of the post. If the `/Users` grep finds tracked files, run `scripts/sanitize_paths.sh`, then redo step 9. Also check the post for a leftover `TODO`, `TBD` or "draft".

## 11. Commit, push, make the repos public (15 min)

```bash
git status                       # nothing unexpected; no data/, checkpoints/, .venv
git add -A && git commit -m "Release materials: Space text, model and dataset cards, video script, checklist"
git push origin main
gh run list --limit 3            # wait for CI to go green
gh repo edit msarwar20/genco-certified --visibility public --accept-visibility-change-consequences
hf repos settings $HF_USER/genco-ieee118-finetunes --public --type model
hf repos settings $HF_USER/genco-contingency-ieee118 --public --type dataset
hf repos settings $HF_USER/genco-certified --public --type space
```

Before flipping GitHub, look once at the repo history for anything you would not want public (the repo has been private, so older commits are included). Open all four links in a private browser window. Check that the model card front matter is accepted (no "metadata error" banner on the model page).

## 12. Post (15 min)

Paste `docs/POST.md` (without the draft comment) into a new discussion in the Community tab of the challenge Space (https://huggingface.co/spaces/gridfm/genco-challenge), with the figure links from step 8. Check Table 0 and the five figures in the preview. After posting, add the discussion URL to `README.md`. The video (`docs/VIDEO_SCRIPT.md`, about 20 min to record and 20 min to edit) is optional; link it in the post only if it is up before the deadline.

## Time summary

| Step | Minutes |
|---|---|
| 0 Names | 5 |
| 1 GPU timing (parallel) | 40 to 60 |
| 2 Pre-flight | 10 |
| 3 Create repos | 5 |
| 4 Model upload | 15 to 30 |
| 5 Dataset upload | 10 to 20 |
| 6 Placeholders | 15 |
| 7 Space | 20 to 30 |
| 8 Figures | 10 |
| 9 Checksums | 5 |
| 10 Greps | 5 |
| 11 Commit and publish | 15 |
| 12 Post | 15 |
| Total without GPU | about 130 to 170 |

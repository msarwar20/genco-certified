# What is GENCO worth to a Newton solver? Pricing a grid foundation model against the pre-contingency state

This is the entry of Muhammad Sarwar (Iowa State University) and Muhammad Rizwan (Pakistan Institute of Engineering & Applied Sciences) to the GridFM GENCO Challenge 2026. The write-up is [docs/POST.md](docs/POST.md), and the models and data are at [GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console).

GENCO predicts an AC power-flow state in one forward pass, but in contingency analysis an operator rarely solves from scratch, because the solved pre-contingency case is already on the screen and is the warm start that production tools use. This repository asks a narrow question about the released GENCO checkpoints: if the prediction is handed to a float64 Newton solver, how much solver work does it save against the pre-contingency start, and can that saving be taken without missing an overload? Everything here, from the Newton solvers to the figure scripts, is built to answer that question and to let you re-check the answer, and every number in the write-up is traced to a source file in [docs/NUMBERS.md](docs/NUMBERS.md).

## Results at a glance

In contingency analysis a power-flow prediction is only as valuable as the solver work it saves, and that solver does not start from scratch: it starts from the solved pre-contingency case and usually reuses that case's factorization in a chord or fast-decoupled method. We put the prediction of GENCO [1] in front of each of these solvers on three grids and measured what it saves at equal safety.

Fig. 1 summarizes the result: on IEEE 118 a GENCO start saves a fifth of the work for Newton and a twentieth for fast-decoupled, while at 2,000 buses it saves 2% for chord Newton and costs extra for the other two. The screening savings in Sec. IV are counted in full-Newton factorizations, and Sec. III shows they do not pay back in wall-clock time.

TABLE 0. Results at a glance (IEEE 118, released checkpoints, unless stated).

| Question | Answer |
|---|---|
| Newton iterations saved over the pre-contingency start, to 10⁻⁴ MW | 0.49 [0.45, 0.53] at N-1, 1.0 at N-5; 0.33 on GOC 500; −0.63 on GOC 2000 |
| Work saved under factor reuse | chord: 21% of triangular solves; fast-decoupled: 5%; at 2,000 buses −18% to +2% |
| Wall-clock break-even | inference must cost under 0.015–0.6 ms per contingency; GENCO Base takes 2.5 ms on a T4 GPU and 8.7 ms on a laptop GPU. Only the smallest model in fp16 crosses one threshold (full Newton, N-5). Even with free inference, GENCO + Newton is slower than fast-decoupled from the pre-contingency state |
| Screening with a calibrated Newton check | 0 missed violations in 16,449 held out, and 0 in 17,095 more generated after the rule was frozen, from a different load profile. 37–44% fewer factorizations than the same check from the pre-contingency state |
| What GENCO reports vs what it computes | its power balance looks 32–60× better than the physical one on IEEE 14–118 (about 20× on GOC 500 and 2000), because the slack-bus voltage setpoint is masked and predicted |
| Conditioning GENCO on the base case | a residual decoder (GENCO-Δ) cuts N-1 state error 2.5× but saves no iterations at 10⁻⁴ MW; mixed N-k training trades the N-1 gain for an N-5 fix |

![Triangular solves saved by a GENCO start, by solver and grid size](results/figures/post/figH_hero.png)

*Triangular solves saved by a GENCO start over the pre-contingency start, by solver and grid size (to 10⁻⁴ MW, 95% CI).*

Every number is traced to a result file in [docs/NUMBERS.md](docs/NUMBERS.md).

![Newton iterations from each start, and the GENCO minus pre-contingency difference by grid size](results/figures/post/figB_exchange_rate.png)

*Newton iterations from flat, DC, pre-contingency and GENCO starts (left) and the paired GENCO minus pre-contingency difference (right), with 95% cluster-bootstrap intervals over load scenarios.*

![Missed violating elements against factorizations per contingency](results/figures/post/figC_equal_cost.png)

*Missed violations against factorizations per contingency on IEEE 118 (four 100-load-scenario test sets). Hollow markers are zero-miss cells, drawn at their one-sided 95% Clopper–Pearson bound.*

The remaining figures (workflow, factor-reuse break-even, slack audit, GENCO-Δ) are in [results/figures/post/](results/figures/post/), with the plotted values next to each one in `data/`.

## What is in this repo

- `src/genco_certified/` is the installable package. It holds `acpf.py`, the float64 AC power flow on exactly the admittance graph GENCO sees, and `fastpf.py`, the chord and fast-decoupled solvers that reuse one factorization per operating point. It also holds `tracing.py`, which captures GENCO's state after each correction layer, `graphkit_plugin.py`, the slack-aware PF task for gridfm-graphkit, `basecase_store.py` and `genco_residual.py`, the pre-contingency store and the residual GENCO-Δ decoder, and `loading.py`, which builds a task and test dataloader in-process.
- `scripts/` has one script per experiment. `reproduce.py` and `summarize_repro.py` re-evaluate the released checkpoints, `exp01_layers_newton.py` covers layer dynamics (Newton from each layer), and `exp06_slack_audit.py` covers the slack convention. `exp02_screening.py` with `analyze_exp02.py` produces the exchange rate, sequential policy, calibration and confidence intervals, while `exp05_best_start.py` and `exp07_industrial_baselines.py` cover the best-of-two start, the chord and fast-decoupled baselines, and timing. `exp04_embedding_probe.py` and `exp08_bias_correction.py` are the two negative results, `finetune.py`, `finetune_residual.py` and the `run_genco_delta*.sh` drivers run the slack-aware fine-tune and GENCO-Δ, and the rest are `make_post_figures.py`, `build_space_bundle.py`, `verify_artifacts.py` and the `run_batch3*.sh` drivers.
- `configs/datakit/` holds the gridfm-datakit configs for every new scenario set (N-1 to N-5, calibration, exhaustive), and `tests/` holds unit tests for the solvers, the screening policy and the GENCO-Δ code.
- `space/` is a small Gradio console that walks one contingency through the policy; it is prepared but not published. `notebooks/` has two Colab notebooks, one for a GPU timing run and one for GENCO-Δ on GOC 2000, described in `notebooks/README.md`.
- `results/` has the JSON, CSV and logs for every table, one directory per experiment (`exp01` to `exp08`, `genco_delta`, `repro`, `runtime`), plus `figures/post/`, and `SHA256SUMS` covers the tracked files.
- `docs/` has `POST.md` (write-up), `NUMBERS.md` (source of truth for every number), `REPRODUCE.md` (commands) and `report.md` (earlier report, see its header).

## Reproduce

The reference machine is a MacBook Air M4 with Python 3.12, and a full run needs about 25 GB of disk, of which the core reproduction needs about 8 GB.

```bash
./scripts/setup_env.sh && source .venv/bin/activate && pip install -e .
python scripts/fetch_data.py --task pf --partitions 30
python scripts/reproduce.py --task pf --sizes base small tiny
python scripts/exp02_screening.py --grid case118_ieee --size base
python scripts/analyze_exp02.py --v3 && python scripts/analyze_exp02.py --ci
python scripts/make_post_figures.py
python -m pytest -m "not data" -q
python scripts/verify_artifacts.py
```

The last command checks `SHA256SUMS`, that every result JSON parses, and that its numbers are finite; add `--checkpoints` to also match the released weights. Data generation for the stress sets, the GOC runs, fine-tuning and GENCO-Δ are covered in [docs/REPRODUCE.md](docs/REPRODUCE.md). Counts of iterations and factorizations reproduce to the printed precision on the same platform; timings and the last digits of mismatches will vary by machine.

## Negative results

Several things we tried did not work. Flat-start Newton never failed up to N-5, so there was no convergence to rescue, and accepting GENCO with no Newton step failed because its own residual never certified a whole contingency. The embedding probe ("the latents know where GENCO is wrong") is withdrawn, since bus identity alone explains about as much, and per-bus bias correction removes 0.3% of squared error in-sample and nothing out of sample. A slack fix by mask surgery on Base does not work either: feeding the setpoint in multiplies state error by 17×, and re-anchoring voltages is worse in 95–100% of scenarios. The slack-aware fine-tune gives an honest power balance, but its state error is about 29% worse than a matched control and it brings no iteration gain.

Plain GENCO-Δ reaches base-case quality at best, and one seed diverged, while residual GENCO-Δ regresses at N-5. Calibrating the error bound on 29 operating points was not exchangeable across outage order (up to 33% exceedance at N-2), and a 100-point calibration fixes this without making it a guarantee. Plain chord Newton from the base case fails the 50-iteration cap in 20–70% of contingencies, depending on outage order. Finally, we found no wall-clock gain on an M4 laptop, where inference is 12–1,600× above break-even.

## Limitations

The zero-miss result is empirical, on datakit scenarios. The main calibration and test sets share datakit's daily load trajectory, although the confirmation run uses a different load profile, and a Newton–Kantorovich certificate that would give a proof does not hold at GENCO's raw output. Factor-reuse results cover IEEE 118, GOC 500 and GOC 2000 at seed 0, and GENCO-Δ was run on IEEE 118 only. The reproduction of the released checkpoints uses about 3% of each paper test set, with mean metrics within about 6% (worst 13%). Timings come from Python and SciPy, so a compiled solver would push the break-even lower, and the GPU timings come from a single Colab session on a Tesla T4.

## License and attribution

The code is released under Apache-2.0 ([LICENSE](LICENSE)), and [NOTICE](NOTICE) lists the third-party software, models and data this work builds on, with their terms: gridfm-graphkit, gridfm-datakit and the released GENCO checkpoints. To cite this work, use [CITATION.cff](CITATION.cff).

Contact: Muhammad Sarwar, msarwar@iastate.edu

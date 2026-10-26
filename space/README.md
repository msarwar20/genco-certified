---
title: GENCO Operator Console
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.0
python_version: 3.12
app_file: app.py
pinned: false
license: apache-2.0
short_description: Walk one contingency through GENCO and a calibrated Newton check
---

# GENCO Operator Console

This Space is the interactive companion to our entry for the GridFM GENCO Challenge 2026, "What Is GENCO Worth to a Newton Solver? Savings, Safety and Break-Even Against the Pre-Contingency Start". The write-up has the full tables, and every number in it traces to `docs/NUMBERS.md` in the code repository.

The console runs the released GENCO power-flow model (`gridfm/genco-pf-datakit`, IEEE 118, Base, seed 0) on a bundle of 30 contingencies each at N-1, N-2, N-3 and N-5. For a chosen contingency it shows:
- the post-contingency grid, with true loadings and GENCO's local power mismatch at each bus (evaluated at the true slack-voltage setpoint);
- the max power mismatch through GENCO's twelve layers and the Newton steps that follow, next to Newton–Raphson started from the pre-contingency state and from a flat start;
- the decision card: whether the sequential policy accepted after Newton step 1, 2 or 3, or escalated to a full solve, and how many factorizations it used against the other starts;
- each violation as seen by the true AC solution and by GENCO alone, GENCO plus one Newton step, and the pre-contingency state plus one step.

The policy is the one in the write-up (Fig. 3): after each Newton step a calibrated bound on every monitored branch loading and PQ voltage decides accept or continue, and after three steps the case goes to a full solve. The constant is a split-conformal quantile (α = 0.10) from a separate 100-load-scenario IEEE 118 N-1 set, stored in `bundle/calibration.json`. All safety evidence is empirical.

## Results at a glance

IEEE 118, released checkpoints, unless stated. The Findings tab has the full table and the figures.

| Question | Answer |
|---|---|
| Newton iterations saved over the pre-contingency start, to 10⁻⁴ MW | 0.49 [0.45, 0.53] at N-1, 1.0 at N-5; 0.33 on GOC 500; −0.63 on GOC 2000 |
| Work saved under factor reuse | chord: 21% of triangular solves; fast-decoupled: 5%; at 2,000 buses −18% to +2% |
| Screening with a calibrated Newton check | 0 missed violations in 16,449 held out, and 0 in 17,095 more generated after the rule was frozen, from a different load profile. 37–44% fewer factorizations than the same check from the pre-contingency state |
| What GENCO reports vs what it computes | its power balance looks 32–60× better than the physical one on IEEE 14–118 (about 20× on GOC 500 and 2000), because the slack-bus voltage setpoint is masked and predicted |

## Notes on the bundle

- The 30 scenarios per order include the 8 hardest by GENCO mismatch, so the "Screen a whole set" tab looks worse than the full-set results above. N-1 comes from the released model's paper test split; N-2, N-3 and N-5 come from datakit sets generated for this work.
- The first scenario load downloads the GENCO checkpoint (about 80 MB) from `gridfm/genco-pf-datakit` and runs 30 scenarios on CPU, so the first view of each order takes a few seconds.
- Code and data: [GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar21255/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar21255/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar21255/genco-operator-console)

## Authors

Muhammad Sarwar (Iowa State University) and Muhammad Rizwan (Pakistan Institute of Engineering & Applied Sciences). Contact: msarwar@iastate.edu.

---
title: GENCO, Certified
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.0
python_version: 3.12
app_file: app.py
pinned: false
license: apache-2.0
short_description: Certified contingency screening with the GENCO grid model
---

# GENCO, Certified

This Space is an interactive companion to our entry for the GridFM GENCO Challenge 2026.

It runs the released GENCO power-flow model (`gridfm/genco-pf-datakit`, IEEE 118, Base) on a small bundle of N-1 to N-5 contingencies. For each one it shows:
- the post-contingency grid, with true loadings and GENCO's local power mismatch;
- how the mismatch evolves through GENCO's twelve layers and the Newton steps that follow, next to Newton–Raphson started from the base case and from a flat start;
- whether the screening decision was accepted after one Newton step or escalated;
- each violation as seen by the true AC solution and by each screening method.

The bundle includes the hardest scenarios of each set on purpose, so its statistics are worse than the full-set results reported in the write-up.

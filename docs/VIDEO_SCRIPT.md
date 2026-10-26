# Video script: "What is GENCO worth to a Newton solver?" (2:30)

Screen recording by the author, with voice-over. Narration is the italic text; about 315 words at a calm pace (130 words per minute leaves room for clicks). Record at 1080p, browser zoom 110%, light theme. Do the first scenario load before recording: the first view of each contingency order takes a few seconds (GENCO downloads once, then 30 scenarios run on CPU).

Setup before recording:
- Open the Space (`[GitHub](https://github.com/msarwar20/genco-certified), [checkpoints](https://huggingface.co/sarwar2155/genco-ieee118-finetunes), [data](https://huggingface.co/datasets/sarwar2155/genco-contingency-ieee118), [demo](https://huggingface.co/spaces/sarwar2155/genco-operator-console)`, the Space tab) and wait for the grid to appear.
- In a second tab, have `figH_hero.png` open full-width (or the Findings tab scrolled to Fig. 1).
- Close other windows. Set the mouse pointer to a large size so clicks are visible.

| Time | Screen | Narration |
|---|---|---|
| 0:00 to 0:25 | Fig. 1 (`figH_hero.png`). Slowly move the pointer along the IEEE 118 bars, then to the GOC 2000 bars, then to the shaded region. | *A learned power-flow model is worth what it saves the solver it feeds. In contingency analysis that solver starts from the solved pre-contingency case, so we priced GENCO against that start. On IEEE 118 it saves a fifth of Newton's work. At two thousand buses it saves almost nothing, or costs extra.* (52 words) |
| 0:25 to 0:40 | Switch to the Space, tab "Inspect a contingency". Click "N-2" in the order radio. Open the Scenario dropdown and pick `#21 · out: line 60–61, line 68–69 · GENCO mismatch 0.69 MW · accepted after step 1`. | *Here is one N-2 contingency on IEEE 118. Lines 60 to 61 and 68 to 69 are out, and the red lines are overloaded. GENCO runs live in this Space.* (30 words) |
| 0:40 to 1:00 | Point at the convergence plot. Trace the dotted orange line (layers 1 to 12), then the green line after layer 12. Hover over the first green point after layer 12. | *In the convergence plot, GENCO's twelve layers take the mismatch from thousands of megawatts down to under one. We then hand its answer to a float64 Newton solver, and the green line shows each step.* (35 words) |
| 1:00 to 1:25 | Point at the red and pink lines (Newton from the pre-contingency state, Newton from a flat start). Then point at the decision card. Read the card: accepted after step 1, factorizations 1 against 3 and 4. | *To full convergence Newton needs three steps from GENCO or from the pre-contingency state, and four from a flat start. The saving is in the decision. After one step, a calibrated bound says every branch and voltage is clear of its limit, so the policy accepts with one factorization, not three.* (51 words) |
| 1:25 to 1:35 | Scroll to the violations table. Point at row `line 68–74`: "GENCO alone" shows a warning, the next column a tick. | *GENCO alone would have flagged line 68 to 74. One Newton step corrects it, and nothing is missed.* (18 words) |
| 1:35 to 2:00 | Back to the grid map. Hover over the darkest orange bus, which should sit next to the slack star, and read its mismatch. Then point at the dotted orange line's end (0.69 MW), then at the green line's start (about 69 MW) in the plot. | *Now the slack bus. Graphkit masks the slack voltage, which is a known setpoint, and asks GENCO to predict it. GENCO reports 0.69 megawatts of mismatch. Put the setpoint back and the same state is off by 69. Across scenarios the reported balance looks 32 to 60 times better than the physical one.* (53 words) |
| 2:00 to 2:10 | Findings tab, scroll to Fig. 5. | *We also fed GENCO the pre-contingency state through a residual decoder, GENCO-Δ. It cuts N-1 state error two and a half times, and saves no Newton iterations.* (27 words) |
| 2:10 to 2:25 | Findings tab, scroll to "What did not work", then "Limitations". | *Much of the rest did not work: embedding error warnings, bias correction, accepting GENCO with no Newton step. And the safety evidence is empirical: zero misses in sixteen thousand held-out violations, and zero in seventeen thousand more.* (37 words) |
| 2:25 to 2:30 | Title card or the repository page with the link. | *Code, checkpoints, data and this console are linked below.* (9 words) |

## Notes

- Every number above is in the post's Table 0 or comes from the scenario on screen. The scenario numbers (0.69 MW, 69 MW, 1 against 3 and 4 factorizations, line 68 to 74) are from the bundled N-2 scenario #21 at c = 0.236 with the conf0.10 rule. If the bundle or calibration changes, re-check them in the Space before recording. Scenario ids are fixed by `space/bundle/manifest.json`.
- The Space shows 12 layers and the orange dotted line in GENCO's own slack voltage; the green line starts at the true setpoint. Do not describe the 69 MW as an error of the Newton solver: it is GENCO's state evaluated at the physical setpoint.
- Do not say "certified" or "guaranteed". The post says empirical: the bound was exceeded in 1 to 10% of test operating points at a nominal 10%, and the 400 confirmation operating points share one load trajectory.
- If the GPU timing is in the post by the time you record, add one sentence to the first segment: "Inference takes 8.7 milliseconds per contingency on a laptop GPU, against a break-even of under 0.6." and cut the same length elsewhere. Use the Colab number if you have it.
- Captions: export the narration column as the subtitle file. The text is plain, so the timings above can be used as cues.

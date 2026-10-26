# Supplement to the post

This file holds the material that was moved out of the post to keep it short. All numbers are traced to result files in [NUMBERS.md](NUMBERS.md).

## S1. Best-of-two start

A best-of-two rule starts Newton from whichever candidate (GENCO's prediction or the pre-contingency state) has the smaller mismatch, at no extra factorization. It ends within 0.1 iteration (to 10⁻⁶ MW) of the better start on both IEEE 118 and GOC 2000: 2.25 against 2.15 for GENCO alone on IEEE 118, and 2.92 against 2.93 for the pre-contingency start on GOC 2000.

## S2. GPU timing details (Tesla T4)

Without batching, a single contingency takes 57–63 ms on the T4, most of which is kernel-launch overhead. Only one configuration crosses a break-even: GENCO Tiny in fp16 or with early exit, at 0.38–0.51 ms, against the 0.63 ms full-Newton break-even at N-5. That comparison is generous to GENCO for three reasons:
- the break-even was measured for Base's saving;
- the break-even was measured against single-threaded Python Newton;
- fp16 moves the predicted angles by a median 0.001 rad, and we did not test what that does to the Newton saving.

On GOC 2000, GENCO Small takes 19.5 ms on the T4 (13.4 ms in fp16).

## S3. Calibration history and the GOC 2000 policy

An earlier calibration on only 29 operating points reached 33% exceedance at N-2, which is why calibration now has its own 100-operating-point set. On GOC 2000 both starts reach zero misses, but GENCO never clears the bound after one step, so its 2.00 factorizations (vs 2.16) come from the two-step floor rather than from prediction quality.

## S4. Deviations in the confirmation run

The protocol (`results/confirm/PROTOCOL.md`) lists three operational deviations:
- a standalone splits script;
- 4 worker processes instead of 6;
- a load profile picked from its summary statistics before any data was generated.

The run also reproduced all six frozen constants. The bound was exceeded in 5–10% of operating points at the nominal 10%, and the Newton saving to 10⁻⁶ MW replicated at 0.83–0.98 iterations.

## S5. Repairs of the slack convention

We tried three repairs:
- **Setpoint as input.** Feeding the setpoint into Base zero-shot multiplies its state error by 17×.
- **Output clamp.** Clamping the output works on Small and Tiny, but degrades Base by 5.7×.
- **Slack-aware fine-tune.** A short fine-tune brings the physical mismatch from 17.2 to 1.0 MW over three seeds. It costs 29% more state error than a matched control and does not change Newton iterations.

## S6. Layer dynamics

GENCO's mean mismatch on IEEE 118 falls from 261 MW after the first of its 12 correction layers to 0.054 MW after the last. Newton iterations (to 10⁻⁶ MW) started from intermediate layers stay near 3 until layer 7 and then drop, and stopping after layer 9 saves 26% of inference for 0.12 extra iterations.

## S7. GENCO-Δ details

**Plain Δ.** Of the three seeds, two trained and ended at about base-case quality, with 2.85–2.95 iterations to 10⁻⁶ MW against 2.15 for the released model. The third seed diverged.

**Residual Δ.** The variant shown is the best of three by validation loss. At 10⁻⁴ MW it gives no gain (2.05 vs 2.01).

**Mixed N-k retraining.** We retrained the residual model on 5,056 fresh scenarios with one to five elements out. This brings N-5 one-step misses to 9 against 18 of 1,763, which is still above the 6.3 of the released model. The stress N-1 error rises to 0.056 against 0.022. Because the N-k run also changed the training scenarios, we cannot separate the effect of the outage mix from that of the data.

**The slack voltage.** The base case carries the true slack voltage, but the slack-aware arm alone gains nothing, so the slack voltage is not the source of the residual gain.

## S8. Other negative results

The table below lists the negative and null results that are not in the post's main text.

| Idea | Outcome |
|---|---|
| Error warnings from GENCO's embeddings | A 118-entry bus lookup explains R² 0.55 of the per-bus voltage error, the embeddings 0.58, a randomly initialised GENCO 0.52 |
| Per-bus bias correction | 0.3% of squared error in-sample, nothing out of sample |
| Accepting GENCO with no Newton step | Its own residual never certifies a whole contingency |
| Convergence rescue | Flat-start Newton converged on every IEEE 118 datakit scenario that solved, up to N-5 |
| Plain GENCO-Δ | Base-case quality; one seed diverged |

## S9. Reproduction of the released metrics

Our reproduction uses about 3% of each paper test set. Mean metrics agree within 6% for most runs, with a worst case of 13%.

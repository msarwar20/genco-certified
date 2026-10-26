# GENCO GPU timing: Tesla T4

Run 2026-10-26T00:26:47Z. torch 2.11.0+cu130 (CUDA 13.0), torch_geometric 2.8.1, scatter backend: torch_scatter 2.1.2+pt211cu130. Host CPU: Intel(R) Xeon(R) CPU @ 2.00GHz (8 cores).

Forward pass only, data resident on the device, synchronised, median of the repeats (see `reps` in the CSV). `ms/scenario` = batch time / batch size. `best` = the batch size with the lowest ms/scenario; `best + h2d` adds the host-to-device copy of that batch (pageable memory) per scenario.

## IEEE 118 vs the break-even thresholds, paper set (N-1)

Thresholds (ms per contingency, M4 / scipy): full NR 0.306, compensated chord 0.0368, FDLF 0.015. A ratio below 1 means GENCO inference is cheaper than the solver time its start saves.

| model | dtype | layers | batch 1 (ms) | best (ms/scenario) | at batch | best + h2d (ms) | best / full NR | best / compensated chord | best / FDLF | below break-even for |
|---|---|---|---|---|---|---|---|---|---|---|
| base | bf16 | 12 | 62.5 | 2.83 | 64 | 2.85 | 9.24 | 76.8 | 188 | none |
| base | fp16 | 12 | 62.5 | 1.59 | 256 | 1.6 | 5.19 | 43.1 | 106 | none |
| base | fp32 | 12 | 56.7 | 2.52 | 64 | 2.55 | 8.25 | 68.6 | 168 | none |
| base | bf16 | 9 | 46.9 | 2.12 | 64 | 2.14 | 6.93 | 57.6 | 141 | none |
| base | fp16 | 9 | 46.9 | 1.23 | 64 | 1.25 | 4.02 | 33.4 | 81.9 | none |
| base | fp32 | 9 | 43 | 1.91 | 64 | 1.93 | 6.24 | 51.9 | 127 | none |
| small | bf16 | 12 | 63.6 | 1.29 | 1024 | 1.3 | 4.2 | 35 | 85.8 | none |
| small | fp16 | 12 | 63.6 | 0.885 | 1024 | 0.9 | 2.89 | 24.1 | 59 | none |
| small | fp32 | 12 | 55.8 | 1.3 | 64 | 1.32 | 4.24 | 35.2 | 86.4 | none |
| small | bf16 | 9 | 46.5 | 0.982 | 1024 | 0.996 | 3.21 | 26.7 | 65.5 | none |
| small | fp16 | 9 | 48.7 | 0.662 | 1024 | 0.677 | 2.16 | 18 | 44.2 | none |
| small | fp32 | 9 | 41.6 | 0.988 | 64 | 1.01 | 3.23 | 26.8 | 65.8 | none |
| tiny | bf16 | 12 | 63.7 | 0.651 | 256 | 0.667 | 2.13 | 17.7 | 43.4 | none |
| tiny | fp16 | 12 | 64.7 | 0.509 | 1024 | 0.524 | 1.66 | 13.8 | 34 | none |
| tiny | fp32 | 12 | 55.6 | 0.649 | 256 | 0.664 | 2.12 | 17.6 | 43.3 | none |
| tiny | bf16 | 9 | 47.2 | 0.476 | 1024 | 0.491 | 1.56 | 12.9 | 31.7 | none |
| tiny | fp16 | 9 | 47.5 | 0.381 | 1024 | 0.396 | 1.25 | 10.4 | 25.4 | none |
| tiny | fp32 | 9 | 41.9 | 0.473 | 1024 | 0.488 | 1.55 | 12.9 | 31.5 | none |

## IEEE 118 vs the break-even thresholds, N-5 stress

Thresholds (ms per contingency, M4 / scipy): full NR 0.63, compensated chord 0.0577, FDLF 0.0415. A ratio below 1 means GENCO inference is cheaper than the solver time its start saves.

| model | dtype | layers | batch 1 (ms) | best (ms/scenario) | at batch | best + h2d (ms) | best / full NR | best / compensated chord | best / FDLF | below break-even for |
|---|---|---|---|---|---|---|---|---|---|---|
| base | bf16 | 12 | 62.5 | 2.83 | 64 | 2.85 | 4.49 | 49 | 68.1 | none |
| base | fp16 | 12 | 62.5 | 1.59 | 256 | 1.6 | 2.52 | 27.5 | 38.3 | none |
| base | fp32 | 12 | 56.7 | 2.52 | 64 | 2.55 | 4.01 | 43.8 | 60.8 | none |
| base | bf16 | 9 | 46.9 | 2.12 | 64 | 2.14 | 3.36 | 36.7 | 51.1 | none |
| base | fp16 | 9 | 46.9 | 1.23 | 64 | 1.25 | 1.95 | 21.3 | 29.6 | none |
| base | fp32 | 9 | 43 | 1.91 | 64 | 1.93 | 3.03 | 33.1 | 46 | none |
| small | bf16 | 12 | 63.6 | 1.29 | 1024 | 1.3 | 2.04 | 22.3 | 31 | none |
| small | fp16 | 12 | 63.6 | 0.885 | 1024 | 0.9 | 1.41 | 15.3 | 21.3 | none |
| small | fp32 | 12 | 55.8 | 1.3 | 64 | 1.32 | 2.06 | 22.5 | 31.2 | none |
| small | bf16 | 9 | 46.5 | 0.982 | 1024 | 0.996 | 1.56 | 17 | 23.7 | none |
| small | fp16 | 9 | 48.7 | 0.662 | 1024 | 0.677 | 1.05 | 11.5 | 16 | none |
| small | fp32 | 9 | 41.6 | 0.988 | 64 | 1.01 | 1.57 | 17.1 | 23.8 | none |
| tiny | bf16 | 12 | 63.7 | 0.651 | 256 | 0.667 | 1.03 | 11.3 | 15.7 | none |
| tiny | fp16 | 12 | 64.7 | 0.509 | 1024 | 0.524 | 0.809 | 8.83 | 12.3 | full NR |
| tiny | fp32 | 12 | 55.6 | 0.649 | 256 | 0.664 | 1.03 | 11.2 | 15.6 | none |
| tiny | bf16 | 9 | 47.2 | 0.476 | 1024 | 0.491 | 0.755 | 8.25 | 11.5 | full NR |
| tiny | fp16 | 9 | 47.5 | 0.381 | 1024 | 0.396 | 0.605 | 6.61 | 9.18 | full NR |
| tiny | fp32 | 9 | 41.9 | 0.473 | 1024 | 0.488 | 0.751 | 8.2 | 11.4 | full NR |

CPU calibration: a sparse LU solve of the IEEE 118 Jacobian takes 0.3301 ms on this runtime against 0.0641 ms on the M4 (factor 5.15). The classical solvers would be about 5.15x slower here as well, so the thresholds scale to roughly full NR 1.58, compensated chord 0.189, FDLF 0.0772 ms (paper set).

## case500_goc

No break-even thresholds exist for this grid (the contingency study is IEEE 118 only).

| model | dtype | layers | batch 1 (ms) | best (ms/scenario) | at batch | best + h2d (ms) |
|---|---|---|---|---|---|---|
| base | bf16 | 12 | 63.7 | 12.6 | 256 | 12.7 |
| base | fp16 | 12 | 68.3 | 6.7 | 64 | 6.76 |
| base | fp32 | 12 | 56.3 | 12 | 16 | 12.1 |
| base | bf16 | 9 | 48.1 | 9.13 | 64 | 9.19 |
| base | fp16 | 9 | 51.8 | 4.98 | 64 | 5.04 |
| base | fp32 | 9 | 42.4 | 8.72 | 64 | 8.78 |
| small | bf16 | 12 | 63.6 | 5.1 | 256 | 5.15 |
| small | fp16 | 12 | 62.4 | 3.47 | 256 | 3.52 |
| small | fp32 | 12 | 61 | 5.06 | 16 | 5.16 |
| small | bf16 | 9 | 50.3 | 3.9 | 256 | 3.95 |
| small | fp16 | 9 | 47.3 | 2.61 | 256 | 2.66 |
| small | fp32 | 9 | 44.6 | 3.85 | 16 | 3.95 |
| tiny | bf16 | 12 | 62.7 | 2.58 | 64 | 2.64 |
| tiny | fp16 | 12 | 63.3 | 2 | 1024 | 2.05 |
| tiny | fp32 | 12 | 55.3 | 2.55 | 1024 | 2.6 |
| tiny | bf16 | 9 | 47.1 | 1.89 | 256 | 1.94 |
| tiny | fp16 | 9 | 48.6 | 1.5 | 1024 | 1.55 |
| tiny | fp32 | 9 | 41.7 | 1.86 | 256 | 1.91 |

## case2000_goc

No break-even thresholds exist for this grid (the contingency study is IEEE 118 only).

| model | dtype | layers | batch 1 (ms) | best (ms/scenario) | at batch | best + h2d (ms) |
|---|---|---|---|---|---|---|
| small | bf16 | 12 | 67.7 | 19.5 | 64 | 19.7 |
| small | fp16 | 12 | 66.9 | 13.4 | 64 | 13.7 |
| small | fp32 | 12 | 59.1 | 19.5 | 16 | 19.7 |
| small | bf16 | 9 | 51.5 | 15.2 | 64 | 15.4 |
| small | fp16 | 9 | 51.5 | 10.2 | 64 | 10.4 |
| small | fp32 | 9 | 45.7 | 15.2 | 64 | 15.4 |
| tiny | bf16 | 12 | 62.2 | 9.84 | 16 | 10.1 |
| tiny | fp16 | 12 | 63.9 | 7.75 | 256 | 7.98 |
| tiny | fp32 | 12 | 55.2 | 9.88 | 16 | 10.1 |
| tiny | bf16 | 9 | 49.2 | 7.36 | 64 | 7.58 |
| tiny | fp16 | 9 | 48.1 | 5.83 | 256 | 6.07 |
| tiny | fp32 | 9 | 41.9 | 7.38 | 64 | 7.6 |

## Batched dense Newton step (IEEE 118, linear solve only)

Optional comparison: `torch.linalg.solve` on stacked 181 x 181 Jacobians versus one sparse LU on the host CPU. The Jacobian assembly and the remaining Newton iterations are not included.

| device | dtype | batch | ms_per_scenario | reps | status | note |
|---|---|---|---|---|---|---|
| host CPU | float64 sparse LU (scipy) | 1 | 0.33 | 48 | ok | M4 reference 0.0641 ms -> this host is 5.1x the M4 time |
| Tesla T4 | float64 | 1 | 1.77 | 15 | ok | n=181; max relative residual 9.1e-16 |
| Tesla T4 | float64 | 16 | - | - | error: AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelIm | - |
| Tesla T4 | float64 | 64 | - | - | error: AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelIm | - |
| Tesla T4 | float64 | 256 | - | - | error: AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelIm | - |
| Tesla T4 | float32 | 1 | 0.966 | 15 | ok | n=181; max relative residual 3.4e-07 |

## Precision check

Largest and median absolute deviation of the predicted bus Vm (p.u.) and Va (rad) from the fp32 prediction of the same model, on 16 scenarios. For scale: GENCO's median state error against the datakit solution is about 0.05 (IEEE 118, Base).

| grid | size | dtype | layers | max_abs_dev_vm | max_abs_dev_va_rad | median_abs_dev_vm | median_abs_dev_va_rad | nonfinite | status |
|---|---|---|---|---|---|---|---|---|---|
| case118_ieee | base | bf16 | 12 | 0.0309 | 0.0731 | 0.000329 | 0.0153 | 0 | ok |
| case118_ieee | base | fp16 | 12 | 0.00291 | 0.00645 | 5.48e-05 | 0.00123 | 0 | ok |
| case118_ieee | base | bf16 | 9 | 0.0328 | 0.106 | 0.000453 | 0.0102 | 0 | ok |
| case118_ieee | base | fp16 | 9 | 0.0031 | 0.00673 | 6.06e-05 | 0.00118 | 0 | ok |
| case118_ieee | small | bf16 | 12 | 0.0263 | 0.0598 | 0.000326 | 0.00912 | 0 | ok |
| case118_ieee | small | fp16 | 12 | 0.00516 | 0.00426 | 4.55e-05 | 0.0012 | 0 | ok |
| case118_ieee | small | bf16 | 9 | 0.0306 | 0.0523 | 0.000334 | 0.00715 | 0 | ok |
| case118_ieee | small | fp16 | 9 | 0.0044 | 0.00417 | 5.16e-05 | 0.00121 | 0 | ok |
| case118_ieee | tiny | bf16 | 12 | 0.061 | 0.151 | 0.000313 | 0.0305 | 0 | ok |
| case118_ieee | tiny | fp16 | 12 | 0.01 | 0.00939 | 5.3e-05 | 0.00164 | 0 | ok |
| case118_ieee | tiny | bf16 | 9 | 0.0619 | 0.146 | 0.000315 | 0.0254 | 0 | ok |
| case118_ieee | tiny | fp16 | 9 | 0.0123 | 0.01 | 5.22e-05 | 0.00161 | 0 | ok |
| case500_goc | base | bf16 | 12 | 0.0574 | 0.52 | 0.00241 | 0.108 | 0 | ok |
| case500_goc | base | fp16 | 12 | 0.00345 | 0.0594 | 0.000254 | 0.00555 | 0 | ok |
| case500_goc | base | bf16 | 9 | 0.02 | 0.0775 | 0.00191 | 0.015 | 0 | ok |
| case500_goc | base | fp16 | 9 | 0.00403 | 0.00766 | 0.000271 | 0.000778 | 0 | ok |
| case500_goc | small | bf16 | 12 | 0.104 | 0.404 | 0.0103 | 0.125 | 0 | ok |
| case500_goc | small | fp16 | 12 | 0.0102 | 0.0747 | 0.000696 | 0.0152 | 0 | ok |
| case500_goc | small | bf16 | 9 | 0.0422 | 0.118 | 0.0068 | 0.0309 | 0 | ok |
| case500_goc | small | fp16 | 9 | 0.00824 | 0.0306 | 0.000437 | 0.00384 | 0 | ok |
| case500_goc | tiny | bf16 | 12 | 0.358 | 0.562 | 0.00624 | 0.0649 | 0 | ok |
| case500_goc | tiny | fp16 | 12 | 0.017 | 0.381 | 0.000551 | 0.0185 | 0 | ok |
| case500_goc | tiny | bf16 | 9 | 0.102 | 0.273 | 0.00407 | 0.0118 | 0 | ok |
| case500_goc | tiny | fp16 | 9 | 0.0118 | 0.0489 | 0.000385 | 0.00281 | 0 | ok |
| case2000_goc | small | bf16 | 12 | 0.132 | 0.282 | 0.00483 | 0.0411 | 0 | ok |
| case2000_goc | small | fp16 | 12 | 0.00929 | 0.0411 | 0.00085 | 0.00222 | 0 | ok |
| case2000_goc | small | bf16 | 9 | 0.063 | 0.197 | 0.00492 | 0.0179 | 0 | ok |
| case2000_goc | small | fp16 | 9 | 0.00971 | 0.0138 | 0.000876 | 0.0016 | 0 | ok |
| case2000_goc | tiny | bf16 | 12 | 0.0618 | 0.127 | 0.0115 | 0.0231 | 0 | ok |
| case2000_goc | tiny | fp16 | 12 | 0.0136 | 0.0331 | 0.00141 | 0.0034 | 0 | ok |
| case2000_goc | tiny | bf16 | 9 | 0.061 | 0.13 | 0.00873 | 0.0224 | 0 | ok |
| case2000_goc | tiny | fp16 | 9 | 0.0124 | 0.0302 | 0.00112 | 0.00334 | 0 | ok |

## Orientation: the same forward pass on the Apple M4

ms per scenario, from `results/runtime/*.json`.

* case118_ieee/base: CPU, batch 16 12.13, MPS, batch 128 10.31, MPS, batch 128, 9 layers 7.67
* case2000_goc/small: CPU, batch 16 102.78, MPS, batch 64 100.33, MPS, batch 64, 9 layers 71.04

## Configurations that did not run

| grid | size | dtype | batch | layers | status |
|---|---|---|---|---|---|
| case500_goc | base | fp32 | 1024 | 12 | OOM |
| case500_goc | base | bf16 | 1024 | 12 | OOM |
| case500_goc | base | fp16 | 1024 | 12 | OOM |
| case500_goc | base | fp32 | 1024 | 9 | OOM |
| case500_goc | base | bf16 | 1024 | 9 | OOM |
| case500_goc | base | fp16 | 1024 | 9 | OOM |
| case2000_goc | small | - | 1024 | 12 | skipped: 1024 x 2000 buses > MAX_BUSES_PER_BATCH |
| case2000_goc | tiny | - | 1024 | 12 | skipped: 1024 x 2000 buses > MAX_BUSES_PER_BATCH |

## Caveats

* The thresholds come from a Python/scipy implementation on one M4 thread; a compiled solver on a server CPU would be faster, so the real break-even is lower.
* Early exit after 9 layers costs about +0.1 Newton iteration (RESEARCH_LOG), which shrinks the saving it is compared with.
* Batch 1 is latency (kernel launches and Python overhead, so it depends on the host CPU as much as on the GPU); large batches measure throughput.
* Scenarios are repeated cyclically when a batch is larger than the pool; the forward pass does not depend on the values.
* Free-tier GPUs are shared and clocks vary between sessions: repeat the run before quoting a number.
* Reduced precision is only useful if the prediction still works as a Newton start; the precision table shows the deviation, not the effect on Newton iterations.
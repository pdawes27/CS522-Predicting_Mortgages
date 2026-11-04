# Controlled training-data-size experiment

Protocol recorded before the full fits.

Compare 200,000 and 1,000,000 training windows from the locally prepared 2016–2025 sample cohorts. Each input uses 12 consecutive monthly observations to predict the next month's seven-class state. Loan-level partitions remain disjoint; training targets end in 2022, validation targets are in 2023, and test targets start in 2024. Use the same 100,000 validation targets and all eligible test targets in every run.

Preserve the existing 200,000-window sample (sampling seed 522). The larger sample includes every smaller-sample window, plus 800,000 uniformly shuffled remaining windows (seed 523). Model seeds are 522, 523, and 524, paired across data sizes. Fit preprocessing separately on each training subset, never on validation or test data. Natural class frequencies are retained.

The ordinary transformer uses hidden size 64, four heads, two layers, batch size 512, AdamW learning rate 0.001 and weight decay 0.0001. Each fit receives exactly 6,000 full-minibatch optimizer updates (3,072,000 example exposures), validating every 400 updates. Shuffle again when a training epoch ends; discard incomplete minibatches. No early stopping or adaptive learning-rate changes. This isolates data diversity at a fixed update budget; it does not claim that either data size has converged or is optimally tuned.

XGBoost uses 200 boosting rounds, depth four, learning rate 0.05, row and column subsampling 0.8, histogram training on CUDA, four CPU threads. Both data sizes receive the same round budget, which is not a matched wall-time budget. Record validation log loss and macro-F1 every round without early stopping.

For each fitted model save and evaluate two checkpoints: minimum validation multiclass log loss and maximum validation seven-class macro-F1. Ties retain the earlier checkpoint (transformer improvement tolerances 0.00001 loss and 0.00000001 F1). Report both criteria; do not choose the better test checkpoint. Persistence copies the last observed state and is a fixed control.

Assess macro-F1 (higher better), multiclass log loss (lower better), all class precision/recall/F1/support, and accuracy as a secondary measure. Report all seeds, paired changes, year breakdowns, training curves, selected updates/rounds, hardware and wall time. No payoff-focused endpoint or selective reporting. All test sets have already been inspected in earlier exploratory experiments; this is not independent confirmation. Rare classes and overlapping months within loans limit inference. Three initialization seeds measure optimization variability, not population uncertainty.

Reproduce with the CUDA environment and GPU access:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python analysis/run_data_size_experiment.py
MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python -m analysis.compare_data_sizes
```

Results from this 12-month-eligible population should be compared within this study. Earlier context-length experiments required 24 months of eligibility and used a different test population and training schedule.

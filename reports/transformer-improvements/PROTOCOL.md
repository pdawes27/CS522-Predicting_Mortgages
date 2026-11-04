# Transformer optimization, transition residuals, state embeddings, and calibration

Protocol recorded before the new full fits on November 3, 2026.

Use exactly the previous million-window training sample, 100,000 checkpoint-selection
validation targets, and 1,009,313 test targets. Cohorts 2016–2025, 12-month inputs
and eligibility, loan-disjoint partitions, training targets through 2022, validation
in 2023, test from 2024 onward. Retain sampling seed 522 and the nested 200k anchor.
Use paired model seeds 522, 523, 524. Preprocessing uses training inputs only.
Each transformer receives 6,000 updates, batch size 512, validation every 400 updates,
hidden size 64, four heads, two layers, dropout 0.1, AdamW weight decay 0.0001,
and gradient clipping at 1. No early stopping in fixed-budget runs.

## Stage 1: training recipes

Compare the existing constant learning rate 0.001 with constant 0.0003 and
warmup/cosine with peak 0.001. Warmup lasts 300 updates; cosine decay finishes
at 5% of peak. Reuse the prior completed million-window ordinary-transformer runs
as the constant-0.001 control, after verifying population signatures and settings.
Run the two new recipes for all three seeds. Report both log-loss-selected and
macro-F1-selected checkpoints for every arm. Select the recipe for Stage 2 solely
by mean checkpoint-selection validation log loss across the three log-loss-selected
checkpoints. Break ties by the listed order. Do not use test metrics for selection.

## Stage 2: separate architecture ablations

At the selected recipe fit two separate variants, each with all three seeds:

1. Transition-residual transformer: estimate the seven-by-seven transition matrix
   from training labels and current states only, adding 0.5 to every count. Predict
   `softmax(log(training_transition[current_state]) + transformer_logits)`.
   Zero-initialize the final correction layer so initial predictions equal the
   transition baseline. The prior is a fixed saved buffer; the network learns
   unconstrained corrections. This initialization is part of the proposed method,
   not a separately isolated factor. No test-frequency adjustment.
2. State-embedding transformer: add a learned four-state embedding to every
   monthly token while retaining the existing numeric delinquency feature.
   Use observed input states only. No residual prior in this ablation.

Also evaluate the training-only transition matrix alone and fit XGBoost with
four explicit state indicators per input month for three seeds, to control for the
embedding model's explicit state representation. Existing XGBoost remains the
reference, with the same 200 rounds and settings. Do not combine modifications
based on the test results.

## Stage 3: calibration diagnostic

For every reported transformer checkpoint and both XGBoost feature variants,
fit a single positive temperature by minimizing multiclass log loss on validation
targets excluded from checkpoint selection. Use all remaining 196,496 validation
windows. No training or test labels enter temperature fitting. These windows are
distinct targets but may share validation loans and input history with selection
targets; calibration is exploratory and does not create an independent validation
population. A strict loan-disjoint calibration subset would be selected unevenly
by the original window sample, so it is not used as a representative calibrator.

Optimize log-temperature within [-log(10), log(10)] using bounded scalar search,
including temperature 1 as a candidate to avoid increasing calibration loss.
Reconstruct scores as log of clipped probabilities, using the same operation
before and after calibration, and report clipping. Temperature scaling preserves
argmax predictions, so macro-F1 must stay unchanged. Record before/after log loss,
multiclass Brier score, temperature, and class metrics. Save temperatures; do not
claim calibration fixes temporal distribution shift or rare-event support.

## Evaluation and interpretation

Report all seeds, both checkpoint rules, macro-F1, multiclass log loss, accuracy,
all class precision/recall/F1/support, prediction-year metrics, fitted updates,
selected checkpoints, curves, runtimes, and GPU verification. Changes are paired
against the ordinary transformer with the same recipe. Retain persistence and
both XGBoost feature variants as controls. No payoff-focused primary endpoint.
Calibration is reported separately from architecture/training changes.

Test data have already been inspected. All results remain exploratory; three
model seeds describe limited optimization variability, not population uncertainty.
No improvement, architectural novelty, or broad superiority is presumed.

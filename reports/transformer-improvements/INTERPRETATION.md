# What improved, and what did not

Prepared from the completed, predeclared experiment.
See [all results](RESULTS.md), [protocol](PROTOCOL.md), and
[validation-only recipe selection](recipe_selection.json).

**Explicit state embeddings produce a modest, repeatable classification gain.
Scalar temperature calibration improves probability scores for every evaluated
checkpoint. Neither finding establishes overall transformer superiority to XGBoost.**

## Training recipes

The original constant learning rate 0.001 had the lowest mean checkpoint-selection
validation log loss: 0.07005, versus 0.07032 at constant 0.0003 and 0.07062 with
warmup/cosine. Architecture experiments therefore retained constant 0.001.

Cosine decay improved mean test macro-F1 and log loss under both checkpoint rules,
but the effects varied by seed. With log-loss checkpoint selection, only one seed
improved test log loss. The lower constant rate did not improve overall results.
The test averages do not override the predeclared validation-based recipe choice.

## Architecture experiments

Means across three paired seeds on the same 1,009,313 test targets:

| Model | Checkpoint selection | Macro-F1 ↑ | Raw log loss ↓ | Calibrated log loss ↓ |
|---|---|---:|---:|---:|
| Ordinary transformer | Validation log loss | 0.3437 | 0.09490 | 0.09072 |
| Transition residual | Validation log loss | 0.3402 | 0.09645 | 0.09067 |
| State embeddings | Validation log loss | 0.3493 | 0.09519 | 0.09095 |
| Ordinary transformer | Validation macro-F1 | 0.3625 | 0.10235 | 0.09423 |
| Transition residual | Validation macro-F1 | 0.3678 | 0.09955 | 0.09186 |
| State embeddings | Validation macro-F1 | 0.3700 | 0.09812 | 0.09190 |

State embeddings increase macro-F1 in all three seeds under both selection rules.
With macro-F1 selection, paired changes are +0.00512, +0.00891, and +0.00852.
The corresponding scores are 0.3685, 0.3712, and 0.3703, each above persistence's
0.3678. Mean macro-F1 also improves in each test calendar year.

The benefit comes mainly from delinquency classes. Mean 30–59-day delinquency F1
rises 0.3738 → 0.4029, and 60–89-day F1 rises 0.3073 → 0.3263. Their recalls rise
0.2788 → 0.3099 and 0.3321 → 0.3616, respectively. Adverse and other exits remain
poorly supported and have zero F1. These are improvements to the existing model,
not a claim that categorical state embeddings are a novel method.

Probability results are less consistent: uncalibrated log loss improves in only
one seed under either selection rule. The mean macro-F1-selected loss improvement
is largely the removal of the ordinary transformer's poor seed-523 result.
Log-loss-selected state embeddings slightly worsen mean log loss.

The transition-residual variant increases macro-F1 in all three seeds with
macro-F1 selection, but raw log loss improves in only one. With log-loss selection,
both mean macro-F1 and mean raw log loss get worse. This does not support it as a
general replacement. Its initialization and prior are tested together.

## Calibration and baseline fairness

One temperature per checkpoint was fitted only on 196,496 validation targets
excluded from checkpoint selection. All **42 of 42** checkpoints improved test
log loss and multiclass Brier score; predicted classes and macro-F1 were unchanged.
This includes both XGBoost feature variants, not just the transformers. Temperatures
are below one in these runs, sharpening the distributions. Improved proper scores
do not establish perfect calibration for every class or future economic period.

The explicit-state XGBoost control reaches macro-F1 0.3686 and calibrated log loss
0.08932 with macro-F1 selection. State embeddings reach slightly higher mean
macro-F1 (0.3700) but worse calibrated log loss (0.09190). The original XGBoost's
corresponding calibrated loss is 0.08893, at macro-F1 0.3642. Overall superiority
still depends on the classification/probability tradeoff.

## Presentation claim and limits

A supported statement is: **representing observed delinquency states explicitly
improved transformer macro-F1 across three seeds, especially for moderate
delinquency; a separate validation-fitted calibration step improved probability
scores for both transformers and XGBoost.** Report the baseline tradeoff alongside
that statement. The residual prior and learning-rate changes were not reliable
overall improvements.

There were 12 new transformer fits and three new XGBoost fits, all on the RTX 5070,
plus reused controls. New full runs took 67–178 seconds and peaked at 5.31 GiB
process RSS. Every transformer used exactly 6,000 updates and 3,072,000 example
exposures. The complete test suite passed all 32 tests; GPU smoke fits and full
artifact checks also passed.

Three seeds measure limited optimization variability. Calibration targets can
share validation loans and input history with selection targets, and the test set
has already been inspected in earlier experiments. No test-selected combination
of state embeddings and cosine decay was fitted. Untouched future data would be
needed for independent confirmation. The optional variants remain available for
reproduction and further validation.

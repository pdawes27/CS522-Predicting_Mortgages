# Interpretation of the controlled data-size experiment

Prepared from the completed six-run comparison. Full results,
all class scores, and the protocol are in [RESULTS.md](RESULTS.md).

Increasing training data helps, but it does not establish a broad transformer
advantage. XGBoost improves both overall macro-F1 and multiclass log loss in all
three seeds under both checkpoint-selection rules. The transformer improves mean
macro-F1 while mean log loss gets worse. No test-based checkpoint selection was used.

| Model / validation checkpoint | Macro-F1, 200k → 1m | Log loss, 200k → 1m |
|---|---:|---:|
| XGBoost / log loss | 0.3545 → 0.3651 | 0.10218 → 0.09678 |
| XGBoost / macro-F1 | 0.3538 → 0.3642 | 0.10192 → 0.09629 |
| Transformer / log loss | 0.3349 → 0.3437 | 0.09387 → 0.09490 |
| Transformer / macro-F1 | 0.3445 → 0.3625 | 0.10022 → 0.10235 |

Higher macro-F1 and lower log loss are better. These are means over three paired
model seeds on the same 1,009,313 held-out targets. Persistence remains at macro-F1
0.3678; it does not supply probabilistic log loss. At one million examples, the
log-loss-selected transformer still has lower mean log loss than the equivalently
selected XGBoost, but lower macro-F1. The macro-F1-selected transformer is below
XGBoost on both mean endpoints. There is no overall transformer win here.

The log-loss-selected transformer's macro-F1 rises substantially in seed 522,
is nearly unchanged in seed 524, and falls in seed 523. Its test log loss
worsens in all three. Macro-F1 selection produces a more consistent classification
gain: 0.3426 → 0.3634, 0.3510 → 0.3623, and 0.3399 → 0.3618. However, its log loss
improves in two seeds and deteriorates sharply in the third (0.09500 → 0.11626),
so the mean worsens. Selecting the best test checkpoint would hide this tradeoff.

The clearest class-level improvement is moderate delinquency. With macro-F1
selection, transformer 60–89-day delinquency F1 rises from 0.2189 to 0.3073 and
recall from 0.1752 to 0.3321. Its 30–59-day F1 rises from 0.3474 to 0.3738.
Mean macro-F1 improves in each test calendar year; the probability-loss issue also
persists in the year-level means. Adverse and other exits remain at zero F1 for
every model. Larger samples raise adverse-exit training support from four to 17,
with zero adverse-exit validation cases and only 35 test cases. More windows alone
have not solved rare-event prediction.

Each transformer fit used 6,000 optimizer updates and exactly 3,072,000 example
exposures, finishing in 16 epochs at 200k and four epochs at 1m. Each XGBoost fit
used 200 rounds. Entire runs took 78–127 seconds and peaked below 4.9 GiB process
RSS. Both fitted model families used CUDA on the RTX 5070. Runtime includes
preprocessing, fitting, and evaluation; checkpoint variants share a fit.

A defensible presentation message is: **more training diversity improves
delinquency classification, but probability quality and checkpoint stability remain
limitations for the transformer; XGBoost benefits more consistently overall.**
These exploratory results do not establish a novel architectural contribution.

The next focused experiment would keep the 1m dataset and examine optimization
stability and learning-rate scheduling using validation curves, with both metrics
reported. Temporal probability calibration is another reasonable validation-only
diagnostic. This study does not yet justify another architecture change or a full
licensed-data download. Any final confirmation needs untouched future data because
the present test population has already been inspected.

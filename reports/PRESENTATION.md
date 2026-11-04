# Mortgage outcome prediction

Draft presentation outline.

## 1. Question

Does a small transformer improve next-month mortgage-state prediction over simple
baselines? Can a state-only variant generate useful multi-month trajectories?

## 2. Data

- Freddie Mac official 2016–2025 sample cohorts: 500,000 loans.
- 20,927,667 monthly records; 15,170,390 eligible 12-month windows.
- Cohort years describe origination, with performance through June 2026.
- Four delinquency buckets plus payoff, adverse exit/REO, and other exit.
- Adverse exits are extremely rare: 487 eligible windows across all cohorts.

Suggested figure: [class distribution](class_distribution.png).

## 3. Evaluation design

- Stable 70/15/15 loan-level assignments; no loan overlap.
- Training-only preprocessing and 12 consecutive observed input months.
- No future losses or disposition information in features.
- Compare models on identical examples and report rare-class support.
- Separate future-period evaluation: train through 2022, validate in 2023, test later.
- Training capped at 200,000 uniformly sampled windows per experiment.

## 4. Models

- Persistence and a smoothed state-transition table.
- XGBoost with flattened history and origination attributes.
- Transformer: two encoder layers, four attention heads, hidden size 64.
- Ordinary/weighted training and a last-month-only ablation.

## 5. One-month results

| Model | 2018 test macro-F1 |
|---|---:|
| Persistence | 0.3476 |
| XGBoost | 0.3664 |
| Transformer | 0.3651 |
| Weighted transformer | 0.3669 |
| Last-month-only transformer | 0.3622 |

Weighting worsened log loss. The 12-month history improvement was small. One seed
and six adverse-exit test examples prevent strong claims about differences or
default detection. See the report for the expansion and future-period results.

## 6. Focused result: next-month payoff ranking

On the future-period test, the transformer ranks next-month payoff/maturity events
better than the current XGBoost baseline.

| Measure | XGBoost | Transformer |
|---|---:|---:|
| Payoff average precision | 0.01832 | 0.04490 |
| Actual payoffs in the top 1% of ranked loan-months | 318 | 605 |
| Precision at that selection budget | 3.15% | 5.99% |

Both select 10,093 of 1,009,313 loan-months. The test contains 6,776 payoff events.
The transformer has higher payoff AP separately in 2024, 2025, and January–June
2026. The paired whole-loan bootstrap interval for the AP difference is
[0.02320, 0.03046]. This is exploratory evidence from one model seed; it does not
include tuning uncertainty or independently confirm a post-hoc selected endpoint.
Raw payoff probabilities overestimate observed rates, so the demonstrated gain
is in ranking. See [error analysis](ERROR_ANALYSIS.md) for the limitations.

Suggested figure: [payoff ranking advantage](diagnostics/payoff_advantage.png).

## 7. Architecture ablation: static-conditioned attention

Add one projected origination-profile token to the 12 monthly tokens. The encoder
can attend to loan attributes, while retaining the original classifier connection.
The addition is optional and adds 1,344 parameters (about 1.8%).

Three paired seeds use the same training windows, validation data, test population,
and optimization settings:

| Model | Mean payoff average precision |
|---|---:|
| Original transformer | 0.04188 |
| Static-context token | 0.03794 |

Conditioning wins in one of three seeds; it does not establish an improvement.
The experiment is exploratory because the test data had been inspected previously.
Keep the original architecture as the reference.

Suggested figure: [paired architecture comparison](static-conditioning/payoff_ap_by_seed.png).

A separate full GPU repeat verified RTX 5070 training. Epochs took approximately
3–5 seconds instead of 30–36 seconds in the CPU runs; timings include validation
and are descriptive rather than an isolated hardware benchmark.

## 8. Context ablation: 12 versus 24 past months

Use the same 809,831 prediction targets, requiring 24 months of observed history
in both arms. Model inputs use either the last 12 or all 24 months. Three GPU
seeds, identical sampled targets, and the same training budget:

| Model | Payoff AP, 12 months | Payoff AP, 24 months |
|---|---:|---:|
| Ordinary transformer | 0.04160 | 0.04206 |
| Static-conditioned transformer | 0.03810 | 0.03408 |
| XGBoost | 0.02198 | 0.02151 |

The ordinary transformer improves in two of three seeds; the small mean gain is
not a consistent improvement. The conditioned variant worsens in all three.
The ordinary transformer retains its focused payoff-ranking advantage over the
current XGBoost baseline. This is exploratory evidence on inspected data.

XGBoost's lag inputs grow from 116 to 212 columns and require refitting. Persistence
uses only the latest state; its predictions and scores are exactly unchanged.
The target remains next month, so this is a past-context experiment rather than
a longer forecast horizon.

Suggested figure: [matched context comparison](context-length/payoff_ap_by_context.png).

## 9. Autoregressive generation

Generate one mortgage-state token, append it, and repeat with a rolling context.
Use fixed origination attributes and state history so future balances/rates are
not required. Stop at terminal states. Sample many paths to summarize uncertainty.

The generation loop is analogous to next-token generation; the model is a small
specialized transformer, not a pretrained language model.

## 10. Multi-month results

Backtest: 1,000 held-out origins, 128 paths each, horizons 1–12 months.

| Model | 12-month Brier score | 12-month macro-F1 |
|---|---:|---:|
| Autoregressive | 0.38094 | 0.1239 |
| Transition | 0.39612 | 0.1239 |
| Persistence | 0.46525 | 0.2049 |

Lower Brier is better. Only 705 origins have observable 12-month outcomes or known
earlier exits. Probability estimates improve over these baselines on this sample;
hard state predictions do not consistently improve over persistence.

Suggested figure: [performance by horizon](autoregressive_horizon_performance.png).

## 11. Controlled data-size comparison

Follow-up: three paired GPU seeds, nested 200k/1m training windows,
identical validation/test populations, and equal transformer optimizer budgets.
Both validation checkpoint criteria are reported in the
[full interpretation](data-size/INTERPRETATION.md).

With log-loss checkpoint selection, XGBoost macro-F1 increases 0.3545 → 0.3651 and
log loss improves 0.10218 → 0.09678. Transformer macro-F1 increases
0.3349 → 0.3437, while log loss worsens 0.09387 → 0.09490.
With macro-F1 selection, transformer macro-F1 increases 0.3445 → 0.3625, but mean
log loss also worsens. Its clearest class improvement is 60–89-day delinquency.

Presentation claim: more data helps delinquency classification; it does not
establish a broad transformer win. XGBoost benefits more consistently across both
overall metrics. See [the overall metric figure](data-size/overall_performance.png).

## 12. State representation and calibration follow-up

Experiment: three paired seeds, one million training windows,
6,000 updates per transformer fit, and both validation checkpoint criteria.
The original constant learning rate won the validation-only recipe selection.

Adding observed delinquency-state embeddings increased transformer macro-F1 in
all three seeds under both checkpoint rules. With macro-F1 selection, the mean
increased **0.3625 → 0.3700**; 30–59-day delinquency F1 increased
0.3738 → 0.4029 and 60–89-day F1 increased 0.3073 → 0.3263.
Raw probability-loss improvements were inconsistent across seeds.

A separate temperature fitted on unused validation targets improved test log loss
and multiclass Brier for all 42 evaluated checkpoints, including both XGBoost
feature variants, without changing predicted classes. State embeddings slightly
exceed the explicit-state XGBoost's mean macro-F1 (0.3700 versus 0.3686), while
calibrated XGBoost has lower log loss (0.08932 versus 0.09190).

Claim: explicit state representation improves delinquency classification, and
validation-fitted calibration helps probability quality; overall model choice
still involves a tradeoff. Residual outputs and learning-rate changes did not
establish a reliable overall improvement. See the
[complete interpretation](transformer-improvements/INTERPRETATION.md).

## 13. Takeaways and limitations

- A simple transformer is enough to complete a meaningful comparison.
- Overall state classification did not establish superiority over the baselines.
- Payoff ranking showed a tangible advantage at the same selection budget.
- Recursive generation works, but errors and omitted future conditions limit it.
- Rare events, censoring, correlated windows, cohort sampling, and single-seed
  initial comparisons constrain interpretation. The architecture ablation repeats
  three paired seeds.
- Three paired seeds and larger training samples have now been tested; results
  show classification/probability tradeoffs and continued rare-event limitations.
- Commands, checkpoints, metrics, tests, and source fingerprints support reproduction.

Project authors should review the work and read the cited papers before submission.

# Would more training data help?

Assessment prepared from saved runs and recomputed training
pool counts. No data-volume learning curve has yet been fitted.

## What has actually been used

All recent comparisons capped training at 200,000 supervised target windows. The
existing official sample files contain substantially more eligible training data
before the December 2022 target cutoff:

| Context/eligibility | Available training windows | Used | Fraction used | Eligible training loans |
|---|---:|---:|---:|---:|
| 12 months | 4,560,776 | 200,000 | 4.39% | 187,745 |
| Common 24-month eligibility | 2,673,875 | 200,000 | 7.48% | 123,649 |

The ordinary 12-month future-period run's selected windows involve 104,272 loans
and 1,973,298 distinct monthly input rows. The common-eligibility run involves
82,480 loans. Fractions in the table refer to supervised targets, not independent
loans or raw input records. Overlapping windows share many observations.

The 12-month pool contains 87,870 payoff/maturity target events, compared with
3,830 in the capped training set. First-exit handling gives one such event per
loan. Using more existing windows can therefore add meaningful exit supervision,
not merely repeat unchanged current states. Adverse-exit/REO targets increase from
four in the capped run to 98 in the full available training pool; that is still a
small absolute number for a heterogeneous rare-outcome class.

Counts and provenance are saved in [data_scale_counts.json](data_scale_counts.json).

## What the previous experiments establish

They establish unused data headroom, not a measured benefit from increasing
training-set size. Adding cohorts under the same 200,000-window cap redistributed
the fixed training budget. Extending history increased input context per example
while keeping the number of supervised targets fixed. Neither is a data-volume
learning curve.

Training loss generally keeps falling while validation loss plateaus or fluctuates.
That is compatible with several explanations: limited sample generalization,
optimizer/checkpoint variability, missing predictive features, or calendar shift.
Training and validation cover different reporting periods and class frequencies,
and dropout is enabled during training, so their raw loss magnitudes should not
be treated as a directly comparable generalization gap.

The model has roughly 74,000 parameters. Language-model scaling results do not
establish a minimum data requirement or guaranteed advantage over boosting for
this mortgage encoder. [Tabular model benchmarking](https://arxiv.org/abs/2207.08815)
also motivates evaluating data and inductive biases rather than assuming a
transformer must eventually win with enough observations. That benchmark is not
direct evidence about this particular monthly-sequence dataset.

## Recommended next test

1. Keep the ordinary transformer at 12 months and hold the current validation/test
   identities, target definition, and date cutoffs fixed.
2. Compare 200,000, 1,000,000, and eventually all 4,560,776 available training
   windows from the existing sample files. Use nested subsets, and record distinct
   loans and outcome counts alongside window counts. A fixed sampling seed alone
   does not make different-sized `choice` samples nested.
3. Give XGBoost the same training examples. Keep persistence as a fixed control.
4. Evaluate validation payoff AP/log loss, natural-distribution calibration, and
   secondary delinquency outcomes over several model seeds. Track runtime and
   optimizer updates. Prefer both a matched-update comparison and sufficiently
   trained fits; fixed epochs alone increase compute with data size.
5. Fit all preprocessing on the selected training inputs only. Use validation for
   choices and treat the already inspected test data as exploratory evidence.

The CLI already accepts `--train-limit 1000000` and `--train-limit 0`; a carefully
controlled scaling driver still needs to implement nested sampling and training
budget accounting. The million-window stage is a sensible first GPU experiment.
Full-window XGBoost currently materializes flattened features in host memory, so
its memory cost should be measured before using the full pool.

Download full origination cohorts only if the learning curve continues improving
and more independent loans or rare events are needed. The full raw dataset would
require stronger streaming/filtering at ingestion; the current small-cohort parser
and in-memory training arrays should not be assumed to scale to the entire tape.

For the presentation, a successful next result would be a measured data-scaling
advantage: the transformer's held-out payoff ranking improves more with additional
training examples than the matched XGBoost baseline. That remains a hypothesis
until the experiment is run.

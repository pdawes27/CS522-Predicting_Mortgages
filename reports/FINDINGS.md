# Findings and next decisions

Prepared from the completed local runs. See
[REPORT.md](REPORT.md) for tables, class support, figures, and methods.

The simple transformer works, but these experiments do not establish an overall
state-classification advantage over XGBoost or persistence. Follow-up exploratory
diagnostics found a task-specific advantage in next-month payoff ranking; see
[the error analysis](ERROR_ANALYSIS.md) for matched-budget comparisons and
loan-cluster bootstrap intervals. This does not establish calibrated probabilities
or superior default prediction.

## One-month predictions

On the fixed 2018 test set of 227,392 windows, macro-F1 was 0.3476 for persistence,
0.3664 for XGBoost, and 0.3651 for the unweighted transformer. The weighted
transformer reached 0.3669, but its log loss was much worse (0.25028 versus 0.16914
for the ordinary transformer). That small F1 difference does not establish a useful
improvement. Only one model seed was run.

The last-month-only transformer scored 0.3622 versus 0.3651 with 12 months. This is
a small descriptive improvement for history; additional seeds would be needed
before making a robust claim. The ordinary transformer's terminal-event accuracy
was only 0.0069, and adverse exits had only six test examples. Most predictive
success came from current and seriously delinquent states, not payoff/default
event detection.

Expanding training to 2016–2020 under the same 200,000-window cap reduced test
macro-F1 to 0.3557 for XGBoost and 0.3434 for the transformer. The 2018 test
population was verified identical using ordered loan/month/label fingerprints.
The expanded run replaces some 2018 training windows with other cohorts; this
does not show that using all available multi-year training data would hurt.

In the separate calendar test, training targets ended in 2022, validation was in
2023, and test targets were after 2023, with disjoint loans. Across all downloaded
cohorts, test macro-F1 was 0.3678 for persistence, 0.3536 for XGBoost, and 0.3219
for the transformer. High accuracy (around 98%) did not imply good transition
detection. These scores use a different population and should not be directly
ranked against the random loan-held-out 2018 experiment.

## Multi-month generation

The separately trained state-only transformer can generate a monthly state,
append it to its rolling history, and continue. Payoff, adverse exit/REO, and other
exit states stop generation. No future balance or interest rate is supplied.
Both a greedy path and sampled alternatives are available.

The initial backtest used the 2018 cohort, the same 2022/2023 calendar cutoffs,
1,000 held-out loans with one random origin per loan, and 128 sampled paths per
origin. At 12 months, 705 origins had observable outcomes or a known earlier
terminal event. Brier scores were 0.38094 for autoregressive simulation, 0.39612
for the absorbing transition model, and 0.46525 for persistence; lower is better.
The autoregressive model also had lower Brier error at 1, 3, and 6 months.

Its most likely state did not beat persistence consistently: 12-month macro-F1
was 0.1239 versus 0.2049 for persistence. Sampled probability forecasts and hard
classification answer different questions. Generation is functional, but these
results do not establish reliable long-term default prediction. Changing observed
support across horizons, sampling error, omitted economic conditions, and limited
rare-event counts remain material limits.

## Transformer training and representation follow-up

The latest [optimization and architecture comparison](transformer-improvements/INTERPRETATION.md)
tested lower learning rates, warmup/cosine decay, transition-residual outputs,
state embeddings, explicit-state XGBoost controls, and calibration. It used three
paired seeds and recorded validation-only recipe selection.
State embeddings improved macro-F1 in every seed under both checkpoint rules;
with macro-F1 selection the mean rose 0.3625 → 0.3700. Raw probability-loss changes
were inconsistent. Temperature calibration improved test log loss and Brier scores
for every one of 42 reported checkpoints, including XGBoost. This establishes a
modest classification improvement and a useful calibration diagnostic, not overall
transformer superiority. See the linked interpretation for all tradeoffs and limits.

## Controlled training-data-size comparison

The subsequent [200k versus 1m experiment](data-size/INTERPRETATION.md) completed
three paired GPU seeds, nested training subsets, fixed transformer update budgets,
and both validation-log-loss and validation-macro-F1 checkpoint selection.
XGBoost improves both overall macro-F1 and log loss in all three seeds under both
selection rules. The transformer improves mean macro-F1 while mean log loss worsens.
This is evidence that data diversity helps classification, but does not establish
a broad transformer advantage. See the complete results for every class and seed.
This follow-up assesses overall performance; earlier payoff-focused
comparisons below remain historical diagnostics.

## Earlier follow-up experiments

The [matched context-length experiment](context-length/RESULTS.md) subsequently
compared 12 and 24 past months using three GPU seeds and the same 809,831 eligible
test targets. Ordinary transformer payoff AP changes only slightly, from 0.04160
to 0.04206, improving in two seeds and worsening in one. Static-conditioned payoff
AP falls from 0.03810 to 0.03408. XGBoost changes from 0.02198 to 0.02151; its
new-delinquency ranking improves slightly. Persistence is identical. Longer context
does not establish a reliable transformer improvement, while the ordinary model's
payoff-ranking advantage over the current XGBoost baseline remains visible across
these paired runs. The common-history population differs from earlier experiments.

The subsequent [static-conditioning experiment](static-conditioning/RESULTS.md)
tested three paired future-period seeds. Mean payoff AP was 0.04188 for the
original transformer and 0.03794 with the extra context token. Conditioning won
in one of three seeds, so this architecture addition has not established an
improvement. A separate [GPU repeat](static-conditioning/GPU.md) also favored the
original model on payoff ranking. The RTX 5070 is now configured and verified;
the original three-seed CPU comparison remains intact.

Keep the current architecture. The most useful optional next steps are repeated
seeds, training on more of the existing windows, and a deliberately defined
fixed-horizon delinquency target. Longer windows and crisis cohorts can follow if
the research question calls for them. Do not tune against the reported test sets
and then present those same scores as untouched final evaluation.

The release label/download date remain unconfirmed, and the project authors should
read the cited papers and review the report before academic submission.

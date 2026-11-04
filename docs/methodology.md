# Data and experiment methodology

This is an academic experiment, not a lending
decision system. All counts and results must come from the actual local runs.

## Sources and provenance

- [Freddie Mac dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
- [General user guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf)
- [File layout, July 2026](https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx)
- [File headers, July 2026](https://www.freddiemac.com/fmac-resources/research/docs/file_headers_july_2026.zip)
- [Release notes](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf)
- [Dataset FAQ](https://www.freddiemac.com/fmac-resources/research/pdf/faq.pdf)

Public documentation was consulted on November 2, 2026. The current documentation
identifies Release 48 (October 2026). The downloaded samples' release label was not
confirmed by the user; do not infer it solely from the filename or ZIP timestamp.
Local files have the July 2026-compatible layout: 31 origination columns and 35
performance columns, without headers. Earlier layouts are rejected. Direct public
reference-file downloads returned HTTP 403 in this environment; the links above
are retained instead of claiming the documents were saved locally.

Raw data was supplied by the user and present locally on November 2, 2026. The exact
download date is unconfirmed. Preparation saves the source name, size, row count,
column count, and SHA-256 checksum in each cohort's `summary.json`. Experiments
copy these summaries to `data_sources.json` so results identify their inputs.
Use `--release` and `--download-date` when those details are known. Keep a single
release across all years; refresh into a separate prepared directory if necessary.

The official sample has 50,000 randomly selected loans per full origination year.
Each cohort includes its available monthly history, not just that calendar year's
observations. Equal-size cohort samples are not a volume-weighted representation
of the complete portfolio across origination years.

## Target and eligibility

An example uses 12 consecutive observed reporting months ending at t to predict the
observed state at t+1. No padding or synthetic future rows are used. At least 13
consecutive records are required. Initial, short, gapped, and unknown-status
histories are excluded when they cannot provide a valid example. These exclusions
change the population being studied: early defaults and early payoffs may be absent.

| Class | Source definition |
|---|---|
| current | Delinquency code 00: current or less than 30 days late |
| 30_59_days | Delinquency code 01 |
| 60_89_days | Delinquency code 02 |
| 90_plus_days | Numeric delinquency codes 03 through 99 |
| payoff | Zero-balance code 01: voluntary payoff or maturity |
| adverse_exit_or_reo | REO acquisition status RA, or zero-balance code 02, 03, or 09 |
| other_exit | Other nonempty zero-balance codes, including 15, 16, and 96 |

Known zero-balance exits take precedence over the delinquency field. RA without a
zero-balance exit is the first observed REO event. The first exit/REO record may be
a target; it is never an input. Subsequent records for that loan are discarded,
including trailing loss/disposition updates. Thus we predict first adverse exit or
REO acquisition, not every later step of the foreclosure process. A loan can be
seriously delinquent without entering the adverse-exit class.

Unavailable or invalid delinquency status without a known exit is unknown (-1),
not current or default. An unknown target is excluded. A window containing unknown
or terminal input states is excluded. A missing next month at the disclosure cutoff
is right-censored and provides no next-month label. Never turn file endings into
outcomes. Effective-date/reporting-date differences are counted; labels remain in
the observed reporting month and are not moved backwards using information from a
later report. Freddie Mac cautions that monthly records combine reporting systems
and cycles; a historical disclosure snapshot cannot guarantee real-time availability
of every field or reproduce unrevised historical data vintages.

## Inputs and preprocessing

Monthly inputs: balance, interest rate, loan age, and numerical months delinquent.
Origination inputs: FICO, LTV, DTI, original balance, term, original interest rate,
purpose, and occupancy. Balances use `log1p`. Out-of-range numeric values and source
missing-value sentinels become missing. Numeric inputs use training-only median
imputation and standardization, with a missingness flag for each numeric feature.
Categorical inputs use a training-only vocabulary and one-hot encoding, including
an unknown category. Entirely missing training numeric columns use median zero.

Only the union of selected training input rows and selected training loans is used
to fit preprocessing. Validation/test inputs and target-only future rows are
excluded. The split is assigned before windows are constructed. Outcome codes,
loan identifiers, eventual proceeds, realized losses, and future disposition
details are never model features. IDs and reporting dates are retained for grouping,
window eligibility, and evaluation only.

Data is validated and prepared one cohort at a time. Duplicated loan IDs or
loan-month rows, unmatched performance records, invalid dates, or wrong column
counts fail preparation. Sorting, gaps, exit counts, unknown states, discarded
post-exit rows, missingness, history lengths, and transition counts are recorded.
Prepared data uses Parquet plus target-row indices; overlapping windows share the
same monthly arrays. An experiment currently loads selected prepared years into
memory. This is sufficient for these samples but is not a full-dataset streaming
training implementation.

## Splits and expansion

Blake2b hashing of seed and loan ID assigns approximately 70%/15%/15% to training,
validation, and test. Seed 522 is the default. Adding a cohort or changing file
order leaves existing loan assignments unchanged. Loan IDs are checked for overlap
across selected cohorts; there is no claim that borrowers with multiple loans can
be identified or separated. Saved `loan_splits.parquet` documents assignments.

The basic experiment evaluates unseen loans across the disclosed reporting years.
It does not measure forecasting into a wholly unseen future economic period.
For a separate future-period experiment, require both date cutoffs:

- Training targets on or before `--train-end` and training-assigned loans only.
- Validation targets after that cutoff through `--validation-end`, on validation loans.
- Test targets after the validation cutoff, on test loans.

Held-out loan input histories may precede those cutoffs; target dates must obey
them. Training/validation/test loans never overlap in either mode.

For fixed-cohort expansion comparisons, use `--test-cohorts 2018` and keep history,
split seed, sample seed, date filters, and test limit identical. The original
eligible 2018 test examples then stay fixed even though global row indices change.
With capped training, pooled expansion replaces some original training windows;
this is a comparison under a fixed training budget. Use `--train-limit 0` to train
on all eligible windows and study the effect of adding all available data.

Uniform window caps are drawn without replacement and independently of outcomes.
Validation/test caps, if used, are also uniform and never class-balanced. Caps
approximate the natural population; all-window evaluation uses limit zero. Long
loans contribute more windows and hence more weight. Windows within a loan are
correlated, so row counts do not represent independent statistical observations.

## Models and evaluation

- Persistence predicts the last observed state; it has no log-loss score.
- A smoothed training-only transition table provides a probabilistic baseline.
- XGBoost uses the same history and origination features, with a fixed seven-class
  softmax objective that permits absent training classes.
- The transformer uses linear monthly projections, learned positions, two encoder
  layers, four heads, hidden size 64, and a static-feature classification head.
  All tokens precede the prediction date, so full attention within the input window
  does not expose the target or future months.
- Optional square-root inverse-frequency class weighting uses training counts only.
  The unweighted and weighted variants use natural-distribution validation log loss
  for early stopping. No test-driven threshold tuning is performed.
- A one-month transformer ablation uses identical eligible target examples and
  origination features, while supplying only the last month to the encoder.

Use validation results for model choices; checkpoints minimize validation log loss.
Report fixed-seven-class macro-F1, supported-class macro-F1, per-class
precision/recall/F1/support, accuracy, confusion matrices, and probabilistic log
loss. Fixed-class macro-F1 assigns zero to absent classes. Supported macro-F1
excludes classes absent in the actual evaluation labels, so it is not comparable
across changing test populations without checking support. State transitions are
summarized as unchanged, worsening, recovery, and terminal, and performance is
also grouped by prediction year and origination cohort.

Rare adverse exits may have too few examples for reliable comparisons, especially
under capped training. Do not interpret a high overall accuracy as good detection
of these events. Do not remove crisis periods to improve scores. Pre-crisis/crisis
cohorts are an optional expansion requiring additional downloads.

Keep model comparisons on the same test set. Repeated seeds assess optimization
variability, not all sources of uncertainty. Any confidence intervals should
resample loans, not overlapping monthly windows.

## Reproduction

Each run saves settings, input fingerprints, environment versions, selected window
indices, loan splits, preprocessing, model checkpoints, learning curves, predictions,
metrics, figures, and resource measurements. Runs and data remain local and ignored
by version control. Consult the current Freddie Mac terms before redistributing
data or derived products; project code and aggregate reporting can be shared as
permitted. Check aggregate tables and figures before publication.

## Context-length ablation

The later 12-versus-24-month experiment requires 24 consecutive active input
months for both arms, then supplies either the last 12 or all 24 months to the
model. This separates context length from eligibility. Prediction dates, loan
assignments, sampled target examples, and labels stay fixed. The stored monthly
Parquet representation is independent of context; effective target indices can be
recomputed without changing raw data or the cached original preparation.

Use `--history` for the input length and `--eligibility-history` for the shared
minimum history. Effective window selection is recorded separately from the base
source manifests. Dynamic preprocessing uses only each arm's selected training
inputs, so means and scales may differ. Static preprocessing and vocabularies
must match. Longer learned positional embeddings add parameters and alter seeded
initialization; several paired seeds are reported.

XGBoost is given the same lag window, flattened into columns, and is refitted for
each length. It has no built-in mortgage history outside the supplied features.
Persistence continues to copy the most recent state and cannot exploit older
months. Its predictions are identical when target examples are held fixed.
Changing eligibility alone can change its score by changing the test population.

The target remains the following month; this ablation is separate from extending
the autoregressive forecast horizon. CUDA is selected for both XGBoost histogram
training and the transformers. The earlier state-only generator retains its own
trained context length. See the recorded context-length protocol and results.

## Autoregressive extension

The user requested multi-month generation after the one-month experiments. A
separate state-only variant uses four-way one-hot encodings of the preceding
active mortgage states and the same fixed origination attributes. It shares the
small transformer architecture but is trained separately; the full-feature model
cannot be rolled forward without assumptions about future balances and rates.

Training uses observed histories to predict the next state. Inference predicts or
samples a next state, appends it to the rolling 12-month context, and repeats until
the requested horizon or a terminal event. This is an autoregressive generation
loop, analogous to next-token generation, rather than a pretrained language model.
No real future state, balance, or interest rate is fed back. Generated terminal
tokens stop the trajectory. Probability accounting carries those outcomes forward
as absorbing states; raw trajectories use -1 after the last generated token.

The `forecast` command reloads the saved model and preprocessing without refitting.
It generates a greedy sequence plus sampled paths and empirical monthly state
probabilities. The state-only model omits evolving loan measurements and economic
conditions; sampling does not fix these modeling limits or establish calibration.

The `evaluate-rollouts` command selects one random forecast origin per uniformly
selected held-out test loan, then evaluates a complete generated rollout. At each
horizon it compares to persistence and a transition table whose terminal states
are absorbing. Actual reporting gaps, unavailable statuses, and right-censored
future months remain unknown; previously observed terminal outcomes can be carried
forward. Unknown outcomes are excluded separately at each horizon and support is
reported, so the evaluated population can change across horizons. This is not a
survival analysis correcting for informative censoring.

Macro-F1, accuracy, and multiclass Brier score are reported. Monte Carlo log loss
is omitted because finite path samples produce unstable zero probabilities for
rare events. Probability estimates have sampling error in addition to model error.
The implemented initial backtest uses the 2018 cohort, training targets through
2022, validation targets in 2023, and test targets after 2023. Forecasting any chosen
historical origin with a model trained later is a simulation, not a valid temporal
backtest; the dedicated evaluation uses the saved held-out partitions.

Submitted academic materials should be reviewed by the project authors.

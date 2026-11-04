# Past-context length experiment

Recorded before fitting the longer-context models.
The previous static-context-token experiment did not establish an improvement.
This experiment tests additional observed history while retaining its negative
result. The prediction target is still the following month.

- Context lengths: 12 and 24 months. All examples in both arms must have 24
  consecutive active input months and a known next-month outcome. This holds
  prediction dates, loans, labels, and eligibility fixed.
- Models in each arm: persistence, XGBoost, ordinary transformer, and the optional
  static-conditioned transformer. XGBoost receives the same context as its
  transformer counterpart, flattened into features. Persistence always copies
  the most recent observed state and must produce identical predictions.
- Cohorts: 2016–2025. Loan assignments use the existing split seed 522. Training
  target months end in 2022, validation targets are in 2023, and test targets follow.
- Identical sampled examples across arms: sampling seed 522, 200,000 training
  windows, 100,000 validation windows, all eligible test windows.
- Model seeds: 522, 523, 524. All fits use the RTX 5070 and the CUDA environment.
  XGBoost uses GPU histogram training as well.
- Architecture: two layers, four heads, hidden size 64; dropout 0.1. Only context
  length and the optional context-token flag vary. Longer positional embeddings
  add parameters and affect seeded initialization; this is not a parameter-matched
  experiment.
- Training: unchanged AdamW/cross-entropy, 0.001 learning rate, batch size 512,
  maximum 15 epochs, patience three, validation-log-loss checkpoint selection.
  XGBoost keeps depth four, rate 0.05, maximum 200 rounds, and patience 15.
- Primary focused endpoint: payoff/maturity average precision. Secondary outcomes:
  macro-F1, multiclass log loss, top-1% payoff ranking, payoff Brier score, new
  delinquency ranking, and per-year performance. Report every seed and both
  transformer variants, including unfavorable outcomes.
- Fit preprocessing separately using each arm's training inputs. Numeric scaling
  may differ because the 24-month arm sees more past records, but training loans,
  categorical vocabularies, static preprocessing, and target populations must match.
- Existing test data has been explored. This is an exploratory controlled ablation,
  not independent confirmation. The common 24-month population excludes early
  outcomes and short-history loans; do not compare its scores directly with the
  previous broader 12-month test population.

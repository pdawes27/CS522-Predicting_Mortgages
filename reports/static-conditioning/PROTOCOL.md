# Static-conditioned attention experiment

Protocol recorded before fitting this variant on November 2, 2026.
Earlier test results were inspected during hypothesis development;
this is a controlled exploratory comparison, not an untouched confirmatory test.

- Intervention: prepend one learned linear projection of existing normalized
  origination attributes to the monthly sequence. Keep the direct classifier
  connection, monthly inputs, positional embeddings, and encoder architecture.
- Primary comparison: `transformer_conditioned` versus `transformer`.
- Primary focused endpoint: next-month payoff/maturity average precision.
- Secondary endpoints: precision/count at the top-1% loan-month selection budget,
  binary payoff Brier score, multiclass log loss and macro-F1, and per-year payoff AP.
- Data: the prepared 2016–2025 official samples, 12-month histories, stable disjoint
  loan partitions. Training targets through 2022; validation in 2023; test after 2023.
- Sampling seed: 522; 200,000 training windows, 100,000 validation windows, all test
  windows. Verify ordered population fingerprints and saved preprocessing match.
- Model seeds: 522, 523, 524. Fit both variants independently for each seed with the
  same shared parameter initialization; the extra projection adds parameters.
- Training: hidden size 64, four heads, two layers, dropout 0.1, AdamW at 0.001,
  weight decay 0.0001, batch size 512, maximum 15 epochs, patience three. CPU, four
  threads per run. Checkpoints minimize multiclass validation log loss, unchanged
  from the existing baseline. Do not choose epochs based on test payoff ranking.
- Do not change architecture, window length, loss weights, or hyperparameters in
  response to these test scores. Report all seeds, including unfavorable results.
- The existing XGBoost seed-522 future-period run is a reference, not a new tuned
  multi-seed baseline. Report absolute scores as well as relative differences.
- Preserve original runs. Store new runs separately.

## GPU availability amendment

After the CPU fits were underway, the user pointed out the local GPU. An
unsandboxed check confirmed an RTX 5070. A separate CUDA environment was installed
and verified with matrix multiplication and real training of both variants.
The CPU fits completed during setup and remain the original three-seed comparison.
A separate seed-522 paired GPU repeat uses the same data and settings, with
`--device cuda`, and is reported separately. No CPU/GPU results are pooled to
select a favorable architecture outcome.

# Mortgage outcome prediction TODO

Based on `proposal.txt` and the agreed scope changes.

Goal: determine whether a small transformer using the previous 12 months of loan
history improves next-month state predictions over persistence and XGBoost.
Start with one official Freddie Mac sample origination year, then add years through
the same pipeline. Each sample year contains loans originated that year and their
subsequent monthly histories.


## Implementation status (November 2, 2026)

The core pipeline and four experiment suites are complete. See `README.md` for
commands, `docs/methodology.md` for assumptions, and `reports/REPORT.md` for measured
results. Raw data has 500,000 loans and 20,927,667 monthly records; preparation
produces 15,170,390 eligible 12-month windows. Runs train on a reproducible cap of
200,000 windows, rather than claiming to use every available training example.
All test windows are retained; expanded/calendar runs cap validation at 100,000.
The fixed 2018 expansion test population was verified identical by fingerprint.
The autoregressive run uses the 2018 cohort and a separate calendar-held-out split.

Remaining optional experiments are unchecked below. The exact downloaded release
label/date and author review of the report/presentation are still outstanding;
no files were submitted or published. Rare-event counts and limited seed coverage limit
what can be concluded from the results.

## 1. Finalize scope and data access

- [x] Update `proposal.txt` to reflect next-month prediction as the main task,
      official sample data, fixed 12-month inputs, and two baselines. Make long-term
      prediction optional and remove AutoGluon from the required scope.
- [x] Register for Freddie Mac Clarity and download completed sample origination
      years with sufficient follow-up history (user completed registration/download).
      Keep use and redistribution subject to the applicable terms.
- [x] Save official guide, layout, header, and release-note source links. Record
      selected years, local availability date, file checksums, and observed coverage.
      The release label and exact download date remain explicitly unconfirmed.
- [x] Measure compressed size, extracted size, loan count, and monthly row count.
      Record these measurements to guide later expansion.
- [x] Keep raw data, credentials, generated datasets, and model checkpoints out of
      version control. Preserve raw downloads unchanged and follow redistribution
      terms when sharing project artifacts.
- [x] Set up the Python environment, dependencies, random seeds, and a short README
      explaining how to reproduce the pipeline.

## 2. Define the prediction problem

- [x] Define an example precisely: months t-11 through t predict the state at t+1.
      Use only information available by t.
- [x] Map delinquency codes into current, 30–59 days late, 60–89 days late, and
      90+ days late. Define handling for unavailable codes and REO separately.
- [x] Inspect termination codes and choose a small, documented outcome mapping,
      such as voluntary payoff and adverse termination/REO in addition to the
      delinquency classes. Keep other or unknown exits explicit rather than
      silently assigning them to default or payoff.
- [x] Define how terminal events become labels, including events reported in the
      last observed record. Stop sequences after termination; do not invent
      subsequent monthly observations.
- [x] Exclude examples without an observable next-month outcome. Distinguish a
      dataset cutoff or missing observation from an actual termination event.
- [x] Document the exact meaning of every class. Avoid a separate generic default
      class unless its definition and source codes are unambiguous.

## 3. Build a reusable data pipeline

- [x] Accept a list of origination years and consistent input/output paths, starting
      with a single year. Keep year selection separate from processing logic.
- [x] Read pipe-delimited files with explicit release-specific column names, types,
      missing-value codes, and date parsing.
- [x] Join origination attributes to performance records by loan ID. Treat the loan
      as the grouping unit; do not assume a unique borrower identifier is available.
- [x] Check join coverage, duplicate loan-month records, invalid values, chronological
      order, reporting gaps, and terminal-event consistency.
- [x] Summarize history lengths, missingness, state frequencies, and state transitions.
      Confirm that there are enough rare outcomes to evaluate meaningfully.
- [x] Select a modest initial feature set: delinquency history, current balance,
      current interest rate, loan age, and origination FICO, LTV, DTI, amount, term,
      purpose, and occupancy where available.
- [x] Audit each feature for future information. Exclude eventual loss, proceeds,
      disposition details, identifiers, and other fields unavailable at prediction
      time from model inputs. Outcome fields may be used to construct labels.
- [x] Save a compact processed representation, such as Parquet, using incremental
      file processing so adding years does not require loading all raw data at once.

## 4. Establish splits and training examples

- [x] Create reproducible train/validation/test loan assignments before constructing
      windows. Save the assignments and reserve validation data for tuning.
- [x] Build windows from 12 consecutive observed months with an observed next-month
      label. Skip windows spanning reporting gaps; report how many are excluded.
- [x] Fit numerical preprocessing and categorical vocabularies on training data only.
      Handle missing values and unseen categories consistently in all splits.
- [x] Generate overlapping windows on demand or store window indices instead of
      duplicating the underlying monthly arrays.
- [x] Check that loan IDs do not overlap across splits and that window dates and
      labels align correctly. Manually inspect a few representative examples.
- [x] Freeze the initial test set before model development. Use validation results
      for feature choices, hyperparameters, and any decision thresholds.

## 5. Implement baselines and evaluation

- [x] Implement persistence: predict that next month's state equals the current
      state. Report its classification metrics; use a smoothed training-only state
      transition table if a probabilistic baseline is needed for finite log loss.
- [x] Train XGBoost using the same 12-month history and origination information,
      flattened into a fixed-length feature vector.
- [x] Evaluate every model on identical held-out examples and label definitions.
- [x] Report macro-F1, per-class precision/recall/F1 and support, a confusion matrix,
      and log loss for probabilistic models. Include accuracy as a secondary metric.
- [x] Inspect class imbalance. If necessary, compare ordinary cross-entropy with
      class-weighted training using weights computed from training data only.
      Preserve the natural class distribution in validation and test data.

## 6. Train the small transformer

- [x] Implement monthly feature projections/embeddings, positional information,
      and a small transformer encoder followed by a next-month classification head.
      Include origination features through a simple concatenation or projection.
- [x] Start with two encoder layers, four attention heads, hidden size 64 or 128,
      and 12-month sequences. Keep initial architecture tuning limited.
- [x] Run a small training smoke check to verify finite loss, valid outputs, correct
      label alignment, and that the model can learn from a tiny subset.
- [x] Train with cross-entropy, validation monitoring, early stopping, and saved
      checkpoints. Record settings, seeds, runtime, and peak memory where practical.
- [x] Compare against persistence and XGBoost. Analyze performance on worsening,
      recovery, and terminal transitions rather than only unchanged states.
- [x] Repeat the future-period transformer comparison with seeds 522, 523, and 524.
      Both original and static-conditioned variants use matched settings and data.
      Other original experiments remain single-seed; do not generalize the repeats
      to XGBoost or the 2018-only fits.

## 7. Expand to multiple origination years

- [x] Prepare all supplied 2016–2025 sample cohorts. Run the 2016–2020 expansion
      comparison and a separate future-period experiment using all ten cohorts.
      Pre-crisis/crisis origination cohorts were not supplied and remain optional.
- [x] Download and inventory the new files; verify schema compatibility and coverage.
- [x] Rebuild the dataset through the same year-list interface. Check duplicate IDs,
      cohort counts, missingness, class frequencies, and resource use.
- [x] Extend the saved split assignments without changing the original test loans.
      Retrain preprocessing and models from scratch using only the expanded training
      set; keep original test loans excluded from all training.
- [x] Compare one-year and multi-year models on the same original test examples.
      Evaluate added cohorts separately so changes in test composition do not obscure
      whether more training data helped.
- [x] Add a separate future-period evaluation: define training, validation, and test
      calendar cutoffs, partition loans to prevent overlap, and ensure all training
      targets precede validation/test target periods. Historical context before a
      cutoff is allowed only for the held-out loan's input window.
- [x] Report results by prediction year and origination cohort where sample sizes
      permit. Distinguish results for unseen loans from performance in future periods.

## 8. Finish analysis and deliverables

- [x] Run a small ablation comparing a short history or current-state-only model
      against 12-month inputs to assess whether temporal history adds value.
- [x] Review common errors, rare-class performance, and the effects of sample size,
      censoring, class imbalance, and changing economic conditions.
- [x] Assemble the final model comparison table and a small set of useful figures:
      class distribution, confusion matrices, and performance over time.
- [x] Document data provenance, label mapping, exclusions, splits, preprocessing,
      model settings, resource requirements, and limitations.
- [x] Provide reproducible commands for data preparation, training, and evaluation,
      with a clear distinction between local licensed data and shareable code.
- [x] Draft the report and presentation. Author
      review and any academic submission remain the user's responsibility.

## 9. Requested autoregressive extension

- [x] Implement recursive forecasting following the user's explicit request. A
      separately trained state-only transformer generates month-by-month paths without
      future balances/rates; trajectories stop at exits. Evaluate 1–12-month rollouts
      on held-out loans with censoring, persistence, and an absorbing transition baseline.

## Optional extensions, after the core comparison is complete

- [x] Compare 12-month and 24-month histories on identical 24-month-eligible targets,
      including both transformer variants, XGBoost, and persistence across three seeds.
- [ ] Implement a separately supervised fixed-horizon outcome, such as serious
      delinquency within six months, with explicit incomplete-follow-up handling.

## 10. Static conditioning and GPU follow-up

- [x] Add an optional static-context token with legacy checkpoint compatibility.
- [x] Test attention-path gradients, checkpoint round trips, and shared initialization.
- [x] Run three paired CPU seeds under a recorded protocol and verify matching
      populations, preprocessing, training settings, and original seed-522 predictions.
- [x] Report all outcomes: conditioning improves payoff AP in one of three CPU
      seeds, with a negative mean change. Keep the original model as the reference.
- [x] Verify the RTX 5070 outside the sandbox and create a separate CUDA environment.
- [x] Run CUDA smoke training and a full seed-522 GPU repeat; record device,
      CUDA version, memory, timings, and results separately from the CPU experiment.

## 11. Matched context-length experiment

- [x] Separate model input length from common history eligibility, reusing prepared
      monthly records and preserving saved split/window identities.
- [x] Test matched inputs, reporting gaps, persistence invariance, and cluster mapping;
      preserve eligibility when reloading autoregressive evaluation partitions.
- [x] Fit all four model types at 12 and 24 months for seeds 522, 523, and 524;
      verify actual CUDA training for XGBoost and both transformer variants.
- [x] Report all seeds and endpoints on 809,831 common test loan-months. Ordinary
      transformer payoff AP changes from 0.04160 to 0.04206 on average, with a
      mixed seed effect; conditioning gets worse. Persistence is unchanged.
- [x] Save reproducible drivers, model/configuration artifacts, per-year and paired
      metrics, protocol, figure, and findings.

## 12. Controlled training-data-size experiment

- [x] Preserve nested 200,000- and 1,000,000-window samples with frozen validation
      and test populations, using existing 12-month-eligible data.
- [x] Implement and test equal transformer update/exposure budgets and separate
      validation-log-loss and validation-macro-F1 checkpoints for both model families.
- [x] Run the ordinary transformer and XGBoost at both data sizes across three
      paired GPU seeds; verify population identities and training budgets.
- [x] Report all overall and per-class results, paired differences, year breakdowns,
      selected checkpoints, runtimes, and limitations.

## 13. Transformer optimization and transition modeling

- [x] Record the staged protocol, fixed data/update budgets, and validation-only
      training-recipe selection before full experiments.
- [x] Implement lower-rate and warmup/cosine training, transition-residual logits,
      observed-state embeddings, and an XGBoost state-indicator control.
- [x] Test schedules, training budgets, prior initialization, gradients, checkpoint
      compatibility, and scalar temperature calibration.
- [x] Fit three paired seeds for every optimization/architecture arm; verify shared
      populations and compare both checkpoint-selection rules.
- [x] Calibrate on validation targets excluded from checkpoint selection, verify
      classification invariance, and report probability-loss/Brier changes.
- [x] Summarize every seed, class, year, paired change, and limitation without
      selecting favorable test checkpoints; update presentation findings.

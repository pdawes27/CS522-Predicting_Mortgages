# Next-month mortgage outcomes

CS522 project comparing a small transformer with persistence,
a transition table, and XGBoost on Freddie Mac's official sample loan histories.
See [methodology](docs/methodology.md) for label definitions and leakage controls,
and [TODO.md](TODO.md) for progress and remaining decisions.

The original Fannie Mae downloader (`download_years.py`), spreadsheet, and
`headings.csv` are retained from the group repository. Their
[original setup notes](docs/legacy-fannie-mae-setup.md) describe a separate data
source; those downloads are not inputs to this Freddie Mac pipeline. Playwright
is only needed for that legacy downloader and is not included in the pipeline
environments.

## Repository layout

| Path | Purpose |
| --- | --- |
| `mortgage/` | Installable package: CLI, data preparation, models, evaluation, forecasts, and report generation. |
| `tests/` | Automated tests with synthetic data; licensed files are not required. |
| `analysis/` | Experiment drivers and analyses that use the package and completed local runs. |
| `docs/` | Methodology and source references. |
| `reports/` | Aggregate results, experiment protocols, tables, and figures. |
| `pyproject.toml` | Package metadata, dependencies, CLI entry point, and test configuration. |
| `environment.yml`, `environment-gpu.yml` | Conda/mamba environments for CPU and NVIDIA GPU workflows. |
| `requirements*-lock.txt` | Recorded CPU and GPU environment versions. |
| `proposal.txt`, `TODO.md` | Project scope and progress. |

Run commands from the repository root with the environment activated. Keep reusable
implementation in `mortgage/`, experiment orchestration in `analysis/`, and
aggregate presentation material in `reports/`.

Git excludes `data/`, `runs/`, downloaded references in `docs/reference/`, virtual
environments, credentials, caches, and scratch work in `temp/` or `tmp/`. Saved
arrays, Parquet files, model checkpoints, serialized models, ZIP archives, and
Freddie Mac sample text files are ignored throughout the repository, including
under `reports/`. Keep report tables, figures, and aggregate JSON summaries in
version control; keep loan-level predictions, partitions, and selection arrays
local. The `.env.example` filename is available for a template with placeholders.

## Setup

Use conda or mamba to install Python 3.12 and the runtime and development
dependencies from conda-forge. From the repository root, create the CPU environment:

```bash
mamba env create -f environment.yml
conda activate mortgage
python -m mortgage.cli --help
python -m pytest -q
```

With conda alone, use `conda env create -f environment.yml` for the first command.
All commands below use `python` from the activated environment. Run analysis
scripts as modules (`python -m analysis.<name>`) so they can import `mortgage`
directly from the repository. No pip install is needed for this workflow.

After changing the environment file, update the environment with:

```bash
mamba env update -n mortgage -f environment.yml --prune
```

Conda users can substitute `conda` for `mamba` in that command. For CUDA training,
create `environment-gpu.yml` instead, as described under [GPU training](#gpu-training).
These YAML files describe dependency requirements rather than exact version locks.
`requirements-lock.txt` and `requirements-gpu-lock.txt` preserve the original pip
environment snapshots used for the recorded experiments; they are not conda input
files, and a fresh conda solve may select different versions.

If you need the installed `mortgage` console command or imports from outside this
directory, optionally register the local package after creating the environment:

```bash
mamba install -c conda-forge pip 'setuptools>=77'
python -m pip install --no-deps --no-build-isolation -e .
```

This optional step installs only the local package; conda/mamba manages dependencies.

Place the licensed files `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt` in
`data/sample/`. The original ZIP files may stay alongside them. Do not commit
credentials or raw/processed data.

## Prepare

```bash
python -m mortgage.cli prepare --years 2018
```

Preparation checks every row's width, dates, IDs, duplicates, and outcomes, then
writes per-year Parquet, window indices, and quality summaries under
`data/processed/`. Existing prepared outputs are rebuilt; preserve separate output
directories if comparing releases. Raw files are never changed.

To prepare all downloaded samples through the same interface:

```bash
python -m mortgage.cli prepare \
  --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

## Train and evaluate

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli run \
  --years 2018 --output runs/2018-example
```

Defaults: 100,000 uniformly sampled training windows, all validation and test
windows, up to ten epochs, and four CPU threads. These limits keep initial runs
manageable. Use `--train-limit 0` for all training windows. Every experiment needs
a new output directory, preventing accidental overwrites.

Add `--models persistence transition xgboost transformer transformer_weighted
transformer_current` on one command line to run class weighting and the one-month
ablation. The ablation keeps the same eligible examples as the 12-month model.
Set `--seed` for repeated model fits; leave `--sample-seed` unchanged to hold
sampled examples fixed. A CUDA installation can use `--device cuda` if accessible.

Expand to more cohorts while retaining the original 2018 test population:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli run \
  --years 2016 2017 2018 2019 2020 --test-cohorts 2018 \
  --output runs/expanded-example
```

Evaluate a separate future period with disjoint loans and date-restricted targets:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli run \
  --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 \
  --train-end 2022-12 --validation-end 2023-12 \
  --output runs/future-example
```

The calendar cutoffs apply to target reporting months, not origination years.
Preprocessing is refitted on training inputs for each run. Predictions, loan splits,
and window indices are local artifacts and must not be committed or redistributed
without reviewing the dataset terms.

## Results and checks

Each run produces `comparison.md`, `comparison.csv`, per-model metrics and confusion
figures, performance over time, learning curves, checkpoints, saved preprocessing,
input fingerprints, and resource usage. A missing `resources.json` means the run
did not finish. Detailed class counts are essential because adverse exits are rare.

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m pytest -q
```

Formatting was checked through Black's Python API because its CLI stalled in this
sandbox. To run the same formatting check:

```bash
python - <<'PY'
from pathlib import Path
import black
paths = [*Path('mortgage').glob('*.py'), *Path('tests').glob('*.py')]
assert all(p.read_text() == black.format_str(p.read_text(), mode=black.Mode()) for p in paths)
print('Black formatting check passed')
PY
```

The tests cover outcome mappings, reporting gaps, split isolation, training-only
preprocessing, terminal records, invalid dates/duplicates, metric definitions, and
learning on a tiny synthetic problem. Real-data smoke runs additionally exercise
the complete parser/training/evaluation path.

Generate an aggregate report after experiments finish:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli report \
  --runs runs/2018-main runs/expanded-fixed-test-main runs/future-period runs/autoregressive \
  --output reports/REPORT.md
```

Only list completed runs. The report includes input sizes and coverage, class
support, model comparisons, the validation-selected model's per-class and
transition performance, and resource measurements.

## Generate several months ahead

The full-feature transformer predicts only the next month because future balances
and interest rates are unknown. A separate `autoregressive` variant uses the last
12 mortgage states and fixed origination attributes. It generates a next state,
appends it, and repeats with a rolling 12-month context. This supports a GPT-like
generation loop without requiring a language model or future monthly features.
Paths stop at payoff, adverse exit/REO, or another exit.

A trained example is available in `runs/autoregressive`. To reproduce its training:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli run \
  --years 2018 --train-end 2022-12 --validation-end 2023-12 \
  --train-limit 200000 --epochs 15 \
  --models persistence transition autoregressive --output runs/autoregressive-new
```

Generate 1,000 possible 12-month paths for a held-out loan that is active at the
latest reporting month. `--loan-index` is the zero-based row in the selected
cohort's sorted `loans.parquet`; the example below uses the supplied local files.

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli forecast \
  --run runs/autoregressive --cohort 2018 --loan-index 758 \
  --horizon 12 --paths 1000 --output runs/autoregressive/forecast-new
```

Use `--as-of YYYY-MM` to choose an observed historical origin. A historical
simulation is a valid future-period test only when the model was trained before
that date and the loan was held out. The dedicated backtest enforces the saved
test partition. A closed loan or a nonconsecutive/incomplete context is rejected.

Outputs include a greedy next-state sequence, sampled paths, per-month state
probabilities, terminal-event summaries, and `forecast.png`. Raw paths use -1 after
termination; probability summaries carry the known exit forward as an absorbing
outcome. These are simulated possibilities rather than guaranteed outcomes.

Backtest recursive generation without feeding real future states back into it:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli evaluate-rollouts \
  --run runs/autoregressive --horizon 12 --origins 1000 --paths 128 \
  --output runs/autoregressive/rollout-evaluation-new
```

This compares sampled autoregressive probabilities with persistence and an
absorbing transition-table baseline. It reports macro-F1, accuracy, and multiclass
Brier score by horizon, with observed support counts. Censored months are excluded
instead of labeled as default. Fixed attributes and state history cannot capture
future economic shocks or changes in loan terms; longer forecasts compound this
limitation. See the report for measured performance rather than assuming that
successful generation implies useful long-range prediction.

## Focused error analysis

[ERROR_ANALYSIS.md](reports/ERROR_ANALYSIS.md) examines state-transition errors,
payoff ranking, calibration drift, context length, and proposed architecture changes.
The presentation now includes the measured payoff-ranking advantage and its limits.
Reproduce the exploratory diagnostics and paired loan-cluster bootstrap with:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.diagnose_errors
```

The analysis uses saved predictions; it does not fit new models or operating
thresholds. Bootstrap intervals hold the fitted models fixed and exclude model-seed
and tuning uncertainty. Future experiments should use a validation-led protocol
and clearly distinguish exploratory findings from independent confirmation.

## Optional static-conditioned attention

Select `transformer_conditioned` to prepend an origination-profile token to the
monthly sequence. The encoder can then attend to FICO, LTV, purpose, and the other
existing static inputs. The original classifier connection remains. The ordinary
`transformer` model is unchanged by default, and older checkpoints remain loadable.
No data preparation changes are needed.

Run both variants with the same sampled windows and training settings:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli run \
  --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 \
  --train-end 2022-12 --validation-end 2023-12 \
  --train-limit 200000 --validation-limit 100000 --epochs 15 --seed 522 \
  --models transformer transformer_conditioned --output runs/static-context-example
```

The checkpoint's architecture metadata records whether conditioning is enabled.
The experiment protocol and its predeclared endpoints are in
[static-conditioning/PROTOCOL.md](reports/static-conditioning/PROTOCOL.md).
The completed paired experiment uses seeds 522, 523, and 524; see
[the results](reports/static-conditioning/RESULTS.md). Rebuild its summary with:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.compare_static_context
```

This produces per-seed and per-year payoff-ranking results, state-classification
scores, parameter counts, compatibility checks, and a presentation figure under
`reports/static-conditioning/`. The extra projection adds parameters and one token;
it is a small optional architecture variant, not a full Temporal Fusion Transformer.

## GPU training

Use **both** the GPU environment and `--device cuda`. On a machine with an NVIDIA
GPU and a compatible system driver, create the environment and check access:

```bash
mamba env create -f environment-gpu.yml
conda activate mortgage-gpu
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -c "import xgboost; assert xgboost.build_info()['USE_CUDA']"
```

Conda users can use `conda env create -f environment-gpu.yml`. The file requests
CUDA builds of both PyTorch and XGBoost. Conda supplies their CUDA runtime
dependencies; the NVIDIA driver must be installed on the host. See the
[conda-forge CUDA guide](https://conda-forge.org/docs/user/cuda/) and
[XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html).
The CPU `environment.yml` explicitly selects CPU builds.

The recorded experiments used pip-installed PyTorch `2.15.0+cu132` and CUDA 13.2
on an RTX 5070, with package versions saved in `requirements-gpu-lock.txt`.
The conda environment is a separate setup and does not reproduce that exact
package snapshot. Verify compatibility before reusing saved checkpoints.

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli run \
  --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 \
  --train-end 2022-12 --validation-end 2023-12 \
  --train-limit 200000 --validation-limit 100000 --epochs 15 --seed 522 \
  --device cuda --models transformer transformer_conditioned \
  --output runs/static-context-gpu-example
```

The runner prints the training device, fails promptly if CUDA is unavailable,
configures deterministic cuBLAS execution, and records the GPU model, CUDA version,
and peak PyTorch-allocated GPU memory. Data loading and preprocessing still use the CPU;
GPU utilization can fluctuate with these small models. The transformer weights,
minibatches, and gradient updates run on CUDA when selected.

The original three paired CPU seeds remain the primary architecture experiment.
A separate full GPU repeat is stored in `runs/static-context-gpu-522`, keeping
CPU and GPU results distinct because floating-point computation and optimization
trajectories may differ across devices.

## Compare past-context lengths fairly

`--history` controls the past monthly inputs supplied to the transformer and
XGBoost. `--eligibility-history` can require a longer history for every example,
even when a model uses only its most recent part:

```bash
# Same eligible examples in both arms; repeat with --history 24.
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m mortgage.cli run \
  --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 \
  --history 12 --eligibility-history 24 --device cuda \
  --train-end 2022-12 --validation-end 2023-12 \
  --train-limit 200000 --validation-limit 100000 --epochs 15 --seed 522 \
  --models persistence xgboost transformer transformer_conditioned \
  --output runs/context-example-12
```

The target is still next month. Both context arms require 24 consecutive active
months and share the same eligible prediction dates. XGBoost's flattened inputs
grow from 116 to 212 columns in the current dataset; it must be refitted. Persistence
uses only the latest state, so its predictions stay identical on these common
examples. Its score could change if the evaluated population changed instead.

Prepared monthly Parquet files can be reused for different contexts: the loader
recalculates window indices without reparsing raw files or overwriting the cached
preparation. Effective window eligibility is recorded in each run's
`window_selection.json`. Numeric preprocessing is fitted separately on each
context's training inputs. Rollout evaluation also reloads saved eligibility so
partition indices stay aligned.

The full recorded comparison uses three paired seeds, both context lengths, both
transformer variants, XGBoost, and persistence:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.run_context_experiment
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.compare_context_lengths
```

See [the context-length results](reports/context-length/RESULTS.md) and
[protocol](reports/context-length/PROTOCOL.md). The driver
needs GPU access outside the agent sandbox; XGBoost now receives `device=cuda`
when CUDA is selected, and its saved training configuration verifies that backend.

## Compare training-data sizes

The data-size driver compares nested 200,000- and 1,000,000-window subsets across
three paired seeds, using the ordinary 12-month transformer and XGBoost:

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.run_data_size_experiment
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.compare_data_sizes
```

`--nested-anchor 200000` preserves the original smaller sample inside larger
subsets. `--max-steps 6000 --validate-every 400` fixes the transformer update budget
instead of the epoch count; full minibatches give equal example exposures across
data sizes. `--dual-checkpoints` saves and evaluates separate validation-log-loss
and validation-macro-F1 checkpoints. XGBoost then fits its full round budget and
saves both validation-selected prefixes. Existing epoch-based defaults remain
available. Validation/test populations are frozen, and preprocessing uses only
the selected training subset.

See the [data-size protocol](reports/data-size/PROTOCOL.md),
[complete results](reports/data-size/RESULTS.md), and
[interpretation](reports/data-size/INTERPRETATION.md). This experiment
reports overall macro-F1, multiclass log loss, and every class's scores.

## Test training stability and transition modeling

The staged follow-up retains the million-window dataset and the
same optimizer budget. It compares constant learning rates 0.001 and 0.0003 with
`--lr-schedule warmup_cosine --warmup-steps 300`. The schedule requires
`--max-steps` and ends at 5% of the initial rate. Defaults remain unchanged.

`--models transformer_residual` adds a smoothed, training-only transition prior
to the output logits and zero-initializes the correction head. The prior is saved
in the checkpoint. `transformer_state_embedding` adds observed delinquency-state
embeddings to monthly tokens while retaining numeric severity. `xgboost_states`
provides the corresponding explicit indicators to the tree baseline.

```bash
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.run_transformer_improvements
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.calibrate_transformer_improvements
MPLCONFIGDIR=/tmp/mortgage-matplotlib python -m analysis.compare_transformer_improvements
```

The driver selects the architecture experiments' training recipe using mean
validation log loss alone. It retains both log-loss- and macro-F1-selected
checkpoints. Calibration fits a positive scalar temperature on remaining validation
targets and preserves predicted classes. See the
[protocol](reports/transformer-improvements/PROTOCOL.md),
[complete results](reports/transformer-improvements/RESULTS.md), and
[interpretation](reports/transformer-improvements/INTERPRETATION.md).

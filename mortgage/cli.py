"""Command line entry points."""

import argparse
from pathlib import Path

from .prepare import prepare_year, write_json

MODELS = [
    "persistence",
    "transition",
    "xgboost",
    "transformer",
    "transformer_weighted",
    "transformer_current",
    "autoregressive",
    "transformer_conditioned",
    "transformer_residual",
    "transformer_state_embedding",
    "xgboost_states",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freddie Mac sample experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Validate and prepare each selected sample year"
    )
    prepare.add_argument("--years", type=int, nargs="+", required=True)
    prepare.add_argument("--raw", default="data/sample")
    prepare.add_argument("--output", default="data/processed")
    prepare.add_argument("--split-seed", type=int, default=522)
    prepare.add_argument("--history", type=int, default=12)
    prepare.add_argument(
        "--release", default="unconfirmed; July 2026 layout compatible"
    )
    prepare.add_argument(
        "--download-date", default="unconfirmed; present locally 2026-11-02"
    )
    run = commands.add_parser("run", help="Fit models and evaluate held-out loans")
    run.add_argument("--years", type=int, nargs="+", required=True)
    run.add_argument("--prepared", default="data/processed")
    run.add_argument("--output", required=True)
    run.add_argument("--models", nargs="+", choices=MODELS, default=MODELS[:4])
    run.add_argument("--history", type=int, default=12)
    run.add_argument(
        "--eligibility-history",
        type=int,
        help="Require this much contiguous history for every example, even when the model uses a shorter context",
    )
    run.add_argument(
        "--train-limit",
        type=int,
        default=100_000,
        help="Uniform window sample; 0 means all",
    )
    run.add_argument("--validation-limit", type=int, default=0)
    run.add_argument("--test-limit", type=int, default=0)
    run.add_argument("--seed", type=int, default=522, help="Model initialization seed")
    run.add_argument(
        "--sample-seed", type=int, default=522, help="Fixed window sampling seed"
    )
    run.add_argument(
        "--train-end", help="Optional inclusive training target cutoff, YYYY-MM"
    )
    run.add_argument(
        "--validation-end", help="Optional inclusive validation target cutoff, YYYY-MM"
    )
    run.add_argument(
        "--test-cohorts",
        type=int,
        nargs="+",
        help="Restrict test loans to these cohorts",
    )
    run.add_argument("--epochs", type=int, default=10)
    run.add_argument(
        "--nested-anchor",
        type=int,
        help="Include the original sample of this size within every larger training subset",
    )
    run.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Fixed optimizer-update budget; 0 retains epoch-based training",
    )
    run.add_argument(
        "--validate-every",
        type=int,
        default=400,
        help="Validation cadence in optimizer updates for fixed-step training",
    )
    run.add_argument(
        "--dual-checkpoints",
        action="store_true",
        help="Evaluate separately the validation-log-loss and validation-macro-F1 checkpoints",
    )
    run.add_argument("--patience", type=int, default=3)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--hidden", type=int, default=64)
    run.add_argument("--heads", type=int, default=4)
    run.add_argument("--layers", type=int, default=2)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument(
        "--lr-schedule", choices=["constant", "warmup_cosine"], default="constant"
    )
    run.add_argument("--warmup-steps", type=int, default=300)
    run.add_argument("--boost-rounds", type=int, default=200)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    report = commands.add_parser(
        "report", help="Build an aggregate report from completed experiments"
    )
    report.add_argument("--prepared", default="data/processed")
    report.add_argument("--runs", nargs="+", required=True)
    report.add_argument("--output", default="reports/REPORT.md")
    forecast = commands.add_parser(
        "forecast",
        help="Generate future monthly state paths from a state-only checkpoint",
    )
    rollout = commands.add_parser(
        "evaluate-rollouts",
        help="Backtest recursive state forecasts at multiple horizons",
    )
    for command in [forecast, rollout]:
        command.add_argument("--run", required=True)
        command.add_argument("--prepared", default="data/processed")
        command.add_argument("--output", required=True)
        command.add_argument("--horizon", type=int, default=12)
        command.add_argument(
            "--paths", type=int, default=1000 if command is forecast else 128
        )
        command.add_argument("--seed", type=int, default=522)
        command.add_argument("--threads", type=int, default=4)
    forecast.add_argument("--cohort", type=int, required=True)
    forecast.add_argument(
        "--loan-index",
        type=int,
        required=True,
        help="Zero-based row in the cohort's sorted loans.parquet",
    )
    forecast.add_argument(
        "--as-of",
        help="Observed forecast origin month, YYYY-MM; defaults to latest record",
    )
    rollout.add_argument("--origins", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "prepare":
        summaries = [
            prepare_year(
                Path(args.raw),
                Path(args.output),
                year,
                args.split_seed,
                args.history,
                args.release,
                args.download_date,
            )
            for year in sorted(set(args.years))
        ]
        write_json(Path(args.output) / "inventory.json", summaries)
    elif args.command in ["forecast", "evaluate-rollouts"]:
        from .forecast import forecast as generate_forecast, evaluate_rollouts

        if args.command == "forecast":
            generate_forecast(args)
        else:
            evaluate_rollouts(args)
    elif args.command == "report":
        from .report import build_report

        build_report(
            Path(args.prepared), [Path(path) for path in args.runs], Path(args.output)
        )
    else:
        from .experiment import run as run_experiment

        run_experiment(args)


if __name__ == "__main__":
    main()

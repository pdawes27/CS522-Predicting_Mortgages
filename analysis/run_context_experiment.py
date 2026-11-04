"""Run the predeclared matched 12/24-month comparison.

Run with the CUDA environment and GPU access:
MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python analysis/run_context_experiment.py
"""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    years = list(range(2016, 2026))
    models = ["persistence", "xgboost", "transformer", "transformer_conditioned"]
    for seed in [522, 523, 524]:
        for history in [12, 24]:
            output = Path("runs") / f"context-{history}-{seed}"
            expected = {
                "years": years,
                "history": history,
                "eligibility_history": 24,
                "seed": seed,
                "sample_seed": 522,
                "train_end": "2022-12",
                "validation_end": "2023-12",
                "train_limit": 200_000,
                "validation_limit": 100_000,
                "test_limit": 0,
                "epochs": 15,
                "patience": 3,
                "batch_size": 512,
                "hidden": 64,
                "heads": 4,
                "layers": 2,
                "learning_rate": 0.001,
                "boost_rounds": 200,
                "threads": 4,
                "device": "cuda",
                "models": models,
            }
            if (root / output / "resources.json").exists():
                actual = json.loads((root / output / "config.json").read_text())
                if any(actual[key] != value for key, value in expected.items()):
                    raise ValueError(f"Existing run does not match protocol: {output}")
                print(f"Already complete: {output}", flush=True)
                continue
            print(f"Starting {history}-month context, seed {seed}", flush=True)
            command = [
                sys.executable,
                "-m",
                "mortgage.cli",
                "run",
                "--years",
                *map(str, years),
                "--history",
                str(history),
                "--eligibility-history",
                "24",
                "--train-end",
                "2022-12",
                "--validation-end",
                "2023-12",
                "--train-limit",
                "200000",
                "--validation-limit",
                "100000",
                "--epochs",
                "15",
                "--patience",
                "3",
                "--batch-size",
                "512",
                "--hidden",
                "64",
                "--heads",
                "4",
                "--layers",
                "2",
                "--learning-rate",
                "0.001",
                "--boost-rounds",
                "200",
                "--threads",
                "4",
                "--sample-seed",
                "522",
                "--test-limit",
                "0",
                "--seed",
                str(seed),
                "--device",
                "cuda",
                "--models",
                *models,
                "--output",
                str(output),
            ]
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()

"""Controlled data-size comparison."""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for seed in [522, 523, 524]:
        for size in [200_000, 1_000_000]:
            output = root / "runs" / f"data-size-{size}-{seed}"
            expected = {
                "years": list(range(2016, 2026)),
                "history": 12,
                "eligibility_history": 12,
                "sample_seed": 522,
                "seed": seed,
                "train_limit": size,
                "nested_anchor": 200_000,
                "train_end": "2022-12",
                "validation_end": "2023-12",
                "validation_limit": 100_000,
                "test_limit": 0,
                "max_steps": 6000,
                "validate_every": 400,
                "dual_checkpoints": True,
                "batch_size": 512,
                "hidden": 64,
                "heads": 4,
                "layers": 2,
                "learning_rate": 0.001,
                "boost_rounds": 200,
                "threads": 4,
                "device": "cuda",
                "models": ["persistence", "xgboost", "transformer"],
            }
            if (output / "resources.json").exists():
                actual = json.loads((output / "config.json").read_text())
                if any(actual[key] != value for key, value in expected.items()):
                    raise ValueError(f"Existing run differs from protocol: {output}")
                print(f"Already complete: {output}", flush=True)
                continue
            command = [sys.executable, "-m", "mortgage.cli", "run"]
            for key, value in expected.items():
                command.append("--" + key.replace("_", "-"))
                if isinstance(value, list):
                    command.extend(map(str, value))
                elif value is not True:
                    command.append(str(value))
            command.extend(["--output", str(output)])
            print(f"Starting {size:,} windows, seed {seed}", flush=True)
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()

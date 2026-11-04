"""Run staged, validation-selected transformer experiments."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "constant": {"learning_rate": 0.001, "lr_schedule": "constant"},
    "lower": {"learning_rate": 0.0003, "lr_schedule": "constant"},
    "cosine": {"learning_rate": 0.001, "lr_schedule": "warmup_cosine"},
}


def ordinary_run(recipe: str, seed: int) -> Path:
    if recipe == "constant":
        return ROOT / f"runs/data-size-1000000-{seed}"
    return ROOT / f"runs/optimization-{recipe}-{seed}"


def experiment_sources() -> list[tuple[str, str, int, Path, str]]:
    """All predeclared arms, including both validation checkpoint criteria."""
    selection = json.loads(
        (ROOT / "reports/transformer-improvements/recipe_selection.json").read_text()
    )
    selected = selection["selected_recipe"]
    sources = []
    for seed in [522, 523, 524]:
        for recipe in RECIPES:
            for model in ["transformer", "transformer_macro_f1"]:
                sources.append(
                    ("optimization", recipe, seed, ordinary_run(recipe, seed), model)
                )
        for model in [
            "transition",
            "xgboost_states",
            "xgboost_states_macro_f1",
            "transformer_residual",
            "transformer_residual_macro_f1",
            "transformer_state_embedding",
            "transformer_state_embedding_macro_f1",
        ]:
            sources.append(
                (
                    "architecture",
                    selected,
                    seed,
                    ROOT / f"runs/architecture-improvements-{seed}",
                    model,
                )
            )
        for model in ["persistence", "xgboost", "xgboost_macro_f1"]:
            sources.append(
                ("reference", "constant", seed, ordinary_run("constant", seed), model)
            )
    return sources


def run_fit(output: Path, seed: int, recipe: str, models: list[str]) -> None:
    expected = {
        "years": list(range(2016, 2026)),
        "history": 12,
        "eligibility_history": 12,
        "sample_seed": 522,
        "seed": seed,
        "train_limit": 1_000_000,
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
        "boost_rounds": 200,
        "threads": 4,
        "device": "cuda",
        "warmup_steps": 300,
        "models": models,
        **RECIPES[recipe],
    }
    if (output / "resources.json").exists():
        config = json.loads((output / "config.json").read_text())
        for key, value in expected.items():
            if (
                config.get(
                    key, {"lr_schedule": "constant", "warmup_steps": 300}.get(key)
                )
                != value
            ):
                raise ValueError(f"Existing run differs: {output}, {key}")
        print(f"Already complete: {output}", flush=True)
        return
    command = [sys.executable, "-m", "mortgage.cli", "run"]
    for key, value in expected.items():
        command.append("--" + key.replace("_", "-"))
        if isinstance(value, list):
            command.extend(map(str, value))
        elif value is not True:
            command.append(str(value))
    command.extend(["--output", str(output)])
    print(f"Starting {output.name}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    for recipe in RECIPES:
        for seed in [522, 523, 524]:
            if recipe == "constant":
                # The existing run also fitted the reference XGBoost and persistence.
                run_fit(
                    ordinary_run(recipe, seed),
                    seed,
                    recipe,
                    ["persistence", "xgboost", "transformer"],
                )
            else:
                run_fit(ordinary_run(recipe, seed), seed, recipe, ["transformer"])
    validation = {}
    for recipe in RECIPES:
        validation[recipe] = [
            json.loads(
                (ordinary_run(recipe, seed) / "transformer/metrics.json").read_text()
            )["validation"]["log_loss"]
            for seed in [522, 523, 524]
        ]
    selected = min(RECIPES, key=lambda recipe: sum(validation[recipe]) / 3)
    output = ROOT / "reports/transformer-improvements/recipe_selection.json"
    output.write_text(
        json.dumps(
            {
                "criterion": "mean validation log loss of log-loss-selected checkpoints",
                "validation_losses": validation,
                "selected_recipe": selected,
                "selected_settings": RECIPES[selected],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Validation-only selected recipe: {selected}", flush=True)
    for seed in [522, 523, 524]:
        run_fit(
            ROOT / f"runs/architecture-improvements-{seed}",
            seed,
            selected,
            [
                "transition",
                "xgboost_states",
                "transformer_residual",
                "transformer_state_embedding",
            ],
        )


if __name__ == "__main__":
    main()

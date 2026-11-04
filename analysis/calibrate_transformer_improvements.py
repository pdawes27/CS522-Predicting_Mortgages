"""Calibrate all declared model arms on unused validation targets."""

import json
import os
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from analysis.run_transformer_improvements import ROOT, experiment_sources
from mortgage.calibration import (
    PROBABILITY_FLOOR,
    apply_temperature,
    fit_temperature,
    multiclass_brier,
)
from mortgage.data import LoanData
from mortgage.evaluate import metrics
from mortgage.experiment import xgboost_predictions
from mortgage.models import MortgageTransformer, predict_transformer
from mortgage.prepare import write_json
from mortgage.schema import LABELS


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("Calibration inference requires CUDA for this protocol")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output = ROOT / "reports/transformer-improvements/calibration"
    output.mkdir(parents=True, exist_ok=True)
    data = LoanData(ROOT / "data/processed", list(range(2016, 2026)), 12, 12)
    _, validation_all, _ = data.partitions("2022-12", "2023-12", None)
    with np.load(ROOT / "runs/data-size-1000000-522/partitions.npz") as indices:
        selection, test = indices["validation"], indices["test"]
    calibration = np.setdiff1d(validation_all, selection, assume_unique=True)
    assert len(calibration) == 196_496
    assert not np.isin(calibration, selection).any()
    assert not np.isin(data.target_loans[calibration], data.target_loans[test]).any()
    np.save(output / "calibration_indices.npy", calibration)
    write_json(
        output / "population.json",
        {
            "examples": len(calibration),
            "signature": data.population_signature(calibration),
            "class_counts": dict(
                zip(
                    LABELS,
                    np.bincount(
                        data.labels[calibration], minlength=len(LABELS)
                    ).tolist(),
                )
            ),
            "selection_targets_disjoint": True,
            "test_loans_disjoint": True,
            "loans_may_overlap_checkpoint_selection": True,
            "probability_floor": PROBABILITY_FLOOR,
            "scipy_version": version("scipy"),
        },
    )
    summary = []
    for stage, recipe, seed, run, name in experiment_sources():
        if not name.startswith(("transformer", "xgboost")):
            continue
        destination = output / f"{stage}-{recipe}-{seed}-{name}.json"
        if destination.exists():
            summary.append(json.loads(destination.read_text())["summary"])
            continue
        started = time.perf_counter()
        data.apply_preprocessing(json.loads((run / "preprocessing.json").read_text()))
        with np.load(run / "partitions.npz") as indices:
            np.testing.assert_array_equal(indices["validation"], selection)
            np.testing.assert_array_equal(indices["test"], test)
        if name.startswith("transformer"):
            saved = torch.load(
                run / name / "transformer.pt", map_location="cuda", weights_only=True
            )
            model = MortgageTransformer(**saved["architecture"]).cuda()
            model.load_state_dict(saved["state_dict"], strict=True)
            probabilities = predict_transformer(
                model, data, calibration, 512, torch.device("cuda"), 12
            )
            del model, saved
        else:
            booster = xgb.Booster()
            booster.load_model(run / name / "model.ubj")
            probabilities = xgboost_predictions(
                booster,
                data,
                calibration,
                state_indicators=name.startswith("xgboost_states"),
            )
            del booster
        temperature = fit_temperature(probabilities, data.labels[calibration])
        calibration_before = metrics(
            data.labels[calibration], apply_temperature(probabilities, 1)
        )
        calibration_after = metrics(
            data.labels[calibration], apply_temperature(probabilities, temperature)
        )
        frame = pd.read_parquet(run / name / "predictions.parquet")
        np.testing.assert_array_equal(frame.actual, data.labels[test])
        raw = frame[["p_" + label for label in LABELS]].to_numpy(dtype=np.float32)
        before_prob = apply_temperature(raw, 1)
        after_prob = apply_temperature(raw, temperature)
        np.testing.assert_array_equal(before_prob.argmax(1), after_prob.argmax(1))
        before, after = metrics(data.labels[test], before_prob), metrics(
            data.labels[test], after_prob
        )
        original = json.loads((run / name / "metrics.json").read_text())["test"]
        assert before["macro_f1"] == original["macro_f1"]
        assert abs(before["log_loss"] - original["log_loss"]) < 1e-6
        assert calibration_after["log_loss"] <= calibration_before["log_loss"] + 1e-7
        row = {
            "stage": stage,
            "recipe": recipe,
            "seed": seed,
            "model": name,
            "temperature": temperature,
            "macro_f1": before["macro_f1"],
            "log_loss_before": before["log_loss"],
            "log_loss_after": after["log_loss"],
            "brier_before": multiclass_brier(before_prob, data.labels[test]),
            "brier_after": multiclass_brier(after_prob, data.labels[test]),
            "calibration_loss_before": calibration_before["log_loss"],
            "calibration_loss_after": calibration_after["log_loss"],
            "seconds": time.perf_counter() - started,
            "test_probabilities_clipped": int((raw < PROBABILITY_FLOOR).sum()),
        }
        yearly = {}
        for year in np.unique(data.target_months[test] // 12):
            mask = data.target_months[test] // 12 == year
            yearly[str(year)] = {
                "before": metrics(data.labels[test][mask], before_prob[mask]),
                "after": metrics(data.labels[test][mask], after_prob[mask]),
            }
        write_json(
            destination,
            {
                "summary": row,
                "source_run": str(run.relative_to(ROOT)),
                "calibration_before": calibration_before,
                "calibration_after": calibration_after,
                "test_before": before,
                "test_after": after,
                "by_year": yearly,
            },
        )
        summary.append(row)
        pd.DataFrame(summary).to_csv(output / "metrics.csv", index=False)
        print(
            f"Calibrated {stage} {recipe} {seed} {name}: T={temperature:.3f}, loss {before['log_loss']:.5f} -> {after['log_loss']:.5f}",
            flush=True,
        )
    pd.DataFrame(summary).to_csv(output / "metrics.csv", index=False)
    write_json(
        output / "validation.json",
        {
            "models": len(summary),
            "classification_unchanged": True,
            "selection_targets_disjoint": True,
            "test_loans_disjoint": True,
        },
    )


if __name__ == "__main__":
    main()

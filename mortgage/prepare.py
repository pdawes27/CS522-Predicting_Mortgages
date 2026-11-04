"""Validate and prepare one vintage at a time."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .schema import (
    DYNAMIC_NUMERIC,
    LABELS,
    ORIG_COLUMNS,
    ORIG_USE,
    PERF_COLUMNS,
    PERF_USE,
    STATIC_CATEGORICAL,
    STATIC_NUMERIC,
    month_numbers,
    outcome_states,
    split_for_loan,
    year_month,
)


class SourceRecord(TypedDict):
    name: str
    bytes: int
    rows: int
    sha256: str
    columns: int


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inspect_file(path: Path, width: int) -> SourceRecord:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for count, line in enumerate(stream, 1):
            digest.update(line)
            if len(line.rstrip(b"\r\n").split(b"|")) != width:
                raise ValueError(f"{path}:{count}: expected {width} fields")
    if not count:
        raise ValueError(f"Empty file: {path}")
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "rows": count,
        "sha256": digest.hexdigest(),
        "columns": width,
    }


def select_windows(
    loans: NDArray[np.int64],
    months: NDArray[np.int32],
    states: NDArray[np.int8],
    history: int,
) -> tuple[NDArray[np.int64], dict[str, int]]:
    """Select contiguous active histories and count mutually exclusive exclusions."""
    if history < 1:
        raise ValueError("history must be positive")
    targets = np.arange(history, len(loans), dtype=np.int64)
    starts = targets - history
    bad = np.r_[0, np.cumsum((states < 0) | (states >= 4))]
    enough = loans[targets] == loans[starts]
    consecutive = months[targets] - months[starts] == history
    active = bad[targets] == bad[starts]
    known_target = states[targets] >= 0
    valid = enough & consecutive & active & known_target
    exclusions = {
        "insufficient_prior_records": min(history, len(loans)) + int((~enough).sum()),
        "reporting_gap": int((enough & ~consecutive).sum()),
        "unknown_or_terminal_input": int((enough & consecutive & ~active).sum()),
        "unknown_target": int((enough & consecutive & active & ~known_target).sum()),
    }
    return targets[valid], exclusions


def prepare_year(
    raw: Path,
    output: Path,
    year: int,
    seed: int,
    history: int,
    release: str,
    download_date: str,
) -> dict[str, object]:
    directory = output / str(year)
    directory.mkdir(parents=True, exist_ok=True)
    # A missing summary marks an interrupted preparation as incomplete.
    (directory / "summary.json").unlink(missing_ok=True)
    orig_path = raw / f"sample_orig_{year}.txt"
    perf_path = raw / f"sample_perf_{year}.txt"
    sources = [inspect_file(orig_path, 31), inspect_file(perf_path, 35)]
    orig = pd.read_csv(
        orig_path,
        sep="|",
        names=ORIG_COLUMNS,
        usecols=ORIG_USE,
        dtype=str,
        keep_default_na=False,
    )
    if orig["loan_id"].eq("").any() or orig["loan_id"].duplicated().any():
        raise ValueError(f"{year}: empty or duplicate origination loan IDs")
    orig = orig.sort_values("loan_id").reset_index(drop=True)
    orig["cohort"] = year
    orig["split"] = (
        orig["loan_id"].map(lambda value: split_for_loan(value, seed)).astype("int8")
    )
    for name in STATIC_NUMERIC:
        orig[name] = pd.to_numeric(orig[name], errors="coerce").astype("float32")
    limits = {
        "fico": (300, 850),
        "ltv": (1, 998),
        "dti": (0, 65),
        "original_upb": (1, 1e8),
        "term": (1, 600),
        "original_rate": (0, 30),
    }
    for name, (low, high) in limits.items():
        orig.loc[~orig[name].between(low, high), name] = np.nan
    for name in STATIC_CATEGORICAL:
        orig[name] = orig[name].replace({"": "MISSING", "9": "MISSING"})

    chunks = pd.read_csv(
        perf_path,
        sep="|",
        names=PERF_COLUMNS,
        usecols=PERF_USE,
        dtype=str,
        keep_default_na=False,
        chunksize=250_000,
    )
    perf = pd.concat(chunks, ignore_index=True)
    perf["loan_index"] = pd.Index(orig["loan_id"]).get_indexer(perf["loan_id"])
    if (perf["loan_index"] < 0).any():
        raise ValueError(f"{year}: performance rows without origination records")
    perf["month"] = month_numbers(perf["period"])
    if perf.duplicated(["loan_index", "month"]).any():
        raise ValueError(f"{year}: duplicate loan-month records")
    unsorted = not pd.MultiIndex.from_frame(
        perf[["loan_index", "month"]]
    ).is_monotonic_increasing
    perf = perf.sort_values(["loan_index", "month"]).reset_index(drop=True)
    perf["state"] = outcome_states(perf)
    code_counts = perf["zero_code"].value_counts().to_dict()
    status_counts = perf["delinquency"].value_counts().to_dict()
    terminal = perf["state"] >= 4
    # Effective dates may differ from reporting dates; retain reporting-time labels
    # and explicitly count mismatches rather than moving future reports backwards.
    dated_terminal = terminal & perf["zero_date"].ne("")
    mismatches = int(
        (
            month_numbers(perf.loc[dated_terminal, "zero_date"])
            != perf.loc[dated_terminal, "month"].to_numpy()
        ).sum()
    )
    first_exit = perf.loc[terminal].groupby("loan_index")["month"].min()
    exits = perf["loan_index"].map(first_exit)
    after_exit = perf["month"] > exits
    dropped_after_exit = int(after_exit.sum())
    perf = perf.loc[~after_exit].reset_index(drop=True)
    for name in ["upb", "rate", "age"]:
        perf[name] = pd.to_numeric(perf[name], errors="coerce").astype("float32")
    for name, low, high in [("upb", 0, 1e8), ("rate", 0, 30), ("age", 0, 1000)]:
        perf.loc[~perf[name].between(low, high), name] = np.nan
    late = pd.to_numeric(perf["delinquency"], errors="coerce")
    perf["delinquency_months"] = late.where(late.between(0, 99)).astype("float32")
    monthly = perf[["loan_index", "month", "state", *DYNAMIC_NUMERIC]].copy()
    monthly = monthly.astype({"loan_index": "int64", "month": "int32", "state": "int8"})
    targets, exclusions = select_windows(
        monthly.loan_index.to_numpy(),
        monthly.month.to_numpy(),
        monthly.state.to_numpy(),
        history,
    )
    loan_rows = monthly.loan_index.to_numpy()
    months = monthly.month.to_numpy()
    states = monthly.state.to_numpy()
    adjacent = (loan_rows[1:] == loan_rows[:-1]) & (np.diff(months) == 1)
    valid_pairs = adjacent & (states[:-1] >= 0) & (states[:-1] < 4) & (states[1:] >= 0)
    transitions = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(transitions, (states[:-1][valid_pairs], states[1:][valid_pairs]), 1)
    split_values = orig["split"].to_numpy()[loan_rows[targets]]
    history_lengths = monthly.groupby("loan_index").size()
    archive = raw / f"sample_{year}.zip"
    summary = {
        "cohort": year,
        "seed": seed,
        "history": history,
        "release": release,
        "download_date": download_date,
        "schema": "July 2026, orig=31/perf=35",
        "sources": sources,
        "compressed_bytes": archive.stat().st_size if archive.exists() else None,
        "loans": len(orig),
        "performance_rows_raw": int(sources[1]["rows"]),
        "performance_rows_retained": len(monthly),
        "sorting_required": unsorted,
        "loans_without_performance": int(len(orig) - monthly.loan_index.nunique()),
        "dropped_after_first_exit": dropped_after_exit,
        "terminal_effective_date_mismatches": mismatches,
        "reporting_gaps": int(
            ((loan_rows[1:] == loan_rows[:-1]) & (np.diff(months) > 1)).sum()
        ),
        "date_range": [year_month(int(months.min())), year_month(int(months.max()))],
        "history_length_quantiles": {
            str(q): float(history_lengths.quantile(q)) for q in [0, 0.25, 0.5, 0.75, 1]
        },
        "raw_status_counts": status_counts,
        "raw_exit_code_counts": code_counts,
        "state_counts": np.bincount(
            states[states >= 0], minlength=len(LABELS)
        ).tolist(),
        "unknown_state_rows": int((states < 0).sum()),
        "transitions": transitions.tolist(),
        "windows": len(targets),
        "rows_without_eligible_window": len(monthly) - len(targets),
        "window_exclusions": exclusions,
        "window_counts_by_split": {
            name: int((split_values == i).sum())
            for i, name in enumerate(["train", "validation", "test"])
        },
        "window_class_counts": np.bincount(
            states[targets], minlength=len(LABELS)
        ).tolist(),
        "missing_fraction": {
            name: float(monthly[name].isna().mean()) for name in DYNAMIC_NUMERIC
        }
        | {name: float(orig[name].isna().mean()) for name in STATIC_NUMERIC},
    }
    orig.to_parquet(directory / "loans.parquet", index=False)
    monthly.to_parquet(directory / "monthly.parquet", index=False)
    np.save(directory / "targets.npy", targets)
    write_json(directory / "summary.json", summary)
    print(
        f"Prepared {year}: {len(orig):,} loans, {len(monthly):,} rows, {len(targets):,} windows",
        flush=True,
    )
    return summary

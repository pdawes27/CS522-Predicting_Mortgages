"""Explicit Freddie Mac July 2026 layout subset and outcome definitions.

Column positions were checked against the official
user guide and local 31/35-column samples. See docs/methodology.md for sources.
"""

import hashlib

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ORIG_COLUMNS = [
    "fico",
    "first_payment",
    "first_time_buyer",
    "maturity",
    "msa",
    "mi",
    "units",
    "occupancy",
    "cltv",
    "dti",
    "original_upb",
    "ltv",
    "original_rate",
    "channel",
    "prepayment_penalty",
    "product",
    "property_state",
    "property_type",
    "postal_code",
    "loan_id",
    "purpose",
    "term",
    "borrowers",
    "seller",
    "super_conforming",
    "pre_harp_loan",
    "program",
    "harp",
    "valuation",
    "interest_only",
    "vantage_score",
]
PERF_COLUMNS = [
    "loan_id",
    "period",
    "upb",
    "delinquency",
    "age",
    "remaining_term",
    "defect_date",
    "modification",
    "zero_code",
    "zero_date",
    "rate",
    "deferred_upb",
    "last_paid",
    "mi_recoveries",
    "net_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "preservation_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "cumulative_modification_cost",
    "interest_step",
    "payment_deferral",
    "estimated_ltv",
    "zero_removal_upb",
    "accrued_interest",
    "disaster",
    "assistance",
    "modification_cost",
    "interest_bearing_upb",
    "mi_cancellation",
    "servicer",
    "cramdown_cost",
]
STATIC_NUMERIC = ["fico", "ltv", "dti", "original_upb", "term", "original_rate"]
STATIC_CATEGORICAL = ["purpose", "occupancy"]
DYNAMIC_NUMERIC = ["upb", "rate", "age", "delinquency_months"]
LABELS = [
    "current",
    "30_59_days",
    "60_89_days",
    "90_plus_days",
    "payoff",
    "adverse_exit_or_reo",
    "other_exit",
]
SPLITS = ["train", "validation", "test"]
ORIG_USE = ["loan_id", *STATIC_NUMERIC, *STATIC_CATEGORICAL]
PERF_USE = [
    "loan_id",
    "period",
    "upb",
    "rate",
    "age",
    "delinquency",
    "zero_code",
    "zero_date",
]


def month_numbers(values: pd.Series) -> NDArray[np.int32]:
    """Convert YYYYMM to an integer month; reject malformed reporting dates."""
    strings = values.astype(str)
    if not strings.str.fullmatch(r"\d{6}").all():
        raise ValueError("Dates must use YYYYMM")
    numbers = strings.astype("int32").to_numpy()
    years, months = numbers // 100, numbers % 100
    if np.any((years < 1990) | (years > 2100) | (months < 1) | (months > 12)):
        raise ValueError("Invalid reporting month")
    return (years * 12 + months - 1).astype(np.int32)


def year_month(month: int) -> str:
    return f"{month // 12:04d}-{month % 12 + 1:02d}"


def split_for_loan(loan_id: str, seed: int = 522) -> int:
    """Stable 70/15/15 assignment, independent of file order and added years."""
    digest = hashlib.blake2b(f"{seed}:{loan_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 10000
    return 0 if bucket < 7000 else 1 if bucket < 8500 else 2


def outcome_states(frame: pd.DataFrame) -> NDArray[np.int8]:
    """Unknown status is -1. Known exits take precedence over delinquency."""
    codes = frame["delinquency"].astype(str).str.strip()
    numeric = pd.to_numeric(codes, errors="coerce").to_numpy(dtype=float)
    valid = (numeric >= 0) & (numeric <= 99) & (numeric == np.floor(numeric))
    states = np.full(len(frame), -1, dtype=np.int8)
    states[valid] = np.minimum(numeric[valid], 3).astype(np.int8)
    states[codes.eq("RA").to_numpy()] = 5
    exits = frame["zero_code"].astype(str).str.strip().str.zfill(2)
    terminal = frame["zero_code"].astype(str).str.strip().ne("").to_numpy()
    states[terminal] = 6
    states[exits.eq("01").to_numpy()] = 4
    states[exits.isin(["02", "03", "09"]).to_numpy()] = 5
    return states

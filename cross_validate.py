"""Expanding-window validation: is the model good in every month, or did we get lucky once?

Each fold trains on all months BEFORE the test month and scores on that month.
Run: python cross_validate.py
"""

import numpy as np
import pandas as pd

from cleaning import clean
from evaluation import score
from features import FEATURES, add_features, build_city_coords
from model import fit_predict

TEST_MONTHS = [
    6,
    7,
    8,
    9,
    10,
]  # Jun .. Oct 2025; each one is predicted using only earlier months

NO_CALENDAR = [f for f in FEATURES if f not in ("month", "day_of_year")]
VARIANTS = {
    "all features": FEATURES,
    "without month / day_of_year": NO_CALENDAR,
    "without month / day_of_year + market_index": NO_CALENDAR
    + ["market_index"],
}


def baseline_2(train, valid):
    """Median $/mile per equipment type (the dumb baseline to beat)."""
    rpm = (
        (train["posted_rate"] / train["distance"])
        .groupby(train["equipment"], observed=True)
        .median()
    )
    return valid["distance"] * valid["equipment"].map(rpm).astype(float)


if __name__ == "__main__":
    df = pd.read_csv("data/train_test.csv")
    month = pd.to_datetime(df["date"]).dt.month

    results = {name: [] for name in ["baseline 2", *VARIANTS]}
    for m in TEST_MONTHS:
        train_raw, valid_raw = df[month < m], df[month == m]

        coords = build_city_coords(train_raw)
        train, medians = clean(train_raw)
        valid, _ = clean(valid_raw, medians)
        train, valid = add_features(train, coords), add_features(valid, coords)

        results["baseline 2"].append(
            score(valid["posted_rate"], baseline_2(train, valid))["MAE"]
        )
        for name, cols in VARIANTS.items():
            _, pred = fit_predict(
                train[cols], train["posted_rate"], valid[cols]
            )
            results[name].append(score(valid["posted_rate"], pred)["MAE"])

    table = pd.DataFrame(results, index=[f"month {m}" for m in TEST_MONTHS]).T
    table["average"] = table.mean(axis=1)
    print("Validation MAE ($) per test month (lower is better)\n")
    print(table.round(1).to_string())

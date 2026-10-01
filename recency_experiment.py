"""Does giving recent months more weight help? Same 5-month expanding-window check.

Weight of a training row = 0.5 ** (age_in_days / half_life), so a row that is
one half-life older than the newest training row counts half as much.
Run: python recency_experiment.py
"""

import numpy as np
import pandas as pd

from cleaning import clean
from evaluation import score
from features import FEATURES, add_features, build_city_coords
from model import fit_predict

TEST_MONTHS = [6, 7, 8, 9, 10]
HALF_LIVES = [
    None,
    120,
    60,
    30,
]  # days; None = every row equal (what we had before)
NO_CALENDAR = [f for f in FEATURES if f not in ("month", "day_of_year")]
FEATURE_SETS = {
    "all features": FEATURES,
    "no month / day_of_year": NO_CALENDAR,
}


def recency_weights(dates: pd.Series, half_life):
    if half_life is None:
        return None
    age_days = (dates.max() - dates).dt.days.to_numpy()
    return 0.5 ** (age_days / half_life)


if __name__ == "__main__":
    df = pd.read_csv("data/train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    month = df["date"].dt.month

    rows = {}
    for fs_name, cols in FEATURE_SETS.items():
        for hl in HALF_LIVES:
            maes = []
            for m in TEST_MONTHS:
                train_raw, valid_raw = df[month < m], df[month == m]
                coords = build_city_coords(train_raw)
                train, medians = clean(train_raw)
                valid, _ = clean(valid_raw, medians)
                train, valid = add_features(train, coords), add_features(
                    valid, coords
                )
                w = recency_weights(train["date"], hl)
                _, pred = fit_predict(
                    train[cols],
                    train["posted_rate"],
                    valid[cols],
                    sample_weight=w,
                )
                maes.append(score(valid["posted_rate"], pred)["MAE"])
            label = f"{fs_name} | half-life {'none' if hl is None else str(hl) + ' days'}"
            rows[label] = maes

    table = pd.DataFrame(rows, index=[f"month {m}" for m in TEST_MONTHS]).T
    table["average"] = table.mean(axis=1)
    print("Validation MAE ($) per test month (lower is better)\n")
    print(table.round(1).to_string())

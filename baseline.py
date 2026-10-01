"""Dumb baselines: the scores every real model has to beat.

Run: python baseline.py
"""

import pandas as pd

from cleaning import clean
from evaluation import CUTOFF, score, time_split

df = pd.read_csv("data/train_test.csv")
train, valid = time_split(df)
print(
    f"train: {len(train):,} rows (before {CUTOFF}) | validation: {len(valid):,} rows (from {CUTOFF})\n"
)

# Clean with medians learned from the training part only.
train, medians = clean(train)
valid, _ = clean(valid, medians)

# Baseline 1: every load costs (median rate per mile) x (distance).
train["rpm"] = train["posted_rate"] / train["distance"]
rpm = train["rpm"].median()
pred_1 = valid["distance"] * rpm
print(f"Baseline 1: {rpm:.3f} $/mile for every load")
print({k: round(v, 2) for k, v in score(valid["posted_rate"], pred_1).items()})

# Baseline 2: same idea, but with a separate median rate per mile per equipment type.
rpm_by_equipment = train.groupby("equipment")["rpm"].median()
pred_2 = valid["distance"] * valid["equipment"].map(rpm_by_equipment)
print("\nBaseline 2: median $/mile per equipment type")
print(rpm_by_equipment.round(3).to_dict())
print({k: round(v, 2) for k, v in score(valid["posted_rate"], pred_2).items()})

"""First real model: LightGBM on log(rate), scored on the time split.

Run: python model.py
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from cleaning import clean
from evaluation import score, time_split
from features import FEATURES, add_features, build_city_coords

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    objective="l1",  # directly minimise the average absolute error
    random_state=42,
    verbose=-1,
)


def fit_predict(X_train, y_train, X_valid, sample_weight=None):
    """Train on log(rate), predict, and convert back to dollars.

    sample_weight: optional per-row importance (e.g. recent rows count more).
    """
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_train, np.log(y_train), sample_weight=sample_weight)
    return model, np.exp(model.predict(X_valid))


if __name__ == "__main__":
    df = pd.read_csv("data/train_test.csv")
    train, valid = time_split(df)

    coords = build_city_coords(train)
    train, medians = clean(train)
    valid, _ = clean(valid, medians)
    train, valid = add_features(train, coords), add_features(valid, coords)

    # Feature sets to compare. month / day_of_year are the risky ones: the model
    # never sees future months, so they may not generalise.
    no_calendar = [f for f in FEATURES if f not in ("month", "day_of_year")]
    variants = {
        "all features": FEATURES,
        "without month / day_of_year": no_calendar,
        "distance + equipment + weight only": [
            "distance",
            "log_distance",
            "weight",
            "equipment",
        ],
    }

    for name, cols in variants.items():
        model, pred_valid = fit_predict(
            train[cols], train["posted_rate"], valid[cols]
        )
        pred_train = np.exp(model.predict(train[cols]))
        s_valid = score(valid["posted_rate"], pred_valid)
        s_train = score(train["posted_rate"], pred_train)
        print(f"{name}")
        print(
            f"   validation MAE ${s_valid['MAE']:.2f} | RMSE ${s_valid['RMSE']:.2f} | MAPE {s_valid['MAPE_%']:.2f}%"
        )
        print(
            f"   training   MAE ${s_train['MAE']:.2f}   (overfitting check: much lower than validation = bad)\n"
        )

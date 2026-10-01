"""Final pipeline: retrain on ALL labeled data, then write the two prediction files.

Run: python train.py
Then: python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
"""

import numpy as np
import pandas as pd

from cleaning import clean
from evaluation import score
from features import FEATURES, add_features, build_city_coords
from model import fit_predict
from recency_experiment import recency_weights

# Final choices (see the five-month check): no month / day_of_year, and a 60-day
# half-life so the newest months count most.
FINAL_FEATURES = [f for f in FEATURES if f not in ("month", "day_of_year")]
HALF_LIFE_DAYS = 60

TRAIN_PATH = "data/train_test.csv"
VALIDATION_PATH = "data/validation.csv"
TEMPLATE_PATH = "data/validation_predictions_template.csv"
DECEMBER_PATH = "data/december_chart_inputs.csv"

VALIDATION_OUT = "validation_predictions.csv"
DECEMBER_OUT = "december_predictions.csv"


def prepare(df_raw, medians, coords):
    """Clean, then add features, using lookups learned from the training data."""
    cleaned, _ = clean(df_raw, medians)
    return add_features(cleaned, coords)


def main():
    train_raw = pd.read_csv(TRAIN_PATH)
    val_raw = pd.read_csv(VALIDATION_PATH)
    dec_raw = pd.read_csv(DECEMBER_PATH)
    template = pd.read_csv(TEMPLATE_PATH)

    # Everything below is learned from the training data only.
    coords = build_city_coords(train_raw)
    train, medians = clean(train_raw)
    train = add_features(train, coords)
    val = prepare(val_raw, medians, coords)
    dec = prepare(dec_raw, medians, coords)

    weights = recency_weights(pd.to_datetime(train["date"]), HALF_LIFE_DAYS)
    model, _ = fit_predict(
        train[FINAL_FEATURES],
        train["posted_rate"],
        val[FINAL_FEATURES],
        sample_weight=weights,
    )

    # Predict validation loads (the model works in log space, so convert back with exp).
    val_pred = pd.DataFrame(
        {
            "load_id": val_raw["load_id"],
            "predicted_rate": np.exp(model.predict(val[FINAL_FEATURES])),
        }
    )

    # Keep the template's exact row order and ids.
    out = template[["load_id"]].merge(val_pred, on="load_id", how="left")
    assert len(out) == 12_000, "expected 12,000 validation rows"
    assert (
        out["predicted_rate"].notna().all()
    ), "some template load_ids have no prediction"
    assert (
        np.isfinite(out["predicted_rate"]).all()
        and (out["predicted_rate"] > 0).all()
    )
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv(VALIDATION_OUT, index=False)

    # December chart rows: same model, only the date changes.
    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = np.exp(
        model.predict(dec[FINAL_FEATURES])
    ).round(2)
    assert len(dec_out) == 31 and (dec_out["predicted_rate"] > 0).all()
    dec_out.to_csv(DECEMBER_OUT, index=False)

    # Sanity report.
    in_sample = np.exp(model.predict(train[FINAL_FEATURES]))
    print(
        f"Trained on {len(train):,} rows, {len(FINAL_FEATURES)} features, half-life {HALF_LIFE_DAYS} days"
    )
    print(
        "In-sample MAE (only a sanity check, not a real score): "
        f"${score(train['posted_rate'], in_sample)['MAE']:.2f}"
    )
    print(
        f"\nWrote {VALIDATION_OUT}: {len(out):,} rows, "
        f"predicted rate mean ${out['predicted_rate'].mean():,.0f} "
        f"(train actual mean ${train['posted_rate'].mean():,.0f}), "
        f"range ${out['predicted_rate'].min():,.0f} to ${out['predicted_rate'].max():,.0f}"
    )
    print(
        f"Wrote {DECEMBER_OUT}: {len(dec_out)} rows, "
        f"${dec_out['predicted_rate'].min():,.2f} to ${dec_out['predicted_rate'].max():,.2f}"
    )
    print(
        "\nNext: python score.py --predictions validation_predictions.csv "
        "--december-predictions december_predictions.csv"
    )


if __name__ == "__main__":
    main()

"""Feature engineering for the freight rate assessment."""

import numpy as np
import pandas as pd

# Fixed category order, so train / validation / December all encode equipment the same way.
EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]

FEATURES = [
    "distance",
    "log_distance",
    "weight",
    "equipment",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "day_of_week",
    "day_of_year",
]


def build_city_coords(train: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) lookup learned from the training data.

    Every city has exactly one coordinate pair, used identically as pickup and
    delivery, so the lookup is unambiguous. It is needed because the December
    file has city names but no coordinate columns.
    """
    pickups = train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(
        ["city", "lat", "lon"], axis=1
    )
    deliveries = train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(
        ["city", "lat", "lon"], axis=1
    )
    return (
        pd.concat([pickups, deliveries])
        .drop_duplicates("city")
        .set_index("city")
    )


def add_features(df: pd.DataFrame, city_coords: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a CLEANED dataframe with the model's input columns added."""
    df = df.copy()

    # Coordinates: if the file has none (December), look them up by city name.
    for role in ["pickup", "delivery"]:
        for col, source in [(f"{role}_lat", "lat"), (f"{role}_lon", "lon")]:
            looked_up = df[role].map(city_coords[source])
            df[col] = (
                df[col].fillna(looked_up) if col in df.columns else looked_up
            )

    # Distance: rate grows roughly with miles but with diminishing returns,
    # so give the model the log as well.
    df["log_distance"] = np.log(df["distance"])

    # Dates: turn one timestamp into parts the model can learn patterns from.
    date = pd.to_datetime(df["date"])
    df["month"] = date.dt.month
    df["day_of_week"] = date.dt.dayofweek  # Monday = 0 ... Sunday = 6
    df["day_of_year"] = date.dt.dayofyear

    # Equipment: a category with a fixed set of values (LightGBM reads this natively).
    df["equipment"] = pd.Categorical(
        df["equipment"], categories=EQUIPMENT_TYPES
    )

    return df


if __name__ == "__main__":
    # Quick demo: python features.py
    from cleaning import clean

    train = pd.read_csv("data/train_test.csv")
    val = pd.read_csv("data/validation.csv")
    dec = pd.read_csv("data/december_chart_inputs.csv")

    train_clean, medians = clean(train)
    coords = build_city_coords(train)

    X_train = add_features(train_clean, coords)[FEATURES]
    X_val = add_features(clean(val, medians)[0], coords)[FEATURES]
    X_dec = add_features(clean(dec, medians)[0], coords)[FEATURES]

    print("shapes:", X_train.shape, X_val.shape, X_dec.shape)
    print(
        "any missing:",
        X_train.isna().any().any(),
        X_val.isna().any().any(),
        X_dec.isna().any().any(),
    )
    print("equipment categories:", list(X_train["equipment"].cat.categories))
    print(
        "months  train:",
        sorted(X_train.month.unique()),
        "| val:",
        sorted(X_val.month.unique()),
    )
    print(X_dec.head(3).to_string())

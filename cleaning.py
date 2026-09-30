"""Data cleaning for the freight rate assessment."""

import pandas as pd

FILL_COLS = ["weight", "market_index"]  # numeric columns with missing values
DROP_COLS = ["quote_signal", "load_id"]  # leaky signal and a pure identifier


def clean(df: pd.DataFrame, medians: dict | None = None):
    """Clean a dataframe and return (clean_df, medians).

    Steps:
      a) weight -> absolute value (negative weights are sign flips)
      b) fill missing weight / market_index with a median
      c) drop quote_signal (leaks the target) and load_id (just an identifier)

    Training data:   call clean(train) with no medians. They are computed here.
    Any other data:  call clean(other, medians) with the TRAIN medians, so
                     nothing is learned from validation or test data.
    """
    df = df.copy()  # never modify the caller's dataframe

    # (a) sign flips: magnitudes are plausible, only the sign is wrong.
    #     Done before the medians so negative values don't pull them down.
    if "weight" in df.columns:
        df["weight"] = df["weight"].abs()

    # (b) medians come from the training data only, then get reused.
    present = [c for c in FILL_COLS if c in df.columns]
    if medians is None:
        medians = df[present].median().to_dict()
    df[present] = df[present].fillna({c: medians[c] for c in present})

    # (c) drop columns if they exist (the December file has no quote_signal)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return df, medians


if __name__ == "__main__":
    # Quick demo: python cleaning.py
    train = pd.read_csv("data/train_test.csv")
    val = pd.read_csv("data/validation.csv")

    train_clean, medians = clean(train)
    val_clean, _ = clean(val, medians)

    print("medians learned from train:", medians)
    print("train columns:", list(train_clean.columns))
    print(
        "missing after cleaning (train):", int(train_clean.isna().sum().sum())
    )
    print("missing after cleaning (val):  ", int(val_clean.isna().sum().sum()))
    print(
        "min weight (train, val):",
        train_clean.weight.min(),
        val_clean.weight.min(),
    )

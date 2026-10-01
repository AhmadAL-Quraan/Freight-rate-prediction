"""Time-based split and metrics, shared by every model we try."""

import numpy as np
import pandas as pd

# Everything before this date trains the model; everything from it onward is
# held out, like the real task (train on the past, predict the future).
CUTOFF = "2025-09-01"


def time_split(df: pd.DataFrame, cutoff: str = CUTOFF):
    """Split by date: earlier rows -> train, later rows -> validation."""
    date = pd.to_datetime(df["date"])
    return df[date < cutoff].copy(), df[date >= cutoff].copy()


def score(y_true, y_pred) -> dict:
    """MAE ($ off on average), RMSE (punishes big misses), MAPE (% off on average)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / y_true) * 100),
    }

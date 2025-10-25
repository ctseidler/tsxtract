# /// script
# dependencies = [
#     "tsfresh",
#     "jax",
#     "setuptools",
#     "pandas",
# ]
# ///
"""Benchmark performance against tsfresh and TSFEL."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute


def generate_random_time_series_dataset(
    n_samples: int = 100,
    n_channels: int = 1,
    sampling_rate: int = 100,
    time_series_length_in_seconds: float = 10.0,
    *,
    random_seed: int | None = None,
) -> jax.Array:
    seed = 0 if random_seed is None else random_seed
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(
        key,
        shape=(n_samples, n_channels, int(sampling_rate * time_series_length_in_seconds)),
    )


def format_for_tsfresh(data: jax.Array, sampling_rate: int = 100) -> pd.DataFrame:
    # data shape: (n_samples, n_channels, time_length)
    n_samples, n_channels, time_length = data.shape
    records = []
    for sample_idx in range(n_samples):
        for ch in range(n_channels):
            # Convert to NumPy for pandas consumption
            ts = np.array(data[sample_idx, ch])
            for t_idx, value in enumerate(ts):
                records.append(
                    {
                        "id": sample_idx * n_channels + ch,
                        "time": t_idx / sampling_rate,
                        "value": float(value),
                    },
                )
    return pd.DataFrame.from_records(records)


def extract_tsfresh(df: pd.DataFrame, settings):
    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
    )
    return features


def main():
    # Parameters
    n_samples = 50
    n_channels = 2
    sampling_rate = 50
    length_seconds = 5.0
    random_seed = 42

    # 1) Generate
    ts_data = generate_random_time_series_dataset(
        n_samples=n_samples,
        n_channels=n_channels,
        sampling_rate=sampling_rate,
        time_series_length_in_seconds=length_seconds,
        random_seed=random_seed,
    )

    # 2) Format for tsfresh
    df = format_for_tsfresh(ts_data, sampling_rate=sampling_rate)

    # 3) Extract features
    settings = EfficientFCParameters()
    features = extract_tsfresh(
        df,
        settings=settings,
    )

    # 4) Handle missing values
    impute(features)

    print("Extracted features shape:", features.shape)
    print(features.head())


if __name__ == "__main__":
    main()

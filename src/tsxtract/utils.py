"""Utility functions for tsxtract."""

import jax


def generate_random_time_series_dataset(
    n_samples: int = 100,
    n_channels: int = 5,
    sampling_rate: int = 100,
    time_series_length_in_seconds: float = 10.0,
    *,
    random_seed: int | None = None,
) -> jax.Array:
    """Generate a random time series dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, default is 100.
    n_channels : int, optional
        Number of channels per sample (1 = univariate, >1 = multivariate), defaults to 1.
    sampling_rate : int, optional
        Sampling rate in Hz, default is 100 (Hz).
    time_series_length_in_seconds : float, optional
        Length of each time series in seconds, defaults to 10.0 seconds.
    random_seed : int | None, optional
        Random Seed for reproducibility, defaults to None, which equals to random_seed = 0.

    Returns
    -------
    jax.Array
        Randomly sample time series dataset with shape
        (n_samples, n_channels, int(sampling_rate * time_series_length_in_seconds))

    Notes
    -----
    The data is sampled using a normal distribution.

    Examples
    --------
    >>> from tsxtract.utils import generate_random_time_series_dataset
    >>> array = generate_random_time_series_dataset(100, 3, 10, 10)
    >>> array.shape
    (100, 3, 100)

    """
    seed: int = 0 if random_seed is None else random_seed
    key: jax.Array = jax.random.key(seed)
    return jax.random.normal(
        key,
        shape=((n_samples, n_channels, int(sampling_rate * time_series_length_in_seconds))),
    )

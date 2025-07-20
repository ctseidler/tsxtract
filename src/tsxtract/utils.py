"""Utility functions used throughout the package.

Contents
--------
generate_random_time_series_dataset() :
    Generate a random time series dataset.

"""

from sys import stdout
from time import time

import jax
import jax.numpy as jnp


def generate_random_time_series_dataset(
    n_samples: int = 100,
    n_channels: int = 1,
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


def measure_runtime(repeats: int = 5, verbose: int = 1):
    """Measure the the runtime of a function across n repeats.

    Parameters
    ----------
    repeats : int, optional
        Number of times to run the function.
    verbose : int, optional
        Verbosity of output (1 = aggregated only, 2 = individual runs)

    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            runtimes = []
            for _ in range(repeats):
                start: float = time()
                result = func(*args, **kwargs)
                stop: float = time()
                runtimes.append(round(stop - start, 6))

            stdout.write(f"Runtime results for {func.__name__}:\n")
            if verbose > 1:
                stdout.write(f"Observed runtimes: {runtimes}\n")
            runtimes: jax.Array = jax.numpy.array(runtimes)
            msg: str = (
                f"Aggregated across {repeats} runs: {jnp.mean(runtimes):.3f}s"
                f" ±{jnp.std(runtimes):.3f}s\n"
            )
            stdout.write(msg)
            return result

        return wrapper

    return decorator


def estimate_computational_complexity_per_length(
    feature_func,
    lower: float = 2e0,
    upper: float = 2e7,
    repeats: int = 10,
) -> None:
    """Estimate the computational complexity of the given feature function."""
    measured_runtimes: dict[str, list[float]] = {}
    current_ts_length: int = int(lower)

    while current_ts_length <= upper:
        test_signal: jax.Array = jax.random.normal(
            key=jax.random.key(0),
            shape=(current_ts_length,),
        )
        measured_runtimes[f"{current_ts_length}"] = []

        for _ in range(repeats):
            start: float = time()
            feature_func(test_signal)
            stop: float = time()
            measured_runtimes[f"{current_ts_length}"].append(stop - start)

        current_ts_length *= 2

    stdout.write(f"\nLength complexity measurement results for {feature_func.__name__}:\n")
    for name, value in measured_runtimes.items():
        mean: jax.Array = jnp.mean(jnp.array(value)) * 1000
        std: jax.Array = jnp.std(jnp.array(value)) * 1000
        worst_case: jax.Array = jnp.max(jnp.array(value)) * 1000
        stdout.write(f"{name}: {mean:.3f}ms ±{std:.3f}ms (Worst case: {worst_case:.3f}ms)\n")


def estimate_computational_complexity_per_dataset_size(
    feature_func,
    lower: float = 2e0,
    upper: float = 2e4,
    repeats: int = 10,
) -> None:
    """Estimate the computational complexity of the given feature function."""
    measured_runtimes: dict[str, list[float]] = {}
    current_dataset_size: int = int(lower)

    while current_dataset_size <= upper:
        test_dataset: jax.Array = generate_random_time_series_dataset(
            n_samples=current_dataset_size,
            n_channels=1,
            sampling_rate=100,
            time_series_length_in_seconds=10,
        )
        measured_runtimes[f"{current_dataset_size}"] = []
        jax_func = jax.vmap(jax.vmap(feature_func))

        for _ in range(repeats):
            start: float = time()
            jax_func(test_dataset)
            stop: float = time()
            measured_runtimes[f"{current_dataset_size}"].append(stop - start)

        current_dataset_size *= 2

    stdout.write(f"\nDataset complexity measurement results for {feature_func.__name__}:\n")
    for name, value in measured_runtimes.items():
        mean: jax.Array = jnp.mean(jnp.array(value)) * 1000
        std: jax.Array = jnp.std(jnp.array(value)) * 1000
        worst_case: jax.Array = jnp.max(jnp.array(value)) * 1000
        stdout.write(f"{name}: {mean:.3f}ms ±{std:.3f}ms (Worst case: {worst_case:.3f}ms)\n")

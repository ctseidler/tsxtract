"""Feature extraction functions."""

from collections.abc import Callable

import jax
import jax.numpy as jnp


@jax.jit
def extract_features(dataset: jax.Array) -> dict[str, jax.Array]:
    """Extract features using tsxtract.

    Parameters
    ----------
    dataset : jax.Array
        Dataset to extract features from. Must be an array of shape
        (samples, channels, length).

    Returns
    -------
    dict[str, jax.Array] :
        Dictionary with feature names as key and extracted features as values.

    """
    extracted_features: dict[str, jax.Array] = {}

    extracted_features["maximum"] = _flat_vmap(maximum, dataset)
    extracted_features["mean"] = _flat_vmap(mean, dataset)
    extracted_features["minimum"] = _flat_vmap(minimum, dataset)
    extracted_features["absolute_maximum"] = _flat_vmap(absolute_maximum, dataset)
    extracted_features["length"] = _flat_vmap(length, dataset)
    extracted_features["median"] = _flat_vmap(median, dataset)
    extracted_features["peak_to_peak_distance"] = _flat_vmap(peak_to_peak_distance, dataset)
    extracted_features["standard_deviation"] = _flat_vmap(standard_deviation, dataset)
    extracted_features["sum_values"] = _flat_vmap(sum_values, dataset)
    extracted_features["variance"] = _flat_vmap(variance, dataset)
    extracted_features["variance_larger_than_standard_deviation"] = _flat_vmap(
        variance_larger_than_standard_deviation,
        dataset,
    )
    extracted_features["zero_crossing_rate"] = _flat_vmap(zero_crossing_rate, dataset)

    return extracted_features


def _flat_vmap(function: Callable, sample: jax.Array) -> jax.Array:
    """Apply vmap on (samples, channels) simultaneously."""
    samples, channels, length = sample.shape
    sample_flat = sample.reshape(samples * channels, length)
    result = jax.vmap(function)(sample_flat)
    return result.reshape(samples, channels, *result.shape[1:])


def absolute_maximum(signal: jax.Array) -> jax.Array:
    """Get the absolute maximum value of the signal."""
    return jnp.max(jnp.abs(signal))


def length(signal: jax.Array) -> int:
    """Get the number of observation (length) of the signal."""
    return signal.size


def maximum(signal: jax.Array) -> jax.Array:
    """Get the maximal value in the signal."""
    return jnp.max(signal)


def mean(signal: jax.Array) -> jax.Array:
    """Calculate the mean value of the signal."""
    return jnp.mean(signal)


def median(signal: jax.Array) -> jax.Array:
    """Calculate the median value of the signal."""
    return jnp.median(signal)


def minimum(signal: jax.Array) -> jax.Array:
    """Get the minimal value of the signal."""
    return jnp.min(signal)


def peak_to_peak_distance(signal: jax.Array) -> jax.Array:
    """Calculate the peak-to-peak distance of the signal."""
    return jnp.max(signal) - jnp.min(signal)


def standard_deviation(signal: jax.Array) -> jax.Array:
    """Calculate the standard deviation of the signal."""
    return jnp.std(signal)


def sum_values(signal: jax.Array) -> jax.Array:
    """Calculate the sum of all values of the signal."""
    return jnp.sum(signal)


def variance(signal: jax.Array) -> jax.Array:
    """Calculate the variance of the signal."""
    return jnp.var(signal)


def variance_larger_than_standard_deviation(signal: jax.Array) -> jax.Array:
    """Determine whether the variance of the signal is larger than its standard deviation."""
    return jnp.var(signal) > jnp.std(signal)


def zero_crossing_rate(signal: jax.Array) -> jax.Array:
    """Calculate the zero-crossing rate of the signal."""
    signs = jnp.sign(signal)
    return jnp.sum(jnp.diff(signs) != 0)

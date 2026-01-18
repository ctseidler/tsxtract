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
    extracted_features["absolute_energy"] = _flat_vmap(absolute_energy, dataset)
    extracted_features["absolute_sum_of_changes"] = _flat_vmap(absolute_sum_of_changes, dataset)
    extracted_features["absolute_sum_of_values"] = _flat_vmap(absolute_sum_of_values, dataset)
    extracted_features["count_above_mean"] = _flat_vmap(count_above_mean, dataset)
    extracted_features["count_below_mean"] = _flat_vmap(count_below_mean, dataset)
    extracted_features["distance"] = _flat_vmap(distance, dataset)
    extracted_features["first_location_of_maximum"] = _flat_vmap(first_location_of_maximum, dataset)
    extracted_features["first_location_of_minimum"] = _flat_vmap(first_location_of_minimum, dataset)
    extracted_features["has_duplicate"] = _flat_vmap(has_duplicate, dataset)
    extracted_features["has_duplicate_max"] = _flat_vmap(has_duplicate_max, dataset)
    extracted_features["has_duplicate_min"] = _flat_vmap(has_duplicate_min, dataset)
    extracted_features["interquartile_range"] = _flat_vmap(interquartile_range, dataset)

    return extracted_features


def _flat_vmap(function: Callable, sample: jax.Array) -> jax.Array:
    """Apply vmap on (samples, channels) simultaneously."""
    samples, channels, length = sample.shape
    sample_flat = sample.reshape(samples * channels, length)
    result = jax.vmap(function)(sample_flat)
    return result.reshape(samples, channels, *result.shape[1:])


def absolute_energy(signal: jax.Array) -> jax.Array:
    """Calculate the absolute energy of the signal."""
    return jnp.dot(signal, signal)


def absolute_maximum(signal: jax.Array) -> jax.Array:
    """Get the absolute maximum value of the signal."""
    if signal.size == 0:
        return jnp.array(jnp.nan)
    return jnp.max(jnp.abs(signal))


def absolute_sum_of_changes(signal: jax.Array) -> jax.Array:
    """Calculate the absolute sum of changes of the signal."""
    if signal.size < 2:
        return jnp.array(0.0)
    return jnp.sum(jnp.abs(jnp.diff(signal)))


def absolute_sum_of_values(signal: jax.Array) -> jax.Array:
    """Calculate the sum of absolute values of the signal."""
    return jnp.sum(jnp.abs(signal))


def count_above_mean(signal: jax.Array) -> jax.Array:
    """Count the number of values in the signal that are greater than the mean."""
    mean_val = jnp.nanmean(signal)
    return jax.lax.cond(
        jnp.isnan(mean_val),
        lambda _: jnp.array(0, dtype=jnp.int32),  # Mean is NaN --> no value above mean
        lambda _: jnp.sum(signal > mean_val),
        operand=None,
    )


def count_below_mean(signal: jax.Array) -> jax.Array:
    """Count the number of values in the signal that are smaller than the mean."""
    mean_val = jnp.nanmean(signal)
    return jax.lax.cond(
        jnp.isnan(mean_val),
        lambda _: jnp.array(0, dtype=jnp.int32),  # Mean is NaN --> no value above mean
        lambda _: jnp.sum(signal < mean_val),
        operand=None,
    )


def distance(signal: jax.Array) -> jax.Array:
    """Calculate the total 'path length' of the signal."""
    if signal.size < 2:
        return jnp.array(jnp.nan)
    differences = jnp.diff(signal).astype(float)
    return jnp.sum(jnp.sqrt(1 + differences**2))


def first_location_of_maximum(signal: jax.Array) -> jax.Array:
    """Get the relative index of the first occurrence of the maximum value in the signal."""
    size = signal.size
    if size == 0:
        return jnp.array(jnp.nan)
    return jax.lax.cond(
        size > 1,
        lambda _: jnp.argmax(signal) / (size - 1),
        lambda _: jnp.array(0.0),  # Scalar array --> value is always maximum
        operand=None,
    )


def first_location_of_minimum(signal: jax.Array) -> jax.Array:
    """Get the relative index of the first occurrence of the minimum value in the signal."""
    size = signal.size
    if size == 0:
        return jnp.array(jnp.nan)
    return jax.lax.cond(
        size > 1,
        lambda _: jnp.argmin(signal) / (size - 1),
        lambda _: jnp.array(0.0),  # Scalar array --> value is always minimum
        operand=None,
    )


def has_duplicate(signal: jax.Array) -> jax.Array:
    """Check whether the signal has any duplicate values."""
    if signal.size == 0:
        return jnp.array(False)

    _, unique_values = jnp.unique_counts(signal, size=signal.size)
    return jnp.any(jnp.greater(unique_values, 1))


def has_duplicate_max(signal: jax.Array) -> jax.Array:
    """Check whether the maximum value in the signal occurs more than once."""
    if signal.size == 0:
        return jnp.array(False)
    return jnp.sum(signal == jnp.max(signal)) > 1


def has_duplicate_min(signal: jax.Array) -> jax.Array:
    """Check whether the minimum value in the signal occurs more than once."""
    if signal.size == 0:
        return jnp.array(False)
    return jnp.sum(signal == jnp.min(signal)) > 1


def interquartile_range(signal: jax.Array) -> jax.Array:
    """Calculate the interquartile range of the signal."""
    if signal.size == 0:
        return jnp.array(jnp.nan)

    return jnp.percentile(signal, 75) - jnp.percentile(signal, 25)


def length(signal: jax.Array) -> int:
    """Get the number of observation (length) of the signal."""
    return signal.size


def maximum(signal: jax.Array) -> jax.Array:
    """Get the maximal value in the signal."""
    if signal.size == 0:
        return jnp.array(jnp.nan)
    return jnp.max(signal)


def mean(signal: jax.Array) -> jax.Array:
    """Calculate the mean value of the signal."""
    return jnp.mean(signal)


def median(signal: jax.Array) -> jax.Array:
    """Calculate the median value of the signal."""
    if signal.size == 0:
        return jnp.array(jnp.nan)
    return jnp.median(signal)


def minimum(signal: jax.Array) -> jax.Array:
    """Get the minimal value of the signal."""
    if signal.size == 0:
        return jnp.array(jnp.nan)
    return jnp.min(signal)


def peak_to_peak_distance(signal: jax.Array) -> jax.Array:
    """Calculate the peak-to-peak distance of the signal."""
    if signal.size == 0:
        return jnp.array(jnp.nan)
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

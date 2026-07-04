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
    extracted_features["std"] = _flat_vmap(std, dataset)
    extracted_features["variance"] = _flat_vmap(variance, dataset)
    extracted_features["median"] = _flat_vmap(median, dataset)
    extracted_features["rms"] = _flat_vmap(rms, dataset)
    extracted_features["mad"] = _flat_vmap(mad, dataset)
    extracted_features["percentile_25"] = _flat_vmap(lambda x: percentile(x, 25), dataset)
    extracted_features["percentile_75"] = _flat_vmap(lambda x: percentile(x, 75), dataset)

    return extracted_features


def _flat_vmap(function: Callable, sample: jax.Array) -> jax.Array:
    """Apply vmap on (samples, channels) simultaneously."""
    samples, channels, length = sample.shape
    sample_flat = sample.reshape(samples * channels, length)
    result = jax.vmap(function)(sample_flat)
    return result.reshape(samples, channels, *result.shape[1:])


def maximum(signal: jax.Array) -> jax.Array:
    """Get the maximal value in the signal."""
    return jnp.max(signal)


def mean(signal: jax.Array) -> jax.Array:
    """Calculate the mean value of the signal."""
    return jnp.mean(signal)


def minimum(signal: jax.Array) -> jax.Array:
    """Get the minimal value of the signal."""
    return jnp.min(signal)


def std(signal: jax.Array) -> jax.Array:
    """Calculate the standard deviation of the signal."""
    return jnp.std(signal)


def variance(signal: jax.Array) -> jax.Array:
    """Calculate the variance of the signal."""
    return jnp.var(signal)


def median(signal: jax.Array) -> jax.Array:
    """Calculate the median of the signal."""
    return jnp.median(signal)

def rms(signal: jax.Array) -> jax.Array:
    """Calculate the root mean square of the signal."""
    return jnp.sqrt(jnp.mean(jnp.square(signal)))

def mad(signal: jax.Array) -> jax.Array:
    """Calculate the mean absolute deviation of the signal."""
    mean_value = jnp.mean(signal)
    return jnp.mean(jnp.abs(signal - mean_value))

def percentile(signal: jax.Array, q: float) -> jax.Array:
    """Calculate the q-th percentile of the signal."""
    return jnp.percentile(signal, q)

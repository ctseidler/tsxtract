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

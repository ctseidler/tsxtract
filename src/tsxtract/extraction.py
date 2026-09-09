"""Batched feature extraction driver."""

from collections.abc import Callable
from functools import partial

import jax

from tsxtract.feature_settings import DEFAULT_FEATURE_SPECS, FeatureSpec



def _flat_vmap(function: Callable, sample: jax.Array) -> jax.Array:
    """Apply vmap on (samples, channels) simultaneously."""
    samples, channels, length = sample.shape
    sample_flat = sample.reshape(samples * channels, length)
    result = jax.vmap(function)(sample_flat)
    return result.reshape(samples, channels, *result.shape[1:])



@partial(jax.jit, static_argnames=("feature_specs",))
def extract_features(
    dataset: jax.Array,
    sampling_rate: float,
    feature_specs: tuple[FeatureSpec, ...] = DEFAULT_FEATURE_SPECS,
) -> dict[str, jax.Array]:
    """Extract features using tsxtract.

    Parameters
    ----------
    dataset : jax.Array
        Dataset to extract features from. Must be an array of shape
        (samples, channels, length).
    sampling_rate : float
        Sampling rate of the dataset. Passed on to every feature whose
        signature accepts it.
    feature_specs : tuple[FeatureSpec, ...], optional
        Which features to extract and with which parameters (default
        ``DEFAULT_FEATURE_SPECS``). This argument is static: it takes
        part in the JIT cache key, so a new configuration triggers one
        recompilation, and every specification must be hashable.

    Returns
    -------
    dict[str, jax.Array] :
        Dictionary with feature names as key and extracted features as
        values. A feature returning several values per signal (for
        example ``autocorrelation`` at several lags) keeps them in a
        trailing axis of its entry.

    """
    extracted_features: dict[str, jax.Array] = {}

    for spec in feature_specs:
        feature_function = spec.to_function(sampling_rate)
        extracted_features[spec.output_name] = _flat_vmap(feature_function, dataset)

    return extracted_features

def to_columns(features: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Flatten multi-valued entries so that every entry is one output column.

    ``extract_features`` keeps the several values of a multi-valued
    feature in a trailing axis, which lets tsxtract compute them in a
    single pass. Benchmarks and table exports need the column-per-feature
    layout that tsfresh and TSFEL produce, which is what this function
    returns: an entry of shape ``(samples, channels, k)`` becomes ``k``
    entries of shape ``(samples, channels)``, suffixed with the position
    of the value inside the parameter tuple of its specification.

    Parameters
    ----------
    features : dict[str, jax.Array]
        Output of ``extract_features``.

    Returns
    -------
    dict[str, jax.Array] :
        Dictionary with one entry per output column, each of shape
        (samples, channels).

    """
    columns: dict[str, jax.Array] = {}
    for name, values in features.items():
        if values.ndim <= 2:
            columns[name] = values
            continue
        for index in range(values.shape[-1]):
            columns[f"{name}__{index}"] = values[..., index]
    return columns
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
    extracted_features["skewness"] = _flat_vmap(skewness, dataset)
    extracted_features["kurtosis"] = _flat_vmap(kurtosis, dataset)
    extracted_features["zero_crossing_rate"] = _flat_vmap(zero_crossing_rate, dataset)
    extracted_features["autocorrelation"] = _flat_vmap(lambda x: autocorrelation(x, [1, 2, 3]), dataset)

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

def skewness(signal: jax.Array) -> jax.Array:
    """Calculate the skewness of the signal.
       Matches scipy.stats.skew with bias=True (population convention, denominator n), 
       consistent with std and variance in this module.
    """
    mean_value = jnp.mean(signal)
    std_value = jnp.std(signal)
    return jnp.mean(jnp.power((signal - mean_value) / std_value, 3))

def kurtosis(signal: jax.Array) -> jax.Array:
    """Calculate the kurtosis of the signal.
        Returns excess kurtosis (Fisher's definition, normal distribution → 0),
        matching scipy.stats.kurtosis with default fisher=True.
    """
    mean_value = jnp.mean(signal)
    std_value = jnp.std(signal)
    return jnp.mean(jnp.power((signal - mean_value) / std_value, 4)) - 3

def zero_crossing_rate(signal: jax.Array) -> jax.Array:
    """Calculate the zero crossing rate of the signal.
        Counts sign changes between adjacent samples, normalised by the number
        of adjacent pairs (n-1). A sample that is exactly 0.0 is counted as two
        crossings (sign goes +1 → 0 → -1); negligible for float sensor data.
    """
    sign_changes = jnp.diff(jnp.sign(signal)) != 0
    return jnp.sum(sign_changes) / (signal.shape[0] - 1)

def autocorrelation(signal: jax.Array, lags: tuple[int, ...]) -> jax.Array:
    """Calculate the autocorrelation of the signal at the given lags.

    Normalised by the total sum of squares so that lag=0 gives 1.0 and the
    result lies in [-1, 1]. This matches statsmodels' acf(adjusted=False)
    (biased estimator), which keeps the autocovariance matrix positive
    semi-definite. Returns NaN for a constant signal (zero variance),
    consistent with skewness and kurtosis in this module.

    Note: lags must be static Python ints, as they determine slice shapes.
    """
    n = signal.shape[0]
    for k in lags:
        if not 0 <= k < n:
            raise ValueError(f"lag {k} must satisfy 0 <= lag < {n}")

    mean_value = jnp.mean(signal)
    centred = signal - mean_value
    denominator = jnp.sum(jnp.square(centred))
    results = [jnp.sum(centred[: n - k] * centred[k:]) / denominator for k in lags]
    return jnp.stack(results)

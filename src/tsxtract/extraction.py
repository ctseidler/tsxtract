"""Feature extraction functions."""

import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


class FeatureSpec:
    """One entry of a feature configuration: which function to call, with
    which parameters, and under which name the result is stored.

    Keeping the configuration separate from ``extract_features`` allows
    the same feature to be requested several times with different
    parameters -- two percentiles, several frequency bands -- without
    editing the extractor:

    >>> FeatureSpec(percentile, q=25.0)
    >>> FeatureSpec(band_energy, low_frequency=0.6, high_frequency=2.5)
    >>> FeatureSpec(spectral_centroid)

    Parameters
    ----------
    function : Callable
        Feature function taking a one-dimensional signal as its first
        argument.
    name : str, optional
        Overrides the generated output name. Keyword-only, so it cannot
        be mistaken for a parameter of the feature function.
    **parameters
        Extra arguments for the feature function. ``sampling_rate`` is
        not passed here: it is injected by :meth:`to_function` for every
        function whose signature accepts it.

    Notes
    -----
    Instances are immutable and hashable because ``extract_features``
    receives them as a static JIT argument. Parameter values must
    therefore be hashable too: use a tuple ``(1, 2, 3)``, never a list.

    """

    def __init__(
        self, function: Callable, *, name: str | None = None, **parameters: Any
    ) -> None:
        # Assigned through object.__setattr__ because __setattr__ below is
        # closed: a hashable object must not change after creation.
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "parameters", tuple(parameters.items()))
        object.__setattr__(self, "name", name)

    def __setattr__(self, attribute: str, value: Any) -> None:
        raise AttributeError(
            f"FeatureSpec is immutable; create a new one instead of setting {attribute!r}"
        )

    @property
    def output_name(self) -> str:
        """Name this specification stores its result under.

        Without parameters the plain function name; with parameters the
        pairs are appended as ``__name_value``, following the naming
        convention of tsfresh so that the mapping between the three
        packages' output columns stays mechanical.
        """
        if self.name is not None:
            return self.name
        parts = [self.function.__name__]
        parts += [
            f"{parameter}_{_format_parameter_value(value)}"
            for parameter, value in self.parameters
        ]
        return "__".join(parts)

    def to_function(self, sampling_rate: float) -> Callable:
        """Return a one-argument function ``signal -> feature``.

        ``sampling_rate`` is forwarded only to those feature functions
        that declare it. The signature is the single source of truth, so
        a feature can never fall out of sync with a manually maintained
        list of "spectral" features.
        """
        arguments = dict(self.parameters)
        if "sampling_rate" in inspect.signature(self.function).parameters:
            arguments["sampling_rate"] = sampling_rate
        return partial(self.function, **arguments)

    def __repr__(self) -> str:
        return f"FeatureSpec({self.output_name})"

    def _fields(self) -> tuple:
        return (self.function, self.parameters, self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureSpec):
            return NotImplemented
        return self._fields() == other._fields()

    def __hash__(self) -> int:
        return hash(self._fields())


def _format_parameter_value(value: Any) -> str:
    """Render one parameter value for use inside a feature name.

    Tuples are joined with underscores (``(1, 2, 3)`` -> ``1_2_3``) and
    whole-numbered floats lose their decimal part (``25.0`` -> ``25``),
    so that the generated names stay short and stable.
    """
    if isinstance(value, tuple):
        return "_".join(_format_parameter_value(item) for item in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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
         

def _spectral_distribution(signal: jax.Array, sampling_rate: float) -> tuple[jax.Array, jax.Array]:
    """Return ``(fft_frequencies, weights)``: the magnitude spectrum as a
    probability distribution over frequency (weights sum to 1).

    Shared first stage of the spectral features. DC is removed before the
    FFT (result reflects oscillation, not offset); magnitude weighting
    follows TSFEL. For a constant signal the weights are NaN (0/0), which
    propagates to NaN in every downstream feature.
    """
    centred = signal - jnp.mean(signal)
    magnitude = jnp.abs(jnp.fft.rfft(centred))
    fft_frequencies = jnp.fft.rfftfreq(signal.shape[0], 1.0 / sampling_rate)
    return fft_frequencies, magnitude / jnp.sum(magnitude)


def spectral_centroid(signal: jax.Array, sampling_rate: float) -> jax.Array:
    """Calculate the spectral centroid (magnitude-weighted mean frequency, in Hz).

    The mean is subtracted before the FFT, so the result reflects the
    oscillatory content rather than the signal's offset. This differs from
    TSFEL (DC kept); on zero-mean signals both agree exactly.

    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    sampling_rate : float
        Sampling rate in Hz, used to convert FFT bins to frequencies.

    Returns
    -------
    jax.Array
        Spectral centroid in Hz. NaN for a constant signal.

    """
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate)
    return jnp.sum(fft_frequencies * weights)


def _spectral_moment(signal: jax.Array, sampling_rate: float, order: int) -> jax.Array:
    """Return the ``order``-th central moment of the spectral distribution.

    order=2 gives the spectral variance, i.e. ``spectral_bandwidth**2``.
    Time-domain analogy: variance/skewness are the 2nd/3rd central
    moments of the signal's values; here the distribution is over frequency.
    """
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate)
    mu = jnp.sum(fft_frequencies * weights)
    return jnp.sum((fft_frequencies - mu) ** order * weights)


def spectral_bandwidth(signal: jax.Array, sampling_rate: float) -> jax.Array:
    """Calculate the spectral bandwidth (weighted std of the spectrum, in Hz).

    Square root of the second central spectral moment: how far the
    spectral energy spreads around the centroid. Matches TSFEL's
    ``spectral_spread`` up to DC handling (removed here, kept in TSFEL);
    on zero-mean signals both agree exactly.

    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    sampling_rate : float
        Sampling rate in Hz, used to convert FFT bins to frequencies.

    Returns
    -------
    jax.Array
        Spectral bandwidth in Hz. NaN for a constant signal.

    """
    return jnp.sqrt(_spectral_moment(signal, sampling_rate, 2))


def spectral_rolloff(
    signal: jax.Array, sampling_rate: float, roll_percent: float = 0.85
) -> jax.Array:
    """Calculate the spectral rolloff (in Hz): the lowest frequency below
    which ``roll_percent`` of the spectral magnitude is contained.

    The quantile of the spectral distribution: the cumulative weight is
    scanned from low to high frequency and the first frequency reaching
    the threshold is returned. Follows the magnitude (not power)
    convention of TSFEL and librosa; note their defaults differ (TSFEL
    hard-codes 0.95, librosa defaults to 0.85 -- adopted here). DC is
    removed first, unlike in both reference packages; on zero-mean
    signals the results agree.

    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    sampling_rate : float
        Sampling rate in Hz, used to convert FFT bins to frequencies.
    roll_percent : float, optional
        Fraction of the total spectral magnitude to accumulate, between
        0 and 1 (default 0.85).

    Returns
    -------
    jax.Array
        Spectral rolloff in Hz. NaN for a constant signal.

    """
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate)
    cumulative = jnp.cumsum(weights)
    index = jnp.argmax(cumulative >= roll_percent)
    # NaN weights (constant signal) make every comparison False, so
    # argmax would silently return index 0 (i.e. 0.0 Hz); restore the
    # NaN contract explicitly.
    return jnp.where(jnp.isnan(cumulative[-1]), jnp.nan, fft_frequencies[index])

def dominant_frequency(signal: jax.Array, sampling_rate: float) -> jax.Array:
    """Calculate the dominant frequency (in Hz): the frequency of the
    strongest component in the spectrum.

    The frequency of the largest magnitude bin. Weighting by magnitude or
    by power gives the same answer, since squaring does not change which
    bin is largest -- so this feature needs no magnitude/power convention.
    Neither tsfresh nor TSFEL provides this feature, so the definition
    follows the Proposal directly. Two properties worth knowing: the
    result is quantised to the FFT grid (spacing ``sampling_rate / N``,
    no interpolation between bins), and for exactly equal peaks the
    lowest frequency is returned. DC is removed first (without that, a
    signal with an offset would peak at 0 Hz).

    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    sampling_rate : float
        Sampling rate in Hz, used to convert FFT bins to frequencies.

    Returns
    -------
    jax.Array
        Dominant frequency in Hz. NaN for a constant signal.

    """
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate)
    index = jnp.argmax(weights)
    # NaN weights (constant signal): argmax would silently return index 0
    # (i.e. 0.0 Hz), so restore the NaN contract explicitly -- same guard
    # as in spectral_rolloff.
    return jnp.where(jnp.isnan(weights).any(), jnp.nan, fft_frequencies[index])

def spectral_entropy(signal: jax.Array, sampling_rate: float) -> jax.Array:
    """Calculate the normalised spectral entropy (dimensionless, in [0, 1]).

    Shannon entropy of the magnitude spectrum treated as a probability
    distribution over frequency, divided by its maximum possible value
    ``log2(number of bins)``. 0 means all energy sits in a single bin
    (pure tone); 1 means it is spread evenly over all bins (white noise).

    Two deliberate deviations from TSFEL's ``spectral_entropy``: the
    distribution is magnitude-weighted (TSFEL squares the magnitude),
    so that every spectral feature in tsxtract shares one distribution
    (see ``_spectral_distribution``); and the normalisation uses the
    total number of bins, not the number of non-zero bins, so that a
    single non-zero bin gives 0 instead of a division by zero. The
    difference in weighting is quantified in the benchmark.

    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    sampling_rate : float
        Sampling rate in Hz. Only kept for a uniform spectral interface;
        the entropy itself does not depend on the frequency axis.

    Returns
    -------
    jax.Array
        Normalised spectral entropy. NaN for a constant signal.

    """
    _, weights = _spectral_distribution(signal, sampling_rate)
    # 0 * log(0) is defined as 0, but log2(0) is -inf and 0 * -inf is NaN.
    # Replace zero weights by 1 inside the log (log2(1) = 0): the product
    # is then 0 * 0 = 0 for those bins and unchanged everywhere else.
    safe_weights = jnp.where(weights > 0, weights, 1.0)
    entropy = -jnp.sum(weights * jnp.log2(safe_weights))
    return entropy / jnp.log2(weights.shape[0])


def _power_distribution(signal: jax.Array, sampling_rate: float) -> tuple[jax.Array, jax.Array]:
    """Return ``(fft_frequencies, weights)``: the power spectrum as a
    probability distribution over frequency (weights sum to 1).

    Power counterpart of ``_spectral_distribution``. Shared first stage
    of the features named for energy or power, which by convention use
    the squared magnitude; the shape descriptors (centroid, bandwidth,
    rolloff, entropy) use the magnitude spectrum instead. DC is removed
    before the FFT, so a constant signal gives NaN weights (0/0).
    """
    centred = signal - jnp.mean(signal)
    power = jnp.abs(jnp.fft.rfft(centred)) ** 2
    fft_frequencies = jnp.fft.rfftfreq(signal.shape[0], 1.0 / sampling_rate)
    return fft_frequencies, power / jnp.sum(power)


def band_energy(
    signal: jax.Array,
    sampling_rate: float,
    low_frequency: float,
    high_frequency: float,
) -> jax.Array:
    """Calculate the fraction of spectral energy inside a frequency band
    (dimensionless, in [0, 1]).

    The band covers every FFT bin whose frequency ``f`` satisfies
    ``low_frequency <= f < high_frequency``; the result is the energy in
    those bins divided by the energy in all bins. Bands that tile the
    frequency axis therefore sum to 1. The mean is subtracted before the
    FFT, so the 0 Hz bin carries no energy and the fraction reflects
    oscillation only.

    Weighting is by power, not magnitude: the feature is named for
    energy, and under the magnitude convention a 3:1 amplitude pair
    would give 0.75 rather than the 0.9 that "energy fraction" means.
    This follows librosa, where band-energy features
    (``melspectrogram``) use ``power=2`` while the ``spectral_*`` shape
    descriptors use ``power=1``.

    This generalises TSFEL's ``human_range_energy``, which fixes the
    band to 0.6-2.5 Hz and keeps DC. TSFEL selects the bins nearest to
    the band edges; when the edges lie on the FFT grid both conventions
    pick the same bins.

    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    sampling_rate : float
        Sampling rate in Hz, used to convert FFT bins to frequencies.
    low_frequency : float
        Lower band edge in Hz (inclusive).
    high_frequency : float
        Upper band edge in Hz (exclusive).

    Returns
    -------
    jax.Array
        Fraction of total energy inside the band. NaN for a constant
        signal (no energy at all).

    """
    fft_frequencies, weights = _power_distribution(signal, sampling_rate)
    in_band = (fft_frequencies >= low_frequency) & (fft_frequencies < high_frequency)
    fraction = jnp.sum(jnp.where(in_band, weights, 0.0))
    # NaN weights (constant signal): if no bin falls inside the band the
    # ``where`` would mask every NaN away and the fraction would silently
    # be 0.0; restore the NaN contract explicitly.
    return jnp.where(jnp.isnan(weights).any(), jnp.nan, fraction)


def power_bandwidth(
    signal: jax.Array, sampling_rate: float, power_fraction: float = 0.90
) -> jax.Array:
    """Calculate the power bandwidth (in Hz): the width of the frequency
    band that carries ``power_fraction`` of the total power.

    The cumulative power distribution is scanned from low to high
    frequency; the lower edge is the frequency reaching
    ``(1 - power_fraction) / 2`` and the upper edge the one reaching
    ``(1 + power_fraction) / 2``, so equal tails are cut off on both
    sides. The result is the distance between them. Frequency-domain
    analogy of the interquartile range, and the two-sided counterpart of
    ``spectral_rolloff``, which reports a single quantile.

    Weighting is by power, following the convention for features named
    for energy or power (see ``_power_distribution``). Being a quantile
    width, the feature ignores a low noise floor, unlike the
    magnitude-weighted ``spectral_bandwidth``.

    TSFEL's ``power_bandwidth`` computes the same quantity from a Welch
    periodogram; its Hann window spreads each tone over neighbouring
    bins, which widens the result by roughly two bins (a pure tone gives
    0.2 Hz there and 0.0 Hz here). Its hard-coded 95% threshold is
    applied from both ends, i.e. it corresponds to ``power_fraction =
    0.90``, the default adopted here.

    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    sampling_rate : float
        Sampling rate in Hz, used to convert FFT bins to frequencies.
    power_fraction : float, optional
        Fraction of the total power the band must carry, between 0 and 1
        (default 0.90). Equal tails of ``(1 - power_fraction) / 2`` are
        excluded at each end.

    Returns
    -------
    jax.Array
        Power bandwidth in Hz. NaN for a constant signal.

    """
    fft_frequencies, weights = _power_distribution(signal, sampling_rate)
    cumulative = jnp.cumsum(weights)
    lower_index = jnp.argmax(cumulative >= (1.0 - power_fraction) / 2.0)
    upper_index = jnp.argmax(cumulative >= (1.0 + power_fraction) / 2.0)
    width = fft_frequencies[upper_index] - fft_frequencies[lower_index]
    # NaN weights (constant signal) make every comparison False, so both
    # argmax calls would return index 0 and the width would silently be
    # 0.0 Hz; restore the NaN contract explicitly -- same guard as in
    # spectral_rolloff.
    return jnp.where(jnp.isnan(cumulative[-1]), jnp.nan, width)


TEMPORAL_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(maximum),
    FeatureSpec(mean),
    FeatureSpec(minimum),
    FeatureSpec(std),
    FeatureSpec(variance),
    FeatureSpec(median),
    FeatureSpec(rms),
    FeatureSpec(mad),
    FeatureSpec(percentile, q=25.0),
    FeatureSpec(percentile, q=75.0),
    FeatureSpec(skewness),
    FeatureSpec(kurtosis),
    FeatureSpec(zero_crossing_rate),
    FeatureSpec(autocorrelation, lags=(1, 2, 3)),
)
"""Default time-domain features."""

SPECTRAL_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(spectral_centroid),
    FeatureSpec(spectral_bandwidth),
    FeatureSpec(spectral_rolloff),
    FeatureSpec(dominant_frequency),
    FeatureSpec(spectral_entropy),
    FeatureSpec(band_energy, low_frequency=0.6, high_frequency=2.5),
    FeatureSpec(power_bandwidth),
)
"""Default frequency-domain features."""

DEFAULT_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    TEMPORAL_FEATURE_SPECS + SPECTRAL_FEATURE_SPECS
)
"""Feature set extracted when no configuration is given."""


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
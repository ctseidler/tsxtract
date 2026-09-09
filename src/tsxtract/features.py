"""Feature functions.

Each function takes a one-dimensional signal and returns one value (or a
small vector). They are independent of the extraction machinery, so they
can be used on their own or reused in another package.
"""

from typing import Literal

import jax
import jax.numpy as jnp


def maximum(signal: jax.Array) -> jax.Array:
    """The largest value in the signal.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Largest value. NaN if the signal contains NaN.
 
    """
    return jnp.max(signal)


def mean(signal: jax.Array) -> jax.Array:
    """The arithmetic mean of the signal's values.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Arithmetic mean. NaN for an empty signal.
 
    """
    return jnp.mean(signal)


def minimum(signal: jax.Array) -> jax.Array:
    """The smallest value in the signal.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Smallest value. NaN if the signal contains NaN.
 
    """
    return jnp.min(signal)


def std(signal: jax.Array) -> jax.Array:
    """The standard deviation of the signal's values.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Population standard deviation (denominator ``n``), matching
        ``numpy.std`` with its default ``ddof=0``. NaN for an empty
        signal.
    """
    return jnp.std(signal)


def variance(signal: jax.Array) -> jax.Array:
    """The variance of the signal's values.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Population variance (denominator ``n``), matching ``numpy.var``
        with its default ``ddof=0``. NaN for an empty signal.
 
    """
    return jnp.var(signal)


def median(signal: jax.Array) -> jax.Array:
    """The median of the signal's values.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Median value; for an even number of samples the mean of the two
        middle values. NaN if the signal contains NaN.
 
    """
    return jnp.median(signal)

def rms(signal: jax.Array) -> jax.Array:
    """The root mean square of the signal's values.
 
    Unlike the standard deviation the mean is not subtracted, so a
    constant offset contributes to the result.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Root mean square. NaN for an empty signal.
 
    """
    return jnp.sqrt(jnp.mean(jnp.square(signal)))

def mad(signal: jax.Array) -> jax.Array:
    """The mean absolute deviation of the signal's values.
 
    Mean of the absolute distances to the signal's mean. Less sensitive
    to outliers than the standard deviation, which squares them.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Mean absolute deviation. NaN for an empty signal.
 
    """
    mean_value = jnp.mean(signal)
    return jnp.mean(jnp.abs(signal - mean_value))

def percentile(signal: jax.Array, q: float) -> jax.Array:
    """The q-th percentile of the signal's values.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    q : float
        Percentile to compute, between 0 and 100.
 
    Returns
    -------
    jax.Array
        The q-th percentile, linearly interpolated between the two
        neighbouring samples when q does not fall on one exactly.
 
    """
    return jnp.percentile(signal, q)

def skewness(signal: jax.Array) -> jax.Array:
    """The skewness of the signal's value distribution.
 
    Third standardised central moment: positive for a right-tailed
    distribution, negative for a left-tailed one, zero for a symmetric
    one. Matches ``scipy.stats.skew`` with ``bias=True`` (population
    convention, denominator ``n``), consistent with ``std`` and
    ``variance`` in this module.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Skewness, dimensionless. NaN for a constant signal (zero
        standard deviation).
 
    """
    mean_value = jnp.mean(signal)
    std_value = jnp.std(signal)
    return jnp.mean(jnp.power((signal - mean_value) / std_value, 3))

def kurtosis(signal: jax.Array) -> jax.Array:
    """The excess kurtosis of the signal's value distribution.
 
    Fourth standardised central moment minus 3, so that a normal
    distribution gives 0. Matches ``scipy.stats.kurtosis`` with its
    defaults ``fisher=True`` and ``bias=True``.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Excess kurtosis, dimensionless. NaN for a constant signal (zero
        standard deviation).
 
    """
    mean_value = jnp.mean(signal)
    std_value = jnp.std(signal)
    return jnp.mean(jnp.power((signal - mean_value) / std_value, 4)) - 3

def zero_crossing_rate(signal: jax.Array) -> jax.Array:
    """The rate at which the signal changes sign.
 
    Counts sign changes between adjacent samples, normalised by the
    number of adjacent pairs (``n - 1``). A sample that is exactly 0.0
    is counted as two crossings (the sign goes +1 -> 0 -> -1); this is
    negligible for floating-point sensor data. Being based on counting
    rather than arithmetic, the feature does not propagate NaN.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
 
    Returns
    -------
    jax.Array
        Sign changes divided by ``n - 1``, between 0 and 1. NaN for a
        signal with a single sample.
 
    """
    sign_changes = jnp.diff(jnp.sign(signal)) != 0
    return jnp.sum(sign_changes) / (signal.shape[0] - 1)

def autocorrelation(signal: jax.Array, lags: tuple[int, ...]) -> jax.Array:
    """The autocorrelation of the signal at the given lags.
 
    Normalised by the total sum of squares so that lag 0 gives 1.0 and
    the result lies in [-1, 1]. This matches statsmodels'
    ``acf(adjusted=False)`` (biased estimator), which keeps the
    autocovariance matrix positive semi-definite.
 
    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    lags : tuple[int, ...]
        Lags to evaluate. Must be static Python ints, since they
        determine the shapes of the slices.
 
    Returns
    -------
    jax.Array
        One value per lag, in the order given. NaN for a constant signal
        (zero variance), consistent with ``skewness`` and ``kurtosis``.
 
    Raises
    ------
    ValueError
        If a lag is negative or not smaller than the signal length.
 
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
         

def _spectral_distribution(
    signal: jax.Array,
    sampling_rate: float,
    weighting: Literal["magnitude", "power"] = "magnitude",
) -> tuple[jax.Array, jax.Array]:
    """Return ``(fft_frequencies, weights)``: the spectrum as a probability
    distribution over frequency (weights sum to 1).

    Shared first stage of every spectral feature. ``weighting`` selects
    which spectrum the distribution is built from: ``"magnitude"`` for
    the shape descriptors (centroid, bandwidth, rolloff, entropy),
    ``"power"`` -- the squared magnitude -- for the features named for
    energy or power. DC is removed before the FFT, so a constant signal
    gives NaN weights (0/0), which every downstream feature reports as
    NaN.
    """
    if weighting not in ("magnitude", "power"):
        raise ValueError(
            f"weighting must be 'magnitude' or 'power', got {weighting!r}"
        )
    centred = signal - jnp.mean(signal)
    magnitude = jnp.abs(jnp.fft.rfft(centred))
    spectrum = magnitude if weighting == "magnitude" else magnitude**2
    fft_frequencies = jnp.fft.rfftfreq(signal.shape[0], 1.0 / sampling_rate)
    return fft_frequencies, spectrum / jnp.sum(spectrum)


def spectral_centroid(signal: jax.Array, sampling_rate: float) -> jax.Array:
    """The magnitude-weighted mean frequency of the spectrum, in Hz.
 
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
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate, weighting="magnitude")
    return jnp.sum(fft_frequencies * weights)


def _spectral_moment(signal: jax.Array, sampling_rate: float, order: int) -> jax.Array:
    """The ``order``-th central moment of the spectral distribution.
 
    order=2 gives the spectral variance, i.e. ``spectral_bandwidth**2``.
    Time-domain analogy: variance/skewness are the 2nd/3rd central
    moments of the signal's values; here the distribution is over frequency.
    """
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate, weighting="magnitude")
    mu = jnp.sum(fft_frequencies * weights)
    return jnp.sum((fft_frequencies - mu) ** order * weights)


def spectral_bandwidth(signal: jax.Array, sampling_rate: float) -> jax.Array:
    """The magnitude-weighted standard deviation of the spectrum, in Hz.
 
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
    """The lowest frequency below which a given share of the magnitude lies.
 
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
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate, weighting="magnitude")
    cumulative = jnp.cumsum(weights)
    index = jnp.argmax(cumulative >= roll_percent)
    # NaN weights (constant signal) make every comparison False, so
    # argmax would silently return index 0 (i.e. 0.0 Hz); restore the
    # NaN contract explicitly.
    return jnp.where(jnp.isnan(cumulative[-1]), jnp.nan, fft_frequencies[index])

def dominant_frequency(signal: jax.Array, sampling_rate: float) -> jax.Array:
    """The frequency of the strongest component in the spectrum, in Hz.
 
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
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate, weighting="magnitude")
    index = jnp.argmax(weights)
    # NaN weights (constant signal): argmax would silently return index 0
    # (i.e. 0.0 Hz), so restore the NaN contract explicitly -- same guard
    # as in spectral_rolloff.
    return jnp.where(jnp.isnan(weights).any(), jnp.nan, fft_frequencies[index])


def spectral_entropy(
    signal: jax.Array,
    sampling_rate: float,
    weighting: Literal["magnitude", "power"] = "magnitude",
) -> jax.Array:
    """The normalised Shannon entropy of the spectrum, between 0 and 1.

    Shannon entropy of the spectrum treated as a probability distribution
    over frequency, divided by its maximum possible value
    ``log2(number of bins)``. 0 means all energy sits in a single bin
    (pure tone); 1 means it is spread evenly over all bins (white noise).

    The default is magnitude weighting, so that the feature shares one
    spectral distribution with the other ``spectral_*`` features (see
    ``_spectral_distribution``). ``weighting="power"`` reproduces the
    convention of TSFEL's ``spectral_entropy``, which squares the
    magnitude. The normalisation always uses the total number of bins,
    not the number of non-zero bins, so that a single non-zero bin gives
    0 instead of a division by zero.

    Parameters
    ----------
    signal : jax.Array
        One-dimensional input signal.
    sampling_rate : float
        Sampling rate in Hz. Only kept for a uniform spectral interface;
        the entropy itself does not depend on the frequency axis.
    weighting : {"magnitude", "power"}, optional
        Whether the distribution is weighted by the magnitude spectrum
        (default) or by the power spectrum (squared magnitude).

    Returns
    -------
    jax.Array
        Normalised spectral entropy. NaN for a constant signal.

    Raises
    ------
    ValueError
        If ``weighting`` is neither ``"magnitude"`` nor ``"power"``.

    """
    _, weights = _spectral_distribution(signal, sampling_rate, weighting)
    # 0 * log(0) is defined as 0, but log2(0) is -inf and 0 * -inf is NaN.
    # Replace zero weights by 1 inside the log (log2(1) = 0): the product
    # is then 0 * 0 = 0 for those bins and unchanged everywhere else.
    safe_weights = jnp.where(weights > 0, weights, 1.0)
    entropy = -jnp.sum(weights * jnp.log2(safe_weights))
    return entropy / jnp.log2(weights.shape[0])


def band_energy(
    signal: jax.Array,
    sampling_rate: float,
    low_frequency: float,
    high_frequency: float,
) -> jax.Array:
    """The fraction of spectral energy inside a frequency band.
 
    Dimensionless, between 0 and 1. The band covers every FFT bin whose
    frequency ``f`` satisfies ``low_frequency <= f < high_frequency``;
    the result is the energy in those bins divided by the energy in all
    bins. Bands that tile the
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

    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate, weighting="power")
    in_band = (fft_frequencies >= low_frequency) & (fft_frequencies < high_frequency)
    fraction = jnp.sum(jnp.where(in_band, weights, 0.0))
    # NaN weights (constant signal): if no bin falls inside the band the
    # ``where`` would mask every NaN away and the fraction would silently
    # be 0.0; restore the NaN contract explicitly.
    return jnp.where(jnp.isnan(weights).any(), jnp.nan, fraction)


def power_bandwidth(
    signal: jax.Array, sampling_rate: float, power_fraction: float = 0.90
) -> jax.Array:
    """The width of the frequency band carrying a given share of the power.
 
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
    fft_frequencies, weights = _spectral_distribution(signal, sampling_rate, weighting="power")
    cumulative = jnp.cumsum(weights)
    lower_index = jnp.argmax(cumulative >= (1.0 - power_fraction) / 2.0)
    upper_index = jnp.argmax(cumulative >= (1.0 + power_fraction) / 2.0)
    width = fft_frequencies[upper_index] - fft_frequencies[lower_index]
    # NaN weights (constant signal) make every comparison False, so both
    # argmax calls would return index 0 and the width would silently be
    # 0.0 Hz; restore the NaN contract explicitly -- same guard as in
    # spectral_rolloff.
    return jnp.where(jnp.isnan(cumulative[-1]), jnp.nan, width)


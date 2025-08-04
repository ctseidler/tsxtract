"""Module containing the feature extractors."""

from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def absolute_energy(time_series: jax.Array) -> jax.Array:
    r"""
    Compute the absolute energy of a time series.

    The absolute energy is defined as the sum of the squared values of the
    time series, which is equivalent to the squared L2 norm. It is often used
    as a measure of the overall magnitude of the signal over time.

    .. math::

        E = \sum_{i=1}^{N} x_i^2

    where :math:`x_i` are the values of the time series and :math:`N` is the
    number of samples.

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Scalar value representing the absolute energy of the time series.

    Notes
    -----
    This implementation uses the dot product :math:`x \cdot x` for efficient
    computation.
    """
    time_series = jnp.asarray(time_series)
    return jnp.dot(time_series, time_series)


@jax.jit
def absolute_maximum(time_series: jax.Array) -> jax.Array:
    r"""Calculate the absolute maximum value of a time series.

    The absolute maximum is defined as:

    .. math::
        \\max(|x_i|) \\quad \text{for} \\quad i = 1, \\ldots, N

    where \\(x_i\\) are the elements of the input time series.

    Parameters
    ----------
    time_series : jax.Array
        Input 1D array (time series) from which to compute the absolute maximum.

    Returns
    -------
    jax.Array
        Scalar array representing the maximum absolute value of the input.

    Raises
    ------
    ValueError
        If the input array is empty.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> ts = jnp.array([-3.0, 2.0, -5.0])
    >>> absolute_maximum(ts)
    5.0

    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        msg: str = "Input array is empty; cannot compute maximum."
        raise ValueError(msg)

    return jnp.max(jnp.abs(time_series))


@jax.jit
def absolute_sum_of_changes(time_series: jax.Array) -> jax.Array:
    r"""Calculate the absolute sum of changes in a time series.

    The absolute sum of changes is defined as:

    .. math::
        \sum_{i=1}^{N-1} \left| x_{i+1} - x_i \right|

    where \(x_i\) are the elements of the input time series.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the sum of absolute differences between consecutive elements.
        Returns ``0`` if the input contains fewer than 2 elements.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> ts = jnp.array([1.0, 2.5, 0.5])
    >>> absolute_sum_of_changes(ts)
    4.0

    """
    time_series = jnp.asarray(time_series)
    if time_series.size < 2:
        return jnp.array(0.0)

    return jnp.sum(jnp.abs(jnp.diff(time_series)))


@jax.jit
def absolute_sum_values(time_series: jax.Array) -> jax.Array:
    r"""Calculate the sum of absolute values of a time series.

    The sum of absolute values is defined as:

    .. math::
        \sum_{i=1}^{N} |x_i|

    where \(x_i\) are the elements of the input time series.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the sum of absolute values.
        Returns ``0`` for an empty array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> ts = jnp.array([1.0, -2.0, 3.0])
    >>> absolute_sum_values(ts)
    6.0

    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(0.0)
    return jnp.sum(jnp.abs(time_series))


@partial(jax.jit, static_argnames=["lag"])
def autocorrelation(time_series: jax.Array, lag: int) -> jax.Array:
    r"""
    Compute the sample autocorrelation of a time series at a specified lag.

    The autocorrelation measures the linear relationship between the time
    series and a lagged version of itself. For lag :math:`\ell`, it is
    calculated as the covariance between :math:`x_t` and :math:`x_{t+\ell}`
    normalized by the variance of the series.

    .. math::

        r_\ell = \frac{\sum_{t=1}^{N-\ell} (x_t - \bar{x})(x_{t+\ell} - \bar{x})}
                       {(N-\ell) \, \sigma_x^2}

    where:
        - :math:`x_t` is the time series value at time :math:`t`
        - :math:`\bar{x}` is the mean of the series
        - :math:`\sigma_x^2` is the variance of the series
        - :math:`N` is the number of samples

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.
    lag : int
        The lag (number of time steps) at which to compute the autocorrelation.
        Must satisfy :math:`0 < \text{lag} < N`.

    Returns
    -------
    jax.Array
        Scalar value representing the sample autocorrelation at the given lag.
        Returns NaN if:
        - `lag >= len(time_series)`
        - variance of the series is zero

    Notes
    -----
    - Uses the unbiased sample autocorrelation definition.
    - Computation is JIT-compiled with `jax.jit` for efficiency.
    """
    time_series = jnp.asarray(time_series)

    # Early exit if lag is too large
    if lag <= 0 or lag >= len(time_series):
        return jnp.array(jnp.nan)

    ts_mean = jnp.mean(time_series)
    subseries_1 = time_series[: len(time_series) - lag]
    subseries_2 = time_series[lag:]

    sum_product = jnp.sum((subseries_1 - ts_mean) * (subseries_2 - ts_mean))
    variance = jnp.var(time_series)

    return jax.lax.cond(
        jnp.isclose(variance, 0),
        lambda _: jnp.array(jnp.nan),
        lambda _: sum_product / ((len(time_series) - lag) * variance),
        operand=None,
    )


@partial(jax.jit, static_argnames=["max_bins"])
def binned_entropy(time_series: jax.Array, max_bins: int) -> jax.Array:
    r"""
    Compute the entropy of a binned time series.

    The values of the time series are first discretized into `max_bins`
    equally spaced bins. The Shannon entropy is then calculated from the
    empirical probability distribution of the bins.

    .. math::

        H = - \sum_{i=1}^{B} p_i \log p_i

    where:
        - :math:`B` is the number of bins
        - :math:`p_i` is the probability of a value falling into bin :math:`i`

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.
    max_bins : int
        Maximum number of equally spaced bins to discretize the values into.
        Must be >= 1.

    Returns
    -------
    jax.Array
        Scalar value representing the Shannon entropy (in nats) of the binned
        distribution. Returns NaN if the input contains fewer than 2 samples.

    Notes
    -----
    - Uses natural logarithm, so the entropy is expressed in **nats**.
    - Bins are of equal width, determined by `jnp.histogram`.
    - Empty bins contribute 0 to the entropy.
    """
    time_series = jnp.asarray(time_series)

    # Early exit for too-short series
    if time_series.size < 2 or max_bins < 1:
        return jnp.array(jnp.nan)

    # Histogram binning
    hist, _ = jnp.histogram(time_series, bins=max_bins)

    # Convert to probabilities
    probs = hist / time_series.size

    # Remove zero-probability bins from calculation to avoid log(0)
    nonzero_probs = jnp.where(probs > 0, probs, jnp.nan)

    # TODO: If we use log2 here, we calculate the entropy in bits instead of nats
    return -jnp.sum(nonzero_probs * jnp.log(nonzero_probs))


@partial(jax.jit, static_argnames=["lag"])
def c3(time_series: jax.Array, lag: int) -> jax.Array:
    r"""
    Compute the C3 statistic for measuring nonlinearity in a time series.

    The C3 statistic is defined as the mean of the triple product of lagged
    versions of the time series. For lag :math:`\ell`, it is:

    .. math::

        C_3(\ell) = \frac{1}{N - 2\ell} \sum_{t=1}^{N - 2\ell}
        x_t \, x_{t+\ell} \, x_{t+2\ell}

    where:
        - :math:`x_t` is the value of the time series at time :math:`t`
        - :math:`N` is the length of the time series
        - :math:`\ell` is the lag (in samples)

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.
    lag : int
        The lag (number of time steps) used for the C3 computation.
        Must satisfy :math:`0 < 2 \cdot \text{lag} < N`.

    Returns
    -------
    jax.Array
        Scalar value of the C3 statistic for the given lag.
        Returns 0.0 if :math:`2 \cdot \text{lag} \ge N`.

    Notes
    -----
    - The C3 statistic is often used to detect cubic nonlinearities in time
      series, as it is sensitive to triple correlations.
    - This implementation avoids using the wrapped-around values from
      `jnp.roll` by explicitly truncating the valid range.
    """
    time_series = jnp.asarray(time_series)
    n = time_series.size

    if lag <= 0 or 2 * lag >= n:
        return jnp.array(0.0)

    # Create lagged versions
    x0 = time_series
    x1 = jnp.roll(time_series, -lag)
    x2 = jnp.roll(time_series, -2 * lag)

    # Compute only over the valid range to avoid wrapped values
    valid_range = slice(0, n - 2 * lag)
    return jnp.mean(x0[valid_range] * x1[valid_range] * x2[valid_range])


@partial(jax.jit, static_argnames=["threshold"])
def count_above(time_series: jax.Array, threshold: float) -> jax.Array:
    r"""Fraction of values in the time series that are greater or equal than the threshold.

    The fraction is defined as:

    .. math::
        \\frac{\\sum_{i=1}^{N} [x_i \\ge t]}{N}

    where :math:`t` is the threshold.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.
    threshold : float
        The threshold value.

    Returns
    -------
    jax.Array
        Fraction of values greater than or equal to ``threshold``.
        Returns ``nan`` for an empty array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> ts = jnp.array([1, 2, 3])
    >>> count_above(ts, 2)
    0.6666667

    """
    time_series = jnp.asarray(time_series)
    size = time_series.size
    if size == 0:
        return jnp.array(jnp.nan)
    return jnp.sum(time_series >= threshold) / size


@jax.jit
def count_above_mean(time_series: jax.Array) -> jax.Array:
    r"""Count the number of values in the time series that are strictly greater than the mean.

    The count is defined as:

    .. math::
        \\sum_{i=1}^{N} [x_i > \\bar{x}]

    where :math:`\\bar{x}` is the mean of the array.

    Notes
    -----
    - NaNs are treated as not above the mean.
    - If the array is empty, returns ``0``.
    - If the mean is NaN (e.g., all values are NaN), returns ``0``.
    - ``Inf`` values are treated as NaNs.
    - NaNs are ignored in the mean calculation.
    - Values equal to the mean are not counted.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Number of values greater than the mean.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> ts = jnp.array([0, 1, 2])
    >>> count_above_mean(ts)
    Array(1, dtype=int32)

    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(0, dtype=jnp.int32)

    mean_val = jnp.nanmean(time_series)

    return jax.lax.cond(
        jnp.isnan(mean_val),
        lambda _: jnp.array(0, dtype=jnp.int32),  # Mean is NaN --> no value above mean
        lambda _: jnp.sum(time_series > mean_val),
        operand=None,
    )


@partial(jax.jit, static_argnames=["threshold"])
def count_below(time_series: jax.Array, threshold: float) -> jax.Array:
    r"""
    Compute the proportion of values in a time series that are less than or equal to a given threshold.

    For a threshold :math:`\tau`, the statistic is defined as:

    .. math::

        p_{\le \tau} = \frac{\#\{\, x_i \mid x_i \le \tau \,\}}{N}

    where:
        - :math:`x_i` is the i-th value in the time series
        - :math:`N` is the number of samples
        - :math:`\#\{\cdot\}` denotes the count of values satisfying the condition

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.
    threshold : float
        The cutoff value :math:`\tau` for comparison.

    Returns
    -------
    jax.Array
        Scalar in the range [0, 1] representing the fraction of values
        less than or equal to the threshold. Returns NaN if the time series
        is empty.

    Notes
    -----
    - Uses :math:`\le` (less than or equal) comparison.
    - This implementation is JIT-compiled with `jax.jit` for efficiency.
    """
    time_series = jnp.asarray(time_series)

    n = time_series.size
    if n == 0:
        return jnp.array(jnp.nan)

    return jnp.sum(time_series <= threshold) / n


@jax.jit
def count_below_mean(time_series: jax.Array) -> jax.Array:
    r"""
    Count the number of values in a time series that are strictly below its mean.

    Let :math:`\bar{x}` denote the mean of the time series. The statistic is:

    .. math::

        c_{<\bar{x}} = \#\{\, x_i \mid x_i < \bar{x} \,\}

    where:
        - :math:`x_i` is the i-th value in the series
        - :math:`\#\{\cdot\}` denotes the count of values satisfying the condition

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Integer scalar representing the count of values strictly below the mean.
        Returns NaN if the series is empty.

    Notes
    -----
    - Uses the strict less-than comparison (:math:`<`), not less-than-or-equal.
    - This implementation is JIT-compiled with `jax.jit` for efficiency.
    """
    time_series = jnp.asarray(time_series)

    if time_series.size == 0:
        return jnp.array(jnp.nan)

    mean_val = jnp.mean(time_series)
    return jnp.sum(time_series < mean_val)


@jax.jit
def distance(time_series: jax.Array) -> jax.Array:
    r"""
    Compute the total "path length" of a time series.

    This is the sum of Euclidean distances between consecutive points, assuming
    the time axis increments by 1 between samples. For a time series
    :math:`x_1, x_2, \dots, x_N`, the statistic is:

    .. math::

        D = \sum_{i=1}^{N-1} \sqrt{1 + (x_{i+1} - x_i)^2}

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Scalar value representing the total path length of the time series.
        Returns NaN if the input has fewer than two samples.

    Notes
    -----
    - The `1` inside the square root corresponds to the fixed unit time step
      between samples.
    - This measure increases with both the variability and amplitude of the
      changes in the time series.
    - This implementation is equivalent to the **"distance traveled"** feature
      in TSFEL.

    References
    ----------
    - Barandas, M. et al. (2020). TSFEL: Time Series Feature Extraction Library.
      *SoftwareX*, 11, 100456. doi:10.1016/j.softx.2020.100456
    """
    time_series = jnp.asarray(time_series)

    if time_series.size < 2:
        return jnp.array(jnp.nan)

    differences = jnp.diff(time_series).astype(float)
    return jnp.sum(jnp.sqrt(1 + differences**2))


@jax.jit
def first_location_of_maximum(time_series: jax.Array) -> jax.Array:
    r"""
    Compute the relative index of the first occurrence of the maximum value in a time series.

    For a time series :math:`x_1, x_2, \dots, x_N`, let

    .. math::

        k = \min\{\, i \mid x_i = \max_j x_j \,\}

    Then the statistic is:

    .. math::

        r_{\max} = \frac{k}{N}

    where:
        - :math:`k` is the index of the first maximum (0-based index in this implementation)
        - :math:`N` is the length of the time series

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Scalar in the range [0, 1) representing the relative location of the first maximum.
        Returns NaN if the series is empty.

    Notes
    -----
    - Uses the first occurrence of the maximum in case of ties.
    - The output is relative to the series length, not an absolute index.
    - This implementation is JIT-compiled with `jax.jit` for efficiency.
    """
    time_series = jnp.asarray(time_series)

    n = time_series.size
    if n == 0:
        return jnp.array(jnp.nan)

    first_max_idx = jnp.argmax(time_series)  # first occurrence by default
    return first_max_idx / n


@jax.jit
def first_location_of_minimum(time_series: jax.Array) -> jax.Array:
    r"""
    Compute the relative index of the first occurrence of the minimum value in a time series.

    For a time series :math:`x_1, x_2, \dots, x_N`, let

    .. math::

        k = \min\{\, i \mid x_i = \min_j x_j \,\}

    Then the statistic is:

    .. math::

        r_{\min} = \frac{k}{N}

    where:
        - :math:`k` is the index of the first minimum (0-based index in this implementation)
        - :math:`N` is the length of the time series

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Scalar in the range [0, 1) representing the relative location of the first minimum.
        Returns NaN if the series is empty.

    Notes
    -----
    - Uses the first occurrence of the minimum in case of ties.
    - The output is relative to the series length, not an absolute index.
    - This implementation is JIT-compiled with `jax.jit` for efficiency.
    """
    time_series = jnp.asarray(time_series)

    n = time_series.size
    if n == 0:
        return jnp.array(jnp.nan)

    first_min_idx = jnp.argmin(time_series)  # first occurrence by default
    return first_min_idx / n


@jax.jit
def interquartile_range(time_series: jax.Array) -> jax.Array:
    r"""
    Calculate the interquartile range (IQR) of a time series.

    The interquartile range is the difference between the 75th and 25th percentiles:

    .. math::

        \text{IQR} = Q_3 - Q_1

    where:
        - :math:`Q_1` is the 25th percentile (first quartile)
        - :math:`Q_3` is the 75th percentile (third quartile)

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Scalar representing the interquartile range.
        Returns NaN if the input array is empty.

    Notes
    -----
    - The IQR is a measure of statistical dispersion and is robust to outliers.
    - This implementation is based on the TSFEL definition.
    - Percentiles are computed using `jnp.percentile` with linear interpolation.
    """
    time_series = jnp.asarray(time_series)

    if time_series.size == 0:
        return jnp.array(jnp.nan)

    q75 = jnp.percentile(time_series, 75)
    q25 = jnp.percentile(time_series, 25)
    return q75 - q25


@jax.jit
def is_symmetric(time_series: jax.Array, tolerance: float) -> jax.Array:
    r"""
    Determine if the distribution of a time series is approximately symmetric.

    This feature compares the absolute difference between the mean and the median
    to a fraction of the data range (max - min). Formally:

    .. math::

        \text{is\_symmetric} = 
        \begin{cases}
        \text{True}, & \text{if } | \bar{x} - \tilde{x} | < \tau (x_{\max} - x_{\min}) \\
        \text{False}, & \text{otherwise}
        \end{cases}

    where:
        - :math:`\bar{x}` is the mean of the time series,
        - :math:`\tilde{x}` is the median,
        - :math:`x_{\max}` and :math:`x_{\min}` are the maximum and minimum values,
        - :math:`\tau` is the tolerance threshold.

    Parameters
    ----------
    time_series : jax.Array
        1D array of time series values.
    tolerance : float
        Threshold factor controlling the acceptable normalized mean-median difference
        for symmetry detection. Typical values are small (e.g., 0.05).

    Returns
    -------
    jax.Array
        Boolean scalar indicating if the time series distribution is symmetric (True)
        or not (False). Returns False if the time series is empty.

    Notes
    -----
    - The method is a heuristic based on mean-median closeness relative to the range.
    - Useful for detecting skewness in the distribution.
    - JIT-compiled with `jax.jit` for performance.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(False)

    mean_val = jnp.mean(time_series)
    median_val = jnp.median(time_series)
    range_val = jnp.max(time_series) - jnp.min(time_series)

    # Avoid division by zero if range is zero (all values identical)
    return jnp.where(
        range_val == 0,
        True,
        jnp.abs(mean_val - median_val) < tolerance * range_val,
    )


@jax.jit
def has_duplicate(time_series: jax.Array) -> jax.Array:
    """
    Check whether the time series contains any duplicate values.

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Boolean scalar: True if duplicates exist, False otherwise.
        Returns False if the series is empty.

    Notes
    -----
    - Uses JAX's `jnp.unique` to determine the number of unique values.
    - Efficient and JIT-compiled with `jax.jit`.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(False)

    unique_values = jnp.unique(time_series, size=time_series.size)
    return jnp.array(time_series.size > unique_values.size)


@jax.jit
def has_duplicate_max(time_series: jax.Array) -> jax.Array:
    """
    Check if the maximum value in the time series occurs more than once.

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Boolean scalar: True if the maximum value appears multiple times,
        False otherwise. Returns False if the series is empty.

    Notes
    -----
    - Efficiently compares all values to the maximum using vectorized operations.
    - JIT-compiled with `jax.jit` for performance.
    """
    time_series = jnp.asarray(time_series)

    if time_series.size == 0:
        return jnp.array(False)

    max_value = jnp.max(time_series)
    count_max = jnp.sum(time_series == max_value)
    return count_max > 1


@jax.jit
def has_duplicate_min(time_series: jax.Array) -> jax.Array:
    """
    Check if the minimum value in the time series occurs more than once.

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Boolean scalar: True if the minimum value appears multiple times,
        False otherwise. Returns False if the series is empty.

    Notes
    -----
    - Efficiently checks duplicates of the minimal value using vectorized operations.
    - JIT-compiled with `jax.jit` for optimized performance.
    """
    time_series = jnp.asarray(time_series)

    if time_series.size == 0:
        return jnp.array(False)

    min_value = jnp.min(time_series)
    count_min = jnp.sum(time_series == min_value)
    return count_min > 1


@partial(jax.jit, static_argnames=["n_bins"])
def hist_mode(time_series: jax.Array, n_bins: int) -> jax.Array:
    """
    Estimate the mode of a time series by calculating the histogram mode with specified bins.

    The mode is approximated as the midpoint of the histogram bin with the highest frequency.

    Formally, let the histogram bins be defined by edges

    .. math::

        b_0, b_1, \dots, b_{K}

    and counts

    .. math::

        h_k = \text{count of values in } [b_k, b_{k+1})

    The histogram mode is approximated as

    .. math::

        \hat{x}_{mode} = \frac{b_{k^*} + b_{k^*+1}}{2}

    where

    .. math::

        k^* = \arg\max_k h_k

    Parameters
    ----------
    time_series : jax.Array
        1D array of time series values.
    n_bins : int
        Number of bins to use for histogram calculation.

    Returns
    -------
    jax.Array
        Scalar representing the approximate mode value of the time series.
        Returns NaN if the input series is empty or if n_bins is not positive.

    Notes
    -----
    - The accuracy of this mode estimate depends on the bin count.
    - Uses `jnp.histogram` to compute histogram counts and bin edges.
    - JIT-compiled with `jax.jit` for efficiency.
    """
    time_series = jnp.asarray(time_series)

    if time_series.size == 0 or n_bins <= 0:
        return jnp.array(jnp.nan)

    hist_values, bin_edges = jnp.histogram(time_series, bins=n_bins)
    max_bin_idx = jnp.argmax(hist_values)

    # Calculate the midpoint of the bin with the highest count
    mode_estimate = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0
    return mode_estimate


@jax.jit
def large_standard_deviation(time_series: jax.Array, threshold_ratio: float) -> jax.Array:
    """
    Check if the standard deviation of the time series is large relative to its range.

    Formally, it returns True if:

    .. math::

        \sigma > \tau (x_{max} - x_{min})

    where:
        - :math:`\sigma` is the standard deviation of the time series,
        - :math:`x_{max}` and :math:`x_{min}` are the maximum and minimum values,
        - :math:`\tau` is the threshold ratio.

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.
    threshold_ratio : float
        Multiplier for the range to determine what qualifies as a "large" standard deviation.

    Returns
    -------
    jax.Array
        Boolean scalar: True if standard deviation is larger than threshold_ratio times the range,
        False otherwise. Returns False if the time series is empty or has zero range.

    Notes
    -----
    - The check normalizes the standard deviation by the data range.
    - If all values are identical (zero range), the function returns False.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(False)

    data_range = jnp.max(time_series) - jnp.min(time_series)
    std_dev = jnp.std(time_series)

    return jnp.where(data_range == 0, False, std_dev > (threshold_ratio * data_range))


@jax.jit
def last_location_of_maximum(time_series: jax.Array) -> jax.Array:
    """
    Compute the relative position of the last occurrence of the maximum value in the time series.

    The relative location is given as a fraction in [0, 1], where 0 corresponds to
    the first element and 1 to the last element.

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Scalar float representing the relative last position of the maximum value.
        Returns NaN if the input time series is empty.

    Notes
    -----
    - The last occurrence index is found by reversing the series and locating the first max.
    - The relative position is calculated as:

    .. math::

        \text{last\_loc} = 1 - \frac{\text{argmax}(x[::-1])}{N}

    where :math:`N` is the length of the time series.
    """
    time_series = jnp.asarray(time_series)
    n = len(time_series)
    if n == 0:
        return jnp.array(jnp.nan)

    last_max_pos_from_end = jnp.argmax(time_series[::-1])
    relative_last_max_pos = 1.0 - (last_max_pos_from_end / n)
    return jnp.array(relative_last_max_pos)


@jax.jit
def last_location_of_minimum(time_series: jax.Array) -> jax.Array:
    """
    Compute the relative position of the last occurrence of the minimum value in the time series.

    The relative location is expressed as a fraction in [0, 1], where 0 corresponds
    to the first element and 1 to the last element.

    Parameters
    ----------
    time_series : jax.Array
        1D array containing the time series values.

    Returns
    -------
    jax.Array
        Scalar float representing the relative last position of the minimum value.
        Returns NaN if the input time series is empty.

    Notes
    -----
    - Finds the last occurrence by reversing the series and locating the first minimum.
    - The relative position is calculated as:

    .. math::

        \text{last\_loc} = 1 - \frac{\text{argmin}(x[::-1])}{N}

    where :math:`N` is the length of the time series.
    """
    time_series = jnp.asarray(time_series)
    n = len(time_series)
    if n == 0:
        return jnp.array(jnp.nan)

    last_min_pos_from_end = jnp.argmin(time_series[::-1])
    relative_last_min_pos = 1.0 - (last_min_pos_from_end / n)
    return jnp.array(relative_last_min_pos)


@jax.jit
def length(time_series: jax.Array) -> int:
    """
    Calculate the length (number of observations) of the time series.

    The length corresponds to the number of data points in the series.
    For example, a signal measured at 10 Hz over 5 seconds has length 10 * 5 = 50.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    int
        Number of observations in the time series.
    """
    time_series = jnp.asarray(time_series)
    return time_series.size


@jax.jit
def maximum(time_series: jax.Array) -> jax.Array:
    """
    Calculate the maximum value in the time series.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar containing the maximum value of the time series.

    Notes
    -----
    - Returns `-inf` if the input array is empty.
    """
    time_series = jnp.asarray(time_series)
    return jnp.max(time_series)


@jax.jit
def mean(time_series: jax.Array) -> jax.Array:
    """
    Calculate the mean (average) value of the time series.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the mean value of the time series.

    Notes
    -----
    - Returns NaN if the input array is empty.
    """
    time_series = jnp.asarray(time_series)
    return jnp.mean(time_series)


@jax.jit
def mean_abs_change(time_series: jax.Array) -> jax.Array:
    """
    Calculate the mean absolute change between consecutive values in the time series.

    This feature measures the average magnitude of the first differences:

    .. math::

        \text{mean\_abs\_change} = \frac{1}{N-1} \sum_{i=1}^{N-1} |x_{i+1} - x_i|

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the average absolute change between consecutive elements.
        Returns NaN if the time series length is less than 2.
    """
    time_series = jnp.asarray(time_series)
    diffs = jnp.diff(time_series)
    return jnp.mean(jnp.abs(diffs))


@jax.jit
def mean_change(time_series: jax.Array) -> jax.Array:
    """
    Calculate the mean change per time step in the time series.

    This feature measures the average slope or rate of change between the first
    and last points of the series:

    .. math::

        \text{mean\_change} = \frac{x_{N-1} - x_0}{N - 1}

    where :math:`N` is the length of the time series.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the average change per step.
        Returns NaN if the time series has fewer than 2 elements.
    """
    time_series = jnp.asarray(time_series)
    n = len(time_series)
    return (time_series[-1] - time_series[0]) / (n - 1) if n > 1 else jnp.array(jnp.nan)


@partial(jax.jit, static_argnames=["number_of_maxima"])
def mean_n_absolute_max(time_series: jax.Array, number_of_maxima: int) -> jax.Array:
    """
    Calculate the mean of the top `number_of_maxima` absolute maximum values in the time series.

    The function first takes the absolute value of all elements, then finds the largest
    `number_of_maxima` values and returns their mean.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.
    number_of_maxima : int
        Number of absolute maximum values to consider (must be > 0).

    Returns
    -------
    jax.Array
        Scalar representing the mean of the top `number_of_maxima` absolute maximum values.
        Returns NaN if the time series length is less than `number_of_maxima`.

    Raises
    ------
    AssertionError
        If `number_of_maxima` is not positive.
    """
    time_series = jnp.asarray(time_series)
    assert number_of_maxima > 0, "number_of_maxima must be positive."

    n = len(time_series)
    if n < number_of_maxima:
        return jnp.array(jnp.nan)

    abs_values_sorted = jnp.sort(jnp.abs(time_series))
    top_n_abs_values = abs_values_sorted[-number_of_maxima:]

    return jnp.mean(top_n_abs_values)


@jax.jit
def mean_second_derivative_central(time_series: jax.Array) -> jax.Array:
    """
    Calculate the mean of the central finite difference approximation of the second derivative of the time series.

    The second derivative at point i is approximated by:

    .. math::

        f''(x_i) \approx f(x_{i+1}) - 2 f(x_i) + f(x_{i-1})

    This function returns the average of these second derivative approximations
    over the valid range i = 1 to N-2.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the mean second derivative approximation.
        Returns NaN if the time series length is less than 3.
    """
    time_series = jnp.asarray(time_series)
    n = len(time_series)
    if n < 3:
        return jnp.array(jnp.nan)

    second_derivatives = time_series[2:] - 2 * time_series[1:-1] + time_series[:-2]
    return jnp.mean(second_derivatives)


@jax.jit
def median(time_series: jax.Array) -> jax.Array:
    """
    Calculate the median value of the time series.

    The median is the middle value that separates the higher half from the lower half of the data.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the median value.
        Returns NaN if the time series is empty.
    """
    time_series = jnp.asarray(time_series)
    return jnp.median(time_series)


@jax.jit
def median_change(time_series: jax.Array) -> jax.Array:
    """
    Calculate the median of the first differences in the time series.

    This measures the typical change between consecutive points by taking the median
    of the differences:

    .. math::

        \text{median\_change} = \text{median}\left(\{x_{i+1} - x_i\}_{i=0}^{N-2}\right)

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the median of the first differences.
        Returns NaN if the time series length is less than 2.
    """
    time_series = jnp.asarray(time_series)
    diffs = jnp.diff(time_series)
    return jnp.median(diffs) if len(diffs) > 0 else jnp.array(jnp.nan)


@jax.jit
def median_abs_change(time_series: jax.Array) -> jax.Array:
    """
    Calculate the median of the absolute first differences in the time series.

    This feature captures the typical magnitude of change between consecutive points by
    computing the median of the absolute differences:

    .. math::

        \text{median\_abs\_change} = \text{median}\left(\left|x_{i+1} - x_i\right|\right), \quad i=0,\dots,N-2

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the median of the absolute first differences.
        Returns NaN if the time series length is less than 2.
    """
    time_series = jnp.asarray(time_series)
    diffs = jnp.diff(time_series)
    return jnp.median(jnp.abs(diffs)) if len(diffs) > 0 else jnp.array(jnp.nan)


@jax.jit
def minimum(time_series: jax.Array) -> jax.Array:
    """
    Calculate the minimum value of the time series.

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the minimum value.
        Returns NaN if the time series is empty.
    """
    time_series = jnp.asarray(time_series)
    return jnp.min(time_series)


@jax.jit
def negative_turning_points(time_series: jax.Array) -> jax.Array:
    """
    Calculate the number of negative turning points in the time series.

    A negative turning point occurs when the first difference changes from negative
    to positive, indicating a local minimum.

    Formally, for differences \( d_i = x_{i+1} - x_i \), a negative turning point is detected at index i
    if:

    .. math::

        d_i < 0 \quad \text{and} \quad d_{i+1} > 0

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the number of negative turning points.
        Returns 0 if the time series length is less than 3.
    """
    time_series = jnp.asarray(time_series)
    differences = jnp.diff(time_series)
    if len(differences) < 2:
        return jnp.array(0)

    # Indices where negative turning points occur
    idx = jnp.where((differences[:-1] < 0) & (differences[1:] > 0), size=time_series.size)[0]

    return jnp.array(idx.size)


@partial(jax.jit, static_argnames=["m"])
def number_crossing_m(time_series: jax.Array, m: float) -> jax.Array:
    """
    Calculate the number of times the time series crosses the threshold value `m`.

    A crossing is counted each time the time series moves from below `m` to above `m` or vice versa.

    Formally, for the indicator sequence:

    .. math::

        I_i = \begin{cases} 1 & x_i > m \\ 0 & \text{otherwise} \end{cases}

    The number of crossings is the count of indices \(i\) where \(I_i \neq I_{i+1}\).

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.
    m : float
        Threshold value to detect crossings.

    Returns
    -------
    jax.Array
        Scalar representing the number of crossings over the value `m`.
        Returns 0 if the time series length is less than 2.
    """
    time_series = jnp.asarray(time_series)
    if len(time_series) < 2:
        return jnp.array(0)

    positive = time_series > m
    crossings = jnp.diff(positive)
    return jnp.array(jnp.sum(crossings != 0))


@partial(jax.jit, static_argnames=["support"])
def number_peaks(time_series: jax.Array, support: int) -> jax.Array:
    """
    Calculate the number of peaks in the time series with a minimum neighborhood support.

    A peak at position \(i\) is a point that is strictly greater than all points within
    `support` distance to the left and right. Formally, \(x_i\) is a peak if:

    .. math::

        x_i > x_{i-k} \quad \text{and} \quad x_i > x_{i+k} \quad \text{for all } k = 1, \ldots, \text{support}

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.
    support : int
        Number of neighboring points on each side to compare for peak detection.
        Must be positive and less than half the length of the time series.

    Returns
    -------
    jax.Array
        Scalar representing the count of peaks with the given support.
        Returns 0 if the time series is too short for the given support.
    """
    time_series = jnp.asarray(time_series)
    n = len(time_series)
    if support <= 0 or 2 * support >= n:
        return jnp.array(0)

    ts_reduced = time_series[support : n - support]

    # Initialize boolean array: True where ts_reduced is greater than neighbors
    is_peak = jnp.ones_like(ts_reduced, dtype=bool)

    for lag in range(1, support + 1):
        left_neighbors = jnp.roll(time_series, lag)[support : n - support]
        right_neighbors = jnp.roll(time_series, -lag)[support : n - support]

        is_peak &= ts_reduced > left_neighbors
        is_peak &= ts_reduced > right_neighbors

    return jnp.sum(is_peak)


@jax.jit
def peak_to_peak_distance(time_series: jax.Array) -> jax.Array:
    """
    Calculate the peak-to-peak distance of the time series.

    The peak-to-peak distance is defined as the absolute difference between the maximum
    and minimum values in the time series:

    .. math::

        \text{peak-to-peak} = \max(x) - \min(x)

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the peak-to-peak distance.
        Returns NaN if the input time series is empty.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(jnp.nan)

    return jnp.max(time_series) - jnp.min(time_series)


@jax.jit
def percentage_of_reoccurring_values_to_all_values(time_series: jax.Array) -> jax.Array:
    """
    Calculate the percentage of unique values in the time series that occur more than once.

    This feature measures how many distinct values repeat in the time series,
    relative to the total number of unique values.

    Mathematically, if \(x\) is the time series and \(U\) is the set of unique values,
    and \(c_u\) is the count of value \(u \in U\):

    .. math::

        \text{percentage} = \frac{|\{ u \in U : c_u > 1 \}|}{|U|}

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar value between 0 and 1 representing the fraction of unique values that
        appear more than once. Returns NaN if the input is empty.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(jnp.nan)

    unique_values, counts = jnp.unique(time_series, return_counts=True, size=time_series.size)

    if unique_values.size == 0:
        return jnp.array(0.0)

    recurring_count = jnp.sum(counts > 1)
    total_unique = unique_values.size

    return recurring_count / total_unique


@jax.jit
def positive_turning_points(time_series: jax.Array) -> jax.Array:
    """
    Calculate the number of positive turning points in the time series.

    A positive turning point is a point where the first difference changes from positive
    to negative, i.e., the time series goes up and then down around that point.

    Formally, if \(\Delta x_t = x_{t+1} - x_t\), then positive turning points occur where:

    .. math::

        \Delta x_t > 0 \quad \text{and} \quad \Delta x_{t+1} < 0

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    Returns
    -------
    jax.Array
        Scalar representing the number of positive turning points.
        Returns 0 if the input time series length is less than 3.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size < 3:
        return jnp.array(0)

    differences = jnp.diff(time_series)
    # We look at pairs (differences[t], differences[t+1]) for t in [0, n-3]
    is_pos_turning_point = (differences[:-1] > 0) & (differences[1:] < 0)

    return jnp.sum(is_pos_turning_point)


@jax.jit
def quantile(time_series: jax.Array, q: float) -> jax.Array:
    """
    Calculate the q-th quantile of the time series.

    The q-th quantile is the value below which a fraction q of the data falls.

    Formally, the q-th quantile \(Q_q\) satisfies:

    .. math::

        P(X \leq Q_q) = q, \quad q \in [0,1]

    Parameters
    ----------
    time_series : jax.Array
        1D array representing the time series.

    q : float
        Quantile to compute, must be between 0 and 1 inclusive.

    Returns
    -------
    jax.Array
        The q-th quantile of the time series.
        Returns NaN if the input time series is empty.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(jnp.nan)
    return jnp.quantile(time_series, q)


@jax.jit
def range_count(time_series: jax.Array, lower_bound: float, upper_bound: float) -> jax.Array:
    """
    Count the number of values in the time series that lie within the interval
    \([lower\_bound, upper\_bound)\).

    Formally, this count is given by:

    .. math::

        \text{count} = \sum_{i=1}^n \mathbf{1}_{[lower\_bound, upper\_bound)}(x_i)
        \quad \text{where} \quad
        \mathbf{1}_{[a,b)}(x) = 
        \begin{cases}
            1 & \text{if } a \leq x < b \\
            0 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    time_series : jax.Array
        1D array of values representing the time series.

    lower_bound : float
        Inclusive lower bound of the interval.

    upper_bound : float
        Exclusive upper bound of the interval.

    Returns
    -------
    jax.Array
        The count of values within the interval \([lower\_bound, upper\_bound)\).
    """
    time_series = jnp.asarray(time_series)
    return jnp.sum((time_series >= lower_bound) & (time_series < upper_bound))


@partial(jax.jit, static_argnames=["r"])
def ratio_beyond_r_sigma(time_series: jax.Array, r: float) -> jax.Array:
    """
    Calculate the ratio of values in the time series that deviate from the mean
    by more than \( r \) times the standard deviation.

    Formally, this ratio is defined as:

    .. math::

        \text{ratio} = \frac{1}{n} \sum_{i=1}^n \mathbf{1} \left( |x_i - \mu| > r \sigma \right)

    where
    - \( n \) is the length of the time series,
    - \( x_i \) are the individual time series values,
    - \( \mu \) is the mean of the time series,
    - \( \sigma \) is the standard deviation of the time series,
    - \( r \) is the given multiplier threshold,
    - \( \mathbf{1}(\cdot) \) is the indicator function.

    Parameters
    ----------
    time_series : jax.Array
        1D array of time series values.

    r : float
        Threshold multiplier for the standard deviation.

    Returns
    -------
    jax.Array
        The ratio (between 0 and 1) of values with absolute deviation
        greater than \( r \times \) standard deviation from the mean.

    Notes
    -----
    Returns NaN if the time series is empty.
    """
    time_series = jnp.asarray(time_series)
    n = time_series.size
    if n == 0:
        return jnp.array(jnp.nan)

    mean_val = jnp.mean(time_series)
    std_val = jnp.std(time_series)

    deviations = jnp.abs(time_series - mean_val)
    count_beyond = jnp.sum(deviations > r * std_val)
    return count_beyond / n


@jax.jit
def ratio_value_number_to_time_series_length(time_series: jax.Array) -> jax.Array:
    """Calculate the ratio of unique values to the total number of values in the time series.

    Returns NaN if the input time series is empty.

    Parameters
    ----------
    time_series : jax.Array
        Input time series vector.

    Returns
    -------
    jax.Array
        Ratio of unique values to total length of the time series.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(jnp.nan)
    unique_values = jnp.unique(time_series, size=time_series.size)
    return unique_values.size / time_series.size


@jax.jit
def root_mean_square(time_series: jax.Array) -> jax.Array:
    """
    Calculate the root mean square (RMS) of the time series.

    The RMS is a measure of the magnitude of the values in the time series,
    defined as the square root of the average of the squared values:

    .. math::
        \mathrm{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}

    where \( x_i \) are the values of the time series and \( N \) is the length.

    Parameters
    ----------
    time_series : jax.Array
        Vector containing the time series values.

    Returns
    -------
    jax.Array
        The root mean square value of the time series.
    """
    time_series = jnp.asarray(time_series)
    return jnp.sqrt(jnp.mean(jnp.square(time_series)))


@jax.jit
def slope(time_series: jax.Array) -> jax.Array:
    """
    Calculate the slope of the time series using linear regression.

    The slope corresponds to the coefficient \( \beta \) in the simple linear model:

    .. math::
        y_i = \beta t_i + \alpha + \epsilon_i

    where \( y_i \) are the time series values, \( t_i \) are time indices,
    and \( \epsilon_i \) is the error term. The slope \( \beta \) is calculated
    via least squares fitting.

    Parameters
    ----------
    time_series : jax.Array
        Vector containing the time series values.

    Returns
    -------
    jax.Array
        The slope (rate of change) of the time series.
        Returns NaN if the time series length is less than 2.
    """
    time_series = jnp.asarray(time_series)
    n = len(time_series)
    if n < 2:
        return jnp.array(jnp.nan)
    t = jnp.linspace(0, n - 1, n)
    return jnp.polyfit(t, time_series, 1)[0]


@jax.jit
def standard_deviation(time_series: jax.Array) -> jax.Array:
    """
    Calculate the standard deviation of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the standard deviation of.

    Returns
    -------
    jax.Array
        The standard deviation of the vector.
        Returns NaN if the time series is empty.
    """
    time_series = jnp.asarray(time_series)
    if time_series.size == 0:
        return jnp.array(jnp.nan)
    return jnp.std(time_series)


# TODO: Check this function seperately
@jax.jit
def sum_of_reoccurring_values(time_series: jax.Array) -> jax.Array:
    """Calculate the sum of all values that appear in the time series more than once."""
    time_series = jnp.asarray(time_series)
    unique, counts = jnp.unique_counts(time_series, size=len(time_series))
    counts = counts.at[jnp.where(counts < 2, counts, 0)].set(0)
    counts = counts.at[jnp.where(counts > 1, counts, 0)].set(1)
    return jnp.sum(counts * unique)


# TODO: Check this function seperatly
@jax.jit
def sum_of_reoccurring_data_points(time_series: jax.Array) -> jax.Array:
    """Calculate the sum of all data points that appear in the time series more than once."""
    time_series = jnp.asarray(time_series)
    unique, counts = jnp.unique_counts(time_series, size=len(time_series))
    counts = jnp.where(counts > 1, counts, 0)  # keep only repeats
    return jnp.sum(counts * unique)


@jax.jit
def sum_values(time_series: jax.Array) -> jax.Array:
    """Calculate the sum of values of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the sum of values of.

    Returns
    -------
    jax.Array :
        The sum of values of the vector, or NaN if empty.

    """
    time_series = jnp.asarray(time_series)
    return jnp.sum(time_series) if time_series.size > 0 else jnp.array(jnp.nan)


@partial(jax.jit, static_argnames=["lag"])
def time_reversal_asymmetry_statistic(time_series: jax.Array, lag: int) -> jax.Array:
    r"""
    Compute the time-reversal asymmetry statistic for a time series.

    This statistic measures the degree of asymmetry in the joint moments of a time
    series when reversed in time, often used to detect nonlinear dynamics.

    The statistic is defined as:

    .. math::

        \\mathrm{TRAS}(\\tau) =
        \\frac{1}{N - 2\\tau} \\sum_{t=1}^{N - 2\\tau}
        \\left[ x_{t+2\\tau}^2 \\, x_{t+\\tau}
        - x_{t+\\tau} \\, x_t^2 \\right]

    where :math:`x_t` is the time series value at time index :math:`t` and
    :math:`\\tau` is the lag.

    Parameters
    ----------
    time_series : jax.Array
        Input time series of shape ``(n,)``.
    lag : int
        The lag parameter :math:`\\tau`. Must satisfy ``2 * lag < len(time_series)``.
        If this condition is not met, the statistic is returned as 0.0.

    Returns
    -------
    jax.Array
        The computed time-reversal asymmetry statistic as a scalar.

    Notes
    -----
    - This implementation uses ``jnp.roll`` for lagging, but excludes the wrapped
      values from the computation.
    - If ``2 * lag >= len(time_series)``, the function returns 0.0 without error.
    """
    time_series = jnp.asarray(time_series)

    n = len(time_series)
    if 2 * lag >= n:
        return jnp.array(0.0)

    # Shifted versions of the series
    lag1 = jnp.roll(time_series, -lag)  # x_{t+τ}
    lag2 = jnp.roll(time_series, -2 * lag)  # x_{t+2τ}

    # Compute statistic excluding wrapped elements
    valid_range = slice(0, n - 2 * lag)
    tras = jnp.mean(
        lag2[valid_range] ** 2 * lag1[valid_range]
        - lag1[valid_range] * time_series[valid_range] ** 2
    )

    return tras


@jax.jit
def value_count(time_series: jax.Array, value: float) -> jax.Array:
    r"""
    Count the occurrences of a given value in the time series.

    Computes the number of elements in the time series equal to the specified scalar value.

    The count is formally:

    .. math::

        \\mathrm{Count}(v) = \\sum_{t=1}^N \\mathbf{1}_{\\{x_t = v\\}}

    where :math:`x_t` is the time series at time :math:`t`, :math:`v` is the target value,
    and :math:`\\mathbf{1}` is the indicator function.

    Parameters
    ----------
    time_series : jax.Array
        Input time series vector of shape ``(n,)``.
    value : float
        Scalar value to count within the time series.

    Returns
    -------
    jax.Array
        Scalar containing the count of occurrences of `value` in `time_series`.
    """
    time_series = jnp.asarray(time_series)
    return jnp.sum(time_series == value)


@jax.jit
def value_range(time_series: jax.Array) -> jax.Array:
    """
    Compute the range of values in the time series.

    The range is defined as the difference between the maximum and minimum
    observed values in the time series:

    .. math::

        \\mathrm{Range} = \\max_{t} x_t - \\min_{t} x_t

    where :math:`x_t` is the time series value at time index :math:`t`.

    Parameters
    ----------
    time_series : jax.Array
        Input time series vector of shape ``(n,)``.

    Returns
    -------
    jax.Array
        Scalar representing the range of the time series values.
    """
    time_series = jnp.asarray(time_series)
    return jnp.max(time_series) - jnp.min(time_series)


@jax.jit
def variance(time_series: jax.Array) -> jax.Array:
    r"""
    Calculate the variance of the time series.

    Variance measures the average squared deviation from the mean:

    .. math::

        \\mathrm{Var}(X) = \\frac{1}{N} \\sum_{t=1}^N (x_t - \\bar{x})^2

    where :math:`x_t` is the time series value at time :math:`t`, and
    :math:`\\bar{x}` is the mean of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Input time series vector of shape ``(n,)``.

    Returns
    -------
    jax.Array
        Scalar representing the variance of the time series.
    """
    time_series = jnp.asarray(time_series)
    return jnp.var(time_series)


@jax.jit
def variance_larger_than_standard_deviation(time_series: jax.Array) -> jax.Array:
    """
    Determine whether the variance of the time series is larger than its standard deviation.

    Given that variance (:math:`\\sigma^2`) is the square of the standard deviation
    (:math:`\\sigma`), this function checks the inequality:

    .. math::

        \\sigma^2 > \\sigma

    which depends on the scale of the data.

    Parameters
    ----------
    time_series : jax.Array
        Input time series vector of shape ``(n,)``.

    Returns
    -------
    jax.Array
        Boolean scalar indicating if variance is greater than standard deviation.

    Notes
    -----
    - The variance is only larger than the standard deviation iff standard deviation > 1.
    """
    time_series = jnp.asarray(time_series)
    variance = jnp.var(time_series)
    std_dev = jnp.std(time_series)
    return variance > std_dev


@jax.jit
def variation_coefficient(time_series: jax.Array) -> jax.Array:
    r"""
    Calculate the coefficient of variation (CV) of the time series.

    The coefficient of variation is the ratio of the standard deviation to the mean,
    providing a normalized measure of dispersion relative to the mean:

    .. math::

        \\mathrm{CV} = \\frac{\\sigma}{\\mu}

    where :math:`\\sigma` is the standard deviation and :math:`\\mu` is the mean
    of the time series.

    If the mean is zero, the coefficient of variation is undefined and NaN is returned.

    Parameters
    ----------
    time_series : jax.Array
        Input time series vector of shape ``(n,)``.

    Returns
    -------
    jax.Array
        Scalar representing the coefficient of variation, or NaN if mean is zero.
    """
    time_series = jnp.asarray(time_series)
    mean_value = jnp.mean(time_series)
    return jnp.where(mean_value == 0, jnp.nan, jnp.std(time_series) / mean_value)


@jax.jit
def zero_cross_rate(time_series: jax.Array) -> jax.Array:
    """
    Compute the zero-crossing rate of the time series.

    The zero-crossing rate is the number of times the time series changes sign,
    i.e., crosses the zero level:

    .. math::

        \\mathrm{ZCR} = \\sum_{t=1}^{N-1} \\mathbf{1}_{\\{ \\mathrm{sign}(x_t) \\neq \\mathrm{sign}(x_{t-1}) \\}}

    where :math:`x_t` is the value of the time series at time index :math:`t`.

    Parameters
    ----------
    time_series : jax.Array
        Input time series vector of shape ``(n,)``.

    Returns
    -------
    jax.Array
        Scalar representing the number of zero crossings in the time series.
    """
    time_series = jnp.asarray(time_series)
    signs = jnp.sign(time_series)
    zero_crossings = jnp.sum(jnp.diff(signs) != 0)
    return zero_crossings

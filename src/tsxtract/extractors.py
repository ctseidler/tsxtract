"""Module containing the feature extractors."""

from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def absolute_energy(time_series: jax.Array) -> jax.Array:
    """Calculate the absolute energy of the time series.

    The absolute energy is the sum over the squared values.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the absolute energy of.

    Returns
    -------
    jax.Array :
        The absolute energy of the vector.

    """
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
    if time_series.size == 0:
        msg = "Input array is empty; cannot compute maximum."
        raise ValueError(msg)

    return jnp.max(jnp.abs(time_series))


@jax.jit
def absolute_sum_of_changes(time_series: jax.Array) -> jax.Array:
    """Calculate the absolute sum of changes of the values.

    Parameters
    ----------
    time_series :
        The vector to calculate the absolute sum of changes of.

    Returns
    -------
    jax.Array :
        Absolute sum of changes of the values.

    """
    return jnp.sum(jnp.abs(jnp.diff(time_series)))


@jax.jit
def absolute_sum_values(time_series: jax.Array) -> jax.Array:
    """Calculate the sum of absolute values of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the sum of absolute values of.

    Returns
    -------
    jax.Array :
        The sum of absolute values of the vector.

    """
    return jnp.sum(jnp.absolute(time_series))


@partial(jax.jit, static_argnames=["lag"])
def autocorrelation(time_series: jax.Array, lag: int) -> jax.Array:
    """Calculate the autocorrelation of the specified lag."""
    if len(time_series) < lag:
        return jnp.array(jnp.nan)

    subseries_1 = time_series[: (len(time_series) - lag)]
    subseries_2 = time_series[lag:]

    ts_mean = jnp.mean(time_series)
    sum_product = jnp.sum((subseries_1 - ts_mean) * (subseries_2 - ts_mean))

    v = jnp.var(time_series)

    return jax.lax.cond(
        jnp.isclose(v, 0),
        lambda _: jnp.array(jnp.nan),
        lambda _: sum_product / ((len(time_series) - lag) * v),
        operand=None,
    )


@partial(jax.jit, static_argnums=1)
def binned_entropy(time_series: jax.Array, max_bins: int) -> jax.Array:
    """Bin the values of the time series and calculate the entropy."""
    hist, _ = jnp.histogram(time_series, bins=max_bins)
    probs = hist / time_series.size
    probs = jnp.where(probs == 0, 1, probs)

    return -jnp.sum(probs * jnp.log(probs))


@partial(jax.jit, static_argnums=1)
def c3(time_series: jax.Array, lag: int) -> jax.Array:
    """Measure the linearity in the time series using c3 statistic."""
    n = time_series.size

    if 2 * lag >= n:
        return jnp.array(0.0)

    return jnp.mean(
        (jnp.roll(time_series, 2 * -lag) * jnp.roll(time_series, -lag) * time_series)[
            0 : (n - 2 * lag)
        ],
    )


@jax.jit
def count_above(time_series: jax.Array, value: float) -> jax.Array:
    """Calculate the percentage of values in the time series that are above value.

    Parameters
    ----------
    time_series : jax.Array
        The vector to calculate the percentage of values above value.
    value : float
        The threshold used.

    Returns
    -------
    jax.Array :
        Percentage of values in the time series that are above value.

    """
    return jnp.sum(jnp.greater_equal(time_series, value)) / len(time_series)


@jax.jit
def count_above_mean(time_series: jax.Array) -> jax.Array:
    """Calculate the number of values above the mean.

    Parameters
    ----------
    time_series :
        The vector to calculate the number of features above the mean.

    Returns
    -------
    jax.Array :
        Number of features above the mean.

    """
    return jnp.sum(time_series > jnp.mean(time_series))


@jax.jit
def count_below(time_series: jax.Array, value: float) -> jax.Array:
    """Calculate the percentage of values in the time series that are below value.

    Parameters
    ----------
    time_series : jax.Array
        The vector to calculate the percentage of values below value.
    value : float
        The threshold used.

    Returns
    -------
    jax.Array :
        Percentage of values in the time series that are below value.

    """
    return jnp.sum(jnp.less_equal(time_series, value)) / len(time_series)


@jax.jit
def count_below_mean(time_series: jax.Array) -> jax.Array:
    """Calculate the number of values below the mean.

    Parameters
    ----------
    time_series :
        The vector to calculate the number of features below the mean.

    Returns
    -------
    jax.Array :
        Number of features below the mean.

    """
    return jnp.sum(time_series < jnp.mean(time_series))


@jax.jit
def distance(time_series: jax.Array) -> jax.Array:
    """Calculate the distance traveled by the time series. FROM TSFEL"""
    differences = jnp.diff(time_series).astype(float)
    return jnp.sum(jnp.sqrt(1 + differences**2))


@jax.jit
def first_location_of_maximum(time_series: jax.Array) -> jax.Array:
    """Get the relative first location of the maximum value of the time series."""
    return (
        jnp.argmax(time_series) / len(time_series) if len(time_series) > 0 else jnp.arange(jnp.nan)
    )


@jax.jit
def first_location_of_minimum(time_series: jax.Array) -> jax.Array:
    """Get the relative first location of the minimal value of the time series."""
    return (
        jnp.argmin(time_series) / len(time_series) if len(time_series) > 0 else jnp.arange(jnp.nan)
    )


@jax.jit
def interquartile_range(time_series: jax.Array) -> jax.Array:
    """Calculate the interquartile range of the time series. FROM TSFEL"""
    return jnp.percentile(time_series, 75) - jnp.percentile(time_series, 25)


@jax.jit
def is_symmetric(time_series: jax.Array, value: jax.Array) -> jax.Array:
    """Check if the distribution of the time series looks symmetric."""
    mean_median_difference: jax.Array = jnp.abs(jnp.mean(time_series) - jnp.median(time_series))
    max_min_difference: jax.Array = jnp.max(time_series) - jnp.min(time_series)

    return mean_median_difference < (value * max_min_difference)


# TODO: Check if this feature is calculated correctly
@jax.jit
def has_duplicate(time_series: jax.Array) -> jax.Array:
    """Check if any value in the time series occurs more than once."""
    return jnp.array(time_series.size != jnp.unique(time_series, size=len(time_series)))


@jax.jit
def has_duplicate_max(time_series: jax.Array) -> jax.Array:
    """Check if the maximum of the time series is observed more than once."""
    return jnp.sum(time_series == jnp.max(time_series)) > 1


@jax.jit
def has_duplicate_min(time_series: jax.Array) -> jax.Array:
    """Check if the minimal value of the time series is observed more than once."""
    return jnp.sum(time_series == jnp.min(time_series)) > 1


@partial(jax.jit, static_argnums=1)
def hist_mode(time_series: jax.Array, n_bins: int) -> jax.Array:
    """Calculate the mode of a histogram using a given number of bins. FROM TSFEL"""
    hist_values, bin_edges = jnp.histogram(time_series, bins=n_bins)
    max_bin_idx = jnp.argmax(hist_values)
    return (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0


@jax.jit
def large_standard_deviation(time_series: jax.Array, value: float) -> jax.Array:
    """Does the time series has a large standard deviation?"""
    return jnp.std(time_series) > (value * (jnp.max(time_series) - jnp.min(time_series)))


@jax.jit
def last_location_of_maximum(time_series: jax.Array) -> jax.Array:
    """Get the relative last location of the maximum value of the time series."""
    return (
        jnp.array(1.0 - jnp.argmax(time_series[::-1]) / len(time_series))
        if len(time_series) > 0
        else jnp.array(jnp.nan)
    )


@jax.jit
def last_location_of_minimum(time_series: jax.Array) -> jax.Array:
    """Get the relative last location of the minimal value of the time series."""
    return (
        jnp.array(1.0 - jnp.argmin(time_series[::-1]) / len(time_series))
        if len(time_series) > 0
        else jnp.array(jnp.nan)
    )


@jax.jit
def length(time_series: jax.Array) -> int:
    """Calculate the length of the time series.

    The length of the time series is the number of observations. A signal measured with 10 Hz for
    5 seconds has a length of 10 Hz * 5 s = 50

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the length

    Returns
    -------
    jax.Array :
        The length of the vector.

    """
    return time_series.shape[-1]


@jax.jit
def maximum(time_series: jax.Array) -> jax.Array:
    """Calculate the maximum of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the maximum of.

    Returns
    -------
    jax.Array :
        The maximum of the vector.

    """
    return jnp.max(time_series)


@jax.jit
def mean(time_series: jax.Array) -> jax.Array:
    """Calculate the mean of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the mean of.

    Returns
    -------
    jax.Array :
        The mean of the vector.

    """
    return jnp.mean(time_series)


@jax.jit
def mean_abs_change(time_series: jax.Array) -> jax.Array:
    """Average over first differences."""
    return jnp.mean(jnp.abs(jnp.diff(time_series)))


@jax.jit
def mean_change(time_series: jax.Array) -> jax.Array:
    """Average over time series differences."""
    return (
        (time_series[-1] - time_series[0]) / (len(time_series) - 1)
        if len(time_series) > 1
        else jnp.array(jnp.nan)
    )


@partial(jax.jit, static_argnums=1)
def mean_n_absolute_max(time_series: jax.Array, number_of_maxima: int) -> jax.Array:
    """Calculate the mean of the n absolute maximum values of the time series."""
    assert number_of_maxima > 0

    n_absolute_maximum_values = jnp.sort(jnp.absolute(time_series))[-number_of_maxima:]

    return (
        jnp.mean(n_absolute_maximum_values)
        if len(time_series) > number_of_maxima
        else jnp.array(jnp.nan)
    )


@jax.jit
def mean_second_derivative_central(time_series: jax.Array) -> jax.Array:
    """Get the mean value of a central approximation of the second derivative."""
    return (
        (time_series[-1] - time_series[-2] - time_series[1] + time_series[0])
        / (2 * (len(time_series) - 2))
        if len(time_series) > 2
        else jnp.array(jnp.nan)
    )


@jax.jit
def median(time_series: jax.Array) -> jax.Array:
    """Calculate the median of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the median of.

    Returns
    -------
    jax.Array :
        The median of the vector.

    """
    return jnp.median(time_series)


@jax.jit
def median_change(time_series: jax.Array) -> jax.Array:
    """Calculate the median differences in the time series. FROM TSFEL"""
    return jnp.median(jnp.diff(time_series))


@jax.jit
def median_abs_change(time_series: jax.Array) -> jax.Array:
    """Calculate the median absoluve differences in the time series. FROM TSFEL"""
    return jnp.median(jnp.abs(jnp.diff(time_series)))


@jax.jit
def minimum(time_series: jax.Array) -> jax.Array:
    """Calculate the minimum of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the minimum of.

    Returns
    -------
    jax.Array :
        The minimum of the vector.

    """
    return jnp.min(time_series)


@jax.jit
def negative_turning_points(time_series: jax.Array) -> jax.Array:
    """Get the number of negative turning points in the time series. FROM TSFEL"""
    differences = jnp.diff(time_series)
    array_signal = jnp.arange(len(differences[:-1]))
    negative_turning_points = jnp.where(
        (differences[array_signal] < 0) & (differences[array_signal + 1] > 0),
        size=len(time_series),
    )[0]

    return jnp.array(negative_turning_points.size)


@partial(jax.jit, static_argnames=["m"])
def number_crossing_m(time_series: jax.Array, m: float) -> jax.Array:
    """Calculate the number of crossings of the time series on m."""
    positive = time_series > m
    return jnp.array(jnp.where(jnp.diff(positive), size=len(time_series))[0].size)


@partial(jax.jit, static_argnums=1)
def number_peaks(time_series: jax.Array, support: int) -> jax.Array:
    """Calculate the number of peaks of at least support in the time series."""
    ts_reduced = time_series[support:-support]

    result = None
    for i in range(1, support + 1):
        result_first = ts_reduced > jnp.roll(time_series, i)[support:-support]

        if result is None:
            result = result_first
        else:
            result &= result_first

        result &= ts_reduced > jnp.roll(time_series, -i)[support:-support]

    return jnp.sum(result)


@jax.jit
def peak_to_peak_distance(time_series: jax.Array) -> jax.Array:
    """Calculate the peak to peak distance."""
    return jnp.abs(jnp.max(time_series) - jnp.min(time_series))


@jax.jit
def percentage_of_reoccurring_values_to_all_values(time_series: jax.Array) -> jax.Array:
    """Get the percentage of values that are present more than once in the time series."""
    if len(time_series) == 0:
        return jnp.array(jnp.nan)

    counts: jax.Array = jnp.array(jnp.unique_counts(time_series, size=len(time_series)))

    if counts.shape[0] == 0:
        return jnp.array(0.0)

    return jnp.sum(counts > 1) / float(counts.shape[0])


@jax.jit
def positive_turning_points(time_series: jax.Array) -> jax.Array:
    """Get the number of positive turning points in the time series. FROM TSFEL"""
    differences = jnp.diff(time_series)
    array_signal = jnp.arange(len(differences[:-1]))
    positive_turning_points = jnp.where(
        (differences[array_signal + 1] < 0) & (differences[array_signal] > 0),
        size=len(time_series),
    )[0]

    return jnp.array(positive_turning_points.size)


@jax.jit
def quantile(time_series: jax.Array, value: float) -> jax.Array:
    """Calculate the q quantile of the time series."""
    if len(time_series) == 0:
        return jnp.array(jnp.nan)
    return jnp.quantile(time_series, value)


@jax.jit
def range_count(time_series: jax.Array, lower_bound: float, upper_bound: float) -> jax.Array:
    """Count the number of values within the interval [lower_bound, upper_bound).

    Parameters
    ----------
    time_series : jax.Array
        The vector to count the number of values within the interval.
    lower_bound : float
        Inclusive lower bound of the range.
    upper_bound : float
        Exclusive upper bound of the range.

    Returns
    -------
    jax.Array :
        Number of values within the interval.

    """
    return jnp.sum((time_series >= lower_bound) & (time_series < upper_bound))


@jax.jit
def ratio_beyond_r_sigma(time_series: jax.Array, value: float) -> jax.Array:
    """Ratio of values that are more than value * std(time_series) away from the mean."""
    return (
        jnp.sum(jnp.abs(time_series - jnp.mean(time_series)) > value * jnp.std(time_series))
        / time_series.size
    )


@jax.jit
def ratio_value_number_to_time_series_length(time_series: jax.Array) -> jax.Array:
    """Ratio of unique values to all values."""
    return (
        jnp.array(jnp.nan)
        if time_series.size == 0
        else jnp.unique(time_series, size=len(time_series)).size / time_series.size
    )


@jax.jit
def root_mean_square(time_series: jax.Array) -> jax.Array:
    """Calculate the root mean square of the values.

    Parameters
    ----------
    time_series :
        The vector to calculate the root mean square of.

    Returns
    -------
    jax.Array :
        Root mean square of the values.

    """
    return jnp.sqrt(jnp.mean(jnp.square(time_series)))


@jax.jit
def slope(time_series: jax.Array) -> jax.Array:
    """Calculate the slope of the time series. FROM TSFEL"""
    t = jnp.linspace(0, len(time_series) - 1, len(time_series))
    return jnp.polyfit(t, time_series, 1)[0]


@jax.jit
def standard_deviation(time_series: jax.Array) -> jax.Array:
    """Calculate the standard deviation of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the standard deviation of.

    Returns
    -------
    jax.Array :
        The standard deviation of the vector.

    """
    return jnp.std(time_series)


@jax.jit
def sum_of_reoccurring_values(time_series: jax.Array) -> jax.Array:
    """Calculate the sum of all values that appear in the time series more than once."""
    unique, counts = jnp.unique_counts(time_series, size=len(time_series))
    counts = counts.at[jnp.where(counts < 2, counts, 0)].set(0)
    counts = counts.at[jnp.where(counts > 1, counts, 0)].set(1)
    return jnp.sum(counts * unique)


@jax.jit
def sum_of_reoccurring_data_points(time_series: jax.Array) -> jax.Array:
    """Calculate the sum of all data points that appear in the time series more than once."""
    unique, counts = jnp.unique_counts(time_series, size=len(time_series))
    counts = counts.at[jnp.where(counts < 2, counts, 0)].set(0)
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
        The sum of values of the vector.

    """
    return jnp.sum(time_series)


@partial(jax.jit, static_argnums=1)
def time_reversal_asymmetry_statistic(time_series: jax.Array, lag: int) -> jax.Array:
    """Calculate the time reversal asymmetry statistic."""
    n = len(time_series)
    if 2 * lag >= n:
        return jnp.array(0.0)

    one_lag = jnp.roll(time_series, -lag)
    two_lag = jnp.roll(time_series, 2 * -lag)

    return jnp.mean(
        (two_lag * two_lag * one_lag - one_lag * time_series * time_series)[0 : (n - 2 * lag)],
    )


@jax.jit
def value_count(time_series: jax.Array, value: float) -> jax.Array:
    """Count the occurrences of value in the time series.

    Parameters
    ----------
    time_series : jax.Array
        The vector to count the occurrences of the value of.
    value : float
        The value to be counted.

    Returns
    -------
    jax.Array :
        Number of occurrences of value in time series.

    """
    return jnp.sum(jnp.equal(time_series, value))


@jax.jit
def value_range(time_series: jax.Array) -> jax.Array:
    """Calculate the range of the values.

    The range is calculated as max observed value - min observed value.

    Parameters
    ----------
    time_series :
        The vector to calculate the range of.

    Returns
    -------
    jax.Array :
        Calculated value range.

    """
    return jnp.max(time_series) - jnp.min(time_series)


@jax.jit
def variance(time_series: jax.Array) -> jax.Array:
    """Calculate the variance of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the variance of.

    Returns
    -------
    jax.Array :
        The variance of the vector.

    """
    return jnp.var(time_series)


@jax.jit
def variance_larger_than_standard_deviation(time_series: jax.Array) -> jax.Array:
    """Check if the variance is larger than the standard deviation.

    Parameters
    ----------
    time_series : jax.Array
        The vector to check.

    Returns
    -------
    jax.Array :
        Bool, if variance is larger than standard deviation.

    """
    return jnp.var(time_series) > jnp.std(time_series)


@jax.jit
def variation_coefficient(time_series: jax.Array) -> jax.Array:
    """Calculate the variation coefficient."""
    avg: jax.Array = jnp.mean(time_series)

    return jnp.where(avg == 0, jnp.array(jnp.nan), jnp.std(time_series) / avg)


@jax.jit
def zero_cross_rate(time_series: jax.Array) -> jax.Array:
    """Calculate the zero-crossing rate of the time series. FROM TSFEL"""
    return jnp.array(len(jnp.where(jnp.diff(jnp.sign(time_series)), size=len(time_series))[0]))

"""Module containing the feature extractors."""

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
    """Calculate the absolute maximum of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the absolute maximum of.

    Returns
    -------
    jax.Array :
        The absolute maximum of the vector.

    """
    return jnp.max(jnp.absolute(time_series))


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

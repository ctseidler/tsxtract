"""Unit tests for tsxtract.extraction.absolute_sum_values."""

import jax.numpy as jnp

import tsxtract.extractors as tsx


def test_ones(ones_array) -> None:
    """tsx.absolute_sum_values should return 100 for a time series with 100 ones."""
    expected_output: int = 100
    assert tsx.absolute_sum_values(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.absolute_sum_values should return 0 for a time series with 100 zeros."""
    expected_output: int = 0
    assert tsx.absolute_sum_values(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.absolute_sum_values should return 100 for a time series with 100 -1 values."""
    expected_output: int = 100
    assert tsx.absolute_sum_values(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.absolute_sum_values should return the value for a single datapoint."""
    expected_output: float = single_point
    assert tsx.absolute_sum_values(single_point) == expected_output


def test_empty(empty_array) -> None:
    """tsx.absolute_sum_values should return 0 for an empty sequence."""
    expected_output: int = 0
    assert tsx.absolute_sum_values(empty_array) == expected_output


def test_nan_values(nan_array) -> None:
    """tsx.absolute_sum_values should return nan for an array with 100 nan values."""
    assert jnp.isnan(tsx.absolute_sum_values(nan_array))


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.absolute_sum_values should return nan for an array with 20 nan values."""
    assert jnp.isnan(tsx.absolute_sum_values(array_with_nan))


def test_inf_values(inf_array) -> None:
    """tsx.absolute_sum_values should return inf for an array with 100 inf values."""
    assert jnp.isinf(tsx.absolute_sum_values(inf_array))


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.absolute_sum_values should return a numeric for an array with 20 inf values."""
    assert jnp.isreal(tsx.absolute_sum_values(array_with_inf))


def test_50_50(array_50_50) -> None:
    """tsx.absolute_sum_values should return 50 for a time series with 50 zeros and 50 ones."""
    expected_output: int = 50
    assert tsx.absolute_sum_values(array_50_50) == expected_output


def test_20_80(array_20_80) -> None:
    """tsx.absolute_sum_values should return 80 for a time series with 20 zeros and 80 ones."""
    expected_output: int = 80
    assert tsx.absolute_sum_values(array_20_80) == expected_output


def test_positive_range(array_positive_range) -> None:
    """tsx.absolute_sum_values should return 5050 for a range from 0 to 100."""
    expected_output: int = 5050
    assert tsx.absolute_sum_values(array_positive_range) == expected_output


def test_negative_range(array_negative_range) -> None:
    """tsx.absolute_sum_values should return 5050 for a range from -100 to 0."""
    expected_output: int = 5050
    assert tsx.absolute_sum_values(array_negative_range) == expected_output


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.absolute_sum_values should return 2550 for a range from -50 to 50."""
    expected_output: int = 2550
    assert tsx.absolute_sum_values(array_positive_and_negative_range) == expected_output

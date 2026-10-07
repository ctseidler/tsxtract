"""Unit tests for tsxtract.extractors.mean_absolute_deviation."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


def test_ones(ones_array) -> None:
    """tsx.mean_absolute_deviation should return 0 for a time series with 5 ones."""
    expected_output: float = 0.0
    assert tsx.mean_absolute_deviation(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.mean_absolute_deviation should return 0 for a time series with 5 zeros."""
    expected_output: float = 0.0
    assert tsx.mean_absolute_deviation(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.mean_absolute_deviation should return 0 for a time series with 5 -1 values."""
    expected_output: float = 0.0
    assert tsx.mean_absolute_deviation(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.mean_absolute_deviation should return 0 for a single datapoint."""
    expected_output: float = 0.0
    assert tsx.mean_absolute_deviation(single_point) == expected_output


def test_empty(empty_array) -> None:
    """tsx.mean_absolute_deviation should return nan for an empty sequence."""
    assert jnp.isnan(tsx.mean_absolute_deviation(empty_array))


def test_nan_values(nan_array) -> None:
    """tsx.mean_absolute_deviation should return nan for an array with 5 nan values."""
    assert jnp.isnan(tsx.mean_absolute_deviation(nan_array))


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.mean_absolute_deviation should return nan for an array with one nan value."""
    assert jnp.isnan(tsx.mean_absolute_deviation(array_with_nan))


def test_inf_values(inf_array) -> None:
    """tsx.mean_absolute_deviation should return nan for an array of inf values."""
    assert jnp.isnan(tsx.mean_absolute_deviation(inf_array))


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.mean_absolute_deviation should return nan for an array with one inf value."""
    assert jnp.isnan(tsx.mean_absolute_deviation(array_with_inf))


def test_50_50(array_50_50) -> None:
    """tsx.mean_absolute_deviation should return 0.5 for [0, 0, 1, 1] (mean 0.5)."""
    expected_output: float = 0.5
    assert tsx.mean_absolute_deviation(array_50_50) == pytest.approx(expected_output)


def test_20_80(array_20_80) -> None:
    """tsx.mean_absolute_deviation should return 0.32 for [0, 1, 1, 1, 1] (mean 0.8)."""
    expected_output: float = 0.32  # (0.8 + 4 * 0.2) / 5
    assert tsx.mean_absolute_deviation(array_20_80) == pytest.approx(expected_output)


def test_positive_range(array_positive_range) -> None:
    """tsx.mean_absolute_deviation should return 2550/101 for a range from 0 to 100."""
    expected_output: float = 2550 / 101  # mean is 50; sum of |i - 50| is 2 * (1 + ... + 50)
    assert tsx.mean_absolute_deviation(array_positive_range) == pytest.approx(expected_output)


def test_negative_range(array_negative_range) -> None:
    """tsx.mean_absolute_deviation should return 2550/101 for a range from -100 to 0."""
    expected_output: float = 2550 / 101
    assert tsx.mean_absolute_deviation(array_negative_range) == pytest.approx(expected_output)


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.mean_absolute_deviation should return 2550/101 for a range from -50 to 50."""
    expected_output: float = 2550 / 101
    assert tsx.mean_absolute_deviation(array_positive_and_negative_range) == pytest.approx(
        expected_output,
    )


def test_large_numbers(large_numbers_array) -> None:
    """tsx.mean_absolute_deviation should return 0 for identical large values."""
    assert tsx.mean_absolute_deviation(large_numbers_array) == 0.0

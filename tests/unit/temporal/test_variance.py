"""Unit tests for tsxtract.extraction.variance."""

import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx


def test_ones(ones_array) -> None:
    """tsx.variance should return 0 for a time series with 100 ones."""
    expected_output: int = 0
    assert tsx.variance(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.variance should return 0 for a time series with 100 zeros."""
    expected_output: int = 0
    assert tsx.variance(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.variance should return 0 for a time series with 100 -1 values."""
    expected_output: int = 0
    assert tsx.variance(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.variance should return 0 for a single datapoint."""
    expected_output: int = 0
    assert tsx.variance(single_point) == expected_output


def test_empty(empty_array) -> None:
    """tsx.variance should return nan for an empty sequence."""
    assert jnp.isnan(tsx.variance(empty_array))


def test_nan_values(nan_array) -> None:
    """tsx.variance should return nan for an array with 100 nan values."""
    assert jnp.isnan(tsx.variance(nan_array))


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.variance should return nan for an array with 20 nan values."""
    assert jnp.isnan(tsx.variance(array_with_nan))


def test_inf_values(inf_array) -> None:
    """tsx.variance should return nan for an array with 100 inf values."""
    assert jnp.isnan(tsx.variance(inf_array))


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.variance should return a numeric for an array with 20 inf values."""
    assert jnp.isreal(tsx.variance(array_with_inf))


def test_50_50(array_50_50) -> None:
    """tsx.variance should return 0.25 for a time series with 50 zeros and 50 ones."""
    expected_output: float = 0.25
    assert tsx.variance(array_50_50) == expected_output


def test_20_80(array_20_80) -> None:
    """tsx.variance should return 0.1599999 for a time series with 20 zeros and 80 ones."""
    expected_output: float = 0.1599999
    assert tsx.variance(array_20_80) == pytest.approx(expected_output)


def test_positive_range(array_positive_range) -> None:
    """tsx.variance should return 850 for a range from 0 to 100."""
    expected_output: int = 850
    assert tsx.variance(array_positive_range) == expected_output


def test_negative_range(array_negative_range) -> None:
    """tsx.variance should return 29.15476 for a range from -100 to 0."""
    expected_output: int = 850
    assert tsx.variance(array_negative_range) == expected_output


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.variance should return 29.15476 for a range from -50 to 50."""
    expected_output: int = 850
    assert tsx.variance(array_positive_and_negative_range) == expected_output

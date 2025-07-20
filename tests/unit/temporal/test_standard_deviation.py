"""Unit tests for tsxtract.extraction.standard_deviation."""

import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx


def test_ones(ones_array) -> None:
    """tsx.standard_deviation should return 0 for a time series with 100 ones."""
    expected_output: int = 0
    assert tsx.standard_deviation(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.standard_deviation should return 0 for a time series with 100 zeros."""
    expected_output: int = 0
    assert tsx.standard_deviation(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.standard_deviation should return 0 for a time series with 100 -1 values."""
    expected_output: int = 0
    assert tsx.standard_deviation(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.standard_deviation should return 0 for a single datapoint."""
    expected_output: float = 0
    assert tsx.standard_deviation(single_point) == expected_output


def test_empty(empty_array) -> None:
    """tsx.standard_deviation should return nan for an empty sequence."""
    assert jnp.isnan(tsx.standard_deviation(empty_array))


def test_nan_values(nan_array) -> None:
    """tsx.standard_deviation should return nan for an array with 100 nan values."""
    assert jnp.isnan(tsx.standard_deviation(nan_array))


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.standard_deviation should return nan for an array with 20 nan values."""
    assert jnp.isnan(tsx.standard_deviation(array_with_nan))


def test_inf_values(inf_array) -> None:
    """tsx.standard_deviation should return nan for an array with 100 inf values."""
    assert jnp.isnan(tsx.standard_deviation(inf_array))


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.standard_deviation should return a numeric for an array with 20 inf values."""
    assert jnp.isreal(tsx.standard_deviation(array_with_inf))


def test_50_50(array_50_50) -> None:
    """tsx.standard_deviation should return 0.5 for a time series with 50 zeros and 50 ones."""
    expected_output: float = 0.5
    assert tsx.standard_deviation(array_50_50) == expected_output


def test_20_80(array_20_80) -> None:
    """tsx.standard_deviation should return 0.4 for a time series with 20 zeros and 80 ones."""
    expected_output: float = 0.4
    assert tsx.standard_deviation(array_20_80) == pytest.approx(expected_output)


def test_positive_range(array_positive_range) -> None:
    """tsx.standard_deviation should return 29.15476 for a range from 0 to 100."""
    expected_output: float = 29.15476
    assert tsx.standard_deviation(array_positive_range) == pytest.approx(expected_output)


def test_negative_range(array_negative_range) -> None:
    """tsx.standard_deviation should return 29.15476 for a range from -100 to 0."""
    expected_output: float = 29.15476
    assert tsx.standard_deviation(array_negative_range) == pytest.approx(expected_output)


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.standard_deviation should return 29.15476 for a range from -50 to 50."""
    expected_output: float = 29.15476
    assert tsx.standard_deviation(array_positive_and_negative_range) == pytest.approx(
        expected_output,
    )

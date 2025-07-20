"""Unit tests for tsxtract.extraction.root_mean_square."""

import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx


def test_ones(ones_array) -> None:
    """tsx.root_mean_square should return 1.0 for a time series with 100 ones."""
    expected_output: float = 1.0
    assert tsx.root_mean_square(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.root_mean_square should return 0.0 for a time series with 100 zeros."""
    expected_output: float = 0.0
    assert tsx.root_mean_square(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.root_mean_square should return 1.0 for a time series with 100 -1 values."""
    expected_output: float = 1.0
    assert tsx.root_mean_square(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.root_mean_square should return 1.6226422 for this single datapoint."""
    expected_output: float = 1.6226422
    assert tsx.root_mean_square(single_point) == pytest.approx(expected_output)


def test_empty(empty_array) -> None:
    """tsx.root_mean_square should return nan for an empty sequence."""
    assert jnp.isnan(tsx.root_mean_square(empty_array))


def test_nan_values(nan_array) -> None:
    """tsx.root_mean_square should return nan for an array with 100 nan values."""
    assert jnp.isnan(tsx.root_mean_square(nan_array))


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.root_mean_square should return nan for an array with 20 nan values."""
    assert jnp.isnan(tsx.root_mean_square(array_with_nan))


def test_inf_values(inf_array) -> None:
    """tsx.root_mean_square should return inf for an array with 100 inf values."""
    assert jnp.isinf(tsx.root_mean_square(inf_array))


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.root_mean_square should return a numeric for an array with 20 inf values."""
    assert jnp.isreal(tsx.root_mean_square(array_with_inf))


def test_50_50(array_50_50) -> None:
    """tsx.root_mean_square should return 0.70710677 for a time series with 50 zeros and 50 ones."""
    expected_output: float = 0.70710677
    assert tsx.root_mean_square(array_50_50) == pytest.approx(expected_output)


def test_20_80(array_20_80) -> None:
    """tsx.root_mean_square should return 0.894427 for a time series with 20 zeros and 80 ones."""
    expected_output: float = 0.894427
    assert tsx.root_mean_square(array_20_80) == pytest.approx(expected_output)


def test_positive_range(array_positive_range) -> None:
    """tsx.root_mean_square should return 57.879185 for a range from 0 to 100."""
    expected_output: float = 57.879185
    assert tsx.root_mean_square(array_positive_range) == pytest.approx(expected_output)


def test_negative_range(array_negative_range) -> None:
    """tsx.root_mean_square should return 57.879185 for a range from -100 to 0."""
    expected_output: float = 57.879185
    assert tsx.root_mean_square(array_negative_range) == pytest.approx(expected_output)


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.root_mean_square should return 29.15476 for a range from -50 to 50."""
    expected_output: float = 29.15476
    assert tsx.root_mean_square(array_positive_and_negative_range) == pytest.approx(expected_output)

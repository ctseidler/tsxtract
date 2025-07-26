"""Unit tests for tsxtract.extraction.count_above."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


def test_ones(ones_array) -> None:
    """tsx.count_above should return 1.0 for a time series with 100 ones."""
    expected_output: float = 1.0
    assert tsx.count_above(ones_array, 0) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.count_above should return 1.0 for a time series with 100 zeros."""
    expected_output: float = 1.0
    assert tsx.count_above(zeros_array, 0) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.count_above should return 100 for a time series with 100 -1 values."""
    expected_output: float = 1.0
    assert tsx.count_above(negatives_array, -1) == expected_output


def test_single_point(single_point) -> None:
    """tsx.count_above should return 1.0 for this single datapoint."""
    expected_output: float = 1.0
    assert tsx.count_above(single_point, single_point.item()) == expected_output


def test_empty(empty_array) -> None:
    """tsx.count_above should return nan for an empty sequence."""
    assert jnp.isnan(tsx.count_above(empty_array, 0))


def test_nan_values(nan_array) -> None:
    """tsx.count_above should return 0.0 for an array with 100 nan values."""
    expected_output: float = 0.0
    assert tsx.count_above(nan_array, 0) == expected_output


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.count_above should return 0.8 for this array with 20 nan values."""
    expected_output: float = 0.8
    assert tsx.count_above(array_with_nan, -100) == pytest.approx(expected_output)


def test_inf_values(inf_array) -> None:
    """tsx.count_above should return 1.0 for an array with 100 inf values."""
    expected_output: float = 1.0
    assert tsx.count_above(inf_array, 0) == expected_output


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.count_above should return 1.0 for this array with 20 inf values."""
    expected_output: float = 1.0
    assert tsx.count_above(array_with_inf, -100) == expected_output


def test_50_50(array_50_50) -> None:
    """tsx.count_above should return 0.5 for this time series with 50 zeros and 50 ones."""
    expected_output: float = 0.5
    assert tsx.count_above(array_50_50, 0.5) == pytest.approx(expected_output)


def test_20_80(array_20_80) -> None:
    """tsx.count_above should return 0.8 for this time series with 20 zeros and 80 ones."""
    expected_output: float = 0.8
    assert tsx.count_above(array_20_80, 0.5) == pytest.approx(expected_output)


def test_positive_range(array_positive_range) -> None:
    """tsx.count_above should return 1.0 for a range from 0 to 100."""
    expected_output: float = 1.0
    assert tsx.count_above(array_positive_range, 0) == expected_output


def test_negative_range(array_negative_range) -> None:
    """tsx.count_above should return 1.0 for a range from -100 to 0."""
    expected_output: float = 1.0
    assert tsx.count_above(array_negative_range, -100) == expected_output


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.count_above should return 0.504950 for a range from -50 to 50."""
    expected_output: float = 0.504950
    assert tsx.count_above(array_positive_and_negative_range, 0) == pytest.approx(expected_output)

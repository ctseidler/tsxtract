"""Unit tests for tsxtract.extraction.value_count."""

import pytest

import tsxtract.extractors as tsx


def test_ones(ones_array) -> None:
    """tsx.value_count should return 100 for a time series with 100 ones."""
    expected_output: int = 100
    assert tsx.value_count(ones_array, 1) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.value_count should return 100 for a time series with 100 zeros."""
    expected_output: int = 100
    assert tsx.value_count(zeros_array, 0) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.value_count should return 100 for a time series with 100 -1 values."""
    expected_output: int = 100
    assert tsx.value_count(negatives_array, -1) == expected_output


def test_single_point(single_point) -> None:
    """tsx.value_count should return 1 for this single datapoint."""
    expected_output: int = 1
    assert tsx.value_count(single_point, single_point.item()) == expected_output


def test_empty(empty_array) -> None:
    """tsx.value_count should return 0 for an empty sequence."""
    expected_output: int = 0
    assert tsx.value_count(empty_array, 0) == expected_output


def test_nan_values(nan_array) -> None:
    """tsx.value_count should return 0 for an array with 100 nan values."""
    expected_output: int = 0
    assert tsx.value_count(nan_array, 0) == expected_output


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.value_count should return 0 for this array with 20 nan values."""
    expected_output: int = 0
    assert tsx.value_count(array_with_nan, 0) == expected_output


def test_inf_values(inf_array) -> None:
    """tsx.value_count should return 0 for an array with 100 inf values."""
    expected_output: int = 0
    assert tsx.value_count(inf_array, 0) == expected_output


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.value_count should return 0 for this array with 20 inf values."""
    expected_output: int = 0
    assert tsx.value_count(array_with_inf, 0) == expected_output


def test_50_50(array_50_50) -> None:
    """tsx.value_count should return 50 for a time series with 50 zeros and 50 ones."""
    expected_output: int = 50
    assert tsx.value_count(array_50_50, 0) == expected_output


def test_20_80(array_20_80) -> None:
    """tsx.value_count should return 80 for a time series with 20 zeros and 80 ones."""
    expected_output: int = 80
    assert tsx.value_count(array_20_80, 1) == pytest.approx(expected_output)


def test_positive_range(array_positive_range) -> None:
    """tsx.value_count should return 1 for a range from 0 to 100."""
    expected_output: int = 1
    assert tsx.value_count(array_positive_range, 100) == expected_output


def test_negative_range(array_negative_range) -> None:
    """tsx.value_count should return 1 for a range from -100 to 0."""
    expected_output: int = 1
    assert tsx.value_count(array_negative_range, -100) == expected_output


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.value_count should return 1 for a range from -50 to 50."""
    expected_output: int = 1
    assert tsx.value_count(array_positive_and_negative_range, 0) == expected_output

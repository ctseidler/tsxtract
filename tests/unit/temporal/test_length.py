"""Unit tests for tsxtract.extraction.length."""

import tsxtract.extraction as tsx


def test_normal(normal_array) -> None:
    """tsx.length should return 100 for a time series with 100 points."""
    expected_output: int = 100
    assert tsx.length(normal_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.length should return 1 for a single datapoint."""
    expected_output: int = 1
    assert tsx.length(single_point) == expected_output


def test_empty(empty_array) -> None:
    """tsx.length should return 0 for an empty sequence."""
    expected_output: int = 0
    assert tsx.length(empty_array) == expected_output


def test_nan_values(nan_array) -> None:
    """tsx.length should return 100 for an array with 100 nan values."""
    expected_output: int = 100
    assert tsx.length(nan_array) == expected_output


def test_inf_values(inf_array) -> None:
    """tsx.length should return 100 for an array with 100 inf values."""
    expected_output: int = 100
    assert tsx.length(inf_array) == expected_output

"""Unit tests for tsxtract.extraction.absolute_maximum."""

import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx


def test_ones(ones_array) -> None:
    """tsx.absolute_maximum should return 1 for a time series with 100 ones."""
    expected_output: int = 1
    assert tsx.absolute_maximum(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.absolute_maximum should return 0 for a time series with 100 zeros."""
    expected_output: int = 0
    assert tsx.absolute_maximum(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.absolute_maximum should return 1 for a time series with 100 -1 values."""
    expected_output: int = 1
    assert tsx.absolute_maximum(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.absolute_maximum should return the value for a single datapoint."""
    expected_output: float = single_point
    assert tsx.absolute_maximum(single_point) == expected_output


def test_empty(empty_array) -> None:
    """tsx.absolute_maximum should raise a ValueError for an empty sequence."""
    with pytest.raises(
        ValueError,
        match="zero-size array to reduction operation max which has no identity",
    ):
        tsx.absolute_maximum(empty_array)


def test_nan_values(nan_array) -> None:
    """tsx.absolute_maximum should return nan for an array with 100 nan values."""
    assert jnp.isnan(tsx.absolute_maximum(nan_array))


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.absolute_maximum should return nan for an array with 20 nan values."""
    assert jnp.isnan(tsx.absolute_maximum(array_with_nan))


def test_inf_values(inf_array) -> None:
    """tsx.absolute_maximum should return inf for an array with 100 inf values."""
    assert jnp.isinf(tsx.absolute_maximum(inf_array))


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.absolute_maximum should return a numeric for an array with 20 inf values."""
    assert jnp.isreal(tsx.absolute_maximum(array_with_inf))


def test_50_50(array_50_50) -> None:
    """tsx.absolute_maximum should return 1 for a time series with 50 zeros and 50 ones."""
    expected_output: int = 1
    assert tsx.absolute_maximum(array_50_50) == expected_output


def test_20_80(array_20_80) -> None:
    """tsx.absolute_maximum should return 80 for a time series with 20 zeros and 80 ones."""
    expected_output: int = 1
    assert tsx.absolute_maximum(array_20_80) == expected_output


def test_positive_range(array_positive_range) -> None:
    """tsx.absolute_maximum should return 100 for a range from 0 to 100."""
    expected_output: int = 100
    assert tsx.absolute_maximum(array_positive_range) == expected_output


def test_negative_range(array_negative_range) -> None:
    """tsx.absolute_maximum should return 100 for a range from -100 to 0."""
    expected_output: int = 100
    assert tsx.absolute_maximum(array_negative_range) == expected_output


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.absolute_maximum should return 50 for a range from -50 to 50."""
    expected_output: int = 50
    assert tsx.absolute_maximum(array_positive_and_negative_range) == expected_output

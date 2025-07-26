"""Unit tests for tsxtract.extraction.minimum."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


def test_ones(ones_array) -> None:
    """tsx.minimum should return 1 for a time series with 100 ones."""
    expected_output: int = 1
    assert tsx.minimum(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.minimum should return 0 for a time series with 100 zeros."""
    expected_output: int = 0
    assert tsx.minimum(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.minimum should return -1 for a time series with 100 -1 values."""
    expected_output: int = -1
    assert tsx.minimum(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.minimum should return the value for a single datapoint."""
    expected_output: float = single_point
    assert tsx.minimum(single_point) == expected_output


def test_empty(empty_array) -> None:
    """tsx.minimum should raise a ValueError for an empty sequence."""
    with pytest.raises(
        ValueError,
        match="zero.size array to reduction operation min which has no identity",
    ):
        tsx.minimum(empty_array)


def test_nan_values(nan_array) -> None:
    """tsx.minimum should return nan for an array with 100 nan values."""
    assert jnp.isnan(tsx.minimum(nan_array))


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.minimum should return nan for an array with 80 normal values and 20 nan values."""
    assert jnp.isnan(tsx.minimum(array_with_nan))


def test_inf_values(inf_array) -> None:
    """tsx.minimum should return inf for an array with 100 inf values."""
    assert jnp.isinf(tsx.minimum(inf_array))


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.minimum should return a numeric for an array with 80 normal values and 20 inf values."""
    assert jnp.isreal(tsx.minimum(array_with_inf))


def test_50_50(array_50_50) -> None:
    """tsx.minimum should return 0 for a time series with 50 zeros and 50 ones."""
    expected_output: int = 0
    assert tsx.minimum(array_50_50) == expected_output


def test_20_80(array_20_80) -> None:
    """tsx.minimum should return 0 for a time series with 20 zeros and 80 ones."""
    expected_output: int = 0
    assert tsx.minimum(array_20_80) == expected_output


def test_positive_range(array_positive_range) -> None:
    """tsx.minimum should return 0 for a time series with increasing value from 0 to 100."""
    expected_output: int = 0
    assert tsx.minimum(array_positive_range) == expected_output


def test_negative_range(array_negative_range) -> None:
    """tsx.minimum should return -100 for a time series with increasing value from -100 to 0."""
    expected_output: int = -100
    assert tsx.minimum(array_negative_range) == expected_output


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """tsx.minimum should return -50 for a time series with increasing value from -50 to 50."""
    expected_output: int = -50
    assert tsx.minimum(array_positive_and_negative_range) == expected_output

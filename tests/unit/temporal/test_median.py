"""Unit tests for tsxtract.extraction.median."""

import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx


def test_ones(ones_array) -> None:
    """tsx.median should return 1 for a time series with 100 ones."""
    expected_output: int = 1
    assert tsx.median(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """tsx.median should return 0 for a time series with 100 zeros."""
    expected_output: int = 0
    assert tsx.median(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """tsx.median should return -1 for a time series with 100 -1 values."""
    expected_output: int = -1
    assert tsx.median(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """tsx.median should return the value for a single datapoint."""
    expected_output: float = single_point
    assert tsx.median(single_point) == expected_output


def test_empty(empty_array) -> None:
    """tsx.median should raise a TypeError nan for an empty sequence."""
    with pytest.raises(TypeError):
        jnp.median(empty_array)


def test_nan_values(nan_array) -> None:
    """tsx.median should return nan for an array with 100 nan values."""
    assert jnp.isnan(tsx.median(nan_array))


def test_array_with_nan_values(array_with_nan) -> None:
    """tsx.median should return nan for an array with 80 normal values and 20 nan values."""
    assert jnp.isnan(tsx.median(array_with_nan))


def test_inf_values(inf_array) -> None:
    """tsx.median should return inf for an array with 100 inf values."""
    assert jnp.isinf(tsx.median(inf_array))


def test_array_with_inf_values(array_with_inf) -> None:
    """tsx.median should return a numeric for an array with 80 normal values and 20 inf values."""
    assert jnp.isreal(tsx.median(array_with_inf))


def test_50_50(array_50_50) -> None:
    """tsx.median should return 0.5 for a time series with 50 zeros and 50 ones."""
    expected_output: float = 0.5
    assert tsx.median(array_50_50) == expected_output


def test_20_80(array_20_80) -> None:
    """tsx.median should return 1 for a time series with 20 zeros and 80 ones."""
    expected_output: int = 1
    assert tsx.median(array_20_80) == expected_output

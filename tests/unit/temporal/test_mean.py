"""Unit tests for tsxtract.extraction.mean."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


@pytest.mark.parametrize(
    "arr, expected",
    [
        ("ones_array", 1),
        ("zeros_array", 0),
        ("negatives_array", -1),
    ],
)
def test_constant_arrays(arr, expected, request) -> None:
    """Mean of constant-valued arrays should equal that constant."""
    data = request.getfixturevalue(arr)
    assert tsx.mean(data) == expected


def test_single_value(single_point) -> None:
    """Mean of a single point should equal that value."""
    assert tsx.mean(single_point) == single_point


def test_empty_array(empty_array) -> None:
    """Mean of empty array should be NaN."""
    assert jnp.isnan(tsx.mean(empty_array))


@pytest.mark.parametrize(
    "arr_fixture",
    ["nan_array", "array_with_nan"],
)
def test_nan_propagation(arr_fixture, request) -> None:
    """Mean should be NaN if array contains only NaNs or any NaNs."""
    arr = request.getfixturevalue(arr_fixture)
    assert jnp.isnan(tsx.mean(arr))


@pytest.mark.parametrize(
    "arr_fixture",
    ["inf_array", "array_with_inf"],
)
def test_inf_propagation(arr_fixture, request) -> None:
    """Mean should be Inf if array contains only Infs or any Inf."""
    arr = request.getfixturevalue(arr_fixture)
    assert jnp.isinf(tsx.mean(arr))


@pytest.mark.parametrize(
    "arr_fixture, expected",
    [
        ("array_50_50", 0.5),
        ("array_20_80", 0.8),
    ],
)
def test_finite_mixed_arrays(arr_fixture, expected, request) -> None:
    """Mean of mixed finite arrays should be computed correctly."""
    arr = request.getfixturevalue(arr_fixture)
    assert tsx.mean(arr) == pytest.approx(expected)


def test_nan_inf_finite_mix(nan_inf_finite_array) -> None:
    """Mean should be NaN if NaN is present, even if Inf and finite values exist."""
    assert jnp.isnan(tsx.mean(nan_inf_finite_array))


def test_large_numbers(large_numbers_array) -> None:
    """Mean should handle very large numbers without overflow to Inf."""
    result = tsx.mean(large_numbers_array)
    assert not jnp.isinf(result)
    assert not jnp.isnan(result)

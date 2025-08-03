"""Unit tests for tsxtract.extraction.absolute_maximum."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.ones(5), 1),  # all ones
        (jnp.zeros(5), 0),  # all zeros
        (-jnp.ones(5), 1),  # all negative ones
        (jnp.array([1, -1, 1, -1]), 1),  # alternating ±1
        (jnp.array([0, 1, 0, 1]), 1),  # half zeros, half ones
    ],
)
def test_constant_and_mixed_arrays(array, expected):
    """absolute_maximum returns the max absolute value for constant and mixed arrays."""
    assert tsx.absolute_maximum(array) == expected


def test_single_point(single_point):
    """absolute_maximum of a single element is the absolute value of that element."""
    expected = jnp.abs(single_point)
    assert tsx.absolute_maximum(single_point) == expected


def test_empty_array(empty_array):
    """absolute_maximum should raise ValueError for empty input."""
    with pytest.raises(ValueError, match="Input array is empty"):
        tsx.absolute_maximum(empty_array)


@pytest.mark.parametrize(
    "array",
    [
        jnp.full(5, jnp.nan),  # all NaNs
        jnp.array([1, 2, jnp.nan, 3], dtype=jnp.float32),  # mixed NaN
    ],
)
def test_nan_handling(array):
    """absolute_maximum returns NaN if any NaN is present."""
    result = tsx.absolute_maximum(array)
    assert jnp.isnan(result)


@pytest.mark.parametrize(
    "array",
    [
        jnp.full(5, jnp.inf),  # all Inf
        jnp.array([1, 2, jnp.inf, 3], dtype=jnp.float32),  # mixed Inf
    ],
)
def test_inf_handling(array):
    """absolute_maximum returns Inf if any Inf is present."""
    result = tsx.absolute_maximum(array)
    assert jnp.isinf(result)


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.array([0, 0, 1, 1]), 1),  # half zeros, half ones
        (jnp.array([0, 1, 1, 1, 1]), 1),  # 20% zeros, 80% ones
        (jnp.arange(0, 101), 100),  # 0..100
        (jnp.arange(-100, 1), 100),  # -100..0
        (jnp.arange(-50, 51), 50),  # -50..50
    ],
)
def test_various_arrays(array, expected):
    """absolute_maximum should return correct max abs value for various inputs."""
    assert tsx.absolute_maximum(array) == expected


def test_large_numbers():
    """absolute_maximum should handle large finite values correctly without overflow."""
    arr = jnp.array([1e18, -1e18, 1e18, -1e18])
    result = tsx.absolute_maximum(arr)
    assert jnp.isfinite(result)
    assert result == 1e18

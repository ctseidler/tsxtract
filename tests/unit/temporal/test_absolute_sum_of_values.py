"""Unit tests for tsxtract.extraction.absolute_sum_values."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.ones(5), 5),  # all ones
        (jnp.zeros(5), 0),  # all zeros
        (-jnp.ones(5), 5),  # all negative ones
        (jnp.array([1, -1, 1]), 3),  # alternating sign
    ],
)
def test_basic_patterns(array, expected):
    """Sum of absolute values for simple patterns."""
    assert tsx.absolute_sum_values(array) == expected


def test_single_point(single_point):
    """Single element returns its absolute value."""
    expected = jnp.abs(single_point)
    assert tsx.absolute_sum_values(single_point) == expected


def test_empty(empty_array):
    """Empty array returns zero."""
    assert tsx.absolute_sum_values(empty_array) == 0


@pytest.mark.parametrize(
    "array",
    [
        jnp.full(5, jnp.nan),  # all NaNs
        jnp.array([1, jnp.nan, -3]),  # some NaNs
    ],
)
def test_nan_handling(array):
    """If any NaN present, result is NaN."""
    result = tsx.absolute_sum_values(array)
    assert jnp.isnan(result)


def test_inf_values():
    """All inf → inf result."""
    arr = jnp.full(5, jnp.inf)
    assert jnp.isinf(tsx.absolute_sum_values(arr))


def test_mixed_inf_values():
    """Mix of finite & inf → inf result."""
    arr = jnp.array([1, 2, jnp.inf, -4])
    assert jnp.isinf(tsx.absolute_sum_values(arr))


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.array([0] * 50 + [1] * 50), 50),  # 50 zeros, 50 ones
        (jnp.array([0] * 20 + [1] * 80), 80),  # 20 zeros, 80 ones
        (jnp.arange(0, 101), 5050),  # positive range
        (jnp.arange(-100, 1), 5050),  # negative range
        (jnp.arange(-50, 51), 2550),  # symmetric range
    ],
)
def test_various_patterns(array, expected):
    """Correct sum for different structured arrays."""
    assert tsx.absolute_sum_values(array) == expected

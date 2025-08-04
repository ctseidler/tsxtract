"""Unit tests for tsxtract.extraction.absolute_sum_of_changes."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.ones(5), 0),  # all ones
        (jnp.zeros(5), 0),  # all zeros
        (-jnp.ones(5), 0),  # all negative ones
        (jnp.array([1, -1, 1, -1]), 6),  # alternating ±1
        (jnp.array([0, 1, 0, 1]), 3),  # up-down pattern
    ],
)
def test_constant_and_mixed_arrays(array, expected):
    """Absolute sum of changes matches manual calculation for constant and mixed arrays."""
    assert tsx.absolute_sum_of_changes(array) == expected


def test_single_point(single_point):
    """Single point has zero total change."""
    assert tsx.absolute_sum_of_changes(single_point) == 0


def test_empty_array(empty_array):
    """Empty array has zero total change."""
    assert tsx.absolute_sum_of_changes(empty_array) == 0


@pytest.mark.parametrize(
    "array",
    [
        jnp.full(5, jnp.nan),  # all NaNs
        jnp.array([1, 2, jnp.nan, 3]),  # mixed NaN
    ],
)
def test_nan_handling(array):
    """If any NaN is present, result is NaN."""
    result = tsx.absolute_sum_of_changes(array)
    assert jnp.isnan(result)


def test_inf_values():
    """If all values are Inf, result is Nan (Inf - Inf = NaN)."""
    arr = jnp.full(5, jnp.inf)
    assert jnp.isnan(tsx.absolute_sum_of_changes(arr))


def test_mixed_inf_values():
    """Mix of finite & inf → inf result."""
    arr = jnp.array([1, 2, jnp.inf, 3])
    assert jnp.isinf(tsx.absolute_sum_of_changes(arr))


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.array([0, 0, 1, 1]), 1.0),  # 50/50 zeros & ones
        (jnp.array([0, 1, 1, 1, 1]), 1.0),  # 20/80 split
        (jnp.arange(0, 101), 100),  # increasing sequence
        (jnp.arange(-100, 1), 100),  # increasing negative to zero
        (jnp.arange(-50, 51), 100),  # symmetric range
    ],
)
def test_various_patterns(array, expected):
    """Correct change sum for typical numeric patterns."""
    assert tsx.absolute_sum_of_changes(array) == expected


def test_large_numbers():
    """Should handle large finite values without overflow."""
    arr = jnp.array([1e12, -1e12])
    result = tsx.absolute_sum_of_changes(arr)
    assert jnp.isfinite(result)
    assert result == pytest.approx(2e12)

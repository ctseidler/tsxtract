"""Unit tests for tsxtract.extraction.count_above_mean."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.ones(5), 0),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), 0),
        (jnp.array([0, 1, 2]), 1),
    ],
)
def test_basic_cases(array, expected):
    """Basic numeric arrays."""
    assert tsx.count_above_mean(array) == expected


def test_single_point(single_point):
    """Single value array should return 0."""
    assert tsx.count_above_mean(single_point) == 0


def test_empty(empty_array):
    """Empty array returns 0."""
    assert tsx.count_above_mean(empty_array) == 0


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.full(5, jnp.nan), 0),  # all NaN
        (jnp.array([1, jnp.nan, 2, 3]), 1),  # ignores NaNs in comparison
    ],
)
def test_nan_handling(array, expected):
    """NaN values should be handled."""
    assert tsx.count_above_mean(array) == expected


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.full(5, jnp.inf), 0),
        (jnp.array([1, jnp.inf, -3]), 0),  # only inf above finite mean, but inf does not count
    ],
)
def test_inf_handling(array, expected):
    """Inf values should be handled."""
    assert tsx.count_above_mean(array) == expected


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.array([0] * 50 + [1] * 50), 50),
        (jnp.array([0] * 20 + [1] * 80), 80),
        (jnp.arange(0, 101), 50),
        (jnp.arange(-100, 1), 50),
        (jnp.arange(-50, 51), 50),
    ],
)
def test_varied_patterns(array, expected):
    """Varied patterns should pose no issue."""
    assert tsx.count_above_mean(array) == expected

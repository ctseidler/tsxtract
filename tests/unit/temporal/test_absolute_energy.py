"""Unit tests for tsxtract.extraction.absolute_energy."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.ones(5), 5),  # all ones
        (jnp.zeros(5), 0),  # all zeros
        (-jnp.ones(5), 5),  # all negative ones
        (jnp.array([1, -1, 1, -1]), 4),  # alternating ±1
        (jnp.array([0, 1, 0, 1]), 2),  # half zeros, half ones
    ],
)
def test_constant_and_mixed_arrays(array, expected):
    """Absolute energy should match sum of squares for constant and mixed arrays."""
    assert tsx.absolute_energy(array) == expected


def test_single_point(single_point):
    """Absolute energy of one value is its square."""
    expected = single_point**2
    assert tsx.absolute_energy(single_point) == expected


def test_empty(empty_array):
    """Empty sequence should have zero absolute energy."""
    assert tsx.absolute_energy(empty_array) == 0


@pytest.mark.parametrize(
    "array",
    [
        jnp.array([jnp.nan] * 5),  # all NaN
        jnp.array([1, 2, jnp.nan, 3], dtype=jnp.float32),  # mixed NaN
    ],
)
def test_nan_handling(array):
    """Absolute energy should be NaN if any NaN is present."""
    assert jnp.isnan(tsx.absolute_energy(array))


@pytest.mark.parametrize(
    "array",
    [
        jnp.array([jnp.inf] * 5),  # all Inf
        jnp.array([1, 2, jnp.inf, 3], dtype=jnp.float32),  # mixed Inf
    ],
)
def test_inf_handling(array):
    """Absolute energy should be Inf if any Inf is present."""
    assert jnp.isinf(tsx.absolute_energy(array))


def test_large_numbers():
    """Large numbers should not overflow to NaN in float32."""
    arr = jnp.array([1e10, 1e10, 1e10, 1e10])
    assert jnp.isfinite(tsx.absolute_energy(arr))


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (jnp.arange(0, 101), 338350),  # 0..100
        (jnp.arange(-100, 1), 338350),  # -100..0
        (jnp.arange(-50, 51), 85850),  # -50..50
    ],
)
def test_ranges(array, expected):
    """Absolute energy for numeric ranges should match manual sum of squares."""
    assert tsx.absolute_energy(array) == expected

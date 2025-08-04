"""Unit tests for tsxtract.extraction.count_above."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


@pytest.mark.parametrize(
    ("array", "threshold", "expected"),
    [
        (jnp.ones(5), 0, 1.0),  # all above 0
        (jnp.zeros(5), 0, 1.0),  # all equal to 0
        (-jnp.ones(5), -1, 1.0),  # all equal to -1
        (jnp.array([0, 1, 2]), 1, 2 / 3),  # two >= 1
    ],
)
def test_basic_cases(array, threshold, expected):
    """Fraction above threshold for common patterns."""
    assert tsx.count_above(array, threshold) == pytest.approx(expected)


def test_single_point(single_point):
    """Single point is compared directly."""
    assert tsx.count_above(single_point, single_point.item()) == 1.0


def test_empty(empty_array):
    """Empty array returns NaN."""
    assert jnp.isnan(tsx.count_above(empty_array, 0))


@pytest.mark.parametrize(
    ("array", "threshold", "expected"),
    [
        (jnp.full(5, jnp.nan), 0, 0.0),  # all NaN
        (jnp.array([1, jnp.nan, 2, 3]), -100, 0.75),  # NaN excluded from numerator
    ],
)
def test_nan_handling(array, threshold, expected):
    """NaNs are treated as not above threshold."""
    assert tsx.count_above(array, threshold) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("array", "threshold", "expected"),
    [
        (jnp.full(5, jnp.inf), 0, 1.0),  # all infinite
        (jnp.array([1, jnp.inf, -3]), -100, 1.0),  # finite + inf all above threshold
    ],
)
def test_inf_handling(array, threshold, expected):
    """Infinities handled correctly."""
    assert tsx.count_above(array, threshold) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("array", "threshold", "expected"),
    [
        (jnp.array([0] * 50 + [1] * 50), 0.5, 0.5),  # 50%
        (jnp.array([0] * 20 + [1] * 80), 0.5, 0.8),  # 80%
        (jnp.arange(0, 101), 0, 1.0),  # all >= 0
        (jnp.arange(-100, 1), -100, 1.0),  # all >= -100
        (jnp.arange(-50, 51), 0, 0.504950),  # symmetric range
    ],
)
def test_varied_patterns(array, threshold, expected):
    """Fractions computed correctly for various structures."""
    assert tsx.count_above(array, threshold) == pytest.approx(expected)

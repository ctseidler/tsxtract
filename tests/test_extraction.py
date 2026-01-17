"""Test suite for `tsxtract.extraction.py`."""

import jax
import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx
from tsxtract.utils import generate_random_time_series_dataset


def test_extract_features():
    """Test feature extraction."""
    # Standard case: Array of shape (samples, channels, length)
    implemented_features = 12
    dataset = generate_random_time_series_dataset(
        n_samples=10,
        n_channels=2,
        sampling_rate=5,
        time_series_length_in_seconds=2,
    )
    features = tsx.extract_features(dataset)
    assert isinstance(features, dict)
    assert len(features) == implemented_features
    assert isinstance(features["mean"], jax.Array)
    assert features["mean"].shape == (10, 2)

    # Error case: Array of shape (samples, length)
    dataset = jax.random.normal(
        jax.random.key(0),
        shape=((5, 10)),
    )
    with pytest.raises(ValueError, match="not enough values to unpack"):
        features = tsx.extract_features(dataset)


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 1),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), 1),
        (jnp.array([-1, 0, 1, 0, -1]), 1),
        (jnp.array([-1, 0, 1, 2, 3]), 3),
        (jnp.array([-4, 0, 1, 2, 3]), 4),
        (jnp.array([1e18, 1e18]), 1e18),
        (jnp.array([-1e18, -1e18]), 1e18),
    ],
)
def test_absolute_maximum(array, expected):
    """Test extraction of absolute maximum."""
    assert tsx.absolute_maximum(signal=array) == expected


def test_absolute_maximum_edge_cases():
    """Test extraction of absolute maximum on edge cases."""
    with pytest.raises(
        ValueError,
        match="zero-size array",
    ):
        tsx.absolute_maximum(signal=jnp.array([]))

    assert jnp.isnan(tsx.absolute_maximum(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isinf(tsx.absolute_maximum(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.absolute_maximum(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isinf(tsx.absolute_maximum(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isinf(tsx.absolute_maximum(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.absolute_maximum(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 5),
        (jnp.zeros(5), 5),
        (-jnp.ones(5), 5),
        (jnp.array([-1, 0, 1, 0, -1]), 5),
        (jnp.array([-1, 0, 1, 2, 3]), 5),
        (jnp.array([1e18, 1e18]), 2),
        (jnp.array([-1e18, -1e18]), 2),
        (jnp.array([]), 0),
        (jnp.array([jnp.nan, jnp.nan]), 2),
        (jnp.array([jnp.inf, -jnp.inf]), 2),
        (jnp.array([0, jnp.nan, 1]), 3),
        (jnp.array([0, jnp.inf, 1]), 3),
        (jnp.array([0, -jnp.inf, 1]), 3),
        (jnp.array([0, jnp.nan, jnp.inf, 1]), 4),
    ],
)
def test_length(array, expected):
    """Test extraction of length."""
    assert tsx.length(signal=array) == expected


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 1),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), -1),
        (jnp.array([-1, 0, 1, 0, -1]), 1),
        (jnp.array([-1, 0, 1, 2, 3]), 3),
        (jnp.array([1e18, 1e18]), 1e18),
        (jnp.array([-1e18, -1e18]), -1e18),
    ],
)
def test_maximum(array, expected):
    """Test extraction of maximum value."""
    assert tsx.maximum(signal=array) == expected


def test_maximum_edge_cases():
    """Test extraction of maximum on edge cases."""
    with pytest.raises(
        ValueError,
        match="zero-size array",
    ):
        tsx.maximum(signal=jnp.array([]))

    assert jnp.isnan(tsx.maximum(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isinf(tsx.maximum(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.maximum(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isinf(tsx.maximum(signal=jnp.array([0, jnp.inf, 1])))
    assert tsx.maximum(signal=jnp.array([0, -jnp.inf, 1])) == 1
    assert jnp.isnan(tsx.maximum(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 1),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), -1),
        (jnp.array([-1, 0, 1, 0, -1]), -0.2),
        (jnp.array([-1, 0, 1, 2, 3]), 1),
        (jnp.array([1e18, 1e18]), 1e18),
        (jnp.array([-1e18, -1e18]), -1e18),
    ],
)
def test_mean(array, expected):
    """Test extraction of mean value."""
    assert tsx.mean(signal=array) == expected


def test_mean_edge_cases():
    """Test extraction of mean on edge cases."""
    assert jnp.isnan(tsx.mean(signal=jnp.array([])))
    assert jnp.isnan(tsx.mean(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isinf(tsx.mean(signal=jnp.array([jnp.inf, jnp.inf])))
    assert jnp.isnan(tsx.mean(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.mean(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isinf(tsx.mean(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isinf(tsx.mean(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.mean(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 1),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), -1),
        (jnp.array([-1, 0, 1, 0, -1]), 0),
        (jnp.array([-1, 0, 1, 2, 3]), 1),
        (jnp.array([1e18, 1e18]), 1e18),
        (jnp.array([-1e18, -1e18]), -1e18),
        (jnp.array([0, jnp.inf, 1]), 1),
        (jnp.array([0, -jnp.inf, 1]), 0),
    ],
)
def test_median(array, expected):
    """Test extraction of median value."""
    assert tsx.median(signal=array) == expected


def test_median_edge_cases():
    """Test extraction of median on edge cases."""
    with pytest.raises(
        TypeError,
        match=r"Slice size at index 0 in gather op is out of range",
    ):
        tsx.median(signal=jnp.array([]))

    assert jnp.isnan(tsx.median(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isinf(tsx.median(signal=jnp.array([jnp.inf, jnp.inf])))
    assert jnp.isnan(tsx.median(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.median(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isnan(tsx.median(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 1),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), -1),
        (jnp.array([-1, 0, 1, 0, -1]), -1),
        (jnp.array([-1, 0, 1, 2, 3]), -1),
        (jnp.array([1e18, 1e18]), 1e18),
        (jnp.array([-1e18, -1e18]), -1e18),
    ],
)
def test_minimum(array, expected):
    """Test extraction of minimum value."""
    assert tsx.minimum(signal=array) == expected


def test_minimum_edge_cases():
    """Test extraction of minimum on edge cases."""
    with pytest.raises(
        ValueError,
        match="zero-size array",
    ):
        tsx.minimum(signal=jnp.array([]))

    assert jnp.isnan(tsx.minimum(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isinf(tsx.minimum(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.minimum(signal=jnp.array([0, jnp.nan, 1])))
    assert tsx.minimum(signal=jnp.array([0, jnp.inf, 1])) == 0
    assert jnp.isinf(tsx.minimum(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.minimum(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 0),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), 0),
        (jnp.array([-1, 0, 1, 0, -1]), 2),
        (jnp.array([-1, 0, 1, 2, 3]), 4),
        (jnp.array([1e18, 1e18]), 0),
        (jnp.array([-1e18, -1e18]), 0),
    ],
)
def test_peak_to_peak_distance(array, expected):
    """Test extraction of peak_to_peak distance."""
    assert tsx.peak_to_peak_distance(signal=array) == expected


def test_peak_to_peak_distance_edge_cases():
    """Test extraction of peak_to_peak distance on edge cases."""
    with pytest.raises(
        ValueError,
        match="zero-size array",
    ):
        tsx.peak_to_peak_distance(signal=jnp.array([]))

    assert jnp.isnan(tsx.peak_to_peak_distance(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isinf(tsx.peak_to_peak_distance(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.peak_to_peak_distance(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isinf(tsx.peak_to_peak_distance(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isinf(tsx.peak_to_peak_distance(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.peak_to_peak_distance(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 0),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), 0),
        (jnp.array([-1, 0, 1, 0, -1]), 0.74833155),
        (jnp.array([-1, 0, 1, 2, 3]), 1.4142135),
        (jnp.array([1e18, 1e18]), 0),
        (jnp.array([-1e18, -1e18]), 0),
    ],
)
def test_standard_deviation(array, expected):
    """Test extraction of standard deviation."""
    assert tsx.standard_deviation(signal=array) == expected


def test_standard_deviation_edge_cases():
    """Test extraction of standard deviation on edge cases."""
    assert jnp.isnan(tsx.standard_deviation(signal=jnp.array([])))
    assert jnp.isnan(tsx.standard_deviation(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isnan(tsx.standard_deviation(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.standard_deviation(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isnan(tsx.standard_deviation(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isnan(tsx.standard_deviation(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.standard_deviation(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 5),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), -5),
        (jnp.array([-1, 0, 1, 0, -1]), -1),
        (jnp.array([-1, 0, 1, 2, 3]), 5),
        (jnp.array([1e18, 1e18]), 2e18),
        (jnp.array([-1e18, -1e18]), -2e18),
        (jnp.array([]), 0),
    ],
)
def test_sum_values(array, expected):
    """Test extraction of sum_values."""
    assert tsx.sum_values(signal=array) == expected


def test_sum_values_edge_cases():
    """Test extraction of sum_values on edge cases."""
    assert jnp.isnan(tsx.sum_values(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isnan(tsx.sum_values(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.sum_values(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isinf(tsx.sum_values(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isinf(tsx.sum_values(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.sum_values(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 0),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), 0),
        (jnp.array([-1, 0, 1, 0, -1]), 0.56000006),
        (jnp.array([-1, 0, 1, 2, 3]), 2.0),
        (jnp.array([1e18, 1e18]), 0),
        (jnp.array([-1e18, -1e18]), 0),
    ],
)
def test_variance(array, expected):
    """Test extraction of variance."""
    assert tsx.variance(signal=array) == expected


def test_variance_edge_cases():
    """Test extraction of variance on edge cases."""
    assert jnp.isnan(tsx.variance(signal=jnp.array([])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), False),
        (jnp.zeros(5), False),
        (-jnp.ones(5), False),
        (jnp.array([-1, 0, 1, 0, -1]), False),
        (jnp.array([-1, 0, 1, 2, 3]), True),
        (jnp.array([1e18, 1e18]), False),
        (jnp.array([-1e18, -1e18]), False),
        (jnp.array([]), False),
        (jnp.array([jnp.nan, jnp.nan]), False),
        (jnp.array([jnp.inf, -jnp.inf]), False),
        (jnp.array([0, jnp.nan, 1]), False),
        (jnp.array([0, jnp.inf, 1]), False),
        (jnp.array([0, -jnp.inf, 1]), False),
        (jnp.array([0, jnp.nan, jnp.inf, 1]), False),
    ],
)
def test_variance_larger_than_standard_deviation(array, expected):
    """Test extraction of variance_larger_than_standard_deviation."""
    assert tsx.variance_larger_than_standard_deviation(signal=array) == expected


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 0),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), 0),
        (jnp.array([-1, 0, 1, 0, -1]), 4),
        (jnp.array([-1, 0, 1, 2, 3]), 2),
        (jnp.array([1e18, 1e18]), 0),
        (jnp.array([-1e18, -1e18]), 0),
        (jnp.array([]), 0),
        (jnp.array([jnp.nan, jnp.nan]), 1),
        (jnp.array([jnp.inf, -jnp.inf]), 1),
        (jnp.array([0, jnp.nan, 1]), 2),
        (jnp.array([0, jnp.inf, 1]), 1),
        (jnp.array([0, -jnp.inf, 1]), 2),
        (jnp.array([0, jnp.nan, jnp.inf, 1]), 2),
    ],
)
def test_zero_crossing_rate(array, expected):
    """Test extraction of zero_crossing_rate."""
    assert tsx.zero_crossing_rate(signal=array) == expected

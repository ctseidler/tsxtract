"""Test suite for `tsxtract.extraction.py`."""

import jax
import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx
from tsxtract.utils import generate_random_time_series_dataset


def test_extract_features():
    """Test feature extraction."""
    # Standard case: Array of shape (samples, channels, length)
    implemented_features = 3
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

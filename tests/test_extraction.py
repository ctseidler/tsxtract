"""Test suite for `tsxtract.extraction.py`."""

import jax
import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx
from tsxtract.utils import generate_random_time_series_dataset


def test_extract_features():
    """Test feature extraction."""
    # Standard case: Array of shape (samples, channels, length)
    implemented_features = 10
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


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 0),
        (jnp.zeros(5), 0),
        (-jnp.ones(5), 0),
        (jnp.array([0.0, 2.0]), 1.0),
        (jnp.array([-1.0, 1.0]), 1.0),
        (jnp.array([-2.0, 2.0]), 2.0),
        (jnp.array([0.0, 4.0]), 2.0),
    ],
)
def test_std(array, expected):
    """Test extraction of standard deviation."""
    assert tsx.std(signal=array) == expected


def test_std_edge_cases():
    """Test extraction of standard deviation on edge cases."""
    assert jnp.isnan(tsx.std(signal=jnp.array([])))
    assert jnp.isnan(tsx.std(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isnan(tsx.std(signal=jnp.array([jnp.inf, jnp.inf])))
    assert jnp.isnan(tsx.std(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.std(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isnan(tsx.std(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isnan(tsx.std(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.std(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))

@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 0),
        (jnp.zeros(5), 0),              
        (-jnp.ones(5), 0),
        (jnp.array([0.0, 2.0]), 1.0),
        (jnp.array([-1.0, 1.0]), 1.0),
        (jnp.array([-2.0, 2.0]), 4.0),      
        (jnp.array([0.0, 4.0]), 4.0),
    ],
)
def test_variance(array, expected):
    """Test extraction of variance."""
    assert tsx.variance(signal=array) == expected   


def test_variance_edge_cases():
    """Test extraction of variance on edge cases."""
    assert jnp.isnan(tsx.variance(signal=jnp.array([])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([jnp.inf, jnp.inf])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.variance(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


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
    ],
)
def test_median(array, expected):
    """Test extraction of median value."""
    assert tsx.median(signal=array) == expected 


def test_median_edge_cases():  
    """Test extraction of median on edge cases."""
    with pytest.raises(TypeError):
        tsx.median(signal=jnp.array([]))    

    assert jnp.isnan(tsx.median(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isinf(tsx.median(signal=jnp.array([jnp.inf, jnp.inf])))
    assert jnp.isnan(tsx.median(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.median(signal=jnp.array([0, jnp.nan, 1])))
    assert tsx.median(signal=jnp.array([0, jnp.inf, 1])) == 1.0
    assert tsx.median(signal=jnp.array([0, -jnp.inf, 1])) == 0.0
    assert jnp.isnan(tsx.median(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 1.0),
        (jnp.zeros(5), 0.0),
        (-jnp.ones(5), 1.0),
        (jnp.array([5.0]), 5.0),
        (jnp.array([-1.0, 0.0, 1.0, 0.0, -1.0]), 0.77),
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0]), 1.73),
        (jnp.array([1.0, 1.0, 1.0, 1.0, 10.0]), 4.56),
        (jnp.array([3.0, 3.0, 3.0, 3.0, 2.0]), 2.83),
        (jnp.array([1e18, 1e18]), 1e18),
        (jnp.array([-1e18, -1e18]), 1e18),
    ],
)
def test_rms(array, expected):
    """Test extraction of root mean square value."""
    assert tsx.rms(signal=array) == pytest.approx(expected, rel=1e-2)

def test_rms_edge_cases():
    """Test extraction of root mean square on edge cases."""
    assert jnp.isnan(tsx.rms(signal=jnp.array([])))
    assert jnp.isnan(tsx.rms(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isinf(tsx.rms(signal=jnp.array([jnp.inf, jnp.inf])))
    assert jnp.isinf(tsx.rms(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.rms(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isinf(tsx.rms(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isinf(tsx.rms(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.rms(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))

@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.ones(5), 0.0),
        (jnp.zeros(5), 0.0),    
        (-jnp.ones(5), 0.0),
        (jnp.array([5.0]), 0.0),
        (jnp.array([-1.0, 0.0, 1.0, 0.0, -1.0]), 0.64),
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0]), 1.2),
        (jnp.array([1.0, 1.0, 1.0, 1.0, 10.0]), 2.88),
        (jnp.array([3.0, 3.0, 3.0, 3.0, 2.0]), 0.32),
        (jnp.array([1e18, 1e18]), 0.0),
        (jnp.array([-1e18, -1e18]), 0.0),
    ],
)
def test_mad(array, expected):
    """Test extraction of mean absolute deviation value."""
    assert tsx.mad(signal=array) == pytest.approx(expected, rel=1e-2)       

def test_mad_edge_cases(): 
    """Test extraction of mean absolute deviation on edge cases."""
    assert jnp.isnan(tsx.mad(signal=jnp.array([])))
    assert jnp.isnan(tsx.mad(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isnan(tsx.mad(signal=jnp.array([jnp.inf, jnp.inf])))
    assert jnp.isnan(tsx.mad(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.mad(signal=jnp.array([0, jnp.nan, 1])))
    assert jnp.isnan(tsx.mad(signal=jnp.array([0, jnp.inf, 1])))
    assert jnp.isnan(tsx.mad(signal=jnp.array([0, -jnp.inf, 1])))
    assert jnp.isnan(tsx.mad(signal=jnp.array([0, jnp.nan, jnp.inf, 1])))

@pytest.mark.parametrize(
    argnames=("array", "q", "expected"),
    argvalues=[
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0]), 50, 1.0),
        (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 25, 2.0),
        (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 75, 4.0),
        (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0, 1.0),
        (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 100, 5.0),
        (jnp.array([0.0, 1.0, 2.0, 3.0]), 50, 1.5),
        (jnp.array([1e18, 1e18]), 50, 1e18),
    ],
)
def test_percentile(array, q, expected):
    """Test extraction of q-th percentile value."""
    assert tsx.percentile(signal=array, q=q) == pytest.approx(expected, rel=1e-2)

def test_percentile_edge_cases():
    """Test extraction of q-th percentile on edge cases."""
    with pytest.raises(TypeError):
        tsx.percentile(signal=jnp.array([]), q=50)
    assert jnp.isnan(tsx.percentile(signal=jnp.array([jnp.nan, jnp.nan]), q=50))
    assert jnp.isinf(tsx.percentile(signal=jnp.array([jnp.inf, jnp.inf]), q=50))
    assert jnp.isnan(tsx.percentile(signal=jnp.array([jnp.inf, -jnp.inf]), q=50))
    assert jnp.isnan(tsx.percentile(signal=jnp.array([0, jnp.nan, 1]), q=50))
    assert tsx.percentile(signal=jnp.array([0, jnp.inf, 1]), q=50) == 1.0
    assert tsx.percentile(signal=jnp.array([0, -jnp.inf, 1]), q=50) == 0.0
    assert jnp.isnan(tsx.percentile(signal=jnp.array([0, jnp.nan, jnp.inf, 1]), q=50))
"""Test suite for `tsxtract.extraction.py`."""

import jax
import jax.numpy as jnp
import pytest

import tsxtract.extraction as tsx
from tsxtract.utils import generate_random_time_series_dataset


def test_extract_features():
    """Test feature extraction."""
    # Standard case: Array of shape (samples, channels, length)
    implemented_features = 14
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


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0]), 0.0),
        (jnp.array([0.0, 1.0, 2.0, 3.0]), 0.0),
        (jnp.array([1.0, 2.0, 3.0, 10.0]), 1.0182), # right-skewed; scipy.stats.skew    
        (jnp.array([1.0, 8.0, 9.0, 10.0]), -1.0182), # left-skewed; scipy.stats.skew
    ],
)
def test_skewness(array, expected):
    """Test skewness against scipy.stats.skew (bias=True) references."""
    assert tsx.skewness(signal=array) == pytest.approx(expected, rel=1e-3, abs=1e-5)


def test_skewness_edge_cases():
    """Edge cases: moment-based features propagate NaN/inf to NaN."""
    assert jnp.isnan(tsx.skewness(signal=jnp.array([])))
    assert jnp.isnan(tsx.skewness(signal=jnp.array([3.0, 3.0, 3.0])))  
    assert jnp.isnan(tsx.skewness(signal=jnp.array([1e18, 1e18])))    
    assert jnp.isnan(tsx.skewness(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isnan(tsx.skewness(signal=jnp.array([jnp.inf, jnp.inf])))  
    assert jnp.isnan(tsx.skewness(signal=jnp.array([jnp.inf, -jnp.inf])))    
    assert jnp.isnan(tsx.skewness(signal=jnp.array([0.0, jnp.nan, 1.0])))
    assert jnp.isnan(tsx.skewness(signal=jnp.array([0.0, jnp.inf, 1.0])))


@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0]), -1.3),
        (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), -1.3),
        (jnp.array([0.0, 1.0, 2.0, 3.0]), -1.36),
        (jnp.array([0.0, 0.0, 0.0, 0.0, 10.0]), 0.25),
        (jnp.array([1.0, 2.0, 3.0, 10.0]), -0.7696),
    ],
)
def test_kurtosis(array, expected):
    """Test excess kurtosis against scipy.stats.kurtosis (fisher=True) references."""
    assert tsx.kurtosis(signal=array) == pytest.approx(expected, rel=1e-3, abs=1e-5)


def test_kurtosis_edge_cases():
    """Edge cases: same NaN-propagation contract as skewness."""
    assert jnp.isnan(tsx.kurtosis(signal=jnp.array([])))
    assert jnp.isnan(tsx.kurtosis(signal=jnp.array([3.0, 3.0, 3.0])))
    assert jnp.isnan(tsx.kurtosis(signal=jnp.array([1e18, 1e18])))
    assert jnp.isnan(tsx.kurtosis(signal=jnp.array([jnp.nan, jnp.nan])))
    assert jnp.isnan(tsx.kurtosis(signal=jnp.array([jnp.inf, jnp.inf])))
    assert jnp.isnan(tsx.kurtosis(signal=jnp.array([jnp.inf, -jnp.inf])))
    assert jnp.isnan(tsx.kurtosis(signal=jnp.array([0.0, jnp.nan, 1.0])))
    assert jnp.isnan(tsx.kurtosis(signal=jnp.array([0.0, jnp.inf, 1.0])))



@pytest.mark.parametrize(
    argnames=("array", "expected"),
    argvalues=[
        (jnp.array([1.0, -1.0, 1.0, -1.0]), 1.0),
        (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.0),
        (jnp.array([-1.0, -2.0, -3.0]), 0.0),
        (jnp.array([2.0, 1.0, -1.0, -3.0, 4.0]), 0.5),
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0]), 0.5),
        (jnp.array([0.0, 1.0, 2.0, 3.0]), 1.0 / 3.0),
    ],
)
def test_zero_crossing_rate(array, expected):
    """Test ZCR: crossings / (n-1) adjacent pairs, exact-zero double-count documented."""
    assert tsx.zero_crossing_rate(signal=array) == pytest.approx(expected, abs=1e-6)


def test_zero_crossing_rate_edge_cases():
    """Edge cases: counting-based feature — NaN/inf do NOT propagate to NaN.

    These assertions pin down the actual (documented) behaviour rather than
    an idealised one; see the docstring of zero_crossing_rate.
    """
    assert jnp.isnan(tsx.zero_crossing_rate(signal=jnp.array([5.0])))
    assert tsx.zero_crossing_rate(signal=jnp.array([jnp.inf, jnp.inf])) == 0.0
    assert tsx.zero_crossing_rate(signal=jnp.array([jnp.inf, -jnp.inf])) == 1.0
    assert tsx.zero_crossing_rate(signal=jnp.array([jnp.nan, jnp.nan])) == 1.0
    assert tsx.zero_crossing_rate(signal=jnp.array([0.0, jnp.nan, 1.0])) == 1.0


@pytest.mark.parametrize(
    argnames=("array", "lags", "expected"),
    argvalues=[
        # increasing ramp [1,2,3,4]: mean=2.5, denom=5.0
        # r1 = 1.25/5 = 0.25, r2 = -1.5/5 = -0.3, r3 = -2.25/5 = -0.45 (hand-calculated)
        (jnp.array([1.0, 2.0, 3.0, 4.0]), (1, 2, 3), jnp.array([0.25, -0.3, -0.45])),
        # alternating short signal [1,-1,1,-1]: boundary effect makes r1=-0.75 (not -1), r2=+0.5
        (jnp.array([1.0, -1.0, 1.0, -1.0]), (1, 2), jnp.array([-0.75, 0.5])),
        # long alternating signal: r1 approaches -1, r2 approaches +1 (period-2 structure)
        (jnp.array([1.0, -1.0] * 25), (2,), jnp.array([0.96])),
    ],
)
def test_autocorrelation(array, lags, expected):
    """Test autocorrelation against hand-calculated reference values."""
    assert jnp.allclose(tsx.autocorrelation(signal=array, lags=lags), expected, atol=1e-2)


def test_autocorrelation_edge_cases():
    """Autocorrelation requires 0 <= lag < signal length."""
    with pytest.raises(ValueError):
        tsx.autocorrelation(signal=jnp.array([]), lags=(1,))
    with pytest.raises(ValueError):
        tsx.autocorrelation(signal=jnp.array([1.0]), lags=(1,))
    with pytest.raises(ValueError):
        tsx.autocorrelation(signal=jnp.array([1.0, 2.0]), lags=(2,))


def test_autocorrelation_constant_signal_is_nan():
    """Constant signal has zero variance -> NaN, consistent with skewness/kurtosis."""
    result = tsx.autocorrelation(signal=jnp.array([3.0, 3.0, 3.0]), lags=(1,))
    assert jnp.isnan(result).all()
    
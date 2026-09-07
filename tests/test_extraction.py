"""Test suite for `tsxtract.extraction.py`."""

import jax
import jax.numpy as jnp
import pytest
import numpy as np
import scipy.stats

import tsxtract.extraction as tsx
from tsxtract.utils import generate_random_time_series_dataset

RELATIVE_TOLERANCE = 1e-9
"""Uniform tolerance for every assertion with a non-zero expected value."""

ABSOLUTE_TOLERANCE = 1e-3
"""Used only where the expected value is exactly zero.
A relative tolerance is meaningless against zero. Note that pytest.approx
takes the larger of the two when both are given, so this constant must
never appear next to a non-zero expected value -- it would silently
widen that assertion.
"""

def _acf_reference(x, lag):
    """Reference autocorrelation with the same convention as tsxtract.

    Denominator = n * variance, i.e.
    r_k = sum((x_t - mean)(x_{t+k} - mean)) / sum((x_t - mean)^2).
    Computed in float64 so it can serve as a NumPy oracle for the
    float32 JAX result.
    """
    x = np.asarray(x, dtype=np.float64)
    centred = x - x.mean()
    return np.sum(centred[:-lag] * centred[lag:]) / np.sum(centred**2)

def test_extract_features():
    """Test feature extraction."""
    # Standard case: Array of shape (samples, channels, length)
    implemented_features = len(tsx.DEFAULT_FEATURE_SPECS)
    dataset = generate_random_time_series_dataset(
        n_samples=10,
        n_channels=2,
        sampling_rate=5,
        time_series_length_in_seconds=2,
    )
    
    features = tsx.extract_features(dataset, sampling_rate=5) 
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
        features = tsx.extract_features(dataset, sampling_rate=5)


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
    argnames="array",
    argvalues=[
        (jnp.ones(5)),
        (jnp.zeros(5)),
        (-jnp.ones(5)),
        (jnp.array([5.0])),
        (jnp.array([-1.0, 0.0, 1.0, 0.0, -1.0])),
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])),
        (jnp.array([1.0, 1.0, 1.0, 1.0, 10.0])),
        (jnp.array([3.0, 3.0, 3.0, 3.0, 2.0])),
        (jnp.array([1e18, 1e18])),
        (jnp.array([-1e18, -1e18])),
    ],
)
def test_rms(array):
    """Test extraction of root mean square value against a NumPy float64 reference."""
    x = np.asarray(array, dtype=np.float64)
    expected = np.sqrt(np.mean(x**2))
    assert tsx.rms(signal=array) == pytest.approx(expected, rel=RELATIVE_TOLERANCE)

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
    argnames="array",
    argvalues=[
        (jnp.ones(5)),
        (jnp.zeros(5)),    
        (-jnp.ones(5)),
        (jnp.array([5.0])),
        (jnp.array([-1.0, 0.0, 1.0, 0.0, -1.0])),
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])),
        (jnp.array([1.0, 1.0, 1.0, 1.0, 10.0])),
        (jnp.array([3.0, 3.0, 3.0, 3.0, 2.0])),
        (jnp.array([1e18, 1e18])),
        (jnp.array([-1e18, -1e18])),
    ],
)
def test_mad(array):
    """Test extraction of mean absolute deviation value against a NumPy float64 reference."""
    x = np.asarray(array, dtype=np.float64)
    expected = np.mean(np.abs(x - np.mean(x)))
    assert tsx.mad(signal=array) == pytest.approx(expected, rel=RELATIVE_TOLERANCE)       

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
    assert tsx.percentile(signal=array, q=q) == pytest.approx(expected, rel=RELATIVE_TOLERANCE)

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
    argnames="array",
    argvalues=[
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])), # symmetric -> 0
        (jnp.array([0.0, 1.0, 2.0, 3.0])),  # symmetric -> 0
        (jnp.array([1.0, 2.0, 3.0, 10.0])), # right-skewed; scipy.stats.skew    
        (jnp.array([1.0, 8.0, 9.0, 10.0])), # left-skewed; scipy.stats.skew
    ],
)
def test_skewness(array):
    """Test skewness against a scipy.stats.skew (bias=True) reference."""
    x = np.asarray(array, dtype=np.float64)
    expected = scipy.stats.skew(x, bias=True)
    assert tsx.skewness(signal=array) == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


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
    argnames="array",
    argvalues=[
        (jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])),
        (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        (jnp.array([0.0, 1.0, 2.0, 3.0])),
        (jnp.array([0.0, 0.0, 0.0, 0.0, 10.0])),
        (jnp.array([1.0, 2.0, 3.0, 10.0])),
    ],
)
def test_kurtosis(array):
    """Test excess kurtosis against a scipy.stats.kurtosis (fisher=True, bias=True) reference."""
    x = np.asarray(array, dtype=np.float64)
    expected = scipy.stats.kurtosis(x, fisher=True, bias=True)
    assert tsx.kurtosis(signal=array) == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


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
    assert tsx.zero_crossing_rate(signal=array) == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


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
    argnames=("array", "lags"),
    argvalues=[
        # increasing ramp [1,2,3,4]: mean=2.5, denom=5.0
        # r1 = 1.25/5 = 0.25, r2 = -1.5/5 = -0.3, r3 = -2.25/5 = -0.45 (hand check)
        (jnp.array([1.0, 2.0, 3.0, 4.0]), (1, 2, 3)),
        # alternating short signal [1,-1,1,-1]: boundary effect makes r1=-0.75 (not -1), r2=+0.5
        (jnp.array([1.0, -1.0, 1.0, -1.0]), (1, 2)),
        # long alternating signal: r2 approaches +1 (period-2 structure)
        (jnp.array([1.0, -1.0] * 25), (2,)),
    ],
)
def test_autocorrelation(array, lags):
    """Test autocorrelation against the NumPy float64 reference `_acf_reference`."""
    expected = jnp.array([_acf_reference(array, lag) for lag in lags])
    assert jnp.allclose(
        tsx.autocorrelation(signal=array, lags=lags),
        expected, 
        rtol=RELATIVE_TOLERANCE,
        atol=0.0
        )


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


# spectral_centroid test
@pytest.mark.parametrize(
    argnames=("frequency", "sampling_rate"),
    argvalues=[(7.0, 100.0), (0.5, 100.0), (25.0, 200.0)],
)
def test_spectral_centroid(frequency, sampling_rate):
    """Pure sine at f Hz -> centroid equals f (analytical ground truth)."""
    t = jnp.arange(1000) / sampling_rate
    signal = jnp.sin(2 * jnp.pi * frequency * t)
    assert tsx.spectral_centroid(signal, sampling_rate) == pytest.approx(frequency, rel=RELATIVE_TOLERANCE)


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        jnp.array([-1.0, 0.0, 1.0, 0.0, -1.0]),
        jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
        jnp.array([1.0, 1.0, 1.0, 1.0, 10.0]),
        jnp.array([3.0, 3.0, 3.0, 3.0, 2.0]),
    ],
)
def test_spectral_centroid_numpy_reference(array):
    """Same convention (DC removed, magnitude, Hz) recomputed in NumPy float64."""
    sampling_rate = 100.0
    x = np.asarray(array, dtype=np.float64)
    x = x - x.mean()                                     #  same convention: DC removed
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sampling_rate) # same sampling rate as the feature
    expected = np.sum(freqs * spectrum) / np.sum(spectrum)
    assert tsx.spectral_centroid(signal=array, sampling_rate=sampling_rate) == \
        pytest.approx(expected, rel=RELATIVE_TOLERANCE)


def test_spectral_centroid_offset_invariant():
    """Adding a constant offset must not change the centroid (DC removed)."""
    t = jnp.arange(1000) / 100.0
    signal = jnp.sin(2 * jnp.pi * 7.0 * t)
    assert tsx.spectral_centroid(signal + 1e4, 100.0) == \
        pytest.approx(tsx.spectral_centroid(signal, 100.0), rel=RELATIVE_TOLERANCE)


def test_spectral_centroid_edge_cases():
    """Constant signal has zero spectral energy -> NaN, consistent with skewness."""
    assert jnp.isnan(tsx.spectral_centroid(jnp.ones(64), 100.0))
    assert jnp.isnan(tsx.spectral_centroid(jnp.zeros(64), 100.0))
    assert jnp.isnan(tsx.spectral_centroid(jnp.array([5.0]), 100.0))
    assert jnp.isnan(tsx.spectral_centroid(jnp.array([1e18, 1e18]), 100.0))


# spectral_bandwidth test
@pytest.mark.parametrize(
    "frequency_a, amplitude_a, frequency_b, amplitude_b, expected_bandwidth",
    [
        # Equal amplitudes: both lines carry weight 1/2, centroid at 15 Hz,
        # each line is 5 Hz away -> bandwidth exactly 5.0.
        (10.0, 1.0, 20.0, 1.0, 5.0),
        # Amplitudes 3:1: centroid (3*10 + 1*20)/4 = 12.5 Hz, variance
        # 3/4 * 2.5**2 + 1/4 * 7.5**2 = 18.75 -> sqrt = 4.330127018922194.
        # Unequal amplitudes on purpose: this case distinguishes magnitude
        # weighting (4.33) from power weighting (3.0); equal amplitudes
        # give 5.0 under both and could not catch that mistake.
        (10.0, 3.0, 20.0, 1.0, 4.330127018922194),
    ],
)


def test_spectral_bandwidth(
    frequency_a: float,
    amplitude_a: float,
    frequency_b: float,
    amplitude_b: float,
    expected_bandwidth: float,
) -> None:
    """
    Two-tone signals whose bandwidth can be derived by hand.
    N=1000 at fs=100 gives a grid spacing of 0.1 Hz, so 10 Hz and 20 Hz fall
    exactly on the grid and no spectral leakage widens the result.
    """

    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = amplitude_a * jnp.sin(
        2 * jnp.pi * frequency_a * time
    ) + amplitude_b * jnp.sin(2 * jnp.pi * frequency_b * time)
    assert tsx.spectral_bandwidth(signal, sampling_rate) == pytest.approx(
        expected_bandwidth, rel=RELATIVE_TOLERANCE
    )


def test_spectral_bandwidth_single_tone() -> None:
    """
    Frequency-domain analogue of "the standard deviation of a constant is 0".
    A single spectral line has no spread around its own centroid. Absolute
    tolerance here: the expected value is zero, where a relative tolerance
    has no meaning.
    """

    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = jnp.sin(2 * jnp.pi * 10.0 * time)
    assert tsx.spectral_bandwidth(signal, sampling_rate) == pytest.approx(
        0.0, abs=ABSOLUTE_TOLERANCE
    )


@pytest.mark.parametrize(
    "signal",
    [
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        jnp.array([0.0, -1.0, 1.0, -1.0]),
        jnp.array([-5.0, 3.0, 0.5, 100.0, -20.0, 7.0]),
        jnp.array([0.1, 0.5, 0.2, 0.9, 0.4]),
    ],
)
def test_spectral_bandwidth_numpy_reference(signal: jax.Array) -> None:
    # Arrays must stay out of the feature's degenerate region: after DC
    # removal the spectrum needs at least two non-zero bins, otherwise
    # the true bandwidth is 0 and only the noise floor would be compared.
    sampling_rate = 100.0
    # Same convention as tsxtract: remove DC before the FFT.
    centred = np.asarray(signal, dtype=np.float64)
    centred = centred - centred.mean()
    magnitude = np.abs(np.fft.rfft(centred))
    fft_frequencies = np.fft.rfftfreq(len(centred), 1.0 / sampling_rate)
    weights = magnitude / magnitude.sum()
    mu = np.sum(fft_frequencies * weights)
    expected_bandwidth = np.sqrt(np.sum((fft_frequencies - mu) ** 2 * weights))
    assert tsx.spectral_bandwidth(signal, sampling_rate) == pytest.approx(
        expected_bandwidth, rel=RELATIVE_TOLERANCE
    )


def test_spectral_bandwidth_offset_invariant() -> None:
    # rel=RELATIVE_TOLERANCE, not tighter: the offset costs float32 significant digits
    # (quantisation noise, measured ~0.6% here), while the failure this
    # test guards against (DC not removed) changes the result by orders
    # of magnitude. Same tolerance principle as the centroid test.
    # Two-tone signal on purpose: a single tone's true bandwidth is ~0,
    # so its value is all noise floor and cannot anchor a relative test.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(
        2 * jnp.pi * 20.0 * time
    )
    assert tsx.spectral_bandwidth(
        signal + 1e4, sampling_rate
    ) == pytest.approx(
        tsx.spectral_bandwidth(signal, sampling_rate), rel=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "signal",
    [
        jnp.ones(64),
        jnp.zeros(64),
        jnp.array([5.0]),
        jnp.array([1e18, 1e18]),
    ],
)
def test_spectral_bandwidth_edge_cases(signal: jax.Array) -> None:
    # Constant signals have an all-zero spectrum after DC removal:
    # the distribution is 0/0 -> NaN, and NaN propagates to the feature.
    assert jnp.isnan(tsx.spectral_bandwidth(signal, 100.0))


# ----------spectral_rolloff test------------------
@pytest.mark.parametrize(
    "roll_percent, expected_rolloff",
    [
        # 3:1 two-tone at 10/20 Hz: weights 0.75/0.25, so the cumulative
        # curve is 0.75 at 10 Hz and 1.0 at 20 Hz. The threshold decides
        # which line answers -- this also exercises the roll_percent
        # parameter itself:
        (0.5, 10.0),   # 0.5  <= 0.75          -> stops at the 10 Hz line
        (0.85, 20.0),  # 0.75 < 0.85 <= 1.0    -> must include the 20 Hz line
        (0.95, 20.0),  # same crossing, TSFEL's default threshold
    ],
)
def test_spectral_rolloff(roll_percent: float, expected_rolloff: float) -> None:
    # N=1000 at fs=100 -> grid spacing 0.1 Hz, both tones on the grid.
    # Tight rel: the result snaps to a grid frequency, so it is either
    # exactly right or off by whole bins -- no gradual noise to absorb.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = 3.0 * jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(
        2 * jnp.pi * 20.0 * time
    )
    assert tsx.spectral_rolloff(
        signal, sampling_rate, roll_percent
    ) == pytest.approx(expected_rolloff, rel=RELATIVE_TOLERANCE)


def test_spectral_rolloff_single_tone() -> None:
    # A single tone is NOT degenerate for rolloff (unlike for bandwidth):
    # all weight sits on one line, so every threshold stops there.
    # Also covers the default roll_percent (parameter omitted).
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = jnp.sin(2 * jnp.pi * 10.0 * time)
    assert tsx.spectral_rolloff(signal, sampling_rate) == pytest.approx(
        10.0, rel=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "signal",
    [
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        jnp.array([0.0, -1.0, 1.0, -1.0]),
        jnp.array([-5.0, 3.0, 0.5, 100.0, -20.0, 7.0]),
        jnp.array([0.1, 0.5, 0.2, 0.9, 0.4]),
    ],
)
def test_spectral_rolloff_numpy_reference(signal: jax.Array) -> None:
    sampling_rate = 100.0
    # Same convention as tsxtract: remove DC before the FFT.
    centred = np.asarray(signal, dtype=np.float64)
    centred = centred - centred.mean()
    magnitude = np.abs(np.fft.rfft(centred))
    fft_frequencies = np.fft.rfftfreq(len(centred), 1.0 / sampling_rate)
    weights = magnitude / magnitude.sum()
    cumulative = np.cumsum(weights)
    expected_rolloff = fft_frequencies[np.argmax(cumulative >= 0.85)]
    assert tsx.spectral_rolloff(signal, sampling_rate, 0.85) == pytest.approx(
        expected_rolloff, rel=RELATIVE_TOLERANCE
    )


def test_spectral_rolloff_offset_invariant() -> None:
    # Without DC removal the crossing would be dragged toward 0 Hz.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = 3.0 * jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(
        2 * jnp.pi * 20.0 * time
    )
    assert tsx.spectral_rolloff(
        signal + 1e4, sampling_rate, 0.85
    ) == pytest.approx(
        tsx.spectral_rolloff(signal, sampling_rate, 0.85), rel=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "signal",
    [
        jnp.ones(64),
        jnp.zeros(64),
        jnp.array([5.0]),
        jnp.array([1e18, 1e18]),
    ],
)
def test_spectral_rolloff_edge_cases(signal: jax.Array) -> None:
    # Constant signals: NaN weights make every comparison False, so
    # without the explicit guard argmax would silently return 0.0 Hz.
    # This test is what keeps that guard honest.
    assert jnp.isnan(tsx.spectral_rolloff(signal, 100.0))


# ---------dominant_frequency test------------------
@pytest.mark.parametrize(
    "amplitude_a, amplitude_b, expected_frequency",
    [
        # Two tones at 10 and 20 Hz; the stronger one wins.
        (3.0, 1.0, 10.0),
        (1.0, 3.0, 20.0),
        # Exactly equal peaks: argmax returns the first (lowest) one.
        # Documented behaviour, not an accident -- this pins it down.
        (1.0, 1.0, 10.0),
    ],
)

def test_dominant_frequency(
    amplitude_a: float, amplitude_b: float, expected_frequency: float
) -> None:
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = amplitude_a * jnp.sin(2 * jnp.pi * 10.0 * time) + amplitude_b * jnp.sin(
        2 * jnp.pi * 20.0 * time
    )
    assert tsx.dominant_frequency(signal, sampling_rate) == pytest.approx(
        expected_frequency, rel=RELATIVE_TOLERANCE
    )

def test_dominant_frequency_grid_quantisation() -> None:
    # A 7.05 Hz tone lies between two grid points (spacing 0.1 Hz) and is
    # reported as 7.0: the feature returns bin centres, it does not
    # interpolate. Pins the documented resolution limit.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = jnp.sin(2 * jnp.pi * 7.05 * time)
    assert tsx.dominant_frequency(signal, sampling_rate) == pytest.approx(
        7.0, rel=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "signal",
    [
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        jnp.array([0.0, -1.0, 1.0, -1.0]),
        jnp.array([-5.0, 3.0, 0.5, 100.0, -20.0, 7.0]),
        jnp.array([0.1, 0.5, 0.2, 0.9, 0.4]),
    ],
)
def test_dominant_frequency_numpy_reference(signal: jax.Array) -> None:
    sampling_rate = 100.0
    # Same convention as tsxtract: remove DC before the FFT.
    centred = np.asarray(signal, dtype=np.float64)
    centred = centred - centred.mean()
    magnitude = np.abs(np.fft.rfft(centred))
    fft_frequencies = np.fft.rfftfreq(len(centred), 1.0 / sampling_rate)
    expected_frequency = fft_frequencies[np.argmax(magnitude)]
    assert tsx.dominant_frequency(signal, sampling_rate) == pytest.approx(
        expected_frequency, rel=RELATIVE_TOLERANCE
    )


def test_dominant_frequency_offset_invariant() -> None:
    # Without DC removal this would return 0.0 Hz for the offset signal
    # (the DC bin dwarfs everything); with it, the peak bin is unchanged.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = 3.0 * jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(
        2 * jnp.pi * 20.0 * time
    )
    assert tsx.dominant_frequency(signal + 1e4, sampling_rate) == pytest.approx(
        tsx.dominant_frequency(signal, sampling_rate), rel=RELATIVE_TOLERANCE
    )

@pytest.mark.parametrize(
    "signal",
    [
        jnp.ones(64),
        jnp.zeros(64),
        jnp.array([5.0]),
        jnp.array([1e18, 1e18]),
    ],
)
def test_dominant_frequency_edge_cases(signal: jax.Array) -> None:
    # Constant signal: NaN weights, argmax would silently return 0.0 Hz.
    assert jnp.isnan(tsx.dominant_frequency(signal, 100.0))


# ---------spectral_entropy test------------------
@pytest.mark.parametrize(
    # Analytical truths (N = 1000, fs = 100 -> 501 bins). Each tone lands
    # exactly on an FFT bin, so the magnitude distribution is known:
    #   single tone       -> one bin, entropy 0 (degenerate, absolute tolerance)
    #   equal two-tone    -> p = (1/2, 1/2), entropy 1 bit
    #   3:1 two-tone      -> p = (3/4, 1/4), entropy 0.8112781244591328 bit
    #   five equal tones  -> p = 1/5 each, entropy log2(5) bit
    # Normalisation divides by log2(501). The 3:1 case is the one that
    # separates magnitude weighting from power weighting (9:1 -> 0.469 bit).
    "amplitudes, frequencies, expected_bits",
    [
        ([1.0], [10.0], 0.0),
        ([1.0, 1.0], [10.0, 20.0], 1.0),
        ([3.0, 1.0], [10.0, 20.0], 0.8112781244591328),
        ([1.0, 1.0, 1.0, 1.0, 1.0], [5.0, 10.0, 15.0, 20.0, 25.0], 2.321928094887362),
    ],
)
def test_spectral_entropy(
    amplitudes: list[float], frequencies: list[float], expected_bits: float
) -> None:
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = sum(
        amplitude * jnp.sin(2 * jnp.pi * frequency * time)
        for amplitude, frequency in zip(amplitudes, frequencies)
    )
    number_of_bins = 501
    expected_entropy = expected_bits / np.log2(number_of_bins)
    if expected_bits == 0.0:
        assert tsx.spectral_entropy(signal, sampling_rate) == pytest.approx(
            0.0, abs=ABSOLUTE_TOLERANCE
        )
    else:
        assert tsx.spectral_entropy(signal, sampling_rate) == pytest.approx(
            expected_entropy, rel=RELATIVE_TOLERANCE
        )


def test_spectral_entropy_bounds() -> None:
    # White noise spreads energy over all bins: entropy close to, but
    # never above, 1. Pins the normalisation (without it, the value would
    # be ~9 bits, not ~0.98).
    rng = np.random.default_rng(0)
    signal = jnp.asarray(rng.standard_normal(1000))
    entropy = tsx.spectral_entropy(signal, 100.0)
    assert 0.9 < entropy <= 1.0


@pytest.mark.parametrize(
    "signal",
    [
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        jnp.array([0.0, -1.0, 1.0, -1.0]),
        jnp.array([-5.0, 3.0, 0.5, 100.0, -20.0, 7.0]),
        jnp.array([0.1, 0.5, 0.2, 0.9, 0.4]),
    ],
)
def test_spectral_entropy_numpy_reference(signal: jax.Array) -> None:
    sampling_rate = 100.0
    # Same convention as tsxtract: remove DC, magnitude weighting,
    # normalise by log2 of the total number of bins.
    centred = np.asarray(signal, dtype=np.float64)
    centred = centred - centred.mean()
    magnitude = np.abs(np.fft.rfft(centred))
    weights = magnitude / magnitude.sum()
    nonzero = weights[weights > 0]
    expected_entropy = -np.sum(nonzero * np.log2(nonzero)) / np.log2(len(weights))
    assert tsx.spectral_entropy(signal, sampling_rate) == pytest.approx(
        expected_entropy, rel=RELATIVE_TOLERANCE
    )


def test_spectral_entropy_offset_invariant() -> None:
    # Without DC removal the offset would put almost all weight into the
    # 0 Hz bin and drive the entropy towards 0.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = 3.0 * jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(
        2 * jnp.pi * 20.0 * time
    )
    assert tsx.spectral_entropy(signal + 1e4, sampling_rate) == pytest.approx(
        tsx.spectral_entropy(signal, sampling_rate), rel=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "signal",
    [
        jnp.ones(64),
        jnp.zeros(64),
        jnp.array([5.0]),
        jnp.array([1e18, 1e18]),
    ],
)
def test_spectral_entropy_edge_cases(signal: jax.Array) -> None:
    # Constant signal: NaN weights propagate through the arithmetic, no
    # explicit guard needed (unlike rolloff / dominant_frequency).
    assert jnp.isnan(tsx.spectral_entropy(signal, 100.0))


# ---------band_energy test------------------
@pytest.mark.parametrize(
    # Analytical truths (N = 1000, fs = 100, grid spacing 0.1 Hz). Tones at
    # 10 Hz and 20 Hz sit exactly on the grid, so each one occupies a
    # single bin and the energy split is amplitude**2:
    #   equal amplitudes -> 1:1 energy -> 0.5 in a band around either tone
    #   3:1 amplitudes   -> 9:1 energy -> 0.9 / 0.1 (separates power from
    #                       magnitude weighting, which would give 0.75)
    #   band with no tone -> 0 (degenerate, absolute tolerance)
    #   band covering everything -> 1
    "amplitudes, low_frequency, high_frequency, expected_fraction",
    [
        ([1.0, 1.0], 5.0, 15.0, 0.5),
        ([3.0, 1.0], 5.0, 15.0, 0.9),
        ([3.0, 1.0], 15.0, 25.0, 0.1),
        ([3.0, 1.0], 30.0, 40.0, 0.0),
        ([3.0, 1.0], 0.0, 50.01, 1.0),
    ],
)
def test_band_energy(
    amplitudes: list[float],
    low_frequency: float,
    high_frequency: float,
    expected_fraction: float,
) -> None:
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = amplitudes[0] * jnp.sin(2 * jnp.pi * 10.0 * time) + amplitudes[
        1
    ] * jnp.sin(2 * jnp.pi * 20.0 * time)
    if expected_fraction == 0.0:
        assert tsx.band_energy(
            signal, sampling_rate, low_frequency, high_frequency
        ) == pytest.approx(0.0, abs=ABSOLUTE_TOLERANCE)
    else:
        assert tsx.band_energy(
            signal, sampling_rate, low_frequency, high_frequency
        ) == pytest.approx(expected_fraction, rel=RELATIVE_TOLERANCE)


def test_band_energy_edges_are_half_open() -> None:
    # A tone exactly on the lower edge is inside the band, one exactly on
    # the upper edge is outside. Pins the [low, high) convention.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(2 * jnp.pi * 20.0 * time)
    assert tsx.band_energy(signal, sampling_rate, 10.0, 20.0) == pytest.approx(
        0.5, rel=RELATIVE_TOLERANCE
    )


def test_band_energy_bands_sum_to_one() -> None:
    # Bands that tile the axis must account for all the energy. Pins the
    # normalisation and the half-open edges together (any overlap or gap
    # between neighbouring bands would break the sum).
    sampling_rate = 100.0
    rng = np.random.default_rng(0)
    signal = jnp.asarray(rng.standard_normal(1000))
    edges = [0.0, 5.0, 12.5, 30.0, 50.01]
    total = sum(
        tsx.band_energy(signal, sampling_rate, low, high)
        for low, high in zip(edges[:-1], edges[1:])
    )
    assert total == pytest.approx(1.0, rel=RELATIVE_TOLERANCE)


@pytest.mark.parametrize(
    "signal",
    [
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        jnp.array([0.0, -1.0, 1.0, -1.0]),
        jnp.array([-5.0, 3.0, 0.5, 100.0, -20.0, 7.0]),
        jnp.array([0.1, 0.5, 0.2, 0.9, 0.4]),
    ],
)
def test_band_energy_numpy_reference(signal: jax.Array) -> None:
    sampling_rate = 100.0
    low_frequency, high_frequency = 10.0, 30.0
    # Same convention as tsxtract: remove DC, squared magnitude,
    # half-open band [low, high).
    centred = np.asarray(signal, dtype=np.float64)
    centred = centred - centred.mean()
    power = np.abs(np.fft.rfft(centred)) ** 2
    fft_frequencies = np.fft.rfftfreq(len(centred), 1.0 / sampling_rate)
    in_band = (fft_frequencies >= low_frequency) & (fft_frequencies < high_frequency)
    expected_fraction = power[in_band].sum() / power.sum()
    assert tsx.band_energy(
        signal, sampling_rate, low_frequency, high_frequency
    ) == pytest.approx(expected_fraction, rel=RELATIVE_TOLERANCE)


def test_band_energy_offset_invariant() -> None:
    # Without DC removal the offset would dominate the total energy and
    # drive every band fraction towards 0.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = 3.0 * jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(
        2 * jnp.pi * 20.0 * time
    )
    assert tsx.band_energy(signal + 1e4, sampling_rate, 5.0, 15.0) == pytest.approx(
        tsx.band_energy(signal, sampling_rate, 5.0, 15.0), rel=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "signal",
    [
        jnp.ones(64),
        jnp.zeros(64),
        jnp.array([5.0]),
        jnp.array([1e18, 1e18]),
    ],
)
def test_band_energy_edge_cases(signal: jax.Array) -> None:
    # Constant signal: no energy at all, 0 / 0 propagates to NaN.
    assert jnp.isnan(tsx.band_energy(signal, 100.0, 5.0, 15.0))


# ---------power_bandwidth test------------------
@pytest.mark.parametrize(
    # Analytical truths (N = 1000, fs = 100, grid spacing 0.1 Hz). Each
    # tone occupies a single bin, so the cumulative power distribution is
    # known exactly and the quantile edges can be read off by hand:
    #   two equal tones at 10/30 Hz, cumulative 0.5 / 1.0
    #       -> any fraction picks 10 Hz and 30 Hz -> width 20 Hz
    #   amplitudes 1:3:1 at 10/20/30 Hz, energies 1:9:1, cumulative
    #   1/11, 10/11, 1
    #       -> 0.90: lower edge at 0.05 -> 10 Hz, upper at 0.95 -> 30 Hz
    #       -> 0.50: lower edge at 0.25 -> 20 Hz, upper at 0.75 -> 20 Hz
    #                (the middle tone alone carries 9/11 of the power)
    #   single tone -> both edges on the same bin -> 0 Hz (degenerate,
    #                  absolute tolerance)
    "amplitudes, frequencies, power_fraction, expected_width",
    [
        ([1.0, 1.0], [10.0, 30.0], 0.90, 20.0),
        ([1.0, 1.0], [10.0, 30.0], 0.50, 20.0),
        ([1.0, 3.0, 1.0], [10.0, 20.0, 30.0], 0.90, 20.0),
        ([1.0, 3.0, 1.0], [10.0, 20.0, 30.0], 0.50, 0.0),
        ([1.0], [10.0], 0.90, 0.0),
    ],
)
def test_power_bandwidth(
    amplitudes: list[float],
    frequencies: list[float],
    power_fraction: float,
    expected_width: float,
) -> None:
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = sum(
        amplitude * jnp.sin(2 * jnp.pi * frequency * time)
        for amplitude, frequency in zip(amplitudes, frequencies)
    )
    if expected_width == 0.0:
        assert tsx.power_bandwidth(
            signal, sampling_rate, power_fraction
        ) == pytest.approx(0.0, abs=ABSOLUTE_TOLERANCE)
    else:
        assert tsx.power_bandwidth(
            signal, sampling_rate, power_fraction
        ) == pytest.approx(expected_width, rel=RELATIVE_TOLERANCE)


def test_power_bandwidth_ignores_noise_floor() -> None:
    # The point of a quantile width: a low noise floor spread over all
    # bins carries little power and does not move the edges, whereas the
    # magnitude-weighted spectral_bandwidth grows noticeably.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    rng = np.random.default_rng(0)
    clean = jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(2 * jnp.pi * 30.0 * time)
    noisy = clean + 0.3 * jnp.asarray(rng.standard_normal(1000))
    assert tsx.power_bandwidth(noisy, sampling_rate) == pytest.approx(
        tsx.power_bandwidth(clean, sampling_rate), rel=RELATIVE_TOLERANCE
    )
    # Measured: 10.00 Hz clean -> 13.81 Hz noisy, a factor of 1.38; the
    # threshold below sits well inside that margin.
    assert tsx.spectral_bandwidth(noisy, sampling_rate) > 1.2 * tsx.spectral_bandwidth(
        clean, sampling_rate
    )


def test_power_bandwidth_widens_with_fraction() -> None:
    # More power demanded means the edges move outwards: the width is
    # monotonically non-decreasing in power_fraction. Pins the direction
    # of the threshold (swapping the two edges would invert this).
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = sum(
        jnp.sin(2 * jnp.pi * frequency * time) for frequency in [5.0, 10.0, 20.0, 40.0]
    )
    widths = [
        tsx.power_bandwidth(signal, sampling_rate, fraction)
        for fraction in [0.25, 0.50, 0.75, 0.95]
    ]
    assert widths == sorted(widths)
    assert widths[-1] > widths[0]


@pytest.mark.parametrize(
    "signal",
    [
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        jnp.array([0.0, -1.0, 1.0, -1.0]),
        jnp.array([-5.0, 3.0, 0.5, 100.0, -20.0, 7.0]),
        jnp.array([0.1, 0.5, 0.2, 0.9, 0.4]),
    ],
)
def test_power_bandwidth_numpy_reference(signal: jax.Array) -> None:
    sampling_rate = 100.0
    power_fraction = 0.90
    # Same convention as tsxtract: remove DC, squared magnitude, equal
    # tails cut off at both ends of the cumulative distribution.
    centred = np.asarray(signal, dtype=np.float64)
    centred = centred - centred.mean()
    power = np.abs(np.fft.rfft(centred)) ** 2
    fft_frequencies = np.fft.rfftfreq(len(centred), 1.0 / sampling_rate)
    cumulative = np.cumsum(power) / power.sum()
    lower = fft_frequencies[np.argmax(cumulative >= (1 - power_fraction) / 2)]
    upper = fft_frequencies[np.argmax(cumulative >= (1 + power_fraction) / 2)]
    expected_width = upper - lower
    assert tsx.power_bandwidth(
        signal, sampling_rate, power_fraction
    ) == pytest.approx(expected_width, rel=RELATIVE_TOLERANCE)


def test_power_bandwidth_offset_invariant() -> None:
    # Without DC removal the 0 Hz bin would hold almost all the power and
    # both edges would collapse onto it, giving a width of 0 Hz.
    sampling_rate = 100.0
    time = jnp.arange(1000) / sampling_rate
    signal = 3.0 * jnp.sin(2 * jnp.pi * 10.0 * time) + jnp.sin(
        2 * jnp.pi * 30.0 * time
    )
    assert tsx.power_bandwidth(signal + 1e4, sampling_rate) == pytest.approx(
        tsx.power_bandwidth(signal, sampling_rate), rel=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "signal",
    [
        jnp.ones(64),
        jnp.zeros(64),
        jnp.array([5.0]),
        jnp.array([1e18, 1e18]),
    ],
)
def test_power_bandwidth_edge_cases(signal: jax.Array) -> None:
    # Constant signal: NaN weights make every comparison False, so both
    # argmax calls return index 0 and the width would silently be 0.0 Hz
    # without the explicit guard.
    assert jnp.isnan(tsx.power_bandwidth(signal, 100.0))


EXPECTED_FEATURE_NAMES = {
    "maximum",
    "mean",
    "minimum",
    "std",
    "variance",
    "median",
    "rms",
    "mad",
    "percentile__q_25",
    "percentile__q_75",
    "skewness",
    "kurtosis",
    "zero_crossing_rate",
    "autocorrelation__lags_1_2_3",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "dominant_frequency",
    "spectral_entropy",
    "band_energy__low_frequency_0.6__high_frequency_2.5",
    "power_bandwidth",
}
"""Every output name of the default configuration, written out in full.

Counting the entries would not notice a renamed, duplicated or swapped
feature; comparing the set does. Update this set deliberately whenever
DEFAULT_FEATURE_SPECS changes.
"""


def test_default_feature_names() -> None:
    dataset = generate_random_time_series_dataset(
        n_samples=4,
        n_channels=2,
        sampling_rate=5,
        time_series_length_in_seconds=2,
    )
    features = tsx.extract_features(dataset, sampling_rate=5)
    assert set(features) == EXPECTED_FEATURE_NAMES
    # A duplicated specification would silently overwrite an entry.
    assert len(features) == len(tsx.DEFAULT_FEATURE_SPECS)


def test_feature_specs_accept_custom_configuration() -> None:
    # The point of the refactoring: a different parameter set without
    # touching the extractor. Also checks that sampling_rate is injected
    # only where the signature asks for it (mean would fail otherwise).
    dataset = generate_random_time_series_dataset(
        n_samples=4,
        n_channels=2,
        sampling_rate=5,
        time_series_length_in_seconds=2,
    )
    feature_specs = (
        tsx.FeatureSpec(tsx.mean),
        tsx.FeatureSpec(tsx.percentile, q=10.0),
        tsx.FeatureSpec(
            tsx.band_energy, low_frequency=1.0, high_frequency=2.0, name="alpha_band"
        ),
    )
    features = tsx.extract_features(dataset, 5.0, feature_specs)
    assert set(features) == {"mean", "percentile__q_10", "alpha_band"}


def test_to_columns_flattens_multi_valued_features() -> None:
    dataset = generate_random_time_series_dataset(
        n_samples=4,
        n_channels=2,
        sampling_rate=5,
        time_series_length_in_seconds=2,
    )
    features = tsx.extract_features(dataset, sampling_rate=5)
    columns = tsx.to_columns(features)
    # autocorrelation holds three lags in one entry -> three columns.
    assert len(columns) == len(features) + 2
    assert all(value.shape == (4, 2) for value in columns.values())
    assert jnp.allclose(
        columns["autocorrelation__lags_1_2_3__0"],
        features["autocorrelation__lags_1_2_3"][..., 0],
    )
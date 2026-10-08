"""Module containing PyTest fixtures used for different tests.

See: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68 for more information.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Run the test suite in float64 so that expected values can be checked with a
# single tight tolerance (see meeting 2026-08-28). Must be set before the first
# array is created, hence here and not inside the individual tests.
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def ones_array() -> jax.Array:
    """Array containing only ones."""
    return jnp.ones(5)


@pytest.fixture
def zeros_array() -> jax.Array:
    """Array containing only zeros."""
    return jnp.zeros(5)


@pytest.fixture
def negatives_array() -> jax.Array:
    """Array containing only -1."""
    return jnp.full(5, -1.0)


@pytest.fixture
def single_point() -> jax.Array:
    """Array with a single value."""
    return jnp.array([3.14])


@pytest.fixture
def empty_array() -> jax.Array:
    """Empty array."""
    return jnp.array([])


@pytest.fixture
def nan_array() -> jax.Array:
    """Array containing only NaNs."""
    return jnp.full(5, jnp.nan)


@pytest.fixture
def array_with_nan() -> jax.Array:
    """Array with finite values and some NaNs."""
    return jnp.array([1.0, 2.0, jnp.nan, 4.0, 5.0])


@pytest.fixture
def inf_array() -> jax.Array:
    """Array containing only +Inf."""
    return jnp.full(5, jnp.inf)


@pytest.fixture
def array_with_inf() -> jax.Array:
    """Array with finite values and some +Inf."""
    return jnp.array([1.0, 2.0, jnp.inf, 4.0, 5.0])


@pytest.fixture
def array_50_50() -> jax.Array:
    """Half zeros, half ones."""
    return jnp.array([0, 0, 1, 1])


@pytest.fixture
def array_20_80() -> jax.Array:
    """20% zeros, 80% ones."""
    return jnp.array([0, 1, 1, 1, 1])


@pytest.fixture
def nan_inf_finite_array() -> jax.Array:
    """Array with NaN, Inf, and finite values."""
    return jnp.array([jnp.nan, jnp.inf, 1.0, 2.0])


@pytest.fixture
def large_numbers_array() -> jax.Array:
    """Array with very large finite values to check overflow handling."""
    return jnp.array([1e18, 1e18, 1e18, 1e18])

@pytest.fixture                              
def array_positive_range() -> jax.Array:     
    """Integers from 0 to 100 (101 values)."""   
    return jnp.arange(0, 101)

@pytest.fixture
def array_negative_range() -> jax.Array:
    """Integers from -100 to 0 (101 values)."""
    return jnp.arange(-100, 1)


@pytest.fixture
def array_positive_and_negative_range() -> jax.Array:
    """Integers from -50 to 50 (101 values)."""
    return jnp.arange(-50, 51)


@pytest.fixture
def normal_array() -> jax.Array:
    """Array with 100 standard-normal values."""
    return jax.random.normal(jax.random.key(0), shape=(100,))


SAMPLING_RATE = 100.0


def _tone(frequency: float, amplitude: float = 1.0, offset: float = 0.0) -> jax.Array:
    """One second of a sine wave at SAMPLING_RATE, generated in float64."""
    t = np.arange(0, 1.0, 1.0 / SAMPLING_RATE)
    return jnp.asarray(amplitude * np.sin(2 * np.pi * frequency * t) + offset)


@pytest.fixture
def sampling_rate() -> float:
    """Sampling rate shared by the spectral fixtures."""
    return SAMPLING_RATE


@pytest.fixture
def single_tone() -> jax.Array:
    """Pure 7 Hz sine wave."""
    return _tone(7.0)


@pytest.fixture
def single_tone_with_offset() -> jax.Array:
    """7 Hz sine wave with a DC offset of 1e4."""
    return _tone(7.0, offset=1e4)


@pytest.fixture
def two_tones() -> jax.Array:
    """Sum of a 5 Hz and a 15 Hz sine wave with equal amplitude."""
    return _tone(5.0) + _tone(15.0)


@pytest.fixture
def loud_and_quiet_tone() -> jax.Array:
    """5 Hz sine wave with amplitude 3 plus a 15 Hz sine wave with amplitude 1."""
    return _tone(5.0, amplitude=3.0) + _tone(15.0, amplitude=1.0)


@pytest.fixture
def aliased_tone() -> jax.Array:
    """60 Hz sine wave, above the Nyquist frequency of 50 Hz; aliases to 40 Hz."""
    return _tone(60.0)
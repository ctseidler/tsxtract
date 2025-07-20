"""Module containing PyTest fixtures used for different tests.

See: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68 for more information.
"""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def normal_array() -> jax.Array:
    """Create a random jax.Array with length 100."""
    return jax.random.normal(jax.random.key(0), shape=(100,))


@pytest.fixture
def ones_array() -> jax.Array:
    """Create a jax.Array with length 100 containing only ones."""
    return jax.numpy.ones(shape=(100,))


@pytest.fixture
def zeros_array() -> jax.Array:
    """Create a jax.Array with length 100 containing only zeros."""
    return jax.numpy.zeros(shape=(100,))


@pytest.fixture
def negatives_array() -> jax.Array:
    """Create a jax.Array with length 100 containing only negative values."""
    signal: jax.Array = jax.numpy.ones(shape=(100,))
    signal: jax.Array = signal.at[:].set(-1)
    return signal


@pytest.fixture
def single_point() -> jax.Array:
    """Create a jax.Array with length 1."""
    return jax.random.normal(key=jax.random.key(0), shape=(1,))


@pytest.fixture
def empty_array() -> jax.Array:
    """Create a jax.Array containing no values."""
    return jax.random.normal(jax.random.key(0), shape=(0,))


@pytest.fixture
def nan_array() -> jax.Array:
    """Create a jax.Array containing only nan values."""
    signal: jax.Array = jax.random.normal(jax.random.key(0), shape=(100,))
    signal: jax.Array = signal.at[:].set(jnp.nan)
    return signal


@pytest.fixture
def array_with_nan() -> jax.Array:
    """Create a jax.Array with 80 normal values and 20 nan values."""
    signal: jax.Array = jax.random.normal(jax.random.key(0), shape=(100,))
    signal: jax.Array = signal.at[:20].set(jnp.nan)
    return signal


@pytest.fixture
def inf_array() -> jax.Array:
    """Create a jax.Array containing only inf values."""
    signal: jax.Array = jax.random.normal(jax.random.key(0), shape=(100,))
    signal: jax.Array = signal.at[:].set(jnp.inf)
    return signal


@pytest.fixture
def array_with_inf() -> jax.Array:
    """Create a jax.Array with 80 normal values and 20 inf values."""
    signal: jax.Array = jax.random.normal(jax.random.key(0), shape=(100,))
    signal: jax.Array = signal.at[:20].set(jnp.inf)
    return signal


@pytest.fixture
def array_50_50() -> jax.Array:
    """Create a jax.Array with 50 zeros and 50 ones."""
    signal: jax.Array = jax.numpy.ones(shape=(100,))
    signal: jax.Array = signal.at[:50].set(0)
    return signal


@pytest.fixture
def array_20_80() -> jax.Array:
    """Create a jax.Array with 20 zeros and 80 ones."""
    signal: jax.Array = jax.numpy.ones(shape=(100,))
    signal: jax.Array = signal.at[:20].set(0)
    return signal


@pytest.fixture
def array_positive_range() -> jax.Array:
    """Create a jax.Array with 100 values from 0 to 100."""
    signal: jax.Array = jax.numpy.arange(start=0, stop=101)
    return signal


@pytest.fixture
def array_negative_range() -> jax.Array:
    """Create a jax.Array with 100 values from 0 to -100."""
    signal: jax.Array = jax.numpy.arange(start=0, stop=-101, step=-1)
    return signal


@pytest.fixture
def array_positive_and_negative_range() -> jax.Array:
    """Create a jax.Array with 100 values from -50 to 50."""
    signal: jax.Array = jax.numpy.arange(start=-50, stop=51)
    return signal

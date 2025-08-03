"""Module containing PyTest fixtures used for different tests.

See: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68 for more information.
"""

import jax
import jax.numpy as jnp
import pytest

# @pytest.fixture
# def normal_array() -> jax.Array:
#     """Random array with 5 normal-distributed values."""
#     return jax.random.normal(jax.random.key(0), shape=(5,))


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


# @pytest.fixture
# def array_positive_range() -> jax.Array:
#     """Positive integer range."""
#     return jnp.arange(5)  # 0..4


# @pytest.fixture
# def array_negative_range() -> jax.Array:
#     """Negative integer range."""
#     return jnp.arange(0, -5, -1)  # 0..-4


# @pytest.fixture
# def array_positive_and_negative_range() -> jax.Array:
#     """Mixed negative and positive range."""
#     return jnp.arange(-2, 3)  # -2..2


@pytest.fixture
def nan_inf_finite_array() -> jax.Array:
    """Array with NaN, Inf, and finite values."""
    return jnp.array([jnp.nan, jnp.inf, 1.0, 2.0])


@pytest.fixture
def large_numbers_array() -> jax.Array:
    """Array with very large finite values to check overflow handling."""
    return jnp.array([1e18, 1e18, 1e18, 1e18])

"""Tests for the spectral_centroid feature."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


def test_single_tone(single_tone, sampling_rate) -> None:
    """tsx.spectral_centroid should return the tone frequency for a pure sine wave."""
    assert tsx.spectral_centroid(single_tone, sampling_rate) == pytest.approx(7.0, rel=1e-9)


def test_single_tone_with_offset(single_tone_with_offset, sampling_rate) -> None:
    """tsx.spectral_centroid should ignore the DC offset."""
    assert tsx.spectral_centroid(single_tone_with_offset, sampling_rate) == pytest.approx(
        7.0, rel=1e-9
    )


def test_loud_and_quiet_tone(loud_and_quiet_tone, sampling_rate) -> None:
    """tsx.spectral_centroid should be pulled towards the louder tone (magnitude-weighted)."""
    assert tsx.spectral_centroid(loud_and_quiet_tone, sampling_rate) == pytest.approx(
        7.5, rel=1e-9
    )


def test_ones(ones_array, sampling_rate) -> None:
    """tsx.spectral_centroid should return nan for a constant signal."""
    assert jnp.isnan(tsx.spectral_centroid(ones_array, sampling_rate))


def test_empty(empty_array, sampling_rate) -> None:
    """tsx.spectral_centroid should return nan for an empty array."""
    assert jnp.isnan(tsx.spectral_centroid(empty_array, sampling_rate))


def test_array_with_nan_values(array_with_nan, sampling_rate) -> None:
    """tsx.spectral_centroid should return nan for an array with a nan value."""
    assert jnp.isnan(tsx.spectral_centroid(array_with_nan, sampling_rate))


def test_array_with_inf_values(array_with_inf, sampling_rate) -> None:
    """tsx.spectral_centroid should return nan for an array with an inf value."""
    assert jnp.isnan(tsx.spectral_centroid(array_with_inf, sampling_rate))
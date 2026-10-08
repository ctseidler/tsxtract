"""Tests for the spectral_bandwidth feature."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


def test_single_tone(single_tone, sampling_rate) -> None:
    """tsx.spectral_bandwidth should return 0 for a pure sine wave."""
    assert tsx.spectral_bandwidth(single_tone, sampling_rate) == pytest.approx(0.0, abs=1e-3)


def test_single_tone_with_offset(single_tone_with_offset, sampling_rate) -> None:
    """tsx.spectral_bandwidth should ignore the DC offset."""
    assert tsx.spectral_bandwidth(single_tone_with_offset, sampling_rate) == pytest.approx(
        0.0, abs=1e-3
    )


def test_two_tones(two_tones, sampling_rate) -> None:
    """tsx.spectral_bandwidth should equal half the distance between two equal tones."""
    assert tsx.spectral_bandwidth(two_tones, sampling_rate) == pytest.approx(5.0, rel=1e-9)


def test_loud_and_quiet_tone(loud_and_quiet_tone, sampling_rate) -> None:
    """tsx.spectral_bandwidth should weight by magnitude (power weighting would give 3.0)."""
    assert tsx.spectral_bandwidth(loud_and_quiet_tone, sampling_rate) == pytest.approx(
        4.330127018922194, rel=1e-9
    )


def test_ones(ones_array, sampling_rate) -> None:
    """tsx.spectral_bandwidth should return nan for a constant signal."""
    assert jnp.isnan(tsx.spectral_bandwidth(ones_array, sampling_rate))


def test_empty(empty_array, sampling_rate) -> None:
    """tsx.spectral_bandwidth should return nan for an empty array."""
    assert jnp.isnan(tsx.spectral_bandwidth(empty_array, sampling_rate))


def test_array_with_nan_values(array_with_nan, sampling_rate) -> None:
    """tsx.spectral_bandwidth should return nan for an array with a nan value."""
    assert jnp.isnan(tsx.spectral_bandwidth(array_with_nan, sampling_rate))


def test_array_with_inf_values(array_with_inf, sampling_rate) -> None:
    """tsx.spectral_bandwidth should return nan for an array with an inf value."""
    assert jnp.isnan(tsx.spectral_bandwidth(array_with_inf, sampling_rate))
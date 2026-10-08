"""Tests for the dominant_frequency feature."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


def test_single_tone(single_tone, sampling_rate) -> None:
    """tsx.dominant_frequency should return the tone frequency for a pure sine wave."""
    assert tsx.dominant_frequency(single_tone, sampling_rate) == pytest.approx(7.0, rel=1e-9)


def test_single_tone_with_offset(single_tone_with_offset, sampling_rate) -> None:
    """tsx.dominant_frequency should ignore the DC offset (without mean removal: 0 Hz)."""
    assert tsx.dominant_frequency(single_tone_with_offset, sampling_rate) == pytest.approx(
        7.0, rel=1e-9
    )


def test_loud_and_quiet_tone(loud_and_quiet_tone, sampling_rate) -> None:
    """tsx.dominant_frequency should return the louder of two tones."""
    assert tsx.dominant_frequency(loud_and_quiet_tone, sampling_rate) == pytest.approx(
        5.0, rel=1e-9
    )


def test_aliased_tone(aliased_tone, sampling_rate) -> None:
    """tsx.dominant_frequency should fold a 60 Hz tone sampled at 100 Hz to 40 Hz."""
    assert tsx.dominant_frequency(aliased_tone, sampling_rate) == pytest.approx(40.0, rel=1e-9)


def test_ones(ones_array, sampling_rate) -> None:
    """tsx.dominant_frequency should return nan for a constant signal."""
    assert jnp.isnan(tsx.dominant_frequency(ones_array, sampling_rate))


def test_empty(empty_array, sampling_rate) -> None:
    """tsx.dominant_frequency should return nan for an empty array."""
    assert jnp.isnan(tsx.dominant_frequency(empty_array, sampling_rate))


def test_array_with_nan_values(array_with_nan, sampling_rate) -> None:
    """tsx.dominant_frequency should return nan for an array with a nan value."""
    assert jnp.isnan(tsx.dominant_frequency(array_with_nan, sampling_rate))


def test_array_with_inf_values(array_with_inf, sampling_rate) -> None:
    """tsx.dominant_frequency should return nan for an array with an inf value."""
    assert jnp.isnan(tsx.dominant_frequency(array_with_inf, sampling_rate))
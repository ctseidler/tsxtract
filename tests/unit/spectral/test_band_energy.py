"""Tests for the band_energy feature."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


def test_single_tone_inside_band(single_tone, sampling_rate) -> None:
    """tsx.band_energy should return 1 when the band contains the only tone."""
    assert tsx.band_energy(single_tone, sampling_rate, 0.0, 10.0) == pytest.approx(1.0, rel=1e-9)


def test_single_tone_outside_band(single_tone, sampling_rate) -> None:
    """tsx.band_energy should return 0 when the band misses the only tone."""
    assert tsx.band_energy(single_tone, sampling_rate, 10.0, 50.0) == pytest.approx(
        0.0, abs=1e-3
    )


def test_single_tone_with_offset(single_tone_with_offset, sampling_rate) -> None:
    """tsx.band_energy should ignore the DC offset (without mean removal: ~0)."""
    assert tsx.band_energy(single_tone_with_offset, sampling_rate, 0.0, 10.0) == pytest.approx(
        1.0, rel=1e-9
    )


def test_loud_and_quiet_tone(loud_and_quiet_tone, sampling_rate) -> None:
    """tsx.band_energy should use power weights (magnitude weighting would give 0.75)."""
    assert tsx.band_energy(loud_and_quiet_tone, sampling_rate, 0.0, 10.0) == pytest.approx(
        0.9, rel=1e-9
    )


def test_band_edges(loud_and_quiet_tone, sampling_rate) -> None:
    """tsx.band_energy should include the lower edge and exclude the upper edge."""
    assert tsx.band_energy(loud_and_quiet_tone, sampling_rate, 5.0, 15.0) == pytest.approx(
        0.9, rel=1e-9
    )
    assert tsx.band_energy(loud_and_quiet_tone, sampling_rate, 5.0, 16.0) == pytest.approx(
        1.0, rel=1e-9
    )


def test_ones(ones_array, sampling_rate) -> None:
    """tsx.band_energy should return nan for a constant signal, even for an empty band."""
    assert jnp.isnan(tsx.band_energy(ones_array, sampling_rate, 0.0, 10.0))
    assert jnp.isnan(tsx.band_energy(ones_array, sampling_rate, 60.0, 70.0))


def test_empty(empty_array, sampling_rate) -> None:
    """tsx.band_energy should return nan for an empty array."""
    assert jnp.isnan(tsx.band_energy(empty_array, sampling_rate, 0.0, 10.0))


def test_array_with_nan_values(array_with_nan, sampling_rate) -> None:
    """tsx.band_energy should return nan for an array with a nan value."""
    assert jnp.isnan(tsx.band_energy(array_with_nan, sampling_rate, 0.0, 10.0))


def test_array_with_inf_values(array_with_inf, sampling_rate) -> None:
    """tsx.band_energy should return nan for an array with an inf value."""
    assert jnp.isnan(tsx.band_energy(array_with_inf, sampling_rate, 0.0, 10.0))
"""Tests for the power_bandwidth feature."""

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx


def test_single_tone(single_tone, sampling_rate) -> None:
    """tsx.power_bandwidth should return 0 for a pure sine wave."""
    assert tsx.power_bandwidth(single_tone, sampling_rate) == pytest.approx(0.0, abs=1e-3)


def test_single_tone_with_offset(single_tone_with_offset, sampling_rate) -> None:
    """tsx.power_bandwidth should ignore the DC offset."""
    assert tsx.power_bandwidth(single_tone_with_offset, sampling_rate) == pytest.approx(
        0.0, abs=1e-3
    )


def test_two_tones(two_tones, sampling_rate) -> None:
    """tsx.power_bandwidth should span both tones when each holds 50 % of the power."""
    assert tsx.power_bandwidth(two_tones, sampling_rate) == pytest.approx(10.0, rel=1e-9)


def test_loud_and_quiet_tone(loud_and_quiet_tone, sampling_rate) -> None:
    """tsx.power_bandwidth should still span both tones at the default 90 %."""
    assert tsx.power_bandwidth(loud_and_quiet_tone, sampling_rate) == pytest.approx(
        10.0, rel=1e-9
    )


def test_power_fraction(loud_and_quiet_tone, sampling_rate) -> None:
    """tsx.power_bandwidth should collapse onto the louder tone at 70 % (magnitude: 10 Hz)."""
    assert tsx.power_bandwidth(
        loud_and_quiet_tone, sampling_rate, power_fraction=0.7
    ) == pytest.approx(0.0, abs=1e-3)


def test_ones(ones_array, sampling_rate) -> None:
    """tsx.power_bandwidth should return nan for a constant signal."""
    assert jnp.isnan(tsx.power_bandwidth(ones_array, sampling_rate))


def test_empty(empty_array, sampling_rate) -> None:
    """tsx.power_bandwidth should return nan for an empty array."""
    assert jnp.isnan(tsx.power_bandwidth(empty_array, sampling_rate))


def test_array_with_nan_values(array_with_nan, sampling_rate) -> None:
    """tsx.power_bandwidth should return nan for an array with a nan value."""
    assert jnp.isnan(tsx.power_bandwidth(array_with_nan, sampling_rate))


def test_array_with_inf_values(array_with_inf, sampling_rate) -> None:
    """tsx.power_bandwidth should return nan for an array with an inf value."""
    assert jnp.isnan(tsx.power_bandwidth(array_with_inf, sampling_rate))
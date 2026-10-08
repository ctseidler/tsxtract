"""Tests for the spectral_entropy feature."""

import math

import jax.numpy as jnp
import pytest

import tsxtract.extractors as tsx

# The 100-sample tone fixtures have 51 rfft bins; entropies in bit are divided by log2(51).
LOG2_BINS = math.log2(51)


def test_single_tone(single_tone, sampling_rate) -> None:
    """tsx.spectral_entropy should return 0 for a pure sine wave."""
    assert tsx.spectral_entropy(single_tone, sampling_rate) == pytest.approx(0.0, abs=1e-3)


def test_single_tone_with_offset(single_tone_with_offset, sampling_rate) -> None:
    """tsx.spectral_entropy should ignore the DC offset."""
    assert tsx.spectral_entropy(single_tone_with_offset, sampling_rate) == pytest.approx(
        0.0, abs=1e-3
    )


def test_two_tones(two_tones, sampling_rate) -> None:
    """tsx.spectral_entropy should return 1 bit (normalised) for two equal tones."""
    assert tsx.spectral_entropy(two_tones, sampling_rate) == pytest.approx(
        1.0 / LOG2_BINS, rel=1e-9
    )


def test_loud_and_quiet_tone(loud_and_quiet_tone, sampling_rate) -> None:
    """tsx.spectral_entropy should use magnitude weights (power weighting would give 0.0827)."""
    assert tsx.spectral_entropy(loud_and_quiet_tone, sampling_rate) == pytest.approx(
        0.8112781244591328 / LOG2_BINS, rel=1e-9
    )


def test_ones(ones_array, sampling_rate) -> None:
    """tsx.spectral_entropy should return nan for a constant signal."""
    assert jnp.isnan(tsx.spectral_entropy(ones_array, sampling_rate))


def test_empty(empty_array, sampling_rate) -> None:
    """tsx.spectral_entropy should return nan for an empty array."""
    assert jnp.isnan(tsx.spectral_entropy(empty_array, sampling_rate))


def test_array_with_nan_values(array_with_nan, sampling_rate) -> None:
    """tsx.spectral_entropy should return nan for an array with a nan value."""
    assert jnp.isnan(tsx.spectral_entropy(array_with_nan, sampling_rate))


def test_array_with_inf_values(array_with_inf, sampling_rate) -> None:
    """tsx.spectral_entropy should return nan for an array with an inf value."""
    assert jnp.isnan(tsx.spectral_entropy(array_with_inf, sampling_rate))
"""Test suite for `tsxtract.utils.py`."""

import tsxtract.utils as tsutils


def test_generate_random_time_series_dataset():
    """Test dataset generation."""
    dataset = tsutils.generate_random_time_series_dataset(
        n_samples=10,
        n_channels=2,
        sampling_rate=5,
        time_series_length_in_seconds=2,
    )
    assert dataset.shape == (10, 2, 5 * 2)

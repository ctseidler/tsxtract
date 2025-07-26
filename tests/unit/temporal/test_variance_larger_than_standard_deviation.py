"""Unit tests for tsxtract.extraction.variance_larger_than_standard_deviation."""

import tsxtract.extractors as tsx


def test_ones(ones_array) -> None:
    """Extraction should return False for a time series with 100 ones."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(ones_array) == expected_output


def test_zeros(zeros_array) -> None:
    """Extraction should return False for a time series with 100 zeros."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(zeros_array) == expected_output


def test_negatives(negatives_array) -> None:
    """Extraction should return False for a time series with 100 -1 values."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(negatives_array) == expected_output


def test_single_point(single_point) -> None:
    """Extraction should return False for a single datapoint."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(single_point) == expected_output


def test_empty(empty_array) -> None:
    """Extraction should return False for an empty sequence."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(empty_array) == expected_output


def test_nan_values(nan_array) -> None:
    """Extraction should return False for an array with 100 nan values."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(nan_array) == expected_output


def test_array_with_nan_values(array_with_nan) -> None:
    """Extraction should return False for an array with 20 nan values."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(array_with_nan) == expected_output


def test_inf_values(inf_array) -> None:
    """Extraction should return False for an array with 100 inf values."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(inf_array) == expected_output


def test_array_with_inf_values(array_with_inf) -> None:
    """Extraction should return False for this array with 20 inf values."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(array_with_inf) == expected_output


def test_50_50(array_50_50) -> None:
    """Extraction should return False for a time series with 50 zeros and 50 ones."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(array_50_50) == expected_output


def test_20_80(array_20_80) -> None:
    """Extraction should return False for a time series with 20 zeros and 80 ones."""
    expected_output: bool = False
    assert tsx.variance_larger_than_standard_deviation(array_20_80) == expected_output


def test_positive_range(array_positive_range) -> None:
    """Extraction should return True for a range from 0 to 100."""
    expected_output: bool = True
    assert tsx.variance_larger_than_standard_deviation(array_positive_range) == expected_output


def test_negative_range(array_negative_range) -> None:
    """Extraction should return True for a range from -100 to 0."""
    expected_output: bool = True
    assert tsx.variance_larger_than_standard_deviation(array_negative_range) == expected_output


def test_positive_and_negative_range(array_positive_and_negative_range) -> None:
    """Extraction should return True for a range from -50 to 50."""
    expected_output: bool = True
    assert (
        tsx.variance_larger_than_standard_deviation(array_positive_and_negative_range)
        == expected_output
    )

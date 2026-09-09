"""Example script on how to use tsxtract."""

from time import time

from tsxtract.extraction import extract_features
from tsxtract.utils import generate_random_time_series_dataset


def main() -> None:
    """Run the example script."""
    sampling_rate = 100
    test_data = generate_random_time_series_dataset(sampling_rate=sampling_rate)
    print(test_data.shape)  # (100, 5, 1000)

    start = time()
    features = extract_features(test_data, sampling_rate=sampling_rate)
    stop = time()
    print(features["mean"].shape)  # (100, 5)
    print(f"{stop - start}s")  # ~0.08s


if __name__ == "__main__":
    main()

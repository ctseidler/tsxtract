"""Script to test the package.

Dev dependencies: https://stackoverflow.com/questions/78902565/how-do-i-install-python-dev-dependencies-using-uv

"""

from tsxtract.extraction import extract_features, median
from tsxtract.utils import (
    estimate_computational_complexity_per_length,
    generate_random_time_series_dataset,
)

ESTIMATE_COMPLEXITY = False


def main() -> None:
    """Run the script."""
    dataset = generate_random_time_series_dataset(
        n_samples=10,
        n_channels=2,
        sampling_rate=10,
        time_series_length_in_seconds=1,
    )
    print(dataset.shape)

    features: dict = extract_features(dataset)
    print(features.keys())
    print("Number of implemented features:", len(features.keys()))

    print(features["absolute_energy"][0])
    print(dataset[0])

    # Estimate computational complexity of the features
    if ESTIMATE_COMPLEXITY:
        estimate_computational_complexity_per_length(median, repeats=1000)


if __name__ == "__main__":
    main()

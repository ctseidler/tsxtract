"""Script to test the package.

Dev dependencies: https://stackoverflow.com/questions/78902565/how-do-i-install-python-dev-dependencies-using-uv

"""

import jax
import jax.numpy as jnp

from tsxtract.configuration import ExtractionConfiguration
from tsxtract.extraction import extract_features
from tsxtract.extractors import median
from tsxtract.utils import (
    estimate_computational_complexity_per_length,
    generate_random_time_series_dataset,
)

ESTIMATE_COMPLEXITY = False

# TODO: Add test cases for extract_features and ExtractionConfiguration


def main() -> None:
    """Run the script."""
    dataset = generate_random_time_series_dataset(
        n_samples=10,
        n_channels=2,
        sampling_rate=10,
        time_series_length_in_seconds=1,
    )
    print(dataset.shape)

    cfg: ExtractionConfiguration = ExtractionConfiguration()
    cfg.add_feature("count_above", [{"value": 2}])
    cfg.add_feature(custom_sum_values, None)
    # cfg.remove_feature("count_above__value_2")

    settings = cfg.to_dict()
    cfg = ExtractionConfiguration.from_dict(settings)

    cfg.to_json("extraction_settings")
    cfg = ExtractionConfiguration.from_json("extraction_settings.json")

    features: dict = extract_features(dataset, cfg)
    print(features.keys())
    print("Number of implemented features:", len(features.keys()))

    print(features["absolute_energy"][0])
    print(dataset[0])

    # Estimate computational complexity of the features
    if ESTIMATE_COMPLEXITY:
        estimate_computational_complexity_per_length(median, repeats=1000)


def custom_sum_values(time_series: jax.Array) -> jax.Array:
    """Calculate the sum of values of the time series.

    Parameters
    ----------
    time_series : jax.Array
        Vector to calculate the sum of values of.

    Returns
    -------
    jax.Array :
        The sum of values of the vector.

    """
    return jnp.sum(time_series)


if __name__ == "__main__":
    main()

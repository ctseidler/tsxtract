"""Module containing feature extraction functions.

Docstring example: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

Tsfresh feature list: https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import jax

from tsxtract.configuration import ExtractionConfiguration
from tsxtract.utils import measure_runtime


@measure_runtime()
def extract_features(dataset: jax.Array, config: ExtractionConfiguration) -> dict[str, jax.Array]:
    """Extract a bunch of features.

    Expected input: Dataset of shape (samples, channels, time_series)

    """
    extracted_features: dict[str, jax.Array] = {}
    feature_extractors: dict[str, Callable] = config.map_settings_to_feature_extractors()

    for name, extractor in feature_extractors.items():
        extracted_features[name] = jax.vmap(jax.vmap(extractor))(dataset)

    return extracted_features

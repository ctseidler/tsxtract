"""Module containing feature extraction functions.

Docstring example: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

Tsfresh feature list: https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


from functools import partial

import jax

from tsxtract.configuration import ExtractionConfiguration
from tsxtract.utils import measure_runtime


@measure_runtime()
@partial(jax.jit, static_argnames=["config"])
def extract_features(dataset: jax.Array, config: ExtractionConfiguration) -> dict[str, jax.Array]:
    """Extract a bunch of features.

    Expected input: Dataset of shape (samples, channels, time_series)

    """
    extracted_features: dict[str, jax.Array] = {}
    feature_extractors: dict[str, Callable] = config.map_settings_to_feature_extractors()

    for name, extractor in feature_extractors.items():
        extracted_features[name] = flat_vmap(extractor, dataset)

    return extracted_features


def flat_vmap(fn, x):
    """Applying vmap on (samples, channels) simultaneously."""
    s, c, t = x.shape
    x_flat = x.reshape(s * c, t)
    out = jax.vmap(fn)(x_flat)
    return out.reshape(s, c, *out.shape[1:])

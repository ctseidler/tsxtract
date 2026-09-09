"""Feature configuration: which functions to call, with which parameters.

Separating the configuration from the extractor lets the same feature be
requested several times with different parameters without editing
``extract_features``. The layout follows the convention shared by tsfresh
(``settings.py``) and TSFEL (``features_settings.py``).
"""

import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

from tsxtract.features import (
    autocorrelation,
    band_energy,
    dominant_frequency,
    kurtosis,
    mad,
    maximum,
    mean,
    median,
    minimum,
    percentile,
    power_bandwidth,
    rms,
    skewness,
    spectral_bandwidth,
    spectral_centroid,
    spectral_entropy,
    spectral_rolloff,
    std,
    variance,
    zero_crossing_rate,
)

class FeatureSpec:
    """One entry of a feature configuration: which function to call, with
    which parameters, and under which name the result is stored.

    Keeping the configuration separate from ``extract_features`` allows
    the same feature to be requested several times with different
    parameters -- two percentiles, several frequency bands -- without
    editing the extractor:

    >>> FeatureSpec(percentile, q=25.0)
    >>> FeatureSpec(band_energy, low_frequency=0.6, high_frequency=2.5)
    >>> FeatureSpec(spectral_centroid)

    Parameters
    ----------
    function : Callable
        Feature function taking a one-dimensional signal as its first
        argument.
    name : str, optional
        Overrides the generated output name. Keyword-only, so it cannot
        be mistaken for a parameter of the feature function.
    **parameters
        Extra arguments for the feature function. ``sampling_rate`` is
        not passed here: it is injected by :meth:`to_function` for every
        function whose signature accepts it.

    Notes
    -----
    Instances are immutable and hashable because ``extract_features``
    receives them as a static JIT argument. Parameter values must
    therefore be hashable too: use a tuple ``(1, 2, 3)``, never a list.

    """

    def __init__(
        self, function: Callable, *, name: str | None = None, **parameters: Any
    ) -> None:
        # Assigned through object.__setattr__ because __setattr__ below is
        # closed: a hashable object must not change after creation.
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "parameters", tuple(parameters.items()))
        object.__setattr__(self, "name", name)

    def __setattr__(self, attribute: str, value: Any) -> None:
        raise AttributeError(
            f"FeatureSpec is immutable; create a new one instead of setting {attribute!r}"
        )

    @property
    def output_name(self) -> str:
        """Name this specification stores its result under.

        Without parameters the plain function name; with parameters the
        pairs are appended as ``__name_value``, following the naming
        convention of tsfresh so that the mapping between the three
        packages' output columns stays mechanical.
        """
        if self.name is not None:
            return self.name
        parts = [self.function.__name__]
        parts += [
            f"{parameter}_{_format_parameter_value(value)}"
            for parameter, value in self.parameters
        ]
        return "__".join(parts)

    def to_function(self, sampling_rate: float) -> Callable:
        """Return a one-argument function ``signal -> feature``.

        ``sampling_rate`` is forwarded only to those feature functions
        that declare it. The signature is the single source of truth, so
        a feature can never fall out of sync with a manually maintained
        list of "spectral" features.
        """
        arguments = dict(self.parameters)
        if "sampling_rate" in inspect.signature(self.function).parameters:
            arguments["sampling_rate"] = sampling_rate
        return partial(self.function, **arguments)

    def __repr__(self) -> str:
        return f"FeatureSpec({self.output_name})"

    def _fields(self) -> tuple:
        return (self.function, self.parameters, self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureSpec):
            return NotImplemented
        return self._fields() == other._fields()

    def __hash__(self) -> int:
        return hash(self._fields())


def _format_parameter_value(value: Any) -> str:
    """Render one parameter value for use inside a feature name.

    Tuples are joined with underscores (``(1, 2, 3)`` -> ``1_2_3``) and
    whole-numbered floats lose their decimal part (``25.0`` -> ``25``),
    so that the generated names stay short and stable.
    """
    if isinstance(value, tuple):
        return "_".join(_format_parameter_value(item) for item in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


TEMPORAL_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(maximum),
    FeatureSpec(mean),
    FeatureSpec(minimum),
    FeatureSpec(std),
    FeatureSpec(variance),
    FeatureSpec(median),
    FeatureSpec(rms),
    FeatureSpec(mad),
    FeatureSpec(percentile, q=25.0),
    FeatureSpec(percentile, q=75.0),
    FeatureSpec(skewness),
    FeatureSpec(kurtosis),
    FeatureSpec(zero_crossing_rate),
    FeatureSpec(autocorrelation, lags=(1, 2, 3)),
)

"""Default time-domain features."""

SPECTRAL_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(spectral_centroid),
    FeatureSpec(spectral_bandwidth),
    FeatureSpec(spectral_rolloff),
    FeatureSpec(dominant_frequency),
    FeatureSpec(spectral_entropy),
    FeatureSpec(band_energy, low_frequency=0.6, high_frequency=2.5),
    FeatureSpec(power_bandwidth),
)

"""Default frequency-domain features."""

DEFAULT_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    TEMPORAL_FEATURE_SPECS + SPECTRAL_FEATURE_SPECS
)
"""Feature set extracted when no configuration is given."""

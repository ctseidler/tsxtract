"""Module containing the feature configuration class."""

import importlib.util
import inspect
import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax

from tsxtract import extractors
from tsxtract.utils import parse_number


# TODO: Refactor class
# TODO: Add unit tests
class ExtractionConfiguration:
    """Class to define the feature extraction settings.

    Methods
    -------
    add_feature(feature, config) :
        Add a feature to the extraction settings
    clear() :
        Clear the extraction configuration
    from_dict(feature_config) :
        Create a ExtractionConfiguration from dictionary
    from_json(file) :
        Create a ExtractionConfiguration from a json-file
    map_features_to_feature_extractors() :
        Map the settings_per_feature to the feature extraction functions
    remove_feature(feature) :
        Remove the given feature from the extraction configuration
    to_dict() :
        Return the feature extraction settings to a dictionary
    to_json(file) :
        Export the settings to a json-file

    """

    def __init__(self) -> None:
        """Create a new ExtractionConfiguration."""
        self._settings_per_feature = self._initialize_extraction_settings()
        self._set_default_settings(self._settings_per_feature)

    def _initialize_extraction_settings(self) -> dict[str, None]:
        """Initialize the extraction settings.

        Returns
        -------
        dict[str, None] :
            Dictionary, containing the names of the features to extract and `None` as configuration
            per feature.

        """
        settings_per_feature: dict[str, None] = {}

        for name, value in extractors.__dict__.items():
            if name in ["extract_features", "partial"]:
                continue
            if not callable(value):
                continue
            settings_per_feature[name] = None

        return settings_per_feature

    def _set_default_settings(self, settings_per_feature: dict[str, None]) -> None:
        """Set the default extraction settings.

        This updated the `settings_per_feature` dictionary to set values for features with
        parameters.

        Parameters
        ----------
        settings_per_feature : dict[str, None]
            The initialized feature configuration dictionary.

        """
        settings_per_feature.update(
            {
                "count_above": [{"value": 0}],
                "count_below": [{"value": 0}],
                "value_count": [{"value": value} for value in [-1, 0, 1]],
                "range_count": [
                    {"lower_bound": -1, "upper_bound": 1},
                    {"lower_bound": -1e12, "upper_bound": 0},
                    {"lower_bound": 0, "upper_bound": 1e12},
                ],
            },
        )

    def add_feature(
        self,
        feature: str | Callable,
        config: list[dict[str, int | float]] | None,
    ) -> None:
        """Add a feature to the extraction settings.

        Parameters
        ----------
        feature : str | Callable
            Feature extractor to add to the ExtractionSettings. If str, will use the built-in
            feature with the given name. If Callable, will use the given function.
        config : list[dict[str, int | float]] | None
            Parameters to use for the given feature. Set to `None`, if the feature accepts no
            parameters. Pass a dictionary to use one or multiple parameters sets at once.

        Raises
        ------
        ValueError
            If the mapping fails.

        """
        # Case 1: Built-in feature with no parameters
        if isinstance(feature, str) and (config is None):
            self._settings_per_feature[feature] = extractors.__dict__[feature]

        # Case 2: Built-in feature with parameters
        elif isinstance(feature, str) and (config is not None):
            for setting in config:
                parameters: str = "__".join(f"{key}_{value}" for key, value in setting.items())
                feature_name: str = f"{feature}__{parameters}"
                self._settings_per_feature[feature_name] = partial(
                    extractors.__dict__[feature],
                    **setting,
                )

        # Case 3: Custom feature with no parameters
        elif callable(feature) and (config is None):
            self._settings_per_feature[feature.__name__] = jax.jit(feature)

        # Case 4: Custom feature with parameters
        elif callable(feature) and (config is not None):
            for setting in config:
                parameters: str = "__".join(f"{key}_{value}" for key, value in setting.items())
                feature_name: str = f"{feature.__name__}__{parameters}"
                self._settings_per_feature[feature_name] = jax.jit(partial(feature, **setting))

        # Raise error if no case above matches
        else:
            msg: str = f"Feature {feature} cannot be added."
            raise ValueError(msg)

    def clear(self) -> None:
        """Clear the extraction configuration."""
        self._settings_per_feature.clear()

    @classmethod
    def from_dict(
        cls,
        feature_config: dict[str, Callable | list[dict[str, int | float]] | None],
    ) -> "ExtractionConfiguration":
        """Create a ExtractionConfiguration from dictionary.

        Parameters
        ----------
        feature_config : dict[str, Callable | list[dict[str, int | float]] | None]
            Feature dictionary to create the ExtractionConfiguration from. Each key is the name
            of the feature and each value is the configuration of that feature. If `value` is
            a callable, it is interpreted as custom feature.

        """
        instance: ExtractionConfiguration = cls()
        instance.clear()
        instance._settings_per_feature = feature_config
        return instance

    @classmethod
    def from_json(cls, file: Path | str) -> "ExtractionConfiguration":
        """Create a ExtractionConfiguration from a json-file.

        Parameters
        ----------
        file : Path | str
            JSON-file (including file ending!) to load the feature settings from.

        """
        with Path.open(file, "r", encoding="utf-8") as json_file:
            feature_config = json.load(json_file)

        feature_dict = {}
        for name, config in feature_config.items():
            # Case 1: Built-in feature
            if "tsxtract/src/tsxtract" in config["import_path"]:
                # Convert parameter values to numeric
                if config["parameters"] is not None:
                    config["parameters"] = [
                        {k: parse_number(v) for k, v in d.items()} for d in config["parameters"]
                    ]

                feature_dict[name] = config["parameters"]

            # Case 2: Custom feature
            else:
                module_name = config["import_path"].rsplit("/", maxsplit=1)[-1].strip(".py")
                spec = importlib.util.spec_from_file_location(module_name, config["import_path"])
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                func = getattr(module, name)
                if config["parameters"] is None:
                    feature_dict[name] = func
                else:
                    config["parameters"] = [
                        {k: parse_number(v) for d in config["parameters"] for k, v in d.items()},
                    ]
                    feature_dict[name] = partial(func, **config["parameters"])

        instance: ExtractionConfiguration = cls()
        instance.clear()
        instance._settings_per_feature = feature_dict
        return instance

    def map_settings_to_feature_extractors(self) -> dict[str, Callable]:
        """Map the settings_per_feature to the feature extraction functions.

        Returns
        -------
        dict[str, Callable]
            The mapped feature settings. Keys are the names of the features, values are the
            configured feature extraction functions.

        Notes
        -----
         - Parameter values are automatically append to the feature names using underscores.
         - This method is automatically called by `tsxtract.extraction.extract_features()`

        """
        features_to_extract: dict[str, Callable] = {}

        for name, config in self._settings_per_feature.items():
            # Case 1: Built-in feature with no parameters
            if config is None:
                features_to_extract[name] = extractors.__dict__[name]

            # Case 2: Custom feature
            elif callable(config):
                features_to_extract[name] = config

            # Case 3: Built-in feature with parameters
            elif len(config) > 0:
                for setting in config:
                    parameters: str = "__".join(f"{key}_{value}" for key, value in setting.items())
                    feature_name: str = f"{name}__{parameters}"
                    features_to_extract[feature_name] = partial(
                        extractors.__dict__[name],
                        **setting,
                    )

        return features_to_extract

    def remove_feature(self, feature: str) -> None:
        """Remove the given feature from the extraction configuration.

        Parameters
        ----------
        feature : str
            Feature to remove.

        """
        del self._settings_per_feature[feature]

    def to_dict(self) -> dict[str, Callable | list[dict[str, int | float]] | None]:
        """Return the feature extraction settings to a dictionary.

        Returns
        -------
        dict[str, Callable | list[dict[str, int | float]] | None]
            The feature extraction settings as a dictionary. Each key is the feature name, each
            value is the feature configuration.

        Notes
        -----
        Custom features are stored as callable in the feature dictionary.

        """
        return self._settings_per_feature

    def to_json(self, file: Path | str) -> None:
        """Export the settings to a json-file.

        Parameters
        ----------
        file : Path | str
            The json-file to export the feature settings to. File ending can be omitted.

        """
        feature_extractors: dict[str, Callable] = self.map_settings_to_feature_extractors()
        export_data: dict = {}

        for name, extractor in feature_extractors.items():
            feature_name = name.split("__", maxsplit=1)[0] if "__" in name else name

            # Unwrap jit-compiled function in partial to get import path
            if isinstance(extractor, partial):
                extractor = extractor.func
            if hasattr(extractor, "__wrapped__"):
                extractor = extractor.__wrapped__
            import_path: str = str(Path(inspect.getfile(extractor)).resolve())

            # Get the parameters for the function
            parameters: list[str] = name.split("__")[1:]
            kw_parameters: dict[str, str] = {}
            if len(parameters) > 0:
                for kw_pair in parameters:
                    parameter_name, val = kw_pair.rsplit("_", maxsplit=1)
                    kw_parameters[parameter_name] = val

            export_parameters = None if len(kw_parameters) == 0 else [kw_parameters]

            # Check if feature already exists, if yes, append configurations
            if feature_name in export_data:
                export_data[feature_name]["parameters"].extend(export_parameters)
            else:
                export_data[feature_name] = {
                    "import_path": import_path,
                    "parameters": export_parameters,
                }

        if not str(file).endswith(".json"):
            file = Path(str(file) + ".json")

        with Path.open(file, "w", encoding="utf-8") as json_file:
            json.dump(export_data, json_file, indent=4, sort_keys=True)

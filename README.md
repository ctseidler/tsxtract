# tsxtract

Hardware-accelerated time series feature extraction using JAX.

**NOTE**: tsxtract is still under development. Please report any bugs by creating an issue.

Please star the repository if you like tsxtract.

## Why tsxtract?

- Fast: All extraction operations are vectorized
- Hardware-accelerated: Run on CPU, GPU or TPU
- Easy-to-use: One function is all you need 

## Example usage

```python
from tsxtract.extraction import extract_features

# Dataset is a 3d-numpy or jax array with following dimensions:
# (samples, channels, length)
features = extract_features(dataset)

print(type(features)) # dict
print(features["mean"].shape) # jax.Array of size (samples, channels)
```

Check out `main.py` for a complete usage example.

## Installation

### Using uv

Step 1: Install uv: ```pip install uv```

Step 2: Clone this repository: ```git clone https://github.com/ctseidler/tsxtract.git .```

Step 3: Create a new virtual environment: ```uv venv --python 3.12```

Step 4: Activate the virtual environment: ```source .venv/bin/activate```

Step 5: Install the package as editable from source: ```uv pip install -e .```

Step 6: Test your setup by executing the `main.py` script: ```uv run main.py```

## Overview of Extracted Features

- Maximum
- Mean
- Minimum

## Contributing

Contributions are welcome! Please open an issue, if you have any feature request. You can also implement it by forking the repository and creating a pull-request upon completion. Please make sure that your feature is covered by unittests (see test/). Current test coverage is 100%.

**Development setup**:

- Install the package locally as mentioned above.
- Run `uv sync` to install dev dependencies.
- Run the unit tests prior to a commit (pre-commit): `uv run coverage run -m pytest`
- Check the coverage report to identify missing test coverage: `uv run coverage report -m`

## Roadmap

Version 0.2:
- [ ] Test CPU and GPU support
- [ ] Add example notebook for CPU and GPU extraction

Version 0.3:
- [ ] Add additional features
- [ ] Add features with customizable parameters
- [ ] Add configuration options

Version 0.4:
- [ ] Add support to custom features
- [ ] Allow configuration as dict
- [ ] Allow configuration as json

Version 0.5:
- [ ] Add frequency-based features
- [ ] Make package compatible for Python 3.10 and 3.11
- [ ] Allow easy IO, e.g., by integrating polars

Version 1.0:
- [ ] Performance benchmark
- [ ] Add project logo

## Authors

Christian T. Seidler

## See also

- [tsfresh](https://github.com/blue-yonder/tsfresh): Time Series Feature extraction based on scalable hypothesis test
- [TSFEL](https://github.com/fraunhoferportugal/tsfel): Time Series Feature Extraction Library
- [pycatch22](https://github.com/DynamicsAndNeuralSystems/pycatch22): CAnonical Time-series CHaracteristics in Python
- [seglearn](https://github.com/dmbee/seglearn): An sklearn extension for machine learning time series or sequences

## License

MIT
# tsXtract

Hardware-accelerated time series feature extraction using JAX.

## Why tsXtract?

## Installation

### Using uv

Step 1: Install uv: ```pip install uv```

Step 2: Clone this repository: ```git clone https://github.com/ctseidler/tsxtract.git .```

Step 3: Create a new virtual environment: ```uv venv --python 3.12```

Step 4: Activate the virtual environment: ```source .venv/bin/activate```

Step 5: Install the package as editable from source: ```uv pip install -e .```

Step 6: Test your setup by executing the `test.py` script: ```uv run test.py```


## Examples

## Overview of Extracted Features

| Feature Name | Type | Computational Complexity |
| ------------ | ---- | ---------- |
| Length | Temporal | O(1) for time series length smaller 200.000, O(n) otherwise
| Mean | Temporal  | O(1) for time series length smaller 200.000, O(n) otherwise
| Median | Temporal | n/a
| Maximum | Temporal | n/a
| Minimum | Temporal | n/a
| Standard Deviation | Temporal | n/a
| Variance | Temporal | n/a
| Sum of Values | Temporal | n/a
| Absolute Sum of Values | Temporal | n/a
| Absolute Maximum | Temporal | n/a
| Absolute Energy | Temporal | n/a
| Variance larger than Standard Deviation | Temporal | n/a
| Value Range | Temporal | n/a
| Count above Mean | Temporal | n/a
| Count below Mean | Temporal | n/a
| Root Mean Square | Temporal | n/a
| Absolute Sum of Changes | Temporal | n/a
| Range Count in interval | Temporal | n/a
| Value Count of t | Temporal | n/a
| Count above t | Temporal | n/a
| Count below t | Temporal | n/a

## Contributing

We are open to contributions. Please open an Issue first, so that we can discuss your feature.

Then, fork the repository and create a Pull-Request, once your feature is implemented.

**Key Steps**:

- Setup a dev environment: Run `uv sync` to install dev dependencies.
- Run the unit tests prior to a commit (pre-commit): `uv run coverage run -m pytest`
- Check the coverage report to identify missing test coverage: `uv run coverage report -m`

## Roadmap

- [x] Refactor `extract_features` function
- [x] Set default values for feature extractors with parameters
- [x] Add interface for easy configuration of feature extraction configuration
- [x] Add support to set feature extraction config from dict
- [x] Add JSON import and export for feature extraction configuration
- [ ] Add further unit tests to reach coverage of 100%
- [ ] Add usage example
- [ ] Add additional feature extractors
- [x] Add support for custom features
- [ ] Add custom error messages
- [ ] Enhance feature extraction docstrings
- [ ] Benchmark performance against Tsfresh & TSFEL
- [ ] Enhance README
- [ ] Add documentation using Sphinx
- [ ] Estimate computational complexity per extracted feature
- [ ] Add different pre-configured feature sets
- [ ] Test GPU support
- [ ] Launch version 1.0
- [ ] Add package to PyPi

## Authors

Christian T. Seidler

## See also

## License

MIT
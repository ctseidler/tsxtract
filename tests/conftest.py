"""Shared pytest configuration.

JAX defaults to float32. The x64 flag is a global setting that must be set
before the first array is created, so it belongs here: conftest.py is
imported before any test module. Running the tests in float64 means the
assertions compare implementations, not floating-point precision.
"""

import jax

jax.config.update("jax_enable_x64", True)
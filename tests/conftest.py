"""Shared fixtures for fast-decoder tests."""

import numpy as np
import pytest

from fast_decoder import constants


@pytest.fixture(autouse=True)
def _reset_constants():
    """Ensure constants are at default sample rate for every test."""
    constants.configure(constants.DEFAULT_SAMPLE_RATE)

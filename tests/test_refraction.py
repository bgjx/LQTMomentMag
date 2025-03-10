""" Unit test for refraction.py """

import os
from pathlib import Path
import pytest
import numpy as np

from LQTMomentMag.refraction import (
    build_raw_model,
    upward_model,
    downward_model,
    up_refract,
    calculate_inc_angle
)

from LQTMomentMag.config import CONFIG 

@pytest.fixture
def test_data():
    "Fixture providing consistent test data."

    hypo = [37.916973, 126.651613, 200]
    staion = [ 37.916973, 126.700882, 2200]

    epi_dist_m = 4332.291
    return hypo, staion, epi_dist_m


def test_build_raw_model():
    """ Test build_raw_model creates correct layer structure."""

    boundaries = [
                    [-3.00, -1.90], [-1.90, -0.59], [-0.59, 0.22], [0.22, 2.50], [2.50, 7.00]
    ]

    velocities = [2.68, 2.99, 3.95, 4.50, 4.99]
    model = build_raw_model(boundaries, velocities)
    expected = [[3000.0, -1100.0, 2680], [1900.0, -1310.0, 2990], [590.0, -809.9999999999999, 3950], [-220.0, -2280.0, 4500], [-2500.0, -4500.0, 4990]]

    assert(len(model)) == len(expected)
    for layer, exp_layer in zip(model, expected):
        assert layer == pytest.approx(exp_layer, rel=1e-5)
    


""" Unit test for refraction.py """

import os
from pathlib import Path
import pytest
import numpy as np

from src.LQTMomentMag.refraction import (
    build_raw_model,
    upward_model,
    downward_model,
    up_refract,
    calculate_inc_angle
)

from src.LQTMomentMag.config import CONFIG 
"""
LQTMomentMag: A Python package for calculating moment magnitude using full P, SV, and SH energy components in local network settings.

This package provides tools for seismic data processing, ray tracing, and spectral fitting.
"""

from .main import main
from .processing import calculate_moment_magnitude, start_calculate
from .utils import get_user_input, read_waveforms

__version__ = "1.0.0"
__author__ = "Arham Zakki Edelo"
__email__ = "edelo.arham@gmail.com"
__all__ = [
    "main",
    "calculate_moment_magnitude",
    "start_calculate",
    "get_user_input",
    "read_waveforms",
    "calculate_inc_angle"]
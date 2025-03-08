"""  LQTMomentMag: A program for calculating moment magnitude using full P, SV, and SH energy components in local network settings.  """

from .main import main
from .processing import calculate_moment_magnitude, start_calculate
from .utils import get_user_input, read_waveforms

__version__ = "1.0.0"
__author__ = "Arham Zakki Edelo"
__all__ = ["main", "processing", "utils"]
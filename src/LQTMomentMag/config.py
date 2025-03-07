from dataclasses import dataclass
from configparser import ConfigParser
from typing import List
from pathlib import Path

@dataclass
class SeismicConfig:
    """Configuration settings for moment magnitude calculations."""
    WATER_LEVEL: int = 10
    PRE_FILTER: List[float] = None # Initialized in __post_init__
    F_MIN: int = 7
    F_MAX: int = 60
    PADDING_BEFORE_ARRIVAL: float = 0.1
    NOISE_DURATION: float = 0.5
    NOISE_PADDING: float = 0.2
    R_PATTERN_P: float = 0.44
    R_PATTERN_S: float = 0.60
    FREE_SURFACE_FACTOR: float = 2.0
    K_P: float = 0.32
    K_S: float = 0.21
    LAYER_BOUNDARIES: List[List[float]] = None # Initialized in __post_init_
    VELOCITY_VP: List[float] = None
    VELOCITY_VS: List[float] = None
    DENSITY: List[float] = None


    def __post_init__(self):
        if self.PRE_FILTER is None:
            self.PRE_FILTER = [2, 5, 55, 60]
        if self.LAYER_BOUNDARIES is None:
            self.LAYER_BOUNDARIES = [
                [-3.00, -1.90], [-1.90, -0.59], [-0.59, 0.22], [0.22, 2.50],
                [2.50, 7.00], [7.00, 9.00], [9.00, 15.00], [15.00, 33.00], [33.00, 9999]
            ]
        if self.VELOCITY_VP is None:
            self.VELOCITY_VP = [2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00]
        if self.VELOCITY_VS is None:
            self.VELOCITY_VS = [1.60, 1.79, 2.37, 2.69, 2.99, 3.35, 3.47, 3.83, 4.79]
        if self.DENSITY is None:
            self.DENSITY = [2700] * 9

    def load_from_file(self, config_file: str = None) -> None:
        """Load configuration from an INI file, with fallback to defaults."""
        config  = ConfigParser()
        if config_file is None:
            config_file = Path(__file__).parent.parent/ "config.ini"
        if not config.read(config_file):
            return 
        self.WATER_LEVEL = config.getint("Seismic", "water_level", fallback=self.WATER_LEVEL)
        self.PRE_FILTER = [float(x) for x in config.get("Seismic", "pre_filter", fallback="2,5,55,60").split(",")]
        self.F_MIN = config.getfloat("Seismic", "f_min", fallback=self.F_MIN)
        self.F_MAX = config.getfloat("Seismic", "f_max", fallback=self.F_MAX)
        self.PADDING_BEFORE_ARRIVAL = config.getfloat("Seismic", "padding_before_arrival", fallback=self.PADDING_BEFORE_ARRIVAL)
        self.NOISE_DURATION = config.getfloat("Seismic", "noise_duration", fallback=self.NOISE_DURATION)
        self.NOISE_PADDING = config.getfloat("Seismic", "noise_padding", fallback=self.NOISE_PADDING)
        self.R_PATTERN_P = config.getfloat("Seismic", "r_pattern_p", fallback=self.R_PATTERN_P )
        self.R_PATTERN_S = config.getfloat("Seismic", "r_pattern_s", fallback=self.R_PATTERN_S)
        self.FREE_SURFACE_FACTOR = config.getfloat("Seismic", "free_surface_factor", fallback=self.FREE_SURFACE_FACTOR)
        self.K_P = config.getfloat("Seismic", "k_p", fallback=self.K_P)
        self.K_S = config.getfloat("Seismic", "k_s", fallback=self.K_S)
        
        # parse layer boundaries
        boundaries_str = config.get("Seismic", "layer_boundaries", fallback= "-3.00,-1.90; -1.90,-0.59; -0.59, 0.22; 0.22, 2.50; 2.50, 7.00; 7.00,9.00;  9.00,15.00 ; 15.00,33.00; 33.00,9999")
        self.LAYER_BOUNDARIES = [[float(x) for x in layer.split(",")] for layer in boundaries_str.split(";")]
        self.VELOCITY_VP = [float(x) for x in config.get("Seismic", "velocity_vp", fallback="2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00").split(",")]
        self.VELOCITY_VS = [float(x) for x in config.get("Seismic", "velocity_vs", fallback="1.60, 1.79, 2.37, 2.69, 2.99, 3.35, 3.47, 3.83, 4.79").split(",")]
        self.DENSITY = [float(x) for x in config.get("Seismic", "density", fallback="2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700").split(",")]


# singleton instance for easy access
CONFIG = SeismicConfig()
CONFIG.load_from_file()
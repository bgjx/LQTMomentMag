import os, glob, sys, warnings, logging
from typing import Tuple, Callable, Optional
import numpy as np
from scipy import signal
from obspy import UTCDateTime, Stream, Trace, read, read_inventory
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    filename = 'mw_calculator_runtime.log',
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("mw_calculator")



def get_valid_input(prompt: str, validate_func: callable, error_msg: str) -> int:
    
    """
    Get valid user input.
    
    Args:
        prompt(str): Prompt to be shown in the terminal.
        validate_func(callable) : A function to validate the input value.
        error_msg(str): Error messages if get wrong input value.
    
    Returns:
        int: Returns an integer event ID”.
    """
    
    while True:
        value = input(prompt).strip()
        try:
            value = int(value)
            if validate_func(value):
                return int(value)
            print(error_msg)
        except ValueError:
            print(error_msg)
        except KeyboardInterrupt:
            sys.exit("Interrupted by user")



def read_waveforms(path: Path, event_id: int, station:str) -> Stream:
    """
    Read waveforms file (.mseed) from the specified path and event id.

    Args:
        path (Path): Parent path of separated by id waveforms directory.
        event_id (int): Unique identifier for the earthquake event.
        station (str): Station name.
    Returns:
        Stream: A Stream object containing all the waveforms from specific event id.
    """
    
    stream = Stream()
    for w in glob.glob(os.path.join(path.joinpath(f"{event_id}"), f"*{station}*.mseed"), recursive = True):
        try:
            stread = read(w)
            stream += stread
        except Exception as e:
            logger.warning(f"Skip reading waveform {w} for event {event_id}: {e}.", exc_info=True)
            continue
            
    return stream



def instrument_remove (
    stream: Stream, 
    calibration_path: Path, 
    figure_path: Optional[str] = None, 
    figure_statement: bool = False,
    config: "SeismicConfig" = None,
    ) -> Stream:
    """
    Removes instrument response from a Stream of seismic traces using calibration files.

    Args:
        stream (Stream): A Stream object containing seismic traces with instrument responses to be removed.
        calibration_path (str): Path to the directory containing the calibration files in RESP format.
        figure_path (Optional[str]): Directory path where response removal plots will be saved. If None, plots are not saved.
        figure_statement (bool): If True, saves plots of the response removal process. Defaults to False.

    Returns:
        Stream: A Stream object containing traces with instrument responses removed.
    """
    
    displacement_stream = Stream()
    for trace in stream:
        try:
            # Construct the calibration file
            station, component = trace.stats.station, trace.stats.component
            inventory_path = calibration_path.joinpath(f"RESP.RD.{station}..BH{component}")
            
            # Read the calibration file
            inventory = read_inventory(inventory_path, format='RESP')
  
            # Prepare plot path if fig_statement is True
            plot_path = None
            if figure_statement and figure_path:
                plot_path = figure_path.joinpath(f"fig_{station}_BH{component}")
            
            # Remove instrument response
            displacement_trace = trace.remove_response(
                                    inventory = inventory,
                                    pre_filt = PRE_FILTER,
                                    water_level = WATER_LEVEL,
                                    output = 'DISP',
                                    zero_mean = True,
                                    taper = True,
                                    taper_fraction = 0.05,
                                    plot = plot_path
                                    )

            # Re-detrend the trace
            displacement_trace.detrend("linear")
            
            # Add the processed trace to the Stream
            displacement_stream+=displacement_trace
            
        except Exception as e:
            logger.warning(f"Error process instrument removal in trace {trace.id}: {e}.", exc_info=True)
            continue
            
    return displacement_stream
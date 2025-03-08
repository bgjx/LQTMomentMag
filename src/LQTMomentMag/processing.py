import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd 
from obspy import Stream, UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.signal import rotate
from scipy import signal
from tqdm import tqdm

import fitting_spectral as fit
import refraction as ref 
from .config import CONFIG
from .plotting import plot_spectral_fitting
from .utils import get_user_input, instrument_remove, read_waveforms, trace_snr


logger = logging.getLogger("mw_calculator")


def calculate_spectra(trace_data: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Power Spectral DENSITY (PSD) of a given signal using the Welch method.

    Args:
        trace_data (np.ndarray): Array of signal data to analyze.
        sampling_rate (float): Sampling rate of the signal in Hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - frequency: Array of sample frequencies.
            - displacement_amplitude: Array of displacement amplitudes in nm·s corresponding to the frequencies.
    """
    if not trace_data.size:
        raise ValueError("trace_data cannot be empty.")
    
    # Calculate Power Spectral DENSITY using Welch's method
    freq, psd = signal.welch(trace_data, sampling_rate, nperseg = len(trace_data))
    
    return freq, np.sqrt(psd)*1e9


def rotate_component(stream: Stream, azimuth: float, incidence: float) -> Stream  :
    """
    Rotates a stream of seismic traces from the ZNE (Vertical-North-East) component system
    to the LQT (Longitudinal-Transverse-Vertical) component system based on given azimuth
    and incidence angles.

    Args:
        stream (Stream): A Stream object containing the Z, N, and E components as traces.
        azimuth (float): The azimuth angle (in degrees) for rotation.
        incidence (float): The incidence angle (in degrees) for rotation.

    Returns:
        Stream: A Stream object containing the rotated L, Q, and T components as traces.
    """
    
    # Create an empty Stream object to hold the rotated traces

    z, n, e = [stream.select(component=comp)[0] for comp in ['Z', 'N', 'E']]
    l_data, q_data, t_data = rotate.rotate_zne_lqt(z.data, n.data, e.data)
    rotated_stream = Stream()
    for data, comp in zip([l_data, q_data, t_data], ['L', 'Q', 'T']):
        trace = z.copy()
        trace.data =  data
        trace.stats.component = comp
        rotated_stream+=trace

    return rotated_stream 



def window_trace(stream: Stream, P_arr: float, S_arr: float) -> Tuple[np.ndarray, ...]:
    """
    Windows seismic trace data around P, SV, and SH phase and extracts noise data.

    Args:
    
        trace (Trace): A Trace object containing the seismic data.
        P_arr (float): The arrival time of the P phase (in seconds from the trace start).
        S_arr (float): The arrival time of the S phase (in seconds from the trace start).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - P_data: The data windowed around the P phase in the L component.
            - SV_data: The data windowed around the S phase in the Q component.
            - SH_data: The data windowed around the S phase in the T component.
            - P_noise: The data windowed around the noise period before the P phase in the L component.
            - SV_noise: The data windowed around the noise period before the P phase in the Q component.
            - SH_noise: The data windowed around the noise period before the P phase in the T component.
    """
    
    # Extract the vertical, radial, and transverse components
    [trace_L, trace_Q, trace_T] = [stream.select(component = comp)[0] for comp in ['L', 'Q', 'T']]
    
    # Dynamic window parameters
    s_p_time = S_arr - P_arr    
    time_after_pick_p = 0.75 * s_p_time
    time_after_pick_s = 1.75 * s_p_time
    
    # Find the data index for phase windowing
    p_phase_start_index = int(round((P_arr - trace_L.stats.starttime - CONFIG.PADDING_BEFORE_ARRIVAL)/trace_L.stats.delta), 4)
    p_phase_end_index = int(round((P_arr - trace_L.stats.starttimem + time_after_pick_p )/trace_L.stats.delta, 4))
    s_phase_start_index = int(round((S_arr - trace_Q.stats.starttime - CONFIG.PADDING_BEFORE_ARRIVAL)/trace_Q.stats.delta), 4)
    s_phase_end_index = int(round((S_arr - trace_Q.stats.starttime + time_after_pick_s )/ trace_Q.stats.delta, 4))
    noise_start_index = int(round((P_arr - trace_L.stats.starttime - CONFIG.NOISE_DURATION)/trace_L.stats.delta, 4))                             
    noise_end_index  = int(round((P_arr - trace_L.stats.starttime - CONFIG.NOISE_PADDING )/trace_L.stats.delta, 4))

    # Window the data by the index
    P_data     = trace_L.data[p_phase_start_index : p_phase_end_index + 1]
    SV_data     = trace_Q.data[s_phase_start_index : s_phase_end_index + 1]
    SH_data     = trace_T.data[s_phase_start_index : s_phase_end_index + 1]

    P_noise  = trace_L.data[noise_start_index : noise_end_index + 1]
    SV_noise = trace_Q.data[noise_start_index : noise_end_index + 1]
    SH_noise = trace_T.data[noise_start_index : noise_end_index + 1]

    return P_data, SV_data, SH_data, P_noise, SV_noise, SH_noise
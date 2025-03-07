#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print(
"""
Created on Thu Dec 15 19:32:03 2022.
Python code to calculate moment magnitude.


Developed by arham zakki edelo.


contact: 
- edelo.arham@gmail.com
- https://github.com/bgjx

Pre-requisite modules:
->[pathlib, tqdm, numpy, pandas, obspy] 

Before you run this program, make sure you have changed all the file path correctly.      


"""
)

import os, glob, sys, warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from obspy import UTCDateTime, Stream, read, Trace, read_inventory
from obspy.signal import rotate
from obspy.geodetics import gps2dist_azimuth

import refraction as ref
import fitting_spectral as fit
import instrument_remove_alt as remove_alt

import logging

warnings.filterwarnings("ignore")

logging.basicConfig(
    filename = 'mw_calculator_runtime.log',
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("mw_calculator")


# Global Parameters
#============================================================================================
# instrument correction parameters
WATER_LEVEL = 10                     # water level to prevent low frequency clipping
PRE_FILTER = [2, 5, 55, 60]  # these values need to be customized for a specific bandwidth

# time padding and position parameters
TIME_BEFORE_PICK = 0.1

# noise spectra duration before P pick
NOISE_DURATION = 0.5
NOISE_PADDING = 0.2

# setting frequency range for spectral fitting (default spectral f-band frequency range)
F_MIN = 7
F_MAX = 60
#============================================================================================


def start_calculate(
    wave_path: Path,
    calibration_path: Path,
    figure_path: Path,
    hypo_data: pd.DataFrame,
    pick_data: pd.DataFrame,
    station_data: pd.DataFrame
    ) -> Tuple [pd.DataFrame, pd.DataFrame, str]:

    """
    Start the process of moment magnitude calculation.
    
    Args:
        wave_path (Path): Path to the waveforms file.
        calibration_path (Path) : Path to the calibration file (.RESP format).
        figure_path (Path) : Path to the directory where the image of peak-to-peak amplitude will be stored.
        hypo_data (pd.DataFrame): Dataframe of hypocenter catalog.
        pick_data (pd.DataFrame): Dataframe of detail data picking.
        station_data (pd.DataFrame) : Dataframe of stations.
        
    Returns:
        Tuple [pd.Dataframe, pd.DataFrame, str]: DataFrames for magnitude results and fitting results, and the output file name.
    """
        
    prompt=input('Please type yes/no if you had changed the path :').strip().lower()
    if prompt != 'yes':
        sys.exit("Ok, please correct the path first!")
    else:
        sys.stdout.write("Process the program ....\n")
    
    # Get the user input.
    id_start, id_end, mw_output, figure_statement = get_user_input()

    # initiate dataframe for magnitude calculation results
    df_result   = pd.DataFrame(
                        columns = ["ID", "Fc_avg", "Fc_std", "Src_rad_avg_(m)", "Src_rad_std_(m)", "Stress_drop_(bar)", "Mw_average", "Mw_std"] 
                        )
    df_fitting  = pd.DataFrame(
                        columns = ["ID", "Station", "F_corner_P", "F_corner_SV", "F_corner_SH", "Qfactor_P", "Qfactor_SV", "Qfactor_SH", "Omega_0_P_(nms)", "Omega_0_SV_(nms)",  "Omega_0_SH_(nms)", "RMS_e_P_(nms)", "RMS_e_SV_(nms)", "RMS_e_SH_(nms)", "Moment_P_(Nm)", "Moment_S_(Nm)"] 
                        )

    failed_events=0
    with tqdm(
        total = id_end - id_start + 1,
        file=sys.stderr,
        position=0,
        leave=True,
        desc="Processing events",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ncols=80,
        smoothing=0.1
    ) as pbar:
        for event_id in range (id_start, id_end + 1):
            logging.info(f"  Calculate moment magnitude for event ID {event_id} ...")
            
            # get the dataframe 
            hypo_data_handler   = hypo_data[hypo_data["ID"] == event_id]
            pick_data_handler   = pick_data[pick_data["Event ID"] == event_id]
            
            # check empty data frame
            if hypo_data_handler.empty or pick_data_handler.empty:
                logger.warning(f"Event_{id}: No data for event {event_id}, skipping...")
                failed_events+=1
            else:
                # start calculating moment magnitude
                try:
                    # calculate the moment magnitude
                    mw_results, fitting_result = calculate_moment_magnitude(wave_path, hypo_data_handler, pick_data_handler, station_data, calibration_path, event_id, figure_path, figure_statement)

                    # create the dataframe from calculate_ml_magnitude results
                    mw_magnitude_result = pd.DataFrame.from_dict(mw_results)
                    mw_fitting_result   = pd.DataFrame.from_dict(fitting_result)
                    
                    # concatinate the dataframe
                    df_result = pd.concat([df_result, mw_magnitude_result], ignore_index = True)
                    df_fitting = pd.concat([df_fitting, mw_fitting_result], ignore_index = True)
                
                except Exception as e:
                    logger.warning(f"Event_{event_id}: An error occurred during calculation for event {event_id}, {e}", exc_info=True)
                    logger.warning(f"  There may be errors during calculation for event {event_id}, check runtime.log file")
                    failed_events += 1
                    continue
                    
            pbar.set_postfix({"Failed": failed_events})
            pbar.update(1)
        sys.stdout.write("Finished..., check the runtime.log file for detail of errors that migth've happened during calculation")
    return df_result, df_fitting, mw_output



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



def get_user_input() -> Tuple[int, int, str, bool]:
    """
    Get user inputs for processing parameters.
    
    Returns:
        Tuple[int, int, str, bool]: Start ID, end ID, output name, and whether to generate figures.
    """
    
    id_start    = get_valid_input("Event ID to start: ", lambda x: int(x) >= 0, "Please input non-negative integer")
    id_end      = get_valid_input("Event ID to end: ", lambda x: int(x) >= id_start, f"Please input an integer >= {id_start}")
    
    while True:
        mw_output   = input("Result file name? (ex. mw_out, press ENTER for default): ").strip() or "mw_output"
        if not any(c in mw_output for c in r'<>:"/\\|?*'):
            break
        print("Enter a valid filename without special characters")
        
    while True:
        figure_statement = input("Do you want to produce the spectral fitting figures [yes/no]?: ").strip().lower()
        if figure_statement in ['yes', 'no']:
            figure_statement = (figure_statement == 'yes')
            break
        print("Pleases enter 'yes' or 'no'")
        
    return id_start, id_end, mw_output, figure_statement



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
    figure_statement: bool = False
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
    rotated_stream = Stream()
    
    # Extract the traces and their metadata
    trace_Z = stream.select(component='Z')[0]
    trace_L_status = trace_Z.stats
    
    trace_N = stream.select(component='N')[0]
    trace_T_status = trace_N.stats
    
    trace_E = stream.select(component='E')[0]
    trace_Q_status = trace_E.stats
    
    # Perform the rotation using the provided azimuth and inclination
    trace_L_data, trace_Q_data, trace_T_data = rotate.rotate_zne_lqt(trace_Z.data, trace_N.data, trace_E.data, azimuth, incidence)
    
    # Convert numpy ndarrays to Trace objects and update their metadata
    trace_L = Trace(trace_L_data, header = trace_L_status)
    trace_L.stats.component = 'L'
    trace_Q = Trace(trace_Q_data, header = trace_Q_status)
    trace_Q.stats.component = 'Q'
    trace_T = Trace(trace_T_data, header = trace_T_status)
    trace_T.stats.component = 'T'

    # Add the rotated traces to the Stream
    rotated_stream.extend([trace_L, trace_Q, trace_T])
    
    return rotated_stream
    
    
    
def window_trace(trace: Trace, P_arr: float, S_arr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    [trace_L, trace_Q, trace_T] = [trace.select(component = comp)[0] for comp in ['L', 'Q', 'T']]
    
    # Dynamic window parameters
    s_p_time = float(S_arr - P_arr)    
    time_after_pick_p = 0.75 * s_p_time
    time_after_pick_s = 1.75 * s_p_time
    
    # Calculate indices for P phase windowing
    p_phase_start_index = int(round( (P_arr - trace_L.stats.starttime )/ trace_L.stats.delta,4)) - \
                            int(round(TIME_BEFORE_PICK  / trace_L.stats.delta,4))
    p_phase_finish_index = int(round((P_arr - trace_L.stats.starttime )/ trace_L.stats.delta,4))+ \
                            int(round(time_after_pick_p / trace_L.stats.delta,4))
                            
    P_data     = trace_L.data[p_phase_start_index : p_phase_finish_index + 1]
    
    # Calculate indices for SV and SH phase windowing
    s_phase_start_index = int(round( (S_arr - trace_Q.stats.starttime )/ trace_Q.stats.delta,4))- \
                            int(round(TIME_BEFORE_PICK / trace_Q.stats.delta,4))
    s_phase_finish_index = int(round((S_arr - trace_Q.stats.starttime )/ trace_Q.stats.delta,4))+ \
                            int(round(time_after_pick_s / trace_Q.stats.delta,4))

    SV_data     = trace_Q.data[s_phase_start_index : s_phase_finish_index + 1]
    SH_data     = trace_T.data[s_phase_start_index : s_phase_finish_index + 1]
    
    # Calculate indices for noise data windowing
    noise_start_index = int(round( (P_arr - trace_L.stats.starttime )/ trace_L.stats.delta,4)) - \
                            int(round( NOISE_DURATION / trace_L.stats.delta,4))
    noise_finish_index  = int(round( (P_arr - trace_L.stats.starttime )/ trace_L.stats.delta,4)) - \
                            int(round( NOISE_PADDING / trace_L.stats.delta,4))

    P_noise  = trace_L.data[noise_start_index : noise_finish_index + 1]
    SV_noise = trace_Q.data[noise_start_index : noise_finish_index + 1]
    SH_noise = trace_T.data[noise_start_index : noise_finish_index + 1]

    return P_data, SV_data, SH_data, P_noise, SV_noise, SH_noise



def trace_snr(data: np.ndarray, noise: np.ndarray) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) based on the RMS (Root Mean Square) of the signal and noise.

    Args:
        data (np.ndarray): Array of signal data.
        noise (np.ndarray): Array of noise data.

    Returns:
        float: The Signal-to-Noise Ratio (SNR), calculated as the ratio of the RMS of the signal to the RMS of the noise.
    """
    
    # Compute RMS of the signal
    data_rms = np.sqrt(np.mean(np.square(data)))
    
    # Compute RMS of the noise
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    
    return data_rms / noise_rms



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
    
    # Ensure the input data is a numpy array
    trace_data = np.asarray(trace_data)
    
    # Check if the trace_data is non-empty
    if len(trace_data) == 0:
        raise ValueError("trace_data cannot be empty.")
    
    # Calculate Power Spectral DENSITY using Welch's method
    frequency, power_spectra = signal.welch(trace_data, sampling_rate, nperseg = len(trace_data))
    displacement_amplitude_ms = np.sqrt(power_spectra)
    displacement_amplitude_nms = displacement_amplitude_ms*1e9 # return displacement amplitude in nms
    
    return frequency, displacement_amplitude_nms



def window_band(frequencies: np.ndarray, spectrums: np.ndarray, f_min: float, f_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a subset of the frequency spectrum within a specified frequency band.

    Args:
        frequencies (np.ndarray): Array of frequency values.
        spectrums (np.ndarray): Array of spectral values corresponding to the frequencies.
        f_min (float): Minimum frequency of the band to extract (inclusive).
        f_max (float): Maximum frequency of the band to extract (inclusive).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - freq: Array of frequencies within the specified band.
            - spec: Array of spectral values corresponding to the extracted frequencies.
    """
    
    # Ensure input arrays are numpy arrays
    frequencies = np.asarray(frequencies)
    spectrums = np.asarray(spectrums)
    
    # Validate that frequencies and spectrums have the same length
    if frequencies.shape != spectrums.shape:
        raise ValueError("frequencies and spectrums must have the same shape.")

    # Extract indices for the specified frequency band
    indices = np.where((frequencies >= f_min) & (frequencies <= f_max))
    
    # Extract the frequencies and spectra within the specified band
    freqs = frequencies[indices]
    specs = spectrums[indices]
    
    return  freqs, specs



def calculate_moment_magnitude(
    wave_path: Path, 
    hypo_df: pd.DataFrame, 
    pick_df: pd.DataFrame, 
    station_df: pd.DataFrame, 
    calibration_path: Path, 
    event_id: int, 
    figure_path: Path, 
    figure_statement: bool = False
    ) -> Tuple[Dict[str, str], Dict[str,List]]:
    
    """
    Calculate the moment magnitude of an earthquake event and generate a spectral fitting profile.

    Args:
        wave_path (Path): Path to the directory containing waveform files.
        hypo_df (pd.DataFrame): DataFrame containing hypocenter information (latitude, longitude, depth).
        pick_df (pd.DataFrame): DataFrame containing pick information (arrival times).
        station_df (pd.DataFrame): DataFrame containing station information (latitude, longitude, elevation).
        calibration_path (Path): Path to the calibration files for instrument response removal.
        event_id (int): Unique identifier for the earthquake event.
        figure_path (Path): Path to save the generated figures.
        figure_statement (bool): Whether to generate and save figures (default is False).

    Returns:
        Tuple[Dict[str, str], Dict[str, List]]:
            - results (Dict[str, str]): A Dictionary containing calculated moment magnitude and related metrics.
            - fitting_result (Dict[str, List]): A dictionary of detailed fitting results for each station.
    """
    
    # initialize figure if needed
    if figure_statement:
        try:
            num_stations = len(pick_df['Station'].unique())
            fig, axs= plt.subplots(num_stations*3, 2, figsize=(20,60))
            plt.subplots_adjust(hspace=0.5)
            axs[0,0].set_title("Phase Window", fontsize='20')
            axs[0,1].set_title("Spectra Fitting Profile", fontsize='20')
            counter = 0
        except Exception as e:
            logger.warning(f"Event_{event_id}: Error initializing figures for event {event_id}: {e}.", exc_info=True)
            fig_statement = False
    
    # Predefined parameters 
    R_PATTERN_P = 0.440 # P phase radiation pattern
    R_PATTERN_S = 0.600 # S phase radiation pattern
    FREE_SURFACE_FACTOR = 2.0
    k_P = 0.32 # kappa parameter for P phase
    k_S = 0.21 # kappa parameter for S phase
    LAYER_TOP   = [ [-3.00,-1.90],[-1.90,-0.59],[-0.59, 0.22],[0.22, 2.50], [2.50, 7.00], [7.00,9.00] , [9.00,15.00], [15.00,33.00], [33.00,9999]]
    VELOCITY_VP = [2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00]                                    # km/s
    VELOCITY_VS = [1.60, 1.79, 2.37, 2.69, 2.99, 3.35, 3.47, 3.83, 4.79]                                    # km/s
    DENSITY     = [ 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700]                   # kg/m3
    

    fitting_result = {
        "ID":[],
        "Station":[],
        "F_corner_P":[],
        "F_corner_SV":[],
        "F_corner_SH":[],
        "Qfactor_P":[],
        "Qfactor_SV":[],
        "Qfactor_SH":[],
        "Omega_0_P_(nms)":[],
        "Omega_0_SV_(nms)":[],
        "Omega_0_SH_(nms)":[],
        "RMS_e_P_(nms)":[],
        "RMS_e_SV_(nms)":[],
        "RMS_e_SH_(nms)":[],
        "Moment_P_(Nm)":[],
        "Moment_S_(Nm)":[]
    }

    # get hypocenter details
    hypo_info = hypo_df.iloc[0]
    origin_time = UTCDateTime(f"{int(hypo_info.Year):04d}-{int(hypo_info.Month):02d}-{int(hypo_info.Day):02d}T{int(hypo_info.Hour):02d}:{int(hypo_info.Minute):02d}:{float(hypo_info.T0):012.9f}") 
    hypo_lat, hypo_lon , hypo_depth =  hypo_info.Lat, hypo_info.Lon, hypo_info.Depth

    # find the correct velocity and DENSITY value for the spesific layer depth
    velocity_P, velocity_S, density_value = None, None, None
    for layer, (top, bottom) in enumerate(LAYER_TOP):
        if (top*1000)   <= hypo_depth <= (bottom*1000):
            velocity_P = VELOCITY_VP[layer]*1000  # velocity in m/s
            velocity_S = VELOCITY_VS[layer]*1000  # velocity in m/s
            density_value = DENSITY[layer]
    if any(value is None for value in [velocity_P, velocity_S, density_value]):
        logger.warning(f"Event_{event_id}: Hypo depth not within the defined layers.")
    
    # start spectrum fitting and magnitude estimation
    moments, corner_frequencies, source_radius = [],[],[]
    for station in list(pick_df.get("Station")):
        # get the station coordinat
        station_xyz_df = station_df[station_df.Stations == station]
        if station_xyz_df.empty:
            logger.warning(f"Event_{event_id}: No staion data for {station}, skipping...")
            continue
        station_xyz_info = station_xyz_df.iloc[0]
        station_lat, station_lon, station_elev = station_xyz_info.Lat, station_xyz_info.Lon, station_xyz_info.Elev
        
        # calculate the source distance and the azimuth (hypo to station azimuth)
        epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, station_lat, station_lon)
        source_distance = np.sqrt(epicentral_distance**2 + ((hypo_depth + station_elev)**2))
        
        # get the pick_df data for P arrival and S arrival
        pick_info_df = pick_df[pick_df.Station == station]
        if pick_info_df.empty:
            logger.warning(f"Event_{event_id}: No pick data for {station}, skipping...")
            continue
        pick_info = pick_info_df.iloc[0]
        P_pick_time = UTCDateTime(
            f"{pick_info.Year}-{int(pick_info.Month):02d}-{int(pick_info.Day):02d}T"
            f"{int(pick_info.Hour):02d}:{int(pick_info.Minutes_P):02d}:{float(pick_info.P_Arr_Sec):012.9f}"
        )
        S_pick_time = UTCDateTime(
            f"{pick_info.Year}-{int(pick_info.Month):02d}-{int(pick_info.Day):02d}T"
            f"{int(pick_info.Hour):02d}:{int(pick_info.Minutes_S):02d}:{float(pick_info.S_Arr_Sec):012.9f}"
        )
        
        # read the waveform 
        stream = read_waveforms(wave_path, event_id, station)
        stream_copy = stream.copy()
        if len(stream_copy) < 3:
            logger.warning(f"Event_{event_id}: Not all components available for station {station} to calculate event {event_id} moment magnitude")
            continue
        
        # perform the instrument removal
        try:
            stream_displacement = instrument_remove(stream_copy, calibration_path, figure_path, figure_statement=False)
            # # if you want to use Martin Mityska code
            # poles_vector = [-8.8857 + 8.8857j, -8.8857 - 8.8857j]
            # zeros_vector = [0, 0]
            # sensitivity = 1/1.57907e9             # sensitivity reftex
            # sensitivity_tim_itb = 1/5.29129e10    # ???
            # sample_rate = 500
            # stream_displacement = remove_alt.instrument_remove_alt(stream_copy, poles_vector, zeros_vector, sensitivity_tim_itb, sample_rate, figure_path, figure_statement = False )
        except Exception as e:
            logger.warning(f"Event_{event_id}: An error occured when correcting instrument for station {station}: {e}", exc_info=True)
            continue
        
        # perform station rotation form ZNE to LQT 
        try:
            try:
                hypo_coordinate = [hypo_lat, hypo_lon , -1*hypo_depth]  # depth must be in negative notation
                station_coordinate = [station_lat, station_lon, station_elev]
                take_off_angle, total_traveltime, incidence_angle = ref.calculate_inc_angle(hypo_coordinate, station_coordinate, LAYER_TOP, VELOCITY_VP)
            except Exception as e:
                logger.warning(f"Event_{event_id}: An error occured when calculating incidence angle for station {station}.", exc_info=True)
            rotated_stream = rotate_component(stream_displacement, azimuth, incidence_angle) # do the component rotation from ZNE to LQT
        except Exception as e:
            logger.warning(f"Event_{event_id}: An error occured when rotating component for station {station}.", exc_info=True)
            continue
 
        # Window the trace
        p_window_data, sv_window_data, sh_window_data, p_noise_data, sv_noise_data, sh_noise_data = window_trace(rotated_stream, P_pick_time, S_pick_time)
        
        # Check the data quality (SNR must be above or equal to 1)
        if any(trace_snr(data, noise) <= 1.15 for data, noise in zip ([p_window_data, sv_window_data, sh_window_data], [p_noise_data, sv_noise_data, sh_noise_data])):
            logger.warning(f"Event_{event_id}: SNR below threshold for station {station} to calculate moment magnitude")
            continue
            
        # check sampling rate
        fs = 1 / rotated_stream[0].stats.delta
        try:
            # calculate source spectra
            freq_P , spec_P  = calculate_spectra(p_window_data, fs)
            freq_SV, spec_SV = calculate_spectra(sv_window_data, fs)
            freq_SH, spec_SH = calculate_spectra(sh_window_data, fs)
            
            # calculate the noise spectra
            freq_N_P,  spec_N_P  = calculate_spectra(p_noise_data, fs)
            freq_N_SV, spec_N_SV = calculate_spectra(sv_noise_data, fs)
            freq_N_SH, spec_N_SH = calculate_spectra(sh_noise_data, fs)
        except Exception as e:
            logger.warning(f"Event_{event_id}: An error occured during spectra calculation for station {station}, {e}.", exc_info=True)
            continue

        # fitting the spectrum, find the optimal value of Omega_O, corner frequency and Q using systematic/stochastic algorithm available
        try:
            fit_P  = fit.fit_spectrum_qmc(freq_P,  spec_P,  abs(float(P_pick_time - origin_time)), F_MIN, F_MAX, 3000)
            fit_SV = fit.fit_spectrum_qmc(freq_SV, spec_SV, abs(float(S_pick_time - origin_time)), F_MIN, F_MAX, 3000)
            fit_SH = fit.fit_spectrum_qmc(freq_SH, spec_SH, abs(float(S_pick_time - origin_time)), F_MIN, F_MAX, 3000)
        except Exception as e:
            logger.warning(f"Event_{event_id}: Error during spectral fitting for event {event_id}, {e}.", exc_info=True)
            continue
        if None in [fit_SV, fit_SH, fit_P]:
            continue

        # fitting spectrum output
        Omega_0_P,  Q_factor_p,  f_c_P,  err_P,  x_fit_P,  y_fit_P  = fit_P
        Omega_0_SV, Q_factor_SV, f_c_SV, err_SV, x_fit_SV, y_fit_SV = fit_SV
        Omega_0_SH, Q_factor_SH, f_c_SH, err_SH, x_fit_SH, y_fit_SH = fit_SH

        # updating the fitting dict handler 
        fitting_result["ID"].append(event_id)
        fitting_result["Station"].append(station)
        fitting_result["F_corner_P"].append(f_c_P)
        fitting_result["F_corner_SV"].append(f_c_SV)
        fitting_result["F_corner_SH"].append(f_c_SH)
        fitting_result["Qfactor_P"].append(Q_factor_p)
        fitting_result["Qfactor_SV"].append(Q_factor_SV)
        fitting_result["Qfactor_SH"].append(Q_factor_SH)
        fitting_result["Omega_0_P_(nms)"].append((Omega_0_P))
        fitting_result["Omega_0_SV_(nms)"].append((Omega_0_SV))
        fitting_result["Omega_0_SH_(nms)"].append((Omega_0_SH))
        fitting_result["RMS_e_P_(nms)"].append((err_P))
        fitting_result["RMS_e_SV_(nms)"].append((err_SV))
        fitting_result["RMS_e_SH_(nms)"].append((err_SH))

        # create figure
        if figure_statement:
            # frequency window for plotting purposes
            f_min_plot = F_MIN
            f_max_plot = F_MAX*1.75
            freq_P, spec_P = window_band(freq_P, spec_P, f_min_plot, f_max_plot)
            freq_SV, spec_SV = window_band(freq_SV, spec_SV, f_min_plot, f_max_plot)
            freq_SH, spec_SH = window_band(freq_SH, spec_SH, f_min_plot, f_max_plot)
            freq_N_P, spec_N_P = window_band(freq_N_P, spec_N_P, f_min_plot, f_max_plot)
            freq_N_SV, spec_N_SV = window_band(freq_N_SV, spec_N_SV, f_min_plot, f_max_plot)
            freq_N_SH, spec_N_SH = window_band(freq_N_SH, spec_N_SH, f_min_plot, f_max_plot)

            # dinamic window parameter
            s_p_time = float(S_pick_time - P_pick_time)    
            time_after_pick_p = 0.80 * s_p_time
            time_after_pick_s = 1.75 * s_p_time
            
            try:
                # plot for phase windowing
                # 1. For P phase or vertical component
                trace_L = rotated_stream.select(component = 'L')[0]
                start_time = trace_L.stats.starttime
                before = (P_pick_time - start_time) - 2.0
                after = (S_pick_time - start_time) + 6.0
                trace_L.trim(start_time+before, start_time+after)
                axs[counter][0].plot(trace_L.times(), trace_L.data, 'k')
                axs[counter][0].axvline( x= (P_pick_time - trace_L.stats.starttime ), color='r', linestyle='-', label='P arrival')
                axs[counter][0].axvline( x= (S_pick_time - trace_L.stats.starttime ), color='b', linestyle='-', label='S arrival')
                axs[counter][0].axvline( x= (P_pick_time - TIME_BEFORE_PICK -  trace_L.stats.starttime), color='g', linestyle='--')
                axs[counter][0].axvline( x= (P_pick_time + time_after_pick_p - trace_L.stats.starttime), color='g', linestyle='--', label='P phase window')
                axs[counter][0].axvline( x= (P_pick_time - NOISE_DURATION -  trace_L.stats.starttime), color='gray', linestyle='--')
                axs[counter][0].axvline( x= (P_pick_time - NOISE_PADDING  - trace_L.stats.starttime), color='gray', linestyle='--', label='Noise window')
                axs[counter][0].set_title("{}_BH{}".format(trace_L.stats.station, trace_L.stats.component), loc="right",va='center')
                axs[counter][0].legend()
                axs[counter][0].set_xlabel("Relative Time (s)")
                axs[counter][0].set_ylabel("Amp (m)")
               
                # 2. For SV phase or radial component
                axis = counter + 1
                trace_Q = rotated_stream.select(component = 'Q')[0]
                start_time = trace_Q.stats.starttime
                before = (P_pick_time - start_time) - 2.0
                after = (S_pick_time - start_time) + 6.0
                trace_Q.trim(start_time+before, start_time+after)
                axs[counter+1][0].plot(trace_Q.times(), trace_Q.data, 'k')
                axs[counter+1][0].axvline( x= (P_pick_time - trace_Q.stats.starttime ), color='r', linestyle='-', label='P arrival')
                axs[counter+1][0].axvline( x= (S_pick_time - trace_Q.stats.starttime), color='b', linestyle='-', label='S arrival')
                axs[counter+1][0].axvline( x= (S_pick_time - TIME_BEFORE_PICK -  trace_Q.stats.starttime  ), color='g', linestyle='--')
                axs[counter+1][0].axvline( x= (S_pick_time + time_after_pick_s - trace_Q.stats.starttime ), color='g', linestyle='--', label='SV phase window')
                axs[counter+1][0].axvline( x= (P_pick_time - NOISE_DURATION -  trace_Q.stats.starttime), color='gray', linestyle='--')
                axs[counter+1][0].axvline( x= (P_pick_time - NOISE_PADDING  - trace_Q.stats.starttime), color='gray', linestyle='--', label='Noise window')
                axs[counter+1][0].set_title("{}_BH{}".format(trace_Q.stats.station, trace_Q.stats.component), loc="right",va='center')
                axs[counter+1][0].legend()
                axs[counter+1][0].set_xlabel("Relative Time (s)")
                axs[counter+1][0].set_ylabel("Amp (m)")
                
                # 3. For SH phase or transverse component
                trace_T = rotated_stream.select(component = 'T')[0]
                start_time = trace_T.stats.starttime
                before = (P_pick_time - start_time) - 2.0
                after = (S_pick_time - start_time) + 6.0
                trace_T.trim(start_time+before, start_time+after)
                axs[counter+2][0].plot(trace_T.times(), trace_T.data, 'k')
                axs[counter+2][0].axvline( x= (P_pick_time - trace_T.stats.starttime ), color='r', linestyle='-', label='P arrival')
                axs[counter+2][0].axvline( x= (S_pick_time - trace_T.stats.starttime), color='b', linestyle='-', label='S arrival')
                axs[counter+2][0].axvline( x= (S_pick_time - TIME_BEFORE_PICK -  trace_T.stats.starttime  ), color='g', linestyle='--')
                axs[counter+2][0].axvline( x= (S_pick_time + time_after_pick_s - trace_T.stats.starttime ), color='g', linestyle='--', label='SH phase window')
                axs[counter+2][0].axvline( x= (P_pick_time - NOISE_DURATION -  trace_T.stats.starttime), color='gray', linestyle='--')
                axs[counter+2][0].axvline( x= (P_pick_time - NOISE_PADDING  - trace_T.stats.starttime), color='gray', linestyle='--', label='Noise window')
                axs[counter+2][0].set_title("{}_BH{}".format(trace_T.stats.station, trace_T.stats.component), loc="right",va='center')
                axs[counter+2][0].legend()
                axs[counter+2][0].set_xlabel("Relative Time (s)")
                axs[counter+2][0].set_ylabel("Amp (m)")
               
                # plot the spectra (P, SV, SH and Noise spectra)
                # 1. For P spectra
                axs[counter][1].loglog(freq_P, spec_P, color='black', label='P spectra')
                axs[counter][1].loglog(freq_N_P, spec_N_P, color='gray', label='Noise spectra')
                axs[counter][1].loglog(x_fit_P, y_fit_P, 'b-', label='Fitted P Spectra')
                axs[counter][1].set_title("{}_BH{}".format(trace_L.stats.station, trace_L.stats.component), loc="right",va='center')
                axs[counter][1].legend()
                axs[counter][1].set_xlabel("Frequencies (Hz)")
                axs[counter][1].set_ylabel("Amp (nms)")
               
               
                # 2. For SV spectra
                axs[counter+1][1].loglog(freq_SV, spec_SV, color='black', label='SV spectra')
                axs[counter+1][1].loglog(freq_N_SV, spec_N_SV, color='gray', label='Noise spectra')
                axs[counter+1][1].loglog(x_fit_SV, y_fit_SV, 'b-', label='Fitted SV Spectra')
                axs[counter+1][1].set_title("{}_BH{}".format(trace_Q.stats.station, trace_Q.stats.component), loc="right",va='center')
                axs[counter+1][1].legend()
                axs[counter+1][1].set_xlabel("Frequencies (Hz)")
                axs[counter+1][1].set_ylabel("Amp (nms)")
                
                
                # 3. For SH spectra
                axs[counter+2][1].loglog(freq_SH, spec_SH, color='black', label='SH spectra')
                axs[counter+2][1].loglog(freq_N_SH, spec_N_SH, color='gray', label='Noise spectra')
                axs[counter+2][1].loglog(x_fit_SH, y_fit_SH, 'b-', label='Fitted SH Spectra')
                axs[counter+2][1].set_title("{}_BH{}".format(trace_T.stats.station, trace_T.stats.component), loc="right",va='center')
                axs[counter+2][1].legend()
                axs[counter+2][1].set_xlabel("Frequencies (Hz)")
                axs[counter+2][1].set_ylabel("Amp (nms)")

                counter +=3
            except Exception as e:
                logger.warning(f"Event_{event_id}: Failed to plot the fitting spectral for station {station}, {e}.")
                continue

        # calculate the moment magnitude
        try:
            # calculate the  resultant omega
            omega_P = Omega_0_P*1e-9
            omega_S = ((Omega_0_SV**2 + Omega_0_SH**2)**0.5)*1e-9
         
            # calculate seismic moment
            M_0_P = (4.0 * np.pi * density_value * (velocity_P ** 3) * source_distance * omega_P) / (R_PATTERN_P * FREE_SURFACE_FACTOR)
            M_0_S = (4.0 * np.pi * density_value * (velocity_S ** 3) * source_distance * omega_S) / (R_PATTERN_S * FREE_SURFACE_FACTOR)
            fitting_result["Moment_P_(Nm)"].append(M_0_P)
            fitting_result["Moment_S_(Nm)"].append(M_0_S)
            
            # calculate average seismic moment at station
            average_moment = (M_0_P + M_0_S)/2
            moments.append(average_moment)
            
            # calculate source radius
            r_P = (k_P * velocity_P)/f_c_P
            r_S = (2 * k_S * velocity_S)/(f_c_SV + f_c_SH)
            r_average = (r_P + r_S)/2
            source_radius.append(r_average)
            
            # calculate corner frequency mean
            corner_freq_S = (f_c_SV + f_c_SH)/2
            corner_freq_avg = (f_c_P + corner_freq_S)/2
            corner_frequencies.append(corner_freq_avg)

        except Exception as e:
            logger.warning(f" Event_{event_id}: Failed to calculate seismic moment for event {event_id}, {e}.")
            continue
            
    # calculate average and std of moment magnitude
    moment_average = np.mean(moments)
    moment_std = np.std(moments)
    mw = ((2.0 / 3.0) * np.log10(moment_average)) - 6.07
    mw_std = (2.0 /3.0) * moment_std/(moment_average * np.log(10))
    
    # calculate the average and std of corner frequency
    corner_frequency_average = np.mean(corner_frequencies)
    corner_frequency_std = np.std(corner_frequencies)

    # calculate the average and std of source radius
    source_radius_average = np.mean(source_radius)
    source_radius_std = np.std(source_radius)
    
    # calculate the stress drop
    stress_drop = (7 * moment_average) / (16 * source_radius_average** 3) *1e-5
 
    results = {"ID":[f"{event_id}"], 
                "Fc_avg":[f"{corner_frequency_average:.3f}"],
                "Fc_std":[f"{corner_frequency_std:.3f}"],
                "Src_rad_avg_(m)":[f"{source_radius_average:.3f}"],
                "Src_rad_std_(m)":[f"{source_radius_std:.3f}"],
                "Stress_drop_(bar)":[f"{stress_drop:.3f}"],
                "Mw_average":[f"{mw:.3f}"],
                "Mw_std":[f"{mw_std:.3f}"]
                }
                
    if figure_statement and 'fig' in locals() : 
        fig.suptitle(f"Event {event_id} {mw:.3f}Mw Spesctral Fitting Profile", fontsize='24', fontweight='bold')
        #plt.title("Event {} Spectral Fitting Profile".format(ID), fontsize='20')
        plt.savefig(figure_path.joinpath(f"event_{event_id}.png"))
        plt.close(fig)
    
    return results, fitting_result



def main():
    # path object containing all necessary paths
    paths = {
        "hypo_input": Path(r"E:\BACK UP KERJA\SUPREME ENERGY\SERD\SERD Catalog\hypo_relocated.xlsx"),
        "pick_input": Path(r"E:\BACK UP KERJA\SUPREME ENERGY\SERD\PICKING SERD\2024\all combined\all_serd_2024.xlsx"),
        "station_input": Path(r"E:\BACK UP KERJA\SUPREME ENERGY\SERD\SERD Station\SERD_station.xlsx"),
        "wave_dir": Path(r"E:\BACK UP KERJA\SUPREME ENERGY\SERD\DATA TRIMMING\combined"),
        "cal_dir": Path(r"E:\BACK UP KERJA\SUPREME ENERGY\SERD\calibration file"),
        "mw_result_dir": Path(r"E:\BACK UP KERJA\SUPREME ENERGY\SERD\Magnitude Calculation\MW"),
        "fig_output_dir": Path(r"E:\BACK UP KERJA\SUPREME ENERGY\SERD\Magnitude Calculation\MW\fig_out")
    }

    # cek input integrity
    for path_name, path_dir in paths.items():
        if "input" in path_name and not path_dir.exists():
            raise FileNotFoundError(f"{path_name} not found: {path_dir}")
        if path_name in ["wave_dir", "cal_dir"] and not path_dir.is_dir():
            raise NotADirectoryError(f"{path_name} must be a directory: {path_dir}")
        if path_name in ["mw_result_dir", "fig_output_dir"]:
            path_dir.mkdir(parents=True, exist_ok=True)
            
    # preload all the excel file
    try:
        hypo_df = pd.read_excel(paths["hypo_input"], index_col=None)
        pick_df = pd.read_excel(paths["pick_input"], index_col=None)
        station_df = pd.read_excel(paths["station_input"], index_col=None)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load Excel files: {e}")
        
    # Call the function to start calculating moment magnitude
    mw_result_df, mw_fitting_df, output_name = start_calculate(paths["wave_dir"], paths["cal_dir"], paths["fig_output_dir"], 
                                                               hypo_df, pick_df, station_df)

    # save and set dataframe index
    mw_result_df.to_excel(paths["mw_result_dir"].joinpath(f"{output_name}_result.xlsx"), index = False)
    mw_fitting_df.to_excel(paths["mw_result_dir"].joinpath(f"{output_name}_fitting_result.xlsx"), index = False)
    
    return None



if __name__ == "__main__" :
    main()
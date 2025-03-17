#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:32:03 2022.
Python code to calculate moment magnitude.


Developed by arham zakki edelo.


contact: 
- edelo.arham@gmail.com
- https://github.com/bgjx

Pre-requisite modules:
->[pathlib, tqdm, numpy, pandas, obspy, scipy] 

"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd 
from obspy import Stream, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel
from scipy import signal
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

import LQTMomentMag.fitting_spectral as fit
import LQTMomentMag.refraction as ref 
from .config import CONFIG
from .plotting import plot_spectral_fitting
from .utils import get_user_input, instrument_remove, read_waveforms, trace_snr


logger = logging.getLogger("mw_calculator")


def calculate_spectra(
    trace_data: np.ndarray,
    sampling_rate: float,
    smooth_window: int = None,
    apply_window: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the amplitude spectrum of a given signal using the Fourier transform,
    with optional smoothing and windowing.

    Args:
        trace_data (np.ndarray): Array of displacement signal ( in meters).
        sampling_rate (float): Sampling rate of the signal in Hz.
        smooth_window (int, optional): Window size for moving average smoothing.
            if None, no smoothing is applied.
        apply_window (bool, optional): Apply a Hann window to reduce spectral leakage.
            Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - frequencies: Array of sample frequencies in Hz.
            - displacement_amplitudes: Array of displacement amplitudes in nm.
    Raises:
        ValueError: If trace_data is empty.
    """

    if not trace_data.data.size:
        raise ValueError("Trace data cannot be empty")
    
    n_samples = len(trace_data)

    # Apply Hann window to reduce spectral leakage
    if apply_window:
        window = signal.windows.hann(n_samples)
        trace_data_windowed = trace_data * window
    else:
        trace_data_windowed = trace_data
    
    # Compute the FFT
    frequencies = rfftfreq(n_samples, 1 / sampling_rate)
    fft_data = rfft(trace_data_windowed)
    displacement_amplitudes = np.abs(fft_data) * 1e9

    # Normalize by the window to account for amplitude reduction
    if apply_window:
        window_sum = np.mean(signal.windows.hann(n_samples)) * n_samples
        displacement_amplitudes /= window_sum
    
    # Apply smoothing if specified
    if smooth_window is not None and smooth_window > 1:
        window_smooth = signal.windows.boxcar(smooth_window)
        displacement_amplitudes = np.convolve(displacement_amplitudes, window_smooth / window_smooth.sum(), mode='same')
    
    return frequencies, displacement_amplitudes


def window_trace(streams: Stream, P_arr: float, S_arr: float) -> Tuple[np.ndarray, ...]:
    """
    Windows seismic trace data around P, SV, and SH phase and extracts noise data.

    Args:
    
        streams (Stream): A stream object containing the seismic data.
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
    [trace_L, trace_Q, trace_T] = [streams.select(component = comp)[0] for comp in ['L', 'Q', 'T']]
    
    # Dynamic window parameters
    s_p_time = S_arr - P_arr    
    time_after_pick_p = 0.75 * s_p_time
    time_after_pick_s = 1.75 * s_p_time
    
    # Find the data index for phase windowing
    p_phase_start_index = int(round((P_arr - trace_L.stats.starttime - CONFIG.magnitude.PADDING_BEFORE_ARRIVAL)/trace_L.stats.delta), 4)
    p_phase_end_index = int(round((P_arr - trace_L.stats.starttime + time_after_pick_p )/trace_L.stats.delta, 4))
    s_phase_start_index = int(round((S_arr - trace_Q.stats.starttime - CONFIG.magnitude.PADDING_BEFORE_ARRIVAL)/trace_Q.stats.delta), 4)
    s_phase_end_index = int(round((S_arr - trace_Q.stats.starttime + time_after_pick_s )/ trace_Q.stats.delta, 4))
    noise_start_index = int(round((P_arr - trace_L.stats.starttime - CONFIG.magnitude.NOISE_DURATION)/trace_L.stats.delta, 4))                             
    noise_end_index  = int(round((P_arr - trace_L.stats.starttime - CONFIG.magnitude.NOISE_PADDING )/trace_L.stats.delta, 4))

    # Window the data by the index
    P_data     = trace_L.data[p_phase_start_index : p_phase_end_index + 1]
    SV_data     = trace_Q.data[s_phase_start_index : s_phase_end_index + 1]
    SH_data     = trace_T.data[s_phase_start_index : s_phase_end_index + 1]

    P_noise  = trace_L.data[noise_start_index : noise_end_index + 1]
    SV_noise = trace_Q.data[noise_start_index : noise_end_index + 1]
    SH_noise = trace_T.data[noise_start_index : noise_end_index + 1]

    return P_data, SV_data, SH_data, P_noise, SV_noise, SH_noise


def calculate_moment_magnitude(
    wave_path: Path, 
    source_df: pd.DataFrame, 
    pick_df: pd.DataFrame,
    calibration_path: Path, 
    source_id: int, 
    figure_path: Path, 
    figure_statement: bool = False,
    lqt_mode: bool = True
    ) -> Tuple[Dict[str, str], Dict[str,List]]:
    
    """
    This function processes moment magnitude calculation for an earthquake from given
    hypocenter dataframe and picking dataframe. This function handle the waveform instrument 
    response removal, seismogram rotation, spectral fitting, moment magnitude calculation, and
    figure creation. It return two dictionary objects, magnitude and fitting result. 
    
    The whole process in this function following these steps:
    1. Remove instrument response using calibration file.
    2. Rotate waveforms from ZNE to LQT (or ZRT for non-LQT mode) based on earthquake type.
    3. Window P and S waves, compute their spectra, and fit spectral parameters (
        corner_frequency, omega_0, q_factor) using optimized algorithm (default: QMC).
    4. Calculate seismic moment (M_0) using the formula:
        M_0 = (4 * pi * rho * v^3 * r * Omega_0) / (R * F),
        where rho is density, v is wave velocity, r is distance, R is radiation pattern, and F is free surface factor.
    5. Compute moment magnitude (Mw) using.
        Mw = (2/3) * (log10(M_0) - 6.07), where M_0 is in Nm.
       

    Args:
        wave_path (Path): Path to the directory containing waveform files.
        source_df (pd.DataFrame): DataFrame containing hypocenter information (latitude, longitude, depth).
        pick_df (pd.DataFrame): DataFrame containing pick information (arrival times).
        calibration_path (Path): Path to the calibration files for instrument response removal.
        source_id (int): Unique identifier for the earthquake.
        figure_path (Path): Path to save the generated figures.
        figure_statement (bool): Boolean statement to generate and save figures (default is False).
        lqt_mode (bool): If True, perform LQT rotation; otherwise, use ZRT for very local earthquakes
                            default to True.

    Returns:
        Tuple[Dict[str, str], Dict[str, List]]:
            - results (Dict[str, str]): A Dictionary containing calculated moment magnitude and related metrics.
            - fitting_result (Dict[str, List]): A dictionary of detailed fitting results for each station.
    
    Raises:
        ValueError: if source_df or pick_df are empty or wrong format.
        IOError: If waveform or calibration files cannot be read.
    """ 

    # Validate all config parameter before doing calculation
    required_config = [
        "LAYER_BOUNDARIES", "VELOCITY_VP", "VELOCITY_VS", "DENSITY", "SNR_THRESHOLD"
    ]
    missing_config = [attr for attr in required_config if not hasattr(CONFIG.magnitude, attr)]
    if missing_config:
        logger.error(f"Earthquake_{source_id}: Missing config attributes: {missing_config}")
        raise ValueError(f"Missing config attributes: {missing_config}")
    
    # Create object collector for fitting result
    fitting_result = {
        "source_id":[],
        "station":[],
        "f_corner_p":[],
        "f_corner_sv":[],
        "f_corner_sh":[],
        "q_factor_p":[],
        "q_factor_sv":[],
        "q_factor_sh":[],
        "omega_0_p_nms":[],
        "omega_0_sv_nms":[],
        "omega_0_sh_nms":[],
        "rms_e_p_nms":[],
        "rms_e_sv_nms":[],
        "rms_e_sh_nms":[],
        "moment_p_Nm":[],
        "moment_s_Nm":[]
    }

    # Create object collector for plotting
    if figure_statement:
        all_streams, all_p_times, all_s_times = [], [], []
        all_freqs = {
            "P": [],  "SV":[], "SH":[], "N_P":[], "N_SV":[], "N_SH":[] 
        }
        all_specs = {
            "P": [],  "SV":[], "SH":[], "N_P":[], "N_SV":[], "N_SH":[] 
        }
        all_fits = {
            "P":[], "SV":[], "SH":[]
        }
        station_names = []

    # Get hypocenter details
    source_info = source_df.iloc[0]
    source_origin_time = UTCDateTime(source_info.source_origin_time)
    source_lat, source_lon , source_depth_m =  source_info.source_lat, source_info.source_lon, source_info.source_depth_m
    source_type = source_info.earthquake_type

    # Find the correct velocity and DENSITY value for the specific layer depth
    velocity_P, velocity_S, density_value = None, None, None
    for (top, bottom), vp, vs, rho in zip(CONFIG.magnitude.LAYER_BOUNDARIES, CONFIG.magnitude.VELOCITY_VP, CONFIG.magnitude.VELOCITY_VS, CONFIG.magnitude.DENSITY):
        if (top*1000)   <= source_depth_m <= (bottom*1000):
            velocity_P, velocity_S, density_value = vp*1000, vs*1000, rho
            break
        else:
            logger.warning(f"Earthquake_{source_id}: Hypocenter depth not within the defined layers.")
            return {}, fitting_result
    
    # Start spectrum fitting and magnitude estimation
    moments, corner_frequencies, source_radius = [],[],[]
    for station in pick_df.get("station_code").unique():
        # Get the station coordinate
        station_info = pick_df[pick_df.station_code == station].iloc[0]
        station_lat, station_lon, station_elev_m = station_info.station_lat, station_info.station_lon, station_info.station_elev_m
        p_arr_time = UTCDateTime(station_info.p_arr_time)
        s_arr_time = UTCDateTime(station_info.s_arr_time)
        
        # Calculate the source distance and the azimuth (hypo to station azimuth)
        epicentral_distance, azimuth, _ = gps2dist_azimuth(source_lat, source_lon, station_lat, station_lon)
        source_distance_m = np.sqrt(epicentral_distance**2 + ((source_depth_m + station_elev_m)**2))
        source_distance_degrees = locations2degrees(source_lat, source_lon, station_lat, station_lon)
            
        # Read the waveform 
        stream = read_waveforms(wave_path, source_id, station)
        stream_copy = stream.copy()
        if len(stream_copy) < 3:
            logger.warning(f"Earthquake_{source_id}: Not all components available for station {station} to calculate earthquake {source_id} moment magnitude")
            continue
        
        # Perform the instrument removal
        try:
            stream_displacement = instrument_remove(stream_copy, calibration_path, figure_path, figure_statement=False)
        except (ValueError, IOError) as e:
            logger.warning(f"Earthquake_{source_id}: An error occurred when correcting instrument for station {station}: {e}", exc_info=True)
            continue
        
        # Perform station rotation form ZNE to LQT in earthquake type dependent
        source_coordinate = [source_lat, source_lon , -1*source_depth_m]  # depth must be in negative notation
        station_coordinate = [station_lat, station_lon, station_elev_m]
        try:
            if source_type == 'very_local_earthquake' and not lqt_mode:
                stream_zrt = stream_displacement.copy()
                stream_zrt.rotate(method="NE->RT", back_azimuth=azimuth)
                p, sv, sh = stream_zrt.traces # Z, R, T components
            elif source_type =='teleseismic_earthquake':
                model = TauPyModel(model="iasp91")
                arrivals = model.get_travel_times(
                    source_depth_in_km=(source_depth_m/1e3),
                    distance_in_degree=source_distance_degrees,
                    phase_list=["P", "S"]
                )
                incidence_angle_p = arrivals[0].incident_angle
                incidence_angle_s = arrivals[1].incident_angle
                stream_lqt_p = stream_displacement.copy()
                stream_lqt_s = stream_displacement.copy()
                stream_lqt_p.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p)
                stream_lqt_s.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s)
                p, _, _ = stream_lqt_p.traces # L, Q, T components
                _, sv, sh = stream_lqt_s.traces
            else:
                _, _, incidence_angle_p = ref.calculate_inc_angle(source_coordinate, station_coordinate,
                                                                CONFIG.magnitude.LAYER_BOUNDARIES,
                                                                CONFIG.magnitude.VELOCITY_VP)
                _, _, incidence_angle_s = ref.calculate_inc_angle(source_coordinate, station_coordinate,
                                                                CONFIG.magnitude.LAYER_BOUNDARIES,
                                                                CONFIG.magnitude.VELOCITY_VS)
                stream_lqt_p = stream_displacement.copy()
                stream_lqt_s = stream_displacement.copy()
                stream_lqt_p.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p)
                stream_lqt_s.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s)
                p, _, _ = stream_lqt_p.traces # L, Q, T components
                _, sv, sh = stream_lqt_s.traces
            rotated_stream = Stream(traces=[p, sv, sh])
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: An error occurred when rotating component for station {station}.", exc_info=True)
            continue
 
        # Window the trace
        p_window_data, sv_window_data, sh_window_data, p_noise_data, sv_noise_data, sh_noise_data = window_trace(rotated_stream, p_arr_time, s_arr_time)
        
        # Check the data quality (SNR must be above or equal to 1)
        snr_threshold = CONFIG.magnitude.SNR_THRESHOLD
        if any(trace_snr(data, noise) <= snr_threshold for data, noise in zip ([p_window_data, sv_window_data, sh_window_data], [p_noise_data, sv_noise_data, sh_noise_data])):
            logger.warning(f"Earthquake_{source_id}: SNR below threshold for station {station} to calculate moment magnitude")
            continue
            
        # check sampling rate
        fs = 1 / rotated_stream[0].stats.delta
        try:
            # Calculate source spectra
            freq_P , spec_P  = calculate_spectra(p_window_data, fs)
            freq_SV, spec_SV = calculate_spectra(sv_window_data, fs)
            freq_SH, spec_SH = calculate_spectra(sh_window_data, fs)
            
            # Calculate the noise spectra
            freq_N_P,  spec_N_P  = calculate_spectra(p_noise_data, fs)
            freq_N_SV, spec_N_SV = calculate_spectra(sv_noise_data, fs)
            freq_N_SH, spec_N_SH = calculate_spectra(sh_noise_data, fs)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: An error occurred during spectra calculation for station {station}, {e}.", exc_info=True)
            continue

        # Fitting the spectrum, find the optimal value of Omega_O, corner frequency and Q using systematic/stochastic algorithm available
        try:
            fit_P  = fit.fit_spectrum_qmc(freq_P,  spec_P,  abs(float(p_arr_time - source_origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
            fit_SV = fit.fit_spectrum_qmc(freq_SV, spec_SV, abs(float(s_arr_time - source_origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
            fit_SH = fit.fit_spectrum_qmc(freq_SH, spec_SH, abs(float(s_arr_time - source_origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: Error during spectral fitting for event {source_id}, {e}.", exc_info=True)
            continue
        if any(f is None for f in [fit_P, fit_SV, fit_SH]):
            continue

        # Extract fitting spectrum output
        Omega_0_P,  Q_factor_p,  f_c_P,  err_P,  x_fit_P,  y_fit_P  = fit_P
        Omega_0_SV, Q_factor_SV, f_c_SV, err_SV, x_fit_SV, y_fit_SV = fit_SV
        Omega_0_SH, Q_factor_SH, f_c_SH, err_SH, x_fit_SH, y_fit_SH = fit_SH

        # Updating the fitting object collector 
        fitting_result["id"].append(source_id)
        fitting_result["station"].append(station)
        fitting_result["f_corner_p"].append(f_c_P)
        fitting_result["f_corner_sv"].append(f_c_SV)
        fitting_result["f_corner_sh"].append(f_c_SH)
        fitting_result["q_factor_p"].append(Q_factor_p)
        fitting_result["q_factor_sv"].append(Q_factor_SV)
        fitting_result["q_factor_sh"].append(Q_factor_SH)
        fitting_result["omega_0_p_nms"].append((Omega_0_P))
        fitting_result["omega_0_sv_nms"].append((Omega_0_SV))
        fitting_result["omega_0_sh_nms"].append((Omega_0_SH))
        fitting_result["rms_e_p_nms"].append((err_P))
        fitting_result["rms_e_sv_nms"].append((err_SV))
        fitting_result["rms_e_sh_nms"].append((err_SH))


        # Calculate the moment magnitude
        try:
            # Calculate the  resultant omega
            omega_P = Omega_0_P*1e-9
            omega_S = ((Omega_0_SV**2 + Omega_0_SH**2)**0.5)*1e-9
         
            # Calculate seismic moment
            M_0_P = (4.0 * np.pi * density_value * (velocity_P ** 3) * source_distance_m * omega_P) / (CONFIG.magnitude.R_PATTERN_P * CONFIG.magnitude.FREE_SURFACE_FACTOR)
            M_0_S = (4.0 * np.pi * density_value * (velocity_S ** 3) * source_distance_m * omega_S) / (CONFIG.magnitude.R_PATTERN_S * CONFIG.magnitude.FREE_SURFACE_FACTOR)
            fitting_result["Moment_p_Nm"].append(M_0_P)
            fitting_result["Moment_s_Nm"].append(M_0_S)
            
            # Calculate average seismic moment at station
            moments.append((M_0_P + M_0_S)/2)
            
            # Calculate source radius
            r_P = (CONFIG.magnitude.K_P * velocity_P)/f_c_P
            r_S = (2 * CONFIG.magnitude.K_S * velocity_S)/(f_c_SV + f_c_SH)
            source_radius.append((r_P + r_S)/2)
            
            # Calculate corner frequency mean
            corner_freq_S = (f_c_SV + f_c_SH)/2
            corner_frequencies.append((f_c_P + corner_freq_S)/2)

        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f" Earthquake_{source_id}: Failed to calculate seismic moment for earthquake {source_id}, {e}.", exc_info=True)
            continue
        
        
        # Update fitting spectral object collectro for plotting
        if figure_statement:
            all_streams.append(rotated_stream)
            all_p_times.append(p_arr_time)
            all_s_times.append(s_arr_time)
            all_freqs["P"].append(freq_P)
            all_freqs["SV"].append(freq_SV)
            all_freqs["SH"].append(freq_SH)
            all_freqs["N_P"].append(freq_N_P)
            all_freqs["N_SV"].append(freq_N_SV)
            all_freqs["N_SH"].append(freq_N_SH)
            all_specs["P"].append(spec_P)
            all_specs["SV"].append(spec_SV)
            all_specs["SH"].append(spec_SH)
            all_specs["N_P"].append(spec_N_P)
            all_specs["N_SV"].append(spec_N_SV)
            all_specs["N_SH"].append(spec_N_SH)
            all_fits["P"].append(fit_P)
            all_fits["SV"].append(fit_SV)
            all_fits["SH"].append(fit_SH)
            station_names.append(station)

    if not moments:
        return {}, fitting_result
    
    # Calculate average and std of moment magnitude
    moment_average, moment_std  = np.mean(moments), np.std(moments)
    mw = ((2.0 / 3.0) * np.log10(moment_average)) - 6.07
    mw_std = (2.0 /3.0) * moment_std/(moment_average * np.log(10))
 
    results = {"ID":[f"{source_id}"], 
                "Fc_avg":[f"{np.mean(corner_frequencies):.3f}"],
                "Fc_std":[f"{np.std(corner_frequencies):.3f}"],
                "Src_rad_avg_(m)":[f"{np.mean(source_radius):.3f}"],
                "Src_rad_std_(m)":[f"{np.std(source_radius):.3f}"],
                "Stress_drop_(bar)":[f"{(7 * moment_average) / (16 * np.mean(source_radius)** 3) *1e-5:.3f}"],
                "Mw_average":[f"{mw:.3f}"],
                "Mw_std":[f"{mw_std:.3f}"]
                }
    
    # Create fitting spectral plot
    if figure_statement and all_streams:
        try:
            plot_spectral_fitting(source_id, all_streams, all_p_times, all_s_times, all_freqs, all_specs, all_fits, station_names, figure_path)
        except (ValueError, IOError) as e:
            logger.warning(f"Earthquake_{source_id}: Failed to create spectral fitting plot for event {source_id}, {e}.", exc_info=True)
    
    return results, fitting_result


def start_calculate(
    wave_path: Path,
    calibration_path: Path,
    figure_path: Path,
    catalog_data: pd.DataFrame
    ) -> Tuple [pd.DataFrame, pd.DataFrame, str]:

    """
    This function processes moment magnitude calculation by iterating over a user-specified range
    of earthquake IDs. For each event of earthquake, it extracts source and station data, and 
    computes moment magnitudes using waveform and response file, and aggregates results into
    two DataFrames: magnitude results and spectral fitting parameters.
    
    Args:
        wave_path (Path): Path to the directory containing waveforms file (.miniSEED format).
        calibration_path (Path) : Path to the directory containing calibration file (.RESP format).
        figure_path (Path) : Path to the directory where spectral fitting figures will be saved.
        catalog_data (pd.DataFrame): Catalog DataFrame in LQTMomentMag format.
        
    Returns:
        Tuple [pd.Dataframe, pd.DataFrame, str]:
            - First DataFrame: Magnitude results with columns ['source_id', 'fc_avg', 'fc_std', ...].
            - Second DataFrame: Fitting results with columns ['source_id', 'station', 'f_corner_p', ...].
            - Output filename (str) for saving results.
    
    Raises:
        ValueError: If catalog_data is empty or missing required columns.
    
    Example:
        >>> catalog = pd.read_excel("lqt_catalog.xlsx")
        >>> result_df, fitting_df, output_name = start_calculate(
        ...     Path("data/waveforms"), Path("data/calibration"),
        ...     Path("figures"), catalog)
    """
    
    # Validate catalog columns
    required_columns = [
        "source_id", "source_lat", "source_lon", "source_depth_m",
        "source_origin_time", "earthquake_type", "station_code", 
        "station_lat", "station_lon", "station_elev_m", "p_arr_time",
        "s_arr_time" 
    ]

    missing_columns = [col for col in required_columns if col not in catalog_data.columns]
    if missing_columns:
        logger.error(f"Catalog missing required columns: {missing_columns}")
        raise ValueError(f"catalog missing required columns: {missing_columns}")
    
    # Get the user input.
    id_start, id_end, mw_output, figure_statement, lqt_mode = get_user_input()

    # Initiate dataframe for magnitude calculation results
    df_result = pd.DataFrame(
            columns=["source_id", "fc_avg", "fc_std", "Src_rad_avg_m",
                    "Src_rad_std_m", "Stress_drop_bar",
                    "mw_average", "mw_std"] 
                        )
    df_fitting = pd.DataFrame(
            columns=["source_id", "station", "f_corner_p", "f_corner_sv",
                    "f_corner_sh", "q_factor_p", "q_factor_sv", "q_factor_sh",
                    "omega_0_P_nms", "omega_0_sv_nms", "omega_0_sh_nms",
                    "rms_e_p_nms", "rms_e_sv_nms", "rms_e_sh_nms",
                    "moment_p_Nm", "moment_s_Nm"] 
                        )

    failed_events=0
    result_list = []
    fitting_list = []

    # Pre-grouping catalog by the id for efficiency
    grouped_data = catalog_data.groupby("source_id")
    total_earthquakes = id_end - id_start + 1
    with tqdm(
        total = total_earthquakes,
        file=sys.stderr,
        position=0,
        leave=True,
        desc="Processing earthquakes",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ncols=80,
        smoothing=0.1
    ) as pbar:
        for source_id in range (id_start, id_end + 1):
            logging.info(f"Earthquake_{source_id}: Calculating moment magnitude for earthquakes ID {source_id}")
            
            # Extract data for the current event
            try:
                catalog_data_handler = grouped_data.get_group(source_id)
            except KeyError:
                logger.warning(f"Earthquake_{source_id}: No data for earthquake ID {source_id}")
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            source_data_handler = catalog_data_handler[["source_lat", "source_lon", 
                                                    "source_depth_m", "source_origin_time", 
                                                    "earthquake_type"]].drop_duplicates()
            pick_data_handler = catalog_data_handler[["station_code", "station_lat",
                                                      "station_lon", "station_elev_m",
                                                      "p_arr_time", "s_arr_time"]].drop_duplicates()
            
            # Check for  empty data frame
            if source_data_handler.empty or pick_data_handler.empty:
                logger.warning(f"Earthquake_{source_id}: No data for earthquake {source_id}")
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            # Calculate the moment magnitude
            try:
                mw_results, fitting_result = calculate_moment_magnitude(
                                            wave_path, source_data_handler,
                                            pick_data_handler, calibration_path,
                                            source_id, figure_path,
                                            figure_statement, lqt_mode
                                            )
                result_list.append(pd.DataFrame.from_dict(mw_results))
                fitting_list.append(pd.DataFrame.from_dict(fitting_result))
            except (ValueError, IOError) as e:
                logger.error(
                    f"Earthquake_{source_id}: Calculation failed for earthquake id {source_id}: {e}",
                    exc_info=True
                )
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            pbar.set_postfix({"Failed": failed_events})
            pbar.update(1)
                
    # Concatenate the dataframe
    df_result = pd.concat(result_list, ignore_index = True) if result_list else df_result
    df_fitting = pd.concat(fitting_list, ignore_index = True) if fitting_list else df_fitting

    # Summary message
    sys.stdout.write(
        f"Finished. Proceed {total_earthquakes - failed_events} earthquakes successfully,"
        f"{failed_events} failed. Check runtime.log for details. \n"
    )
    return df_result, df_fitting, mw_output

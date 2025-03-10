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
from obspy.geodetics import gps2dist_azimuth
from obspy.signal import rotate
from scipy import signal
from tqdm import tqdm

import LQTMomentMag.fitting_spectral as fit
import LQTMomentMag.refraction as ref 
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

    z, n, e = [stream.select(component=comp)[0] for comp in ['Z', 'N', 'E']]
    l_data, q_data, t_data = rotate.rotate_zne_lqt(z.data, n.data, e.data)
    rotated_stream = Stream()
    for data, comp in zip([l_data, q_data, t_data], ['L', 'Q', 'T']):
        trace = z.copy()
        trace.data =  data
        trace.stats.component = comp
        rotated_stream+=trace

    return rotated_stream 



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
    p_phase_end_index = int(round((P_arr - trace_L.stats.starttimem + time_after_pick_p )/trace_L.stats.delta, 4))
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

    # Create object collector for fitting result
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

    # Create object collector for plotting
    all_streams = []
    all_p_times = []
    all_s_times = []
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
    hypo_info = hypo_df.iloc[0]
    origin_time = UTCDateTime(f"{int(hypo_info.Year):04d}-{int(hypo_info.Month):02d}-{int(hypo_info.Day):02d}T{int(hypo_info.Hour):02d}:{int(hypo_info.Minute):02d}:{float(hypo_info.T0):012.9f}") 
    hypo_lat, hypo_lon , hypo_depth =  hypo_info.Lat, hypo_info.Lon, hypo_info.Depth

    # Find the correct velocity and DENSITY value for the spesific layer depth
    velocity_P, velocity_S, density_value = None, None, None
    for (top, bottom), vp, vs, rho in zip(CONFIG.magnitude.LAYER_BOUNDARIES, CONFIG.magnitude.VELOCITY_VP, CONFIG.magnitude.VELOCITY_VS, CONFIG.magnitude.DENSITY):
        if (top*1000)   <= hypo_depth <= (bottom*1000):
            velocity_P, velocity_S, density_value = vp*1000, vs*1000, rho
            break
        else:
            logger.warning(f"Event_{event_id}: Hypo depth not within the defined layers.")
            return {}, fitting_result
    
    # Start spectrum fitting and magnitude estimation
    moments, corner_frequencies, source_radius = [],[],[]
    for station in pick_df.get("Station").unique():
        # Get the station coordinat
        station_xyz_info = station_df[station_df.Stations == station].iloc[0]
        station_lat, station_lon, station_elev = station_xyz_info.Lat, station_xyz_info.Lon, station_xyz_info.Elev
        
        # Calculate the source distance and the azimuth (hypo to station azimuth)
        epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, station_lat, station_lon)
        source_distance = np.sqrt(epicentral_distance**2 + ((hypo_depth + station_elev)**2))
        
        # Get the pick_df data for P arrival and S arrival
        pick_info = pick_df[pick_df.Station == station].iloc[0]
        P_pick_time = UTCDateTime(
            f"{pick_info.Year}-{int(pick_info.Month):02d}-{int(pick_info.Day):02d}T"
            f"{int(pick_info.Hour):02d}:{int(pick_info.Minutes_P):02d}:{float(pick_info.P_Arr_Sec):012.9f}"
        )
        S_pick_time = UTCDateTime(
            f"{pick_info.Year}-{int(pick_info.Month):02d}-{int(pick_info.Day):02d}T"
            f"{int(pick_info.Hour):02d}:{int(pick_info.Minutes_S):02d}:{float(pick_info.S_Arr_Sec):012.9f}"
        )
        
        # Read the waveform 
        stream = read_waveforms(wave_path, event_id, station)
        stream_copy = stream.copy()
        if len(stream_copy) < 3:
            logger.warning(f"Event_{event_id}: Not all components available for station {station} to calculate event {event_id} moment magnitude")
            continue
        
        # Perform the instrument removal
        try:
            stream_displacement = instrument_remove(stream_copy, calibration_path, figure_path, figure_statement=False)
        except Exception as e:
            logger.warning(f"Event_{event_id}: An error occured when correcting instrument for station {station}: {e}", exc_info=True)
            continue
        
        # Perform station rotation form ZNE to LQT 
        try:
            try:
                hypo_coordinate = [hypo_lat, hypo_lon , -1*hypo_depth]  # depth must be in negative notation
                station_coordinate = [station_lat, station_lon, station_elev]
                take_off_angle, total_traveltime, incidence_angle = ref.calculate_inc_angle(hypo_coordinate, station_coordinate, CONFIG.magnitude.LAYER_BOUNDARIES, CONFIG.magnitude.VELOCITY_VP)
            except Exception as e:
                logger.warning(f"Event_{event_id}: An error occured when calculating incidence angle for station {station}.", exc_info=True)
            rotated_stream = rotate_component(stream_displacement, azimuth, incidence_angle) # do the component rotation from ZNE to LQT
        except Exception as e:
            logger.warning(f"Event_{event_id}: An error occured when rotating component for station {station}.", exc_info=True)
            continue
 
        # Window the trace
        p_window_data, sv_window_data, sh_window_data, p_noise_data, sv_noise_data, sh_noise_data = window_trace(rotated_stream, P_pick_time, S_pick_time)
        
        # Check the data quality (SNR must be above or equal to 1)
        if any(trace_snr(data, noise) <= 1.25 for data, noise in zip ([p_window_data, sv_window_data, sh_window_data], [p_noise_data, sv_noise_data, sh_noise_data])):
            logger.warning(f"Event_{event_id}: SNR below threshold for station {station} to calculate moment magnitude")
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
        except Exception as e:
            logger.warning(f"Event_{event_id}: An error occured during spectra calculation for station {station}, {e}.", exc_info=True)
            continue

        # Fitting the spectrum, find the optimal value of Omega_O, corner frequency and Q using systematic/stochastic algorithm available
        try:
            fit_P  = fit.fit_spectrum_qmc(freq_P,  spec_P,  abs(float(P_pick_time - origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
            fit_SV = fit.fit_spectrum_qmc(freq_SV, spec_SV, abs(float(S_pick_time - origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
            fit_SH = fit.fit_spectrum_qmc(freq_SH, spec_SH, abs(float(S_pick_time - origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
        except Exception as e:
            logger.warning(f"Event_{event_id}: Error during spectral fitting for event {event_id}, {e}.", exc_info=True)
            continue
        if any(f is None for f in [fit_P, fit_SV, fit_SH]):
            continue

        # Fitting spectrum output
        Omega_0_P,  Q_factor_p,  f_c_P,  err_P,  x_fit_P,  y_fit_P  = fit_P
        Omega_0_SV, Q_factor_SV, f_c_SV, err_SV, x_fit_SV, y_fit_SV = fit_SV
        Omega_0_SH, Q_factor_SH, f_c_SH, err_SH, x_fit_SH, y_fit_SH = fit_SH

        # Updating the fitting object collector 
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


        # Calculate the moment magnitude
        try:
            # Calculate the  resultant omega
            omega_P = Omega_0_P*1e-9
            omega_S = ((Omega_0_SV**2 + Omega_0_SH**2)**0.5)*1e-9
         
            # Calculate seismic moment
            M_0_P = (4.0 * np.pi * density_value * (velocity_P ** 3) * source_distance * omega_P) / (CONFIG.magnitude.R_PATTERN_P * CONFIG.magnitude.FREE_SURFACE_FACTOR)
            M_0_S = (4.0 * np.pi * density_value * (velocity_S ** 3) * source_distance * omega_S) / (CONFIG.magnitude.R_PATTERN_S * CONFIG.magnitude.FREE_SURFACE_FACTOR)
            fitting_result["Moment_P_(Nm)"].append(M_0_P)
            fitting_result["Moment_S_(Nm)"].append(M_0_S)
            
            # Calculate average seismic moment at station
            moments.append((M_0_P + M_0_S)/2)
            
            # Calculate source radius
            r_P = (CONFIG.magnitude.K_P * velocity_P)/f_c_P
            r_S = (2 * CONFIG.magnitude.K_S * velocity_S)/(f_c_SV + f_c_SH)
            source_radius.append((r_P + r_S)/2)
            
            # Calculate corner frequency mean
            corner_freq_S = (f_c_SV + f_c_SH)/2
            corner_frequencies.append((f_c_P + corner_freq_S)/2)

        except Exception as e:
            logger.warning(f" Event_{event_id}: Failed to calculate seismic moment for event {event_id}, {e}.", exc_info=True)
            continue
        
        
        # Update fitting spectral object collectro for plotting
        if figure_statement:
            all_streams.append(rotated_stream)
            all_p_times.append(P_pick_time)
            all_s_times.append(S_pick_time)
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
 
    results = {"ID":[f"{event_id}"], 
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
            plot_spectral_fitting(event_id, all_streams, all_p_times, all_s_times, all_freqs, all_specs, all_fits, station_names, figure_path)
        except Exception as e:
            logger.warning(f"Event_{event_id}: Failed to create spectral fitting plot for event {event_id}, {e}.", exc_info=True)
    
    return results, fitting_result



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
        
    prompt=input('Have you change the paths? [yes/no] :').strip().lower()
    if prompt != 'yes':
        sys.exit("Ok, please correct the path first!")
    else:
        sys.stdout.write("Process the program ....\n")
    
    # Get the user input.
    id_start, id_end, mw_output, figure_statement = get_user_input()

    # Initiate dataframe for magnitude calculation results
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
            
            # Get the dataframe 
            hypo_data_handler   = hypo_data[hypo_data["ID"] == event_id]
            pick_data_handler   = pick_data[pick_data["Event ID"] == event_id]
            
            # Check empty data frame
            if hypo_data_handler.empty or pick_data_handler.empty:
                logger.warning(f"Event_{id}: No data for event {event_id}, skipping...")
                failed_events+=1
                continue

            else:
                try:
                    # Calculate the moment magnitude
                    mw_results, fitting_result = calculate_moment_magnitude(wave_path, hypo_data_handler, pick_data_handler, station_data, calibration_path, event_id, figure_path, figure_statement)

                    # Create the dataframe from calculate_ml_magnitude results
                    mw_magnitude_result = pd.DataFrame.from_dict(mw_results)
                    mw_fitting_result   = pd.DataFrame.from_dict(fitting_result)
                    
                    # Concatinate the dataframe
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

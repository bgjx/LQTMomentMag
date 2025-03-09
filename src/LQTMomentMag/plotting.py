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
->[matplotlib, obspy, pathlib] 

"""

import logging
import matplotlib.pyplot as plt
from obspy import Stream
from typing import List, Dict, Tuple
from pathlib import Path

from .config import CONFIG

logger = logging.getLogger("mw_calculator")

def plot_spectral_fitting(
        event_id: int,
        streams: List[Stream],
        p_arr_times: List[float],
        s_arr_times: List[float],
        freqs: Dict[str, List],
        specs: Dict[str, List],
        fits: Dict[str, list],
        stations: List[str],
        figure_path: Path):
    """
    Plot phase windows and spectral fitting profiles for all stations in an event.

    Args:
        event_id (int): Unique identifier for the earthquake event.
        streams (Stream): A stream object containing the seismic data.
        p_arr_times (List[float]): List of all P arrival time for an event for each station.
        s_arr_times (List[float]): List of all S arrival time for an event for each station.
        freqs (Dict[str,List]): A dictionary of frequency arrays for P, SV, SH, and noise per station.
        specs (Dict[str, List]): A dictionary of spectral arrays for P, SV, SH, and noise per station.
        stations (List[str]): List of station names.
        figure_path(Path): Directory to save the plot.

    """
    
    # Initiate plotting dimension
    num_stations = len(streams)
    fig, axs = plt.subplot(num_stations*3, 2, figsize=(20, num_stations*15), squeeze=False)
    plt.subplots_adjust(hspace=0.5)
    axs[0,0].set_title("Phase Window", fontsize=20)
    axs[0,1].set_title("Spectra Fitting Profile", fontsize=20)

    for station_idx, (stream, p_time, s_time, station) in  enumerate(zip(streams, p_arr_times, s_arr_times, stations)):
        # Dinamic window parameter
        counter = station_idx*3
        s_p_time = float(s_time - p_time)    
        time_after_pick_p = 0.80 * s_p_time
        time_after_pick_s = 1.75 * s_p_time

        for comp, label in zip(["L", "Q", "T"], ["P", "SV", "Sh"]):
            trace = stream.select(comp=trace)[0]
            start_time = trace.stats.starttime
            trace.trim(start_time+(p_time - start_time) - 2.0, start_time+(s_time - start_time)+6.0)

            ax = axs[counter, 0]
            ax.plot(trace.times(), trace.data, "k")
            ax.axvline(p_time - trace.stats.starttime, color='r', linestyle='-', label='P arrival')
            ax.axvline(s_time - trace.stats.starttiem, color='b', linestyle='-', label='S arrival')
            ax.axvline(p_time - CONFIG.magnitude.PADDING_BEFORE_ARRIVAL - trace.stats.starttime, color='g', linestyle='--')
            ax.axvline(p_time + (time_after_pick_p if comp == "L" else time_after_pick_s) - trace.stats.starttime, color='g', linestyle='--', label='P phase window')
            ax.axvline(p_time - CONFIG.magnitude.NOISE_DURATION - trace.stats.starttime, color='gray', linestyle='--')
            ax.axvline(p_time - CONFIG.magnitude.NOISE_PADDING - trace.stats.starttime, color='gray', linestyle='--')
            ax.set_title(f"{station}_BH{comp}", loc='right', va='center')
            ax.legend()
            ax.set_xlabel("Relative Time (s)")
            ax.set_ylabel("Amp (m)")

            ax = axs[counter, 1]
            ax.loglog(freqs[label][station_idx], specs[label][station_idx], "k", label=f"{label} spectral")
            ax.loglog(freqs[f"N_{label}"][station_idx], specs[f"N_{label}"][station_idx], "gray", label="Noise Spectra")
            ax.loglog(fits[label][station_idx][4], fits[label][station_idx][5], "b-", label=f"Fitted {label} Spectra")
            ax.set_title(f"{station}_BH{comp}", loc="right")
            ax.legend()
            ax.set_xlabel("Frequencies (Hz)")
            ax.set_ylabel("Amp (nms)")

            counter += 1
    
    fig.suptitle(f"Event {event_id} Spectral Fitting Profile", fontsize=24, fontweight='bold')
    plt.savefig(figure_path/f"event_{event_id}.png")
    plt.close(fig)

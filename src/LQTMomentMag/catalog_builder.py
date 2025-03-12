#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:32:03 2022.
Catalog builder utility for LQTMomentMag.

Combines hypocenter, pick, and station data into a single catalog file


Developed by arham zakki edelo.


contact: 
- edelo.arham@gmail.com
- https://github.com/bgjx

Pre-requisite modules:
->[pathlib, tqdm, numpy, pandas, obspy, scipy] 

"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import csv


def build_catalog(
        hypo_path: Path,
        picks_path: Path,
        station_path: Path,
        output_path: Path,
        network:str = "project_0", 
        output_name:str = "combined_catalog",
        ) -> None:
    """
    Build a combined catalog from seperate hypocenter, pick, and station file.

    Args:
        hypo_path (Path): Path to the hypocenter catalog file.
        picks_path (Path): Path to the picking catalog file.
        station_path (Path): Path to the station file.
        output_path (Path): Path to save the combined catalog.
        network (str): Network code to assign to the combine catalog (default: "project_0").   
    """
    hypo_df = pd.read_excel(hypo_path, index_col=None)
    picking_df = pd.read_excel(picks_path, index_col=None)
    station_df = pd.read_excel(station_path, index_col=None)


    csv_file = open(output_path/f"{output_name}.csv", 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = [
        "network", "event_id", "source_lat", "source_lon", "source_depth_m", "source_origin_time", "station_code", "station_lat", "station_lon", "station_elev_m", "p_arr_time", "p_onset", "p_polarity", "s_arr_time", "coda_time", "remarks"
    ]
    csv_writer.writerow(header)
    csv_file.flush()
    for id in hypo_df.get("id"):
        pick_data = picking_df[picking_df["id"] == id]
        if pick_data.empty:
            continue
        hypo_info = hypo_df[hypo_df.ID == id].iloc[0]
        _id, source_lat, source_lon, source_depth_m, year, month, day, hour, minute, t0 = hypo_info.ID, hypo_info.Lat, hypo_info.Lon, hypo_info.Depth, hypo_info.Year, hypo_info.Month, hypo_info.Day, hypo_info.Hour, hypo_info.Minute, hypo_info.T0
        int_t0 = int(t0)
        microsecond = int((t0 - int_t0)*1e6)
        source_origin_time =datetime(int(year), int(month), int(day), int(hour), int(minute), int_t0, microsecond)
        for station in pick_data.get("Station"):
            station_info = pick_data[pick_data.Station == station].iloc[0]
            station_name = station_info.Station
            year, month, day, hour, minute_p, second_p, p_onset, p_polarity, minute_s, second_s= station_info.Year, station_info.Month, station_info.Day, station_info.Hour, station_info.Minutes_P, station_info.P_Arr_Sec, station_info.P_Onset, station_info.P_Polarity, station_info.Minutes_S, station_info.S_Arr_Sec
            int_p_second = int(second_p)
            microsecond_p = int((second_p - int_p_second)*1e6)
            int_s_second = int(second_s)
            microsecond_s = int((second_s - int_s_second)*1e6)

            p_arr_time = datetime(year, month, day, hour, minute_p, int_p_second, microsecond_p)
            s_arr_time = datetime(year, month, day, hour, minute_s, int_s_second, microsecond_s)

            csv_writer.writerow([
                network, id, source_lat, source_lon, source_depth_m, source_origin_time, station_name, p_arr_time, p_onset, p_polarity, s_arr_time, "", ""
            ])
    csv_file.close()

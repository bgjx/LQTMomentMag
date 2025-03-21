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
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from obspy.geodetics import gps2dist_azimuth



def build_catalog(
        hypo_path: Path,
        picks_path: Path,
        station_path: Path,
        network:str = "LQTID", 
        ) -> pd.DataFrame:
    """
    Build a combined catalog from seperate hypocenter, pick, and station file.

    Args:
        hypo_path (Path): Path to the hypocenter catalog file.
        picks_path (Path): Path to the picking catalog file.
        station_path (Path): Path to the station file.
        output_path (Path): Path to save the combined catalog.
        network (str): Network code to assign to the combine catalog (default: "project_0").

    Returns:
        pd.DataFrame : Dataframe object of combined catalog.

    """
    hypo_df = pd.read_excel(hypo_path, index_col=None)
    picking_df = pd.read_excel(picks_path, index_col=None)
    station_df = pd.read_excel(station_path, index_col=None)

    rows = []
    for id in hypo_df.get("id"):
        pick_data = picking_df[picking_df["id"] == id]
        if pick_data.empty:
            continue
        hypo_info = hypo_df[hypo_df.id == id].iloc[0]
        source_lat, source_lon, source_depth_m, year, month, day, hour, minute, t0, hypo_remarks = hypo_info.lat, hypo_info.lon, hypo_info.depth, hypo_info.year, hypo_info.month, hypo_info.day, hypo_info.hour, hypo_info.minute, hypo_info.t_0, hypo_info.remarks
        int_t0 = int(t0)
        microsecond = int((t0 - int_t0)*1e6)
        source_origin_time =datetime(int(year), int(month), int(day), int(hour), int(minute), int_t0, microsecond)
        for station in pick_data.get("station_code"):
            station_data = station_df[station_df.station_code == station]
            if station_data.empty:
                continue
            station_info = station_data.iloc[0]
            station_code, station_lat, station_lon, station_elev = station_info.station_code, station_info.lat, station_info.lon, station_info.elev
            
            # cek earthquake distance to determine earthquake type
            epicentral_distance, _, _ = gps2dist_azimuth(source_lat, source_lon, station_lat, station_lon)
            epicentral_distance = epicentral_distance/1e3
            earthquake_type = "very_local_earthquake" if epicentral_distance < 30 else "local_earthquake" if  30 <= epicentral_distance <300 else "regional_earthquake" if 300 <= epicentral_distance < 1000 else "teleseismic_earthquake"

            pick_data_subset= pick_data[pick_data.station_code == station]
            if pick_data_subset.empty:
                continue
            pick_info = pick_data_subset.iloc[0]
            year, month, day, hour, minute_p, second_p, p_onset, p_polarity, minute_s, second_s = pick_info.year, pick_info.month, pick_info.day, pick_info.hour, pick_info.minute_p, pick_info.p_arr_sec, pick_info.p_onset, pick_info.p_polarity, pick_info.minute_s, pick_info.s_arr_sec
            int_p_second = int(second_p)
            microsecond_p = int((second_p - int_p_second)*1e6)
            int_s_second = int(second_s)
            microsecond_s = int((second_s - int_s_second)*1e6)
            p_arr_time = datetime(year, month, day, hour, minute_p, int_p_second, microsecond_p)
            s_arr_time = datetime(year, month, day, hour, minute_s, int_s_second, microsecond_s)
            row = {
                "network": network,
                "source_id": id,
                "source_lat": source_lat, 
                "source_lon": source_lon,
                "source_depth_m": source_depth_m,
                "source_origin_time": source_origin_time,
                "station_code": station_code,
                "station_lat": station_lat,
                "station_lon": station_lon, 
                "station_elev_m": station_elev,
                "p_arr_time": p_arr_time,
                "p_onset": p_onset,
                "p_polarity": p_polarity,
                "s_arr_time": s_arr_time,
                "earthquake_type": earthquake_type,
                "remarks": hypo_remarks
            }
            rows.append(row)
    return pd.DataFrame(rows)

def main(args=None):
    """  Runs the catalog builder from command line or interactive input  """
    parser = argparse.ArgumentParser(description="Build a combined catalog for LQTMomentMag.")
    parser.add_argument("--hypo-file", type=Path, default="tests/sample_tests_data/catalog/hypo_catalog.xlsx", help="Hypocenter data file")
    parser.add_argument("--pick-file", type=Path, default="tests/sample_tests_data/catalog/picking_catalog.xlsx", help="Arrival picking data file")
    parser.add_argument("--station-file", type=Path, default="tests/sample_tests_data/station/station.xlsx", help="Station data file")
    parser.add_argument("--output-dir", type=Path, default="built_catalog", help="Output directory for results")
    parser.add_argument("--network", type=str, default="LQTID", help="Network code (default: LQTID)")
    args = parser.parse_args(args if args is not None else sys.argv[1:])

    for path in [args.hypo_file, args.pick_file, args.station_file]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conmbined_dataframe = build_catalog(args.hypo_file, args.pick_file, args.station_file, args.network)
    conmbined_dataframe.to_excel(args.output_dir/ f"combined_catalog.xlsx", index=False)
    return None

if __name__ == "__main__":
    main()
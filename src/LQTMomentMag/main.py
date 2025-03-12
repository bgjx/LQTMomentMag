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
->[argparse, logging, pathlib, pandas] 

"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd 
import warnings

from .processing import start_calculate

warnings.filterwarnings("ignore")

logging.basicConfig(
    filename = 'mw_calculator_runtime.log',
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("mw_calculator")


def main(args=None):
    """  Runs the moment magnitude calculation form command line or interactive input  """
    parser = argparse.ArgumentParser(description="Calculate moment magnitude in full LQT component.")
    parser.add_argument("--wave-dir", type=Path, default="data/waveforms", help="Path to wavefrom directory")
    parser.add_argument("--cal-dir", type=Path, default="data/calibration", help="Path to the calibration directory")
    parser.add_argument("--fig-dir", type=Path, default="figures", help="Path to save figures")
    parser.add_argument("--hypo-file", type=Path, default="data/hypocenter/hypo_sample.xlsx", help="Hypocenter data file")
    parser.add_argument("--pick-file", type=Path, default="data/picks/picks_sample.xlsx", help="Arrival picking data file")
    parser.add_argument("--station-file", type=Path, default="data/stations/stations_sample.xlsx", help="Station data file")
    parser.add_argument("--output-dir", type=Path, default="results", help="Output directory for results")
    args = parser.parse_args(args if args is not None else sys.argv[1:])

    for path in [args.wave_dir, args.cal_dir, args.hypo_file, args.pick_file, args.station_file]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
    
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
            
    # preload all the excel file
    hypo_df = pd.read_excel(args.hypo_file, index_col=None)
    pick_df = pd.read_excel(args.pick_file, index_col=None)
    station_df = pd.read_excel(args.station_file, index_col=None)

    if any(False for data in [hypo_df, pick_df, station_df] if data.empty):
        logger.warning(f"Cannot process empty dataframes, please check your data again.")
        
    # Call the function to start calculating moment magnitude
    mw_result_df, mw_fitting_df, output_name = start_calculate(args.wave_dir, args.cal_dir, args.fig_dir, 
                                                               hypo_df, pick_df, station_df)

    # save and set dataframe index
    mw_result_df.to_excel(args.output_dir / f"{output_name}_result.xlsx", index = False)
    mw_fitting_df.to_excel(args.output_dir/ f"{output_name}_fitting_result.xlsx", index = False)
    
    return None

if __name__ == "__main__" :
    main()
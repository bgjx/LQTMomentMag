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
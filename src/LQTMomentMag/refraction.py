#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 19:32:03 2023.
Python code for seismic wave refraction calculations in layered medium.

Developed by Arham Zakki Edelo.
Version: 0.2.0
License: MIT

Contact:
- edelo.arham@gmail.com
- https://github.com/bgjx

Pre-requisite modules:
- [numpy, scipy, matplotlib, obspy, configparser]

This module calculates incidence angles, travel times, and ray paths for seismic waves (P-waves, S-waves)
using a layered velocity model and Snell’s Law-based shooting method, suitable for shallow borehole data
in local earthquake monitoring.

References:
- Aki, K., & Richards, P. G. (2002). Quantitative Seismology, 2nd Edition. University Science Books.

"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from obspy.geodetics import gps2dist_azimuth  

from .config import CONFIG

def build_raw_model(layer_boundaries: List[List[float]], velocities: List) -> List[List[float]]:
    """
    Build a model of layers from the given layer boundaries and velocities.

    Args:
        layer_boundaries (List[List[float]]): List of lists where each sublist contains top and bottom depths for a layer.
        velocities (List): List of layer velocities.

    Returns:
        List[List[float]]: List of [top_depth_m, thickness_m, velocity_m_s]
    
    Raises:
        ValueError: If lengths of layer boundaries and velocities don't match.
    """

    if len(layer_boundaries) != len(velocities):
        raise ValueError("Length of layer_boundaries must match velociites")
    model = []
    for (top_km, bottom_km), velocity_km_s in zip(layer_boundaries, velocities):
        top_m = top_km*-1000
        thickness_m = (top_km - bottom_km)* 1000
        velocity_m_s = velocity_km_s * 1000
        model.append([top_m, thickness_m, velocity_m_s])
    return model


def upward_model(hypo_depth_m: float, sta_elev_m: float, raw_model: List[List[float]]) -> List[List[float]]:
    """
    Build a modified model for direct upward-refracted waves from the raw model by evaluating 
    the hypo depth and the station elevation.

    Args:
        hypo_depth_m (float) : Hypocenter depth in meters (negative).
        sta_elev_m (float): Station elevation in meters (positive).
        raw_model (List[List[float]]): List of [top_m, thickness_m, velocity_m_s]

    Returns:
        List[List[float]] : List of modified model corrected by station elevation and hypocenter depth.
    """
    # correct upper model boundary and last layer thickness
    sta_idx, hypo_idx = -1, -1
    for layer in raw_model:
        if layer[0] >= max(sta_elev_m, hypo_depth_m):
            sta_idx+=1
            hypo_idx+=1
        elif layer[0] >= hypo_depth_m:
            hypo_idx+=1
        else:
            pass
    modified_model = raw_model[sta_idx:hypo_idx+1]
    modified_model[0][0] = sta_elev_m  # adjust top to station elevation
    if len(modified_model) > 1:
        modified_model[0][1] = modified_model[1][0] - sta_elev_m # adjust first layer thickness (corrected by station elevation)
        modified_model[-1][1] = hypo_depth_m - modified_model[-1][0] # adjust last layer thickness (corrected by hypo depth)
    else:
        modified_model[0][1] =  hypo_depth_m - sta_elev_m
    return modified_model
 

def downward_model(hypo_depth_m: float, raw_model: List[List[float]]) -> List[List[float]]:
    """
    Build a modified model for downward critically refracted waves from the raw model by evaluating 
    the hypo depth and the station elevation.

    Args:
        hypo_depth_m (float) : Hypocenter depth in meters (negative).
        raw_model (List[List[float]]): List containing sublist where each sublist represents top depth,
                                thickness, and velocity of each layer.

    Returns:
        List[List[float]] : List of modified model from hypocenter depth downward.
    """
    
    hypo_idx = -1
    for layer in raw_model:
        if layer[0] >= hypo_depth_m:
            hypo_idx+=1
    modified_model = raw_model[hypo_idx:]
    modified_model[0][0] = hypo_depth_m
    if len(modified_model) > 1:
        modified_model[0][1] = float(modified_model[1][0]) - hypo_depth_m # adjust first layer thickness relative to the hypo depth
    return modified_model
   
   
def up_refract (epi_dist_m: float, 
                up_model: List[List[float]], 
                angles_deg: np.ndarray
                ) -> Tuple[Dict[str, List], float]:
    """
    Calculate the refracted angle (relative to the normal line), the cumulative distance traveled, 
    and the total travel time for all layers based on the direct upward refracted wave.

    Args:
        epi_dist_m (float): Epicenter distance in m.
        up_model (List[List[float]]): List of sublists containing modified raw model results from the 'upward_model' function.
        angles (np.ndarray): A numpy array of pre-defined angles for a grid search.

    Returns:
        Tuple[Dict[str, List], float]:
            - result (Dict[str, List]): A dictionary mapping take-off angles to {'refract_angle': [], 'distance': [], 'tt': []}.
            - final_take_off (float): The final take-off angle (degrees) of the refracted wave reaches the station.
        
    """
    result = {}
    final_take_off = 0.0
    for angle in angles_deg:
        data= {"refract_angle": [], "distance": [], "tt": []}
        cumulative_dist = 0.0
        current_angle = angle
        for i in range(len(up_model)-1, -1, -1):
            thickness = up_model[i][1]
            velocity =up_model[i][2]          
            dist = np.tan(np.radians(current_angle))*abs(thickness)  # cumulative distance, in abs since the thickness is in negative
            tt = abs(thickness)/(np.cos(np.radians(current_angle))*velocity) 
            cumulative_dist += dist
            data['refract_angle'].append(current_angle)
            data['distance'].append(cumulative_dist)
            data['tt'].append(tt)
            if cumulative_dist >= epi_dist_m:
                break
            if i > 0:
                current_angle = np.degrees(np.arcsin(np.sin(np.radians(current_angle))*up_model[i - 1][2]/velocity))
        result[f"take_off_{angle}"] = data
        if cumulative_dist >= epi_dist_m:
            final_take_off = angle
            break
    return result, final_take_off
      
         
def down_refract(epi_dist_m: float,
                    up_model: List[List[float]],
                    down_model: List[List[float]]
                    ) -> Tuple[Dict[str, List], Dict[str, List]] :
    """
    Calculate the refracted angle (relative to the normal line), the cumulative distance traveled, 
    and the total travel time for all layers based on the downward critically refracted wave.

    Args:
        epi_dist_m (float): Epicenter distance in m.
        up_model (List[List[float]]): List of sublists containing modified raw model results from the 'upward_model' function.
        down_model (List[List[float]]): List of sublists containing modified raw model results from the 'downward_model' function.

    Returns:
        Tuple[Dict[str, List], Dict[str, List]]:
            - Downward segment results (Dict[str, List]): Dict mapping take-off angles to {'refract_angle': [], 'distance': [], 'tt': []}.
            - Upward segment results (Dict[str, List]): Dict for second half of critically refracted rays.
    """
    half_dist = epi_dist_m/2
    critical_angles = []
    if len(down_model) > 1:
        for i in range(len(down_model) - 1):
            critical_angles.append(np.degrees(np.arcsin(down_model[i][2]/down_model[i+1][2])))

    # find the first take-off angle for every critical angle
    take_off_angles=[]
    for i, crit_angle in enumerate(critical_angles):
        angle = crit_angle
        for j in range(i, -1, -1) :
            angle =  np.degrees(np.arcsin(np.sin(np.radians(angle))*down_model[j][2]/down_model[j+1][2]))
        take_off_angles.append(angle)
    take_off_angles.sort()

    down_seg_result = {}
    up_seg_result = {}
    for angle in take_off_angles:
        cumulative_dist = 0.0
        down_data = {"refract_angle": [], "distance": [], "tt": []}
        current_angle = angle
        for i in range(len(down_model)):
            thickness = down_model[i][1]
            velocity = down_model[i][2]
            dist = np.tan(np.radians(current_angle))*abs(thickness)
            tt = abs(thickness)/(np.cos(np.radians(current_angle))*velocity) 
            cumulative_dist += dist
            down_data['refract_angle'].append(current_angle)
            down_data['distance'].append(cumulative_dist)
            down_data['tt'].append(tt)
            if cumulative_dist > half_dist:
                break
            if i + 1 < len(down_model):
                sin_emit = np.sin(np.radians(current_angle))*down_model[i+1][2]/velocity
            if sin_emit < 1:
                current_angle = np.degrees(np.arcsin(sin_emit))            
            elif sin_emit == 1:
                current_angle = 90.0
                up_data, _ = up_refract(epi_dist_m, up_model, np.array([angle]))
                up_seg_result.update(up_data)
                try:
                    dist_up = up_data[f'take_off_{angle}']['distance'][-1]
                    dist_critical = epi_dist_m - (2*cumulative_dist) - dist_up   # total flat line length
                    if dist_critical >= 0:
                        tt_critical = (dist_critical / velocity)
                        down_data['refract_angle'].append(current_angle)
                        down_data['distance'].append(dist_critical + cumulative_dist)
                        down_data['tt'].append(tt_critical)
                except IndexError:
                    pass
                break               
            else:
                break
        down_seg_result[f"take_off_{angle}"] = down_data
    return  down_seg_result, up_seg_result


def plot_rays (hypo_depth_m: float, 
                sta_elev_m: float,
                velocity: List, 
                base_model: List[List[float]],
                up_model: List[List[float]],
                down_model: List[List[float]],
                reached_up_ref: Dict[str, List],
                critical_ref: Dict[str, List],
                down_ref: Dict[str, List],
                down_up_ref: Dict[str, List],
                epi_dist_m: float,
                figure_path: Path
                ) -> None:
    """
    Plot the raw/base model, hypocenter, station, and the relative distance between the hypocenter and station
    and also plot all waves that reach the station.

    Args:
        hypo_depth_m (float) : Hypocenter depth in meters (negative).
        sta_elev_m (float): Elevation of station
        velocity: List of velocities. 
        base_model (List[List[float]]): List of [top_m, thickness_m, velocity_m_s]
        up_model (List[List]): List of [top_m, thickness_m, velocity_m_s] from the 'upward_model' function.
        down_model (List[List]): List of [top_m, thickness_m, velocity_m_s] from the 'downward_model' function.
        reached_up_ref (Dict[str, List]): A dictionary of {'refract_angle': [], 'distance': [], 'tt': []} from all direct upward refracted waves that reach the station.
        critical_ref (Dict[str, List]): A dictionary of {'refract_angle': [], 'distance': [], 'tt': []} from all critically refracted waves.
        down_ref (Dict[str, List]): A dictionary of {'refract_angle': [], 'distance': [], 'tt': []} from all downward segments of critically refracted waves.
        down_up_ref (Dict[str, List]): A dictionary of {'refract_angle': [], 'distance': [], 'tt': []} from all upward segments of downward critically refracted waves.
        epi_dist_m (float): Epicenter distance in m.
        figure_path(Path): Directory to save the plot.
    """
    
    fig, axs = plt.subplots(figsize=(10,8))
    
    # Define colormaps and normalization
    cmap = cm.Oranges
    norm = mcolors.Normalize(vmin=min(velocity), vmax=max(velocity))
    
    max_width = epi_dist_m + 2000
    for layer in base_model:
        color = cmap(norm(layer[2]))
        rect = patches.Rectangle((-1000, layer[0]), max_width, layer[1], linewidth=1, edgecolor= 'black', facecolor = color)
        axs.add_patch(rect)
    
    # Plot only the last ray of direct upward wave that reaches the station
    x, y = 0, hypo_depth_m
    for dist, layer in zip(reached_up_ref['distance'], reversed(up_model)):
        x_next = dist
        y_next = layer[0]
        axs.plot([x, x_next], [y, y_next], 'k')
        x, y = x_next, y_next

    
    for take_off in critical_ref:
        x, y = 0, hypo_depth_m
        for i , (dist, angle) in enumerate(zip(down_ref[take_off]['distance'], down_ref[take_off]['refract_angle'])):
            x_next = dist
            y_next = down_model[i][0] if i == 0 else down_model[i - 1][0] + down_model[i - 1][1]
            axs.plot([x, x_next], [y,y_next], 'b')
            x, y = x_next, y_next
            if angle == 90:
                for j, dist_up in enumerate(down_up_ref[take_off]['distance']):
                    x_next = x + dist_up
                    y_next = up_model[-j - 1][0]
                    axs.plot([x,x_next], [y, y_next], 'b')
                    x, y = x_next, y_next
                break

    axs.plot(epi_dist_m, sta_elev_m, marker = 'v', color = 'black', markersize = 15, label='Station')
    axs.plot(0, hypo_depth_m, marker = '*', color = 'red', markersize = 12)
    axs.set_xlim(-2000, max_width)
    axs.set_ylim((hypo_depth_m-1000), (sta_elev_m+1000))
    axs.set_ylabel('Depth (m)')
    axs.set_xlabel('Distance (m)')
    axs.set_title("Seismic Ray Paths (Snell's Shooting Method)")
    axs.legend()
    plt.savefig(f"{figure_path}/ray_path_event.png")
    plt.close(fig)


def calculate_inc_angle(hypo: List[float],
                        station: List[float],
                        model: List[List],
                        velocity: List, 
                        figure_statement: bool = False,
                        figure_path: Path = None
                        ) -> Tuple [float, float, float]:
    """
    Calculate the take-off angle, total travel-time and the incidence angle at the station for 
    refracted angle using Snell's shooting method.

    Args:
        hypo (List[float]): A list containing the latitude, longitude, and depth of the hypocenter (depth in negative notation).
        sta (List[float]): A list containing the latitude, longitude, and elevation of the station.
        model (List[List[float]]): List of list where each sublist contains top and bottom depths for a layer.
        velocity (List[float]): List of layer velocities.
        figure_statement (bool): Whether to generate and save figures (default is False).
        figure_path (Path): A directory to save plot figures.
        
    Returns:
        Tuple[float, float, float]: take-off angle, total travel time and incidence angle.
    """
    ANGLE_RESOLUTION = np.linspace(0, 90, 1000) # set grid resolution for direct upward refracted wave
    # initialize hypocenter, station, model, and calculate the epicentral distance
    hypo_lat,hypo_lon, hypo_depth_m = hypo
    sta_lat, sta_lon, sta_elev_m = station
    epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, sta_lat, sta_lon)
    
    # build raw model and modified models
    raw_model = build_raw_model(model, velocity)
    up_model = upward_model (hypo_depth_m, sta_elev_m, raw_model.copy())
    down_model = downward_model(hypo_depth_m, raw_model.copy())
    
    #  start calculating all refracted waves for all layers they may propagate through
    up_ref, final_take_off = up_refract(epicentral_distance, up_model, ANGLE_RESOLUTION)
    down_ref, down_up_ref = down_refract(epicentral_distance, up_model, down_model)
    
    # result from direct upward refracted wave only
    last_ray = up_ref[f"take_off_{final_take_off}"]
    take_off_upward_refract = 180 - last_ray['refract_angle'][0]
    upward_refract_tt = np.sum(last_ray['tt'])
    upward_incidence_angle = last_ray['refract_angle'][-1]

    critical_ref = {} # list of downward critically refracted ray (take_off_angle, total_tt, incidence_angle)
    for take_off_key in down_ref:
        if down_ref[take_off_key]["refract_angle"][-1] == 90:
            tt_down = sum(down_ref[take_off_key]['tt'])
            tt_up_seg = sum(down_up_ref[take_off_key]['tt'])
            total_tt = tt_down + tt_up_seg
            inc_angle = down_up_ref[take_off_key]["refract_angle"][-1]
            critical_ref[take_off_key] = {"total_tt": [total_tt], "incidence_angle": [inc_angle]}
    if critical_ref:
        fastest_tt = min(data["total_tt"][0] for data in critical_ref.values())
        fastest_key = next(k for k, v in critical_ref.items() if v['total_tt'][0] == fastest_tt)
        if fastest_tt < upward_refract_tt:
            take_off = float(fastest_key.split("_")[-1])
            total_tt = fastest_tt
            inc_angle = critical_ref[fastest_key]["incidence_angle"][0]
        else:
            take_off = take_off_upward_refract
            total_tt = upward_refract_tt
            inc_angle = upward_incidence_angle
    else:
        take_off = take_off_upward_refract
        total_tt = upward_refract_tt
        inc_angle = upward_incidence_angle
    
    if figure_statement:
        figure_path = figure_path or "."
        plot_rays(hypo_depth_m, sta_elev_m, velocity, raw_model, up_model, down_model, last_ray, critical_ref, down_ref, down_up_ref, epicentral_distance, figure_path)

    return take_off, total_tt, inc_angle

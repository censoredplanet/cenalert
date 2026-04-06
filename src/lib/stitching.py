import glob
import os
import typing
import pandas as pd
import numpy as np
import sys
import datetime

#from .stitching_lib import *

sys.path.append(os.path.abspath('.'))

# Scale time series values to interval [0, 100]
# This won't give precisely what the GT dashboard will, but it will ensure that the
# scale for all events is the same and will at least preserve the general shape of
# the time series
def min_max_normalize(series: pd.Series):
    return (100 * (series - series.min()) / (series.max() - series.min())).fillna(0)


def combine_stitched_dfs_intersection(dfs: typing.List[pd.DataFrame],
                                      date_column: str = "date",
                                      value_column: str = "value",
                                      use_mean: bool = True,
                                      nonzero_fraction = 1) -> pd.DataFrame:

    # Define a function to apply to each date group
    def value_conditionally(group):
        nonzero_values = group[group > 0]
        if len(nonzero_values) >= len(group) * nonzero_fraction:  # Check if all values are greater than 0
            # If yes, return the average or median value
            return nonzero_values.mean() if use_mean else nonzero_values.median()
        else:
            return 0  # Otherwise, return 0

    # Concatenate the list of DataFrames into one DataFrame
    combined_df = pd.concat(dfs)

    # Group by date
    grouped = combined_df.groupby(date_column)[value_column]

    # Apply the function to each date group and aggregate the results
    max_values = grouped.apply(value_conditionally).reset_index(name=value_column)

    return max_values


def group_paths(dirs, slicer):
    grouped = {}
    
    for s in dirs:
        key = s[slicer]
        if key not in grouped:
            grouped[key] = []  
        grouped[key].append(s)
    
    return grouped

def no_ratio(ratios):
    if len(ratios) == 1 and np.isnan(ratios[0]):
        return True
    return np.all(np.isclose(ratios, 0)) or ratios.size == 0

def get_med_or_mean(ratios):
    median_ratio = np.median(ratios)
    # if the median is 0, take the mean
    if np.isclose(median_ratio, 0):
        scale = np.mean(ratios)
    else:
        scale = median_ratio
    return scale

def get_merge_percent(df, df_coarse):
    return (df.merge(df_coarse, how="left", on='date')
                  .ffill()
                  .fillna(df_coarse['value'].iloc[0]))["value_y"]
    

def save_windows(df1, df2, df1c, df2c):
    save_dir = "saved_windows_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # print(f"{save_dir}/{df2['date'].iloc[0]}")
    path = f"{save_dir}/{df2['date'].iloc[0]}"
    os.makedirs(path, exist_ok=True)
    df1.to_csv(f"{path}/df1.csv",index=False)
    df2.to_csv(f"{path}/df2.csv",index=False)
    df1c.to_csv(f"{path}/df1_coarse.csv",index=False)
    df2c.to_csv(f"{path}/df2_coarse.csv",index=False)

def stitch_two_windows_ratio_coarse(df1: pd.DataFrame,
                             df2: pd.DataFrame,
                             df1_coarse: pd.DataFrame,
                             df2_coarse: pd.DataFrame,
                             overlap: list,
                             write=False) -> pd.DataFrame:
    df1_values = df1[df1['date'].isin(overlap)]['value']
    df2_values = df2[df2['date'].isin(overlap)]['value']

    ratios = np.array([v1 / v2 for v1, v2 in zip(df1_values.to_list(), df2_values.to_list()) if v1 != 0 and v2 != 0])
    new_portion = df2[~df2['date'].isin(overlap)].reset_index(drop=True)

    if write:
        save_windows(df1, df2, df1_coarse, df2_coarse)

    if  no_ratio(ratios):
        if (new_portion['value'] == 0).all():
            return pd.concat([df1, df2[~df2['date'].isin(overlap)]]).reset_index(drop=True)

        # getting mean/median of coarse series over df1 and multiplying new portion by reciprocal
        coarse_overlap_old = get_merge_percent(df1, df1_coarse)
        coarse_overlap_new = get_merge_percent(df1, df2_coarse)
        coarse_overlap_scaling = [coarse_overlap_old[i]/coarse_overlap_new[i] if coarse_overlap_new[i] != 0 else 0 for i in range(len(df1)) ]
        co_sfactor = get_med_or_mean(coarse_overlap_scaling)

        new_portion['value'] = new_portion['value']*co_sfactor

        # scaling new region by coarse
        # new_coarse_scaling = get_merge_percent(new_portion, df2_coarse) / 100
        # new_portion['value'] = new_portion['value']*new_coarse_scaling

        return pd.concat([df1, new_portion]).reset_index(drop=True)

    scale = get_med_or_mean(ratios)
    assert not (scale == 0 or np.isnan(scale)), f"Scale {scale} is invalid {ratios}"

    new_portion['value'] = scale * new_portion['value']
    return pd.concat([df1, new_portion]).reset_index(drop=True)

def combine_window_pair(window_list,index,sample_range):
    if index > 0:
        to_combine = [window_list[i][index-1] for i in sample_range]
        merge_old = combine_stitched_dfs_intersection(to_combine, use_mean=True, nonzero_fraction=1)
    else: merge_old = None
    
    if index >= len(window_list[0]):
        # print(window_list[0][index-1])
        # for i in sample_range:
        #     print (len(window_list[i]) )
        pass
    to_combine = [window_list[i][index] for i in sample_range]
    merge_current = combine_stitched_dfs_intersection(to_combine, use_mean=True, nonzero_fraction=1)

    return merge_old, merge_current

def combine_and_stitch(country_samples,write=False):
    # pass in list of full sample paths for given country
    csample_merge = []
    csample_coarse = []
    for csample_dir in country_samples:
        directory = csample_dir
        files = [file for file in sorted(os.listdir(directory)) if file.endswith(".csv")]
        csample_merge.append([pd.read_csv(os.path.join(directory, file), parse_dates=["date"]) for file in files if
                            file.endswith("multiTimeline.csv") and "coarse" not in file])
        
        coarse_sample = [pd.read_csv(os.path.join(directory, file), parse_dates=["date"]) for file in files if
                            file.endswith("coarseMultiTimeline.csv")]
        if (len(coarse_sample)==0):
            print("[ERROR] run into empty sample list")
            files = [file for file in sorted(os.listdir(directory)) if file.endswith(".csv")]
            print(f"[DEBUG] All CSV files in {directory}: {files}")

            coarse_sample_files = [file for file in files if file.endswith("coarseMultiTimeline.csv")]
            print(f"[DEBUG] Coarse files found: {coarse_sample_files}")

        coarse_sample.insert(0, coarse_sample[0])
        
        csample_coarse.append(coarse_sample)
    
    
    sample_range = range(len(country_samples))

    _, merged_window = combine_window_pair(csample_merge,0,sample_range)
    # starting at 1, processes window_index and window_index-1
    
    for window_index in range(1, len(csample_merge[0])):

        _, merge_next = combine_window_pair(csample_merge,window_index,sample_range)
        
        coarse_old, coarse_next = combine_window_pair(csample_coarse,window_index,sample_range)
        # call stitch on these..
        merged_window = stitch_two_windows_ratio_coarse(merged_window, merge_next,
                                           coarse_old, coarse_next, set(merged_window['date'])&set(merge_next['date']), write=write)
    
    #merged_window['value'] = min_max_normalize(merged_window['value'])

    return merged_window.reset_index(drop=True)
        


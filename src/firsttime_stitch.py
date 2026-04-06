#!/usr/bin/python
import argparse
import os
from multiprocessing import Pool
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from lib.stitching import *

NUM_SAMPLES = 45 # Set as constant 

def collect_normal_sample_dirs(sample_dir, num_samples):
    country_dirs = defaultdict(list)
    for i in range(num_samples):
        sdir = os.path.join(sample_dir, f"sample{i}")
        if not os.path.isdir(sdir):
            continue
        for country_code in os.listdir(sdir):
            path = os.path.join(sdir, country_code)
            if os.path.isdir(path):
                country_dirs[country_code].append(path)
    return country_dirs

def stitch_normal_windows(country_code, sample_paths, output_dir):
    try:
        output_path = os.path.join(output_dir, f"{country_code}_stitched.csv")
        combine_and_stitch(sample_paths, write=False).to_csv(output_path, index=False)
        return f"✅ Normal stitched: {country_code}"
    except Exception as e:
        return f"❌ Normal failed: {country_code}: {e}"

def collect_coarse_windows(sample_dir, num_samples):
    country_coarse_windows = defaultdict(list)
    for i in range(num_samples):
        sdir = os.path.join(sample_dir, f"sample{i}")
        if not os.path.isdir(sdir):
            continue
        for country_code in os.listdir(sdir):
            country_path = os.path.join(sdir, country_code)
            if not os.path.isdir(country_path):
                continue
            files = sorted([
                f for f in os.listdir(country_path)
                if f.endswith("coarseMultiTimeline.csv")
            ])
            if not files:
                continue
            dfs = [pd.read_csv(os.path.join(country_path, f), parse_dates=["date"]) for f in files]
            country_coarse_windows[country_code].append(dfs)
    return country_coarse_windows

def stitch_all_coarse_windows(country_coarse_windows, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for country_code, sample_windows in sorted(country_coarse_windows.items()):
        try:
            num_windows = len(sample_windows[0])
            sample_range = range(len(sample_windows))

            if any(len(windows) != num_windows for windows in sample_windows):
                print(f"[WARNING] Unequal coarse window count for {country_code}, skipping.")
                continue

            # Stitch all coarse windows sequentially
            _, merged = combine_window_pair(sample_windows, 0, sample_range)
            for i in range(1, num_windows):
                _, merge_next = combine_window_pair(sample_windows, i, sample_range)
                merged = pd.concat([merged, merge_next]).drop_duplicates(subset="date").sort_values("date")

            output_path = os.path.join(output_dir, f"{country_code}_coarse_stitched.csv")
            merged.to_csv(output_path, index=False)
            print(f"✅ Coarse stitched: {country_code}")
        except Exception as e:
            print(f"❌ Coarse failed: {country_code}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Stitch normal and coarse windows across all countries.")
    parser.add_argument("--sample_dir", required=True, help="Directory with sample0, sample1, ...")
    parser.add_argument("--output_dir", required=True, help="Where stitched outputs will be saved")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Stitch normal windows (parallel)
    country_dirs = collect_normal_sample_dirs(args.sample_dir, NUM_SAMPLES)
    print(f"🔧 Stitching normal windows for {len(country_dirs)} countries...")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(stitch_normal_windows, country_code, paths, args.output_dir): country_code
            for country_code, paths in country_dirs.items()
        }
        for future in as_completed(futures):
            print(future.result())

    # 2. Combine all coarse windows (sequential)
    print(f"\n🔧 combining all coarse windows...")
    coarse_data = collect_coarse_windows(args.sample_dir, NUM_SAMPLES)
    stitch_all_coarse_windows(coarse_data, args.output_dir)


if __name__ == "__main__":
    main()
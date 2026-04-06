#!/usr/bin/env python3

import argparse
import os
from typing import List
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from lib.stitching import combine_stitched_dfs_intersection, stitch_two_windows_ratio_coarse

def sorted_files(dir_path: str, coarse: bool) -> List[str]:
    flist = [
        f
        for f in os.listdir(dir_path)
        if (
            f.endswith("coarseMultiTimeline.csv") if coarse
            else (f.endswith("multiTimeline.csv") and "coarse" not in f)
        )
    ]
    return sorted(flist)


def combine_window(sample_dirs: List[str], win_idx: int, coarse: bool) -> pd.DataFrame:
    dfs = []
    for d in sample_dirs:
        files = sorted_files(d, coarse)
        if win_idx >= len(files):
            raise IndexError(f"{d} has only {len(files)} windows; need idx {win_idx}, coarse is set to {coarse}")
        df = pd.read_csv(os.path.join(d, files[win_idx]), parse_dates=["date"])
        dfs.append(df)
    return combine_stitched_dfs_intersection(dfs, use_mean=True, nonzero_fraction=1)


def process_country(
    code: str,
    sample_dirs: List[str],
    stitched_root: str,
    out_root: str,
) -> str:
    """Combine & stitch one country; return the code on success."""
    stitched_path = os.path.join(stitched_root, f"{code}_stitched.csv")
    stitched_coarse_path = os.path.join(stitched_root, f"{code}_coarse_stitched.csv")

    if not (os.path.isfile(stitched_path) and os.path.isfile(stitched_coarse_path)):
        raise FileNotFoundError(f"Stitched files missing for {code}")

    stitched_prev = pd.read_csv(stitched_path, parse_dates=["date"])
    stitched_prev_coarse = pd.read_csv(stitched_coarse_path, parse_dates=["date"])

    n_windows = len(sorted_files(sample_dirs[0], coarse=False))

    for win_idx in range(n_windows):
        
        new_win = combine_window(sample_dirs, win_idx, coarse=False)
        new_win_coarse = combine_window(sample_dirs, win_idx, coarse=True)

        overlap = set(stitched_prev["date"]) & set(new_win["date"])
        stitched_prev = stitch_two_windows_ratio_coarse(
            stitched_prev,
            new_win,
            stitched_prev_coarse,
            new_win_coarse,
            overlap,
            write=False,
        )

        stitched_prev_coarse =  new_win_coarse

    # write results
    os.makedirs(out_root, exist_ok=True)
    out_normal = os.path.join(out_root, f"{code}_stitched.csv")
    out_coarse = os.path.join(out_root, f"{code}_coarse_stitched.csv")
    stitched_prev.to_csv(out_normal, index=False)
    stitched_prev_coarse.to_csv(out_coarse, index=False)

    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-root", required=True)
    ap.add_argument("--stitched-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=None,
                    help="Processes (default = cpu count)")
    args = ap.parse_args()

    # gather sample folders
    sample0_dir = os.path.join(args.samples_root, "sample0")
    country_codes = sorted(
        d for d in os.listdir(sample0_dir) if os.path.isdir(os.path.join(sample0_dir, d))
    )
    country_samples = {
        code: [os.path.join(args.samples_root, f"sample{i}", code) for i in range(45)]
        for code in country_codes
    }

    worker = partial(
        process_country,
        stitched_root=args.stitched_root,
        out_root=args.out_root,
    )

    # launch all countries in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, c, dirs): c for c, dirs in country_samples.items()}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Countries"):
            code = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"❌ {code} failed: {e}")
            else:
                print(f"✅ {code} done")

if __name__ == "__main__":
    main()

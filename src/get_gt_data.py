#!/usr/bin/python
import argparse
import datetime
import logging
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from lib.google_trends_utils import *

from googleapiclient.discovery import build

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description='Downloader for Google Trends Data')

    # REQUIRED
    # Country_code is for mass downloading. It gets all GT data from start date for all the countries specified
    # Target windows is for downloading specified windows. The input file should have three columns: country_code, start_time, end_time
    parser.add_argument('--daily_update', action='store_true', help='Enable daily update mode')
    parser.add_argument('--no_daily_update', dest='daily_update', action='store_false')
    parser.set_defaults(daily_update=True)

    parser.add_argument('--country_code',
                        default="ground_truth.csv",
                        help='path to raw ground truth CSV, include ONLY if ground truth with windows is not built')
    parser.add_argument('--target_windows',
                        default=None,
                        help='path to ground truth CSV with windows info')
    parser.add_argument('--data_output_existing',
                        default=None,
                        help='SET IF rerunning for missing files or subregions, existing output directory')
    parser.add_argument('--log_level',
                        default="INFO",
                        help='Log level')
    
    parser.add_argument('--sliding_window_size',
                        default=8,
                        help='length (in months) of sliding windows used to download an arbitrary time range at daily granularity',
                        type=int)
    parser.add_argument('--sliding_window_overlap',
                        default=7,
                        help='overlap (in months) of sliding windows',
                        type=int)

    parser.add_argument('--topic',
                        default=VPN_GT_TOPIC_CODE,
                        help='topic to download')
    
    parser.add_argument('--start_month',
                        default= "2011-01",
                        help='start month formatted like 2011-01')
    parser.add_argument('--end_month',
                        default=datetime.datetime.strptime(CURRENT_DATE_STR, DATE_FORMAT).strftime(DATE_FORMAT_MONTH),
                        help='end month formatted like 2011-01, default is current month')

    args = parser.parse_args()
    print(args)
    
    # build output directiory
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.data_output_existing is not None:
        output_dir = args.data_output_existing
    else: 
        output_dir = "output_{0}_{1}".format(timestamp_str, args.topic.replace("/m/", ""))
        os.makedirs(output_dir)

    if args.target_windows is None:
        # This is for mass downloading
        # read in the country codes 
        df = pd.read_csv(args.country_code)  
    else:
        # This is for specifed windows
        df = pd.read_csv(args.target_windows)
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

    # set up logger
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler = logging.FileHandler(os.path.join(os.path.dirname(output_dir), "log.log"),mode="a")
    stream_handler = logging.StreamHandler()
    
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    global logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    # add some logger mode announcement
    collection_type = "initial collection" if args.data_output_existing is None else "recollection"

    logger.info(f"Running for {collection_type}, writing to {output_dir}")

    # grab dev key
    with open("api_key.txt") as f:
        API_KEY = f.read()

    # Start Service
    service = build('trends', 'v1beta', developerKey=API_KEY, discoveryServiceUrl=DISCOVERY_URL)
    
    # daily update on the dataset
    if args.daily_update == True:
        unique_country_codes = df['country_code'].unique()
        with ProcessPoolExecutor(max_workers=3) as executor:
            # Go through every country code
            for country_code in tqdm(unique_country_codes, desc='Country code progress'):
                country_dir = os.path.join(output_dir, country_code) 
                os.makedirs(country_dir, exist_ok=True)
               
                country_processed = False
               
                # build window dir
                if not country_processed:
                    start_date_str = args.start_month
                    end_date_str = args.end_month
                    country_processed = True
                else:
                    # only download full time series once
                    continue
                window_s_str = (datetime.datetime.today() - relativedelta(days=1, months=args.sliding_window_overlap)).strftime("%Y-%m")
                

                tlvl_args = (service, logger, country_dir, start_date_str, window_s_str, end_date_str, country_code)
                    
                # now build window data
                executor.submit(get_multi_timeline_windows, *tlvl_args, args.topic, args.sliding_window_size, args.sliding_window_overlap)

    # Mass downloading
    elif args.target_windows is None:
        unique_country_codes = df['country_code'].unique()

        with ProcessPoolExecutor(max_workers=3) as executor:
            # Go through every country code
            for country_code in tqdm(unique_country_codes, desc='Country code progress'):
                country_dir = os.path.join(output_dir, country_code) 
                os.makedirs(country_dir, exist_ok=True)
               
                country_processed = False
               
                # build window dir
                if not country_processed:
                    start_date_str = args.start_month
                    end_date_str = args.end_month
                    country_processed = True
                else:
                    # only download full time series once
                    continue
               
                tlvl_args = (service, logger, country_dir, start_date_str, start_date_str, end_date_str, country_code)
                
                # now build window data
                executor.submit(get_multi_timeline_windows, *tlvl_args, args.topic, args.sliding_window_size, args.sliding_window_overlap)
    else:
        with ProcessPoolExecutor(max_workers=3) as executor:
        
            for _, row in tqdm(df.iterrows(), total=len(df), desc='Target windows progress'):
                country_code = row['country_code']
                start_date_str = row['start_month']
                end_date_str = row['end_month']

                country_dir = os.path.join(output_dir, country_code)
                if not os.path.exists(country_dir):
                    os.makedirs(country_dir)
                    logger.warning(f"Running for {collection_type}, writing to {output_dir}, but {country_dir} does not exist")

                tlvl_args = (service, logger, country_dir, start_date_str, start_date_str, end_date_str, country_code)

                executor.submit(get_multi_timeline_windows, *tlvl_args, args.topic, args.sliding_window_size, args.sliding_window_overlap)

if __name__ == "__main__":
    main()

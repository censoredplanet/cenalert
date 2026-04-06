import datetime
import time
import os
import secrets
import string
import pandas as pd
from dateutil.relativedelta import relativedelta

DATE_FORMAT_MONTH = "%Y-%m"
DATE_FORMAT = "%Y-%m-%d"

VPN_GT_TOPIC_CODE = "/m/012t0g"

SERVER = 'https://trends.googleapis.com'
API_VERSION = 'v1beta'
DISCOVERY_URL_SUFFIX = '/$discovery/rest?version=' + API_VERSION
DISCOVERY_URL = SERVER + DISCOVERY_URL_SUFFIX

CURRENT_DATE_STR = datetime.date.today().strftime("%Y-%m-%d")

def generate_random_string(length=20):
    characters = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def error_check(logger, error, delay=60):
    logger.error(error.args)
    if len(error.args) > 1:
        if '"code": 429' in str(error.args[1]):
            logger.error(f"HTTPError: Ran out of queries to the Google Trends API, initiating {delay} sec sleep")
            time.sleep(delay)
    return 0

def tlvl_gparse(graph, dest, fname):
    response = graph.execute()
    if(len(response) > 0):
        datanorm = response['item']
        data = pd.json_normalize(datanorm)
        data.to_csv(os.path.join(dest,fname),index=False)
        return 1
    else:
        return 0

def get_multi_timeline(service, logger, dest,fname, start_date_str, end_date_str, country_code, topic: str = VPN_GT_TOPIC_CODE, attempts=10):
    # append a random string to force api refresh
    random_str = generate_random_string(20)
    topics = f"{topic} + {random_str} + {topic.upper()}"
    for i in range(attempts):
        try:
            graph = service.getGraph(terms=topics, 
                                     restrictions_startDate=start_date_str, restrictions_endDate=end_date_str, 
                                     restrictions_geo=country_code)
            response = graph.execute()
            if(len(response) > 0):
                datanorm = response['lines'][0]['points']
                data = pd.json_normalize(datanorm)
                data.to_csv(os.path.join(dest,fname),index=False)
            return
        except Exception as error:
            error_check(logger, error, (i + 1) * 61)
    time.sleep(1)
    logger.critical(f"Unable to retrieve multiTimeline file for {country_code} with start date {start_date_str} and end date {end_date_str}")     

def get_multi_timeline_windows(service, logger, dest, start_date_str, window_s_str,
end_date_str, country_code, topic=VPN_GT_TOPIC_CODE, window=8, overlap=1):
    start_datetime = datetime.datetime.strptime(start_date_str, '%Y-%m')
    end_datetime = datetime.datetime.strptime(end_date_str, '%Y-%m')
    today = datetime.datetime.strptime(CURRENT_DATE_STR, DATE_FORMAT)
    
    if end_datetime > today:
        end_datetime= today
    window_start =  datetime.datetime.strptime(window_s_str, '%Y-%m')
    done = False
    while window_start <= end_datetime: #and window_start <= today
        window_end = window_start + relativedelta(months=(window - 1))

        if window_end.year > today.year or (window_end.year == today.year and window_end.month > today.month):
            window_end = datetime.datetime(year=today.year, 
                                           month=today.month, day=1)

        if window_end >= end_datetime:
            window_end = end_datetime
            done = True

        window_s_str = window_start.strftime(DATE_FORMAT_MONTH)
        window_e_str = window_end.strftime(DATE_FORMAT_MONTH)
        fname =  "{0}_multiTimeline.csv".format(window_s_str)
        
        
        get_multi_timeline(service, logger, dest, fname, window_s_str, 
                           window_e_str, country_code, topic)
        
            
        get_multi_timeline(service, logger, dest, 
                            f"{window_s_str}_coarseMultiTimeline.csv", start_date_str, window_e_str, 
                            country_code, topic)

        window_start += relativedelta(months=(window - overlap))


        if done:
            # delete the last day of data
            file_path = os.path.join(dest, fname)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    if len(df) > 0:
                        df = df.iloc[:-1]
                        df.to_csv(file_path, index=False)
                        logger.info(f"Removed last row from final window file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean last row of {file_path}: {e}")
            break
   
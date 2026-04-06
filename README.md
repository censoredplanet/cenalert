# CenAlert

This repository contains the implementation of the *CenAlert* deployment.

**Note: CenAlert requires a Google Trends API key and Slack token to run.** 

---

## Setup

### Requirements

- Python 3.8 to 3.12

Install Python dependencies:

    pip3 install -r requirements.txt

---

## Directory Structure

The project directories are organized by Google Trends **topics** and **countries**, with all top-level paths defined in the config file (`bin/config.sh`).

At a high level, the repository contains:

    data/                           # raw data by <topic>/<country>/
    stitched_output/                # stitched time series by topic
    final_output/                   # final processed results (e.g., anomalies)
    ChebyshevPreferredFinal/        # intermediate chebyshev-based output
    ChebyshevPreferredFinal_Scaled/ # scaled chebyshev output
    alerts_output/                  # alerts and anomaly outputs

Inside each **topic** directory, data is organized by country:

    data/
      <topic>/
        <country>/
          ...
    stitched_output/
      <topic>/
        <country>/
    final_output/
      <topic>/
        <country>/
    ChebyshevPreferredFinal/
      <topic>/
        <country files>
    ChebyshevPreferredFinal_Scaled/
      <topic>/
        <country files>/
    alerts_output/
      <topic>/
        data/

---

## Bootstrapping CenAlert

Before the daily pipeline can run, *CenAlert* must be bootstrapped with historical data:

### 1. Download Raw Google Trends Data

Run the update script with a start month:

    python3 src/update_samples.py --start-month YYYY-MM

This downloads raw Google Trends data starting from `YYYY-MM` in the `data/` directory under the chosen topic and country. Each downloaded window is sampled 45 times.

### 2. Stitching

The raw data can be stitched into a single time series with the following command:

    python3 src/firsttime_stitch.py --sample_dir <window_samples_directory> --output_dir <output_dir>

This generates the initial stitched files (e.g., `US_stitched.csv`) inside `stitched_output/`.

---

## Running CenAlert

After bootstrapping, the daily pipeline can be run:

    bash bin/pipeline_driver.sh

The pipeline will:

1. Collect the latest raw data for a specified topic
2. Stitch the newly collected data together with the historical time series
3. Perform spike detection on the resulting time series
4. Send Slack notifications of newly detected spikes  

---

## Config Reference

Directory names and key parameters are defined in `bin/config.sh`. Example:

    SCALED_PARAM="ChebyshevPreferredFinal_Scaled"
    TOPIC="vpn"
    BASE_ROOT="data"
    STITCHED_ROOT="stitched_output"
    RESULTS="final_output"
    PARAM="ChebyshevPreferredFinal"
    ALERTS="alerts_output"

---

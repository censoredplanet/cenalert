#!/usr/bin/env python3
import os
import csv
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
import pycountry
import matplotlib.pyplot as plt
import polars as pl
import pickle
import seaborn as sns
import numpy as np

import importlib.util, importlib.machinery

loader = importlib.machinery.SourceFileLoader("bin/config", "./bin/config")
spec = importlib.util.spec_from_loader("bin/config", loader)
config = importlib.util.module_from_spec(spec)
loader.exec_module(config)

from pathlib import Path
BASE = Path(config.BASE_ROOT)


TOPIC = config.TOPIC

ANOMALY_DIR = Path(config.RESULTS) / TOPIC

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(config.ALERTS)/TOPIC
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def scan_recent_anomalies():
    """
    Scans per-country anomalies.csv under ANOMALY_DIR/<CC>/anomalies.csv
    and partitions events into:
      - ongoing (end >= yesterday)
      - started yesterday
      - ended the day before yesterday
    """
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    a_day_before_yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).date()

    ongoing_events = []   # country_code -> [entry_txt, ...]
    started_yesterday_events = []        # list of dicts
    ended_before_yesterday = []          # list of dicts

    for country_dir in ANOMALY_DIR.iterdir():
        if not country_dir.is_dir():
            continue
        csv_path = country_dir / "anomalies.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    start_d = datetime.strptime(row["start"], "%Y-%m-%d").date()
                    end_d = datetime.strptime(row["end"], "%Y-%m-%d").date()
                    impact = float(row["impact"])
                    score = float(row.get("score", 0) or 0)
                    residual = float(row.get("residual", 0) or 0)

                    if score != 0:
                        metric_name, metric_value = "score", score
                    else:
                        metric_name, metric_value = "residual", residual

                    

                    if end_d >= yesterday:
                        ongoing_events.append({
                            "country": country_dir.name,
                            "start": row["start"],
                            "impact": impact,
                            metric_name: metric_value,
                            "peak": row.get("peak", "N/A"),
                        })

                    if start_d == yesterday:
                        started_yesterday_events.append({
                            "country": country_dir.name,
                            "start": row["start"],
                            "end": row["end"],
                            metric_name: metric_value,
                            
                        })

                    if end_d == a_day_before_yesterday:
                        ended_before_yesterday.append({
                            "country": country_dir.name,
                            "start": row["start"],
                            "end": row["end"],
                            "impact": impact,
                            "peak": row.get("peak", "N/A"),
                        })

                except Exception as e:
                    logging.warning(f"Malformed row in {csv_path}: {e}")

    return ongoing_events, started_yesterday_events, ended_before_yesterday

def country_name(code: str) -> str:
    country = pycountry.countries.get(alpha_2=code.upper())
    return country.name if country else code.upper()

def _country_line(country: str, events: list[str]) -> str:
    return f"*{country_name(country)}*:  " + "  |  ".join(events)

def _metric_pair(ev: dict):
    if ev.get("score", 0):
        return "score", float(ev["score"])
    return "residual", float(ev.get("residual", 0))

def _metric_line(label: str, value: float) -> str:
    if label == "score" and value:
        bound = 1 / (value ** 2)
        return (
            f"> *Score*: {value:.2f}  "
            f"(No more than a {bound*100:.4f}% probability to deviate this much from the mean by random chance.)"
        )
    elif label == "score":
        return f"> *Score*: {value:.2f}"
    else:
        return f"> *Residual*: {value:.2f}"

def _write_text(msg: str, out_dir, country_code, topic, start):
    OUTPUT_FILE = out_dir / f"{country_code}_{topic}_{start}.txt"
    
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n\n")  # blank line between messages
    except Exception as e:
        logging.error(f"Failed writing message: {e}")


def generate_plot(country_code: str, start: datetime, end: datetime, topic: str, out):
    df = pl.read_csv(os.path.join("final_output", topic, country_code, "annotated.csv"), try_parse_dates=True)
    with open(os.path.join("ChebyshevPreferredFinal", topic, country_code), "rb") as file:
        parameters = pickle.load(file)
        window_size = round(parameters[0])
    
    window = df.filter((pl.col("date") >= start - timedelta(days=window_size)) & (pl.col("date") <= end))
    anomaly = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    print(anomaly)
    
    plt.rcParams["figure.figsize"] = (6, 3)

    ax = sns.lineplot(x=window["date"], y=window["value"], color="black", linewidth=1.5)
    sns.lineplot(x=window["date"], y=window["threshold"], linestyle="dashed", color="darkgreen")

    if start < end:
        print("not the first day")
        ax.axvline(start, linestyle="dashed", color="darkorchid", linewidth=2)
        ax.fill_between(anomaly["date"], anomaly["value"], anomaly["threshold"], color="red", alpha=0.3)
    else:
        print("the first day")
        plt.plot(end, anomaly[-1, "value"], marker="o", color="red")
    
    ax.grid(which="both", color="black", linestyle="dashed", alpha=0.3)
    
    ax.set_xticks([window[0, "date"], window[len(window)//4, "date"], window[len(window)//2, "date"], window[3*len(window)//4, "date"], window[-1, "date"]])
    ax.tick_params(axis='y', which='both', labelsize=10)
    yticks = np.linspace(window["value"].min(), window["value"].max(), 5)
    ax.set_yticks([round(ytick, 2) for ytick in yticks])
    
    plt.xlabel("Date")
    plt.ylabel("Search Volume")

    plt.tight_layout()

    plt.savefig(out/f"{country_code}_{topic}_{start.strftime("%Y-%m-%d")}.png", dpi=150)
    plt.close()

def write_notifications(topic: str, ongoing_events, started_yesterday, ended_before_yesterday):
    def country_flag(country_code: str) -> str:
        code = country_code.upper()
        return ''.join(chr(0x1F1E6 + ord(c) - ord('A')) for c in code)
    # New spikes
    out_dir =  OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(out_dir, exist_ok=True)
    for ev in started_yesterday:
        label, val = _metric_pair(ev)
        
       
        msg = (
            f"*New spike detected in {country_name(ev['country'])} {country_flag(ev['country'])}*\n"
            f"> *Topic*: {topic}\n"
            f"> *Date*: {ev['start']}\n"
            f"{_metric_line(label, val)}\n"
            
        )
        _write_text(msg, out_dir, ev['country'], topic, ev['start'])

        # Visualization
        fig_path = generate_plot(
            topic=topic,
            country_code=ev["country"],
            start = datetime.strptime(ev["start"], "%Y-%m-%d"),
            end   = datetime.strptime(ev["end"], "%Y-%m-%d"),
            out = out_dir
        )
    
     
    # Ended spikes
    out_dir =  OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(out_dir, exist_ok=True)
    for ev in ended_before_yesterday:
        label, val = _metric_pair(ev)
        peak = ev.get("peak", "N/A")
        impact = ev['impact']
        #TODO: better way to come up with the thresholds
        if impact > 500:
            color = "🔴 (Very high)"   # Very high
        elif impact > 100: # 100 is the maximum impact factor for a single-day event
            color = "🟠 (High)"   # High
        else:
            color = "🟡"   
        msg = (
            f"*Spike just ended in {country_name(ev['country'])} {country_flag(ev['country'])}*\n"
            f"> *Topic*: {topic}\n"
            f"> *Period*: {ev['start']} → {ev['end']}\n"
            f"> *Peak*: {peak}\n"
            f"> *Impact*: {impact:.2f} {color}"
        )
        _write_text(msg, out_dir, ev['country'], topic, ev['start'])
        # Visualization
        fig_path = generate_plot(
            topic=topic,
            country_code=ev["country"],
            start = datetime.strptime(ev["start"], "%Y-%m-%d"),
            end   = datetime.strptime(ev["end"], "%Y-%m-%d"),
            out = out_dir
            
        )
        
    out_dir =  OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(out_dir, exist_ok=True)
    for ev in ongoing_events:
        label, val = _metric_pair(ev)
        impact = ev['impact']
        if impact > 500:    #TODO: better way to come up with the thresholds
            color = "🔴 (Very high)"   # Very high
        elif impact > 100:
            color = "🟠 (High)"   # High
        else:
            color = "🟡"   
        msg = (
            f"*Spike still ongoing in {country_name(ev['country'])} {country_flag(ev['country'])}*\n"
            f"> *Topic*: {topic} \n"
            f"> *Started on*: {ev['start']}\n"
            f"{_metric_line(label, val)}\n"
            f"> *Impact*: {impact:.2f} {color}"
            
        )
        _write_text(msg, out_dir, ev['country'], topic, ev['start'])
        
        fig_path = generate_plot(
            topic=topic,
            country_code=ev["country"],
            start=datetime.strptime(ev["start"], "%Y-%m-%d"),
            end = datetime.combine(date.today(), datetime.min.time()),
            out = out_dir
        )

def main():
    ongoing, started, ended = scan_recent_anomalies()
    write_notifications(TOPIC, ongoing, started, ended)

if __name__ == "__main__":
    main()

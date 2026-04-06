from abc import ABC, abstractmethod
from enum import StrEnum
from datetime import timedelta

import polars as pl
import numpy as np

from scipy.stats import shapiro
from scipy.stats.mstats import winsorize
from more_itertools import consecutive_groups

import isotree
import sklearn.neighbors
import statsforecast.models
from scipy.optimize import minimize_scalar

class DemandCategorization(StrEnum):
    ERRATIC = "erratic"
    LUMPY = "lumpy"
    SMOOTH = "smooth"
    INTERMITTENT = "intermittent"
    NONE = "none"


class Window():
    def __init__(self, min_observations=None):
        self._min_observations = min_observations
        self._window = np.array([])
        self._interarrivals = np.array([])
        self._last_arrival = None

    def clear(self):
        self._window = np.array([])
        self._interarrivals = np.array([])
        self._last_arrival = None
    
    def _window_operation(self, operation, default):
        #if self._min_observations and self._window.size < self._min_observations: return default
        return operation(self._window)
    
    def sum(self):
        return self._window_operation(np.sum, np.nan)
    
    def mean(self):
        return self._window_operation(np.mean, np.nan)
    
    def std(self):
        return self._window_operation(np.std, np.nan)
    
    def median(self):
        return self._window_operation(np.median, np.nan)
    
    def diff(self):
        return np.diff(self._window)
    
    def normality(self, alpha=0.05):
        try:
            _, p = shapiro(self._window)
            return p >= alpha
        except ValueError: return False
    
    def average_interdemand_interval(self, timestamp):
        return np.mean(np.append(self._interarrivals, timestamp - self._last_arrival)) if self._last_arrival is not None else np.nan

    def cov(self):
        return np.std(self._window) / np.mean(self._window) if len(self) > 0 else np.nan
    
    def classify_demand(self, timestamp):
        cov_squared = self.cov() ** 2
        adi = self.average_interdemand_interval(timestamp)

        if adi <= 1.32 and cov_squared > 0.49: return DemandCategorization.ERRATIC
        elif adi > 1.32 and cov_squared > 0.49: return DemandCategorization.LUMPY
        elif adi <= 1.32 and cov_squared <= 0.49: return DemandCategorization.SMOOTH
        elif adi > 1.32 and cov_squared <= 0.49: return DemandCategorization.INTERMITTENT
        return DemandCategorization.NONE

    def sparsity(self):
        return 1 - (np.count_nonzero(self._window) / len(self._window))
    
    def scale(self, factor):
        self._window = np.multiply(self._window, factor)

    def to_array(self):
        if self._last_arrival is None: return np.array([])
        array = np.zeros(int(self._interarrivals.sum()))
        indices = np.cumsum(self._interarrivals, dtype=int) - 1
        array[indices] = self._window
        return array

    def __len__(self):
        return len(self._window)
    
    def __getitem__(self, index):
        return self._window[index]

    def __str__(self):
        return str(self._window) + " " + str(self._interarrivals)
    
    @property
    def window(self):
        return self._window.__deepcopy__(None)

    def insert(self, value, timestamp):
        return NotImplementedError


class SlidingWindow(Window):
    def __init__(self, capacity):
        super().__init__(min_observations=capacity)
        self._capacity = capacity

    def insert(self, value, timestamp):
        assert value != 0
        assert self._last_arrival is None or timestamp > self._last_arrival, f"{timestamp} <= {self._last_arrival}"

        self._window = np.append(self._window, value)
        self._interarrivals = np.append(self._interarrivals, timestamp - self._last_arrival if self._last_arrival is not None else self._capacity)
        self._last_arrival = timestamp

        if self._window.size > self._capacity: self._window = np.delete(self._window, 0)
        if self._interarrivals.size > self._capacity: self._interarrivals = np.delete(self._interarrivals, 0)


class ExpandingWindow(Window):
    def __init__(self):
        super().__init__()

    def insert(self, value):
        self._window = np.append(self._window, value)


class EfficiencyRatio(ExpandingWindow):
    def efficiency_ratio(self):
        if self._window.size <= 1: return np.nan
        net_change = self[-1] - self[0]
        total_change = np.sum(np.abs(self.diff()))
        return net_change / total_change


class CrostonSBA:
    def __init__(self, window: np.array):
        self._window = window
        self._sba = statsforecast.models.CrostonSBA()

    def forecast(self):
        return self._sba.forecast(self._window, 1)["mean"][0]
        

class AnomalyDetector(ABC):
    def __init__(self, window, min_residual=1, efficiency=0.05):
        self._window = SlidingWindow(window)
        self.croston = None
        self._active_anomaly = False
        self._intermittent_demand_anomaly = False
        self._efficiency_ratio = EfficiencyRatio()
        self._min_residual = min_residual
        self._efficiency = efficiency
        self._always_compute_threshold = True
    
    @abstractmethod
    def score(self, value):
        return NotImplementedError
    
    @abstractmethod
    def threshold(self, initial_guess):
        return NotImplementedError
    
    def anomalies(self, impact_threshold=0):
        points_over_threshold = self.annotated_series.filter(pl.col("anomaly"))
        groups = [list(group) for group in consecutive_groups(points_over_threshold["index"])]
        
        def group_to_anomaly(group):
            collective_anomaly = self.annotated_series[group]
            start_date, end_date = collective_anomaly[[0, -1], "date"]
            peak = collective_anomaly[collective_anomaly["value"].arg_max(), "date"]
            score = collective_anomaly[0, "score"]
            residual = collective_anomaly[0, "residual"]
            impact_factor = collective_anomaly["value"].sum() - collective_anomaly["threshold"].sum()
            return (start_date, end_date, peak, score, residual, impact_factor) #if impact_factor >= impact_threshold else None
        
        anomalies = [anomaly for group in groups if (anomaly := group_to_anomaly(group)) is not None]

        df = pl.DataFrame(anomalies, schema=["start", "end", "peak", ("score", float), ("residual", float), ("impact", float)], orient="row")
        #global_min = self.annotated_series["value"].min()
        global_max = self.annotated_series["value"].max()

        # Apply normalization only if max > 100
        if global_max > 100:
            scaling_factor = global_max / 100.0
            df = df.with_columns(
                (df["residual"] / scaling_factor).alias("residual")
            )
            # TODO: impact might need to be normalized using scaling factor squared
            df = df.with_columns(
                (df["impact"] / scaling_factor).alias("impact")
            )
        

        return df

    def run(self, series: pl.DataFrame):
        self.annotated_series = series.clone().with_row_index().with_columns(
            pl.lit(False).alias("anomaly"),
            pl.lit(np.nan).alias("score"),
            pl.lit(np.nan).alias("residual"),
            pl.lit(np.nan).alias("threshold"),
            pl.lit(np.nan).alias("min_score"),
            pl.lit(np.nan).alias("cov2"),
            pl.lit(np.nan).alias("adi"),
            pl.lit(DemandCategorization.NONE).alias("demand_pattern"))
        
        for idx, (date, value) in enumerate(series.iter_rows()):
            if idx < self._window._min_observations:
                if value > 0: self._window.insert(value, idx + 1)
                continue

            interarrival = idx - self._window._last_arrival if self._window._last_arrival is not None else -1
            if not self._active_anomaly and interarrival >= self._window._capacity: 
                self._window.clear()

            score = 0 if not self._active_anomaly else np.inf
            residual = 0 if not self._active_anomaly else np.inf

            # don't recategorize demand in middle of anomaly
            demand_categorization = self._window.classify_demand(idx) if not self._active_anomaly else DemandCategorization(self.annotated_series[idx - 1, "demand_pattern"])
            
            if value > 0 and demand_categorization in (DemandCategorization.NONE, DemandCategorization.LUMPY, DemandCategorization.INTERMITTENT):
                if not self._active_anomaly:
                    self.croston = CrostonSBA(self._window.to_array())
                
                forecast = 0 if len(self._window) == 0 else self.croston.forecast()
                residual = value - forecast
            elif value > 0 and demand_categorization in (DemandCategorization.SMOOTH, DemandCategorization.ERRATIC):
                if self.annotated_series[idx - 1, "demand_pattern"] in (DemandCategorization.LUMPY, DemandCategorization.INTERMITTENT): # DemandCategorization.NONE,
                    self._window._window = winsorize(self._window._window, limits=(0, 0.05))
                score = self.score(value)
            
            # min residual cannot be 0 when using >=
            new_anomaly = not self._active_anomaly and (score >= self._min_score or residual >= self._min_residual)
            return_to_normal = self._active_anomaly and (score < self._min_score or residual < self._min_residual or np.isclose(value, 0))
            new_normal = self._active_anomaly and self._efficiency_ratio.efficiency_ratio() < self._efficiency

            self._active_anomaly = (new_anomaly or self._active_anomaly) and not (return_to_normal or new_normal)

            if new_anomaly:
                #pl.DataFrame(self._window.to_array()).write_csv(f"{date}.csv")
                self._intermittent_demand_anomaly = demand_categorization in (DemandCategorization.NONE, DemandCategorization.LUMPY, DemandCategorization.INTERMITTENT)
                threshold = self.threshold(value) if not self._intermittent_demand_anomaly else forecast + self._min_residual
                self._efficiency_ratio.insert(self.annotated_series[idx - 1, "value"])
            if self._active_anomaly: 
                self._efficiency_ratio.insert(value)

            if new_normal or return_to_normal:
                anomaly = self._efficiency_ratio[1:]
                target_mean = self._window.mean() if return_to_normal else value
                target_std = self._window.std()

                if self._intermittent_demand_anomaly and new_normal:
                    self._window.clear()
                    self._window._last_arrival = idx - len(anomaly)
                
                for i, point in enumerate(anomaly): self._window.insert(point, idx - len(anomaly) + i + 1)

                # with small smoothing constant, affects of outliers are mitigated during simple exponential
                # smoothing during Croston's method, so don't need to flatten
                if not self._intermittent_demand_anomaly:
                    # if anomaly in smooth window, rescale
                    self._window._window = ((self._window.window - self._window.mean()) / self._window.window.std()) * target_std + target_mean
                elif new_normal:
                    anomaly_mean = anomaly.mean()
                    self._window._window = self._window.window * (target_mean / anomaly_mean)
                
                self._intermittent_demand_anomaly = False
                self._efficiency_ratio.clear()

            self.annotated_series[idx, "anomaly"] = self._active_anomaly
            self.annotated_series[idx, "score"] = score
            self.annotated_series[idx, "residual"] = residual
            if new_anomaly:
                self.annotated_series[idx, "threshold"] = threshold
            elif self._active_anomaly:
                self.annotated_series[idx, "threshold"] = self.annotated_series[idx - 1, "threshold"]
            elif self._always_compute_threshold:
                self.annotated_series[idx, "threshold"] = self.threshold(value)
            else:
                self.annotated_series[idx, "threshold"] = np.nan
            self.annotated_series[idx, "min_score"] = self._min_score
            self.annotated_series[idx, "cov2"] = self._window.cov() ** 2 if (not self._active_anomaly) or new_anomaly else self.annotated_series[idx, "cov2"]
            self.annotated_series[idx, "adi"] = self._window.average_interdemand_interval(idx) if (not self._active_anomaly) or new_anomaly else self.annotated_series[idx, "adi"]
            self.annotated_series[idx, "demand_pattern"] = str(demand_categorization)

            if not self._active_anomaly and value > 0: self._window.insert(value, idx + 1)


        # normalization
        max_value = self.annotated_series["value"].max()
        scaling_factor = max_value / 100 if max_value > 100 else 1

        if scaling_factor > 1:
            # Scale relevant columns
            for col in ["value", "residual", "threshold"]:
                if col in self.annotated_series.columns:
                    self.annotated_series = self.annotated_series.with_columns(
                        (pl.col(col) / scaling_factor).alias(col)
                    )
            
            # Scale internal min_residual threshold
            self._min_residual /= scaling_factor
                    
        return self.annotated_series


class ChebyshevInequality(AnomalyDetector):
    def __init__(self, window=60, z=3, k=6, min_residual=1, efficiency=0.05):
        super().__init__(window, min_residual=min_residual, efficiency=efficiency)
        self.z = z
        self.k = k
        self._min_score = np.inf

    def score(self, x):
        mu = self._window.mean()
        sigma = self._window.std()
        self._min_score = self.z if self._window.normality() else self.k
        return (x - mu) / sigma
    
    def threshold(self, initial_guess):
        mu = self._window.mean()
        sigma = self._window.std()
        return mu + self._min_score * sigma


class MedianMethod(AnomalyDetector):
    def __init__(self, half_neighborhood, min_score, min_residual, efficiency):
        super().__init__(2 * half_neighborhood, min_residual=min_residual, efficiency=efficiency)
        self.half_neighborhood = half_neighborhood
        self._min_score = min_score

    def score(self, x):
        median = self._window.median()
        difference_median = np.median(self._window.diff())
        combined_median = max(median + self.half_neighborhood * difference_median, median)
        return ((x - combined_median) / self._window.mean())

    def threshold(self, initial_guess):
        median = self._window.median()
        difference_median = np.median(self._window.diff())
        combined_median = max(median + self.half_neighborhood * difference_median, median)
        return combined_median + self._min_score * self._window.mean()


class IsolationForest(AnomalyDetector):
    def __init__(self, window, min_score=0.8, min_residual=1, efficiency=0.05):
        super().__init__(window, min_residual=min_residual, efficiency=efficiency)
        self._iforest = isotree.IsolationForest(ntrees=10, categ_cols=None, nthreads=1)
        self._min_score = min_score

    def score(self, value):
        try:
            scores = self._iforest.fit_predict(pl.DataFrame(np.append(self._window.window, value)))
            return scores[-1] if value > self._window.mean() else 0
        except ValueError:
            return np.nan
        
    def threshold(self, initial_guess):
        threshold = minimize_scalar(lambda x: np.abs(self.score(x) - self._min_score), bounds=(self._window.mean(), initial_guess)).x
        return threshold if not np.isclose(threshold, initial_guess) else self._window.mean()


class LocalOutlierFactor(AnomalyDetector):
    def __init__(self, window, min_score=1, min_residual=1, efficiency=0.05):
        super().__init__(window, min_residual=min_residual, efficiency=efficiency)
        self._lof = sklearn.neighbors.LocalOutlierFactor(n_neighbors=window - 1, p=1)
        self._min_score = min_score

    def score(self, value):
        try:
            self._lof.fit_predict(pl.DataFrame(np.append(self._window.window, value)))
            scores = np.abs(self._lof.negative_outlier_factor_)
            return scores[-1] if value > self._window.mean() else 1
        except ValueError:
            return np.nan
        
    def threshold(self, initial_guess):
        threshold = minimize_scalar(lambda x: np.abs(self.score(x) - self._min_score), bounds=(self._window.mean(), initial_guess)).x
        return threshold if not np.isclose(threshold, initial_guess) else self._window.mean()

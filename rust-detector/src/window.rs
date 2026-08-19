use std::collections::VecDeque;

use serde::Serialize;

use crate::croston::croston_sba_forecast;
use crate::statistics::{mean, population_std, winsorize_upper_five_percent};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DemandCategory {
    Erratic,
    Lumpy,
    Smooth,
    Intermittent,
    None,
}

/// Sliding history of non-zero demand values and their interarrival times.
///
/// Timestamps deliberately use the integer conventions in `detection.py`;
/// observations are inserted at `index + 1`, while classification is queried
/// with the current zero-based index.
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    capacity: usize,
    values: VecDeque<f64>,
    sorted_values: Vec<f64>,
    interarrivals: VecDeque<usize>,
    last_arrival: Option<usize>,
    moments: Option<(f64, f64)>,
}

impl SlidingWindow {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            values: VecDeque::with_capacity(capacity),
            sorted_values: Vec::with_capacity(capacity),
            interarrivals: VecDeque::with_capacity(capacity),
            last_arrival: None,
            moments: None,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn last_arrival(&self) -> Option<usize> {
        self.last_arrival
    }

    pub fn set_last_arrival(&mut self, timestamp: usize) {
        self.last_arrival = Some(timestamp);
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.sorted_values.clear();
        self.interarrivals.clear();
        self.last_arrival = None;
        self.moments = None;
    }

    pub fn insert(&mut self, value: f64, timestamp: usize) {
        assert!(value != 0.0);
        assert!(
            self.last_arrival
                .is_none_or(|last_arrival| timestamp > last_arrival),
            "{timestamp} <= {:?}",
            self.last_arrival
        );

        let interarrival = self
            .last_arrival
            .map_or(self.capacity, |last_arrival| timestamp - last_arrival);
        self.values.push_back(value);
        let sorted_position = self
            .sorted_values
            .binary_search_by(|candidate| candidate.total_cmp(&value))
            .unwrap_or_else(|position| position);
        self.sorted_values.insert(sorted_position, value);
        self.interarrivals.push_back(interarrival);
        self.last_arrival = Some(timestamp);

        if self.values.len() > self.capacity {
            let removed = self.values.pop_front().expect("window is not empty");
            let sorted_position = self
                .sorted_values
                .binary_search_by(|candidate| candidate.total_cmp(&removed))
                .expect("sorted values mirror the window");
            self.sorted_values.remove(sorted_position);
        }
        if self.interarrivals.len() > self.capacity {
            self.interarrivals.pop_front();
        }
        self.moments = None;
    }

    pub fn values(&self) -> Vec<f64> {
        self.values.iter().copied().collect()
    }

    pub(crate) fn sorted_values(&self) -> &[f64] {
        &self.sorted_values
    }

    pub(crate) fn shapiro_shift(&self) -> f64 {
        self.values[self.values.len() / 2]
    }

    pub fn replace_values(&mut self, values: impl IntoIterator<Item = f64>) {
        self.values = values.into_iter().collect();
        assert_eq!(self.values.len(), self.interarrivals.len());
        self.rebuild_sorted_values();
        self.moments = None;
    }

    pub fn mean(&mut self) -> f64 {
        self.moments().0
    }

    pub fn std(&mut self) -> f64 {
        self.moments().1
    }

    pub fn coefficient_of_variation_squared(&mut self) -> f64 {
        let mean = self.mean();
        let coefficient = self.std() / mean;
        coefficient * coefficient
    }

    pub fn average_interdemand_interval(&self, timestamp: usize) -> f64 {
        let Some(last_arrival) = self.last_arrival else {
            return f64::NAN;
        };
        let pending = timestamp.saturating_sub(last_arrival);
        let total = self.interarrivals.iter().sum::<usize>() + pending;
        total as f64 / (self.interarrivals.len() + 1) as f64
    }

    pub fn classify_demand(&mut self, timestamp: usize) -> DemandCategory {
        let cv_squared = self.coefficient_of_variation_squared();
        let adi = self.average_interdemand_interval(timestamp);

        if adi <= 1.32 && cv_squared > 0.49 {
            DemandCategory::Erratic
        } else if adi > 1.32 && cv_squared > 0.49 {
            DemandCategory::Lumpy
        } else if adi <= 1.32 && cv_squared <= 0.49 {
            DemandCategory::Smooth
        } else if adi > 1.32 && cv_squared <= 0.49 {
            DemandCategory::Intermittent
        } else {
            DemandCategory::None
        }
    }

    pub fn to_dense_array(&self) -> Vec<f64> {
        if self.last_arrival.is_none() {
            return Vec::new();
        }

        let mut dense = vec![0.0; self.interarrivals.iter().sum()];
        let mut position = 0;
        for (value, interarrival) in self.values.iter().zip(&self.interarrivals) {
            position += interarrival;
            dense[position - 1] = *value;
        }
        dense
    }

    pub(crate) fn croston_sba_forecast(&self) -> f64 {
        croston_sba_forecast(&self.to_dense_array())
    }

    pub fn winsorize_upper_five_percent(&mut self) {
        winsorize_upper_five_percent(self.values.make_contiguous());
        self.rebuild_sorted_values();
        self.moments = None;
    }

    pub fn scale(&mut self, factor: f64) {
        for value in &mut self.values {
            *value *= factor;
        }
        self.rebuild_sorted_values();
        self.moments = None;
    }

    pub fn recenter(&mut self, target_mean: f64, target_std: f64) {
        let current_mean = self.mean();
        let current_std = self.std();
        for value in &mut self.values {
            *value = ((*value - current_mean) / current_std) * target_std + target_mean;
        }
        self.rebuild_sorted_values();
        self.moments = None;
    }

    fn moments(&mut self) -> (f64, f64) {
        *self.moments.get_or_insert_with(|| {
            let values = self.values.make_contiguous();
            (mean(values), population_std(values))
        })
    }

    fn rebuild_sorted_values(&mut self) {
        self.sorted_values.clear();
        self.sorted_values.extend(self.values.iter().copied());
        self.sorted_values.sort_unstable_by(f64::total_cmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_non_zero_values_and_reconstructs_zero_filled_series() {
        let mut window = SlidingWindow::new(3);
        window.insert(4.0, 1);
        window.insert(8.0, 4);
        window.insert(6.0, 6);

        assert_eq!(window.values(), vec![4.0, 8.0, 6.0]);
        assert_eq!(
            window.to_dense_array(),
            vec![0.0, 0.0, 4.0, 0.0, 0.0, 8.0, 0.0, 6.0]
        );
    }

    #[test]
    fn evicts_oldest_non_zero_observation_at_capacity() {
        let mut window = SlidingWindow::new(2);
        window.insert(2.0, 1);
        window.insert(4.0, 2);
        window.insert(8.0, 5);

        assert_eq!(window.values(), vec![4.0, 8.0]);
        assert_eq!(window.to_dense_array(), vec![4.0, 0.0, 0.0, 8.0]);
    }

    #[test]
    fn keeps_sorted_values_in_sync_with_window_mutations() {
        let mut window = SlidingWindow::new(3);
        window.insert(3.0, 1);
        window.insert(1.0, 2);
        window.insert(2.0, 3);
        assert_eq!(window.sorted_values(), &[1.0, 2.0, 3.0]);

        window.insert(4.0, 4);
        assert_eq!(window.values(), vec![1.0, 2.0, 4.0]);
        assert_eq!(window.sorted_values(), &[1.0, 2.0, 4.0]);

        window.scale(2.0);
        assert_eq!(window.sorted_values(), &[2.0, 4.0, 8.0]);
    }

    #[test]
    fn classifies_smooth_and_intermittent_demand() {
        let mut smooth = SlidingWindow::new(4);
        // One extra insertion evicts the synthetic, capacity-sized first
        // interarrival used by the Python implementation.
        for timestamp in 1..=5 {
            smooth.insert(10.0, timestamp);
        }
        assert_eq!(smooth.classify_demand(5), DemandCategory::Smooth);

        let mut intermittent = SlidingWindow::new(4);
        intermittent.insert(10.0, 1);
        intermittent.insert(10.0, 4);
        intermittent.insert(10.0, 7);
        intermittent.insert(10.0, 10);
        assert_eq!(
            intermittent.classify_demand(11),
            DemandCategory::Intermittent
        );
    }
}

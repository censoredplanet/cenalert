use std::collections::VecDeque;

use serde::Serialize;

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
    interarrivals: VecDeque<usize>,
    last_arrival: Option<usize>,
}

impl SlidingWindow {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            values: VecDeque::with_capacity(capacity),
            interarrivals: VecDeque::with_capacity(capacity),
            last_arrival: None,
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
        self.interarrivals.clear();
        self.last_arrival = None;
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
        self.interarrivals.push_back(interarrival);
        self.last_arrival = Some(timestamp);

        if self.values.len() > self.capacity {
            self.values.pop_front();
        }
        if self.interarrivals.len() > self.capacity {
            self.interarrivals.pop_front();
        }
    }

    pub fn values(&self) -> Vec<f64> {
        self.values.iter().copied().collect()
    }

    pub fn replace_values(&mut self, values: impl IntoIterator<Item = f64>) {
        self.values = values.into_iter().collect();
        assert_eq!(self.values.len(), self.interarrivals.len());
    }

    pub fn mean(&mut self) -> f64 {
        mean(self.values.make_contiguous())
    }

    pub fn std(&mut self) -> f64 {
        population_std(self.values.make_contiguous())
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

    pub fn winsorize_upper_five_percent(&mut self) {
        winsorize_upper_five_percent(self.values.make_contiguous());
    }

    pub fn scale(&mut self, factor: f64) {
        for value in &mut self.values {
            *value *= factor;
        }
    }

    pub fn recenter(&mut self, target_mean: f64, target_std: f64) {
        let current_mean = self.mean();
        let current_std = self.std();
        for value in &mut self.values {
            *value = ((*value - current_mean) / current_std) * target_std + target_mean;
        }
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

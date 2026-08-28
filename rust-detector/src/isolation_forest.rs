use crate::detector::{run_detector, DenseEvaluation, DenseScorer, Detector};
use crate::lof::bounded_threshold;
use crate::model::{DetectionResult, Observation, OutlierDetectorParameters};
use crate::window::SlidingWindow;

const TREE_COUNT: u64 = 10;
const RANDOM_SEED: u64 = 1;

pub struct IsolationForestDetector {
    parameters: OutlierDetectorParameters,
}

impl IsolationForestDetector {
    pub fn new(parameters: OutlierDetectorParameters) -> Self {
        Self { parameters }
    }
}

impl Detector for IsolationForestDetector {
    fn detect(&self, observations: &[Observation]) -> DetectionResult {
        run_detector(
            observations,
            self.parameters.window,
            self.parameters.min_residual_ratio,
            self.parameters.efficiency,
            IsolationForestScorer {
                minimum_score: self.parameters.min_score,
            },
        )
    }
}

pub fn isolation_forest_score(window: &[f64], value: f64) -> f64 {
    if window.is_empty() {
        return f64::NAN;
    }
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    if value <= mean {
        return 0.0;
    }

    let mut samples = Vec::with_capacity(window.len() + 1);
    samples.extend_from_slice(window);
    samples.push(value);
    let maximum_depth = (samples.len() as f64).log2().ceil() as usize;
    let expected_depth = expected_average_depth(samples.len());
    if expected_depth == 0.0 {
        return 0.5;
    }

    let mut total_depth = 0.0;
    for tree in 0..TREE_COUNT {
        let mut random = Xoshiro256PlusPlus::seeded(RANDOM_SEED + tree);
        let indices: Vec<usize> = (0..samples.len()).collect();
        let root = build_tree(&samples, &indices, 0, maximum_depth, &mut random);
        total_depth += root.depth(value);
    }
    2.0_f64.powf(-total_depth / (TREE_COUNT as f64 * expected_depth))
}

enum IsolationTree {
    Leaf(f64),
    Split {
        point: f64,
        left: Box<IsolationTree>,
        right: Box<IsolationTree>,
    },
}

impl IsolationTree {
    fn depth(&self, value: f64) -> f64 {
        match self {
            Self::Leaf(depth) => *depth,
            Self::Split { point, left, right } => {
                if value <= *point {
                    left.depth(value)
                } else {
                    right.depth(value)
                }
            }
        }
    }
}

fn build_tree(
    samples: &[f64],
    indices: &[usize],
    depth: usize,
    maximum_depth: usize,
    random: &mut Xoshiro256PlusPlus,
) -> IsolationTree {
    if indices.len() <= 2 || depth >= maximum_depth {
        return IsolationTree::Leaf(depth as f64 + expected_average_depth(indices.len()));
    }

    // isotree draws once for its split-strategy choice even when all strategy
    // probabilities are zero, then draws the numeric split point.
    random.unit_interval();
    let (minimum, maximum) = indices.iter().map(|index| samples[*index]).fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    );
    if minimum >= maximum {
        return IsolationTree::Leaf(depth as f64 + expected_average_depth(indices.len()));
    }
    let split = sample_uniform(minimum, maximum, random.unit_interval());
    let left_indices: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|index| samples[*index] <= split)
        .collect();
    let right_indices: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|index| samples[*index] > split)
        .collect();
    let left = build_tree(samples, &left_indices, depth + 1, maximum_depth, random);
    let right = build_tree(samples, &right_indices, depth + 1, maximum_depth, random);
    IsolationTree::Split {
        point: split,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn sample_uniform(minimum: f64, maximum: f64, unit: f64) -> f64 {
    let mut sample = 2.0 * (0.5 * minimum + unit * (0.5 * maximum - 0.5 * minimum));
    if sample >= maximum {
        sample = next_down(maximum).max(minimum);
    }
    sample.max(minimum)
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        value
    } else if value == 0.0 {
        -f64::from_bits(1)
    } else {
        f64::from_bits(if value > 0.0 {
            value.to_bits() - 1
        } else {
            value.to_bits() + 1
        })
    }
}

fn expected_average_depth(sample_size: usize) -> f64 {
    match sample_size {
        0 | 1 => 0.0,
        _ => {
            let harmonic = (1..=sample_size)
                .map(|value| 1.0 / value as f64)
                .sum::<f64>();
            2.0 * (harmonic - 1.0)
        }
    }
}

pub(crate) struct IsolationForestScorer {
    pub minimum_score: f64,
}

impl DenseScorer for IsolationForestScorer {
    fn initial_minimum_score(&self) -> f64 {
        self.minimum_score
    }

    fn evaluate(&mut self, window: &mut SlidingWindow, value: f64) -> DenseEvaluation {
        DenseEvaluation {
            score: isolation_forest_score(&window.values(), value),
            minimum_score: self.minimum_score,
        }
    }

    fn threshold(&mut self, window: &mut SlidingWindow, value: f64) -> f64 {
        let values = window.values();
        bounded_threshold(
            window.mean(),
            value,
            |candidate| isolation_forest_score(&values, candidate),
            self.minimum_score,
        )
    }
}

struct Xoshiro256PlusPlus {
    state: [u64; 4],
}

impl Xoshiro256PlusPlus {
    fn seeded(seed: u64) -> Self {
        let first = splitmix64(splitmix64(seed));
        let second = splitmix64(first);
        let third = splitmix64(second);
        let fourth = splitmix64(third);
        Self {
            state: [first, second, third, fourth],
        }
    }

    fn next(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let temporary = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn unit_interval(&mut self) -> f64 {
        ((self.next() >> 11) as f64) * 2.0_f64.powi(-53)
    }
}

fn splitmix64(seed: u64) -> u64 {
    let mut value = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_isotree_reference_scores() {
        let cases = [
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], 20.0, 0.750_647_191_173_167_6),
            (vec![0.0, 0.0, 0.0, 1.0, 2.0], 10.0, 0.750_647_191_173_167_6),
            (
                vec![1.0, 1.0, 2.0, 3.0, 5.0, 8.0],
                13.0,
                0.706_008_690_594_711_2,
            ),
        ];
        for (window, value, expected) in cases {
            let actual = isolation_forest_score(&window, value);
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }

    #[test]
    fn below_mean_value_has_zero_score() {
        assert_eq!(isolation_forest_score(&[1.0, 2.0, 20.0], 2.0), 0.0);
    }
}

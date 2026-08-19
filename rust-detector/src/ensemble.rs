use crate::detector::{run_detector, ChebyshevScorer, DenseEvaluation, DenseScorer, Detector};
use crate::isolation_forest::IsolationForestScorer;
use crate::lof::LocalOutlierFactorScorer;
use crate::model::{DetectionResult, DetectorParameters, EnsembleParameters, Observation};
use crate::window::SlidingWindow;

const MAJORITY_SCORE: f64 = 2.0;

pub struct EnsembleDetector {
    parameters: EnsembleParameters,
}

impl EnsembleDetector {
    pub fn new(parameters: EnsembleParameters) -> Self {
        Self { parameters }
    }
}

impl Detector for EnsembleDetector {
    fn detect(&self, observations: &[Observation]) -> DetectionResult {
        let parameters = self.parameters;
        run_detector(
            observations,
            parameters.window,
            parameters.min_residual,
            parameters.efficiency,
            EnsembleScorer {
                chebyshev: ChebyshevScorer::new(DetectorParameters {
                    window: parameters.window,
                    z: parameters.chebyshev_z,
                    k: parameters.chebyshev_k,
                    min_residual: parameters.min_residual,
                    efficiency: parameters.efficiency,
                }),
                isolation_forest: IsolationForestScorer {
                    minimum_score: parameters.isolation_forest_min_score,
                },
                lof: LocalOutlierFactorScorer {
                    window: parameters.window,
                    minimum_score: parameters.lof_min_score,
                },
            },
        )
    }
}

struct EnsembleScorer {
    chebyshev: ChebyshevScorer,
    isolation_forest: IsolationForestScorer,
    lof: LocalOutlierFactorScorer,
}

impl DenseScorer for EnsembleScorer {
    fn initial_minimum_score(&self) -> f64 {
        MAJORITY_SCORE
    }

    fn evaluate(&mut self, window: &mut SlidingWindow, value: f64) -> DenseEvaluation {
        let chebyshev = self.chebyshev.evaluate(window, value);
        let isolation_forest = self.isolation_forest.evaluate(window, value);
        let lof = self.lof.evaluate(window, value);
        DenseEvaluation {
            score: vote_count([chebyshev, isolation_forest, lof]),
            minimum_score: MAJORITY_SCORE,
        }
    }

    fn threshold(&mut self, window: &mut SlidingWindow, value: f64) -> f64 {
        [
            self.chebyshev.threshold(window, value),
            self.isolation_forest.threshold(window, value),
            self.lof.threshold(window, value),
        ]
        .into_iter()
        .fold(f64::INFINITY, f64::min)
    }
}

fn vote_count(evaluations: [DenseEvaluation; 3]) -> f64 {
    evaluations
        .iter()
        .filter(|evaluation| evaluation.score >= evaluation.minimum_score)
        .count() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_member_votes_and_requires_a_majority() {
        let one_vote = vote_count([
            DenseEvaluation {
                score: 4.0,
                minimum_score: 3.0,
            },
            DenseEvaluation {
                score: 0.7,
                minimum_score: 0.8,
            },
            DenseEvaluation {
                score: 1.4,
                minimum_score: 1.5,
            },
        ]);
        let two_votes = vote_count([
            DenseEvaluation {
                score: 4.0,
                minimum_score: 3.0,
            },
            DenseEvaluation {
                score: 0.8,
                minimum_score: 0.8,
            },
            DenseEvaluation {
                score: 1.4,
                minimum_score: 1.5,
            },
        ]);
        assert!(one_vote < MAJORITY_SCORE);
        assert!(two_votes >= MAJORITY_SCORE);
    }
}

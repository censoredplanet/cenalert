use crate::detector::{run_detector, ChebyshevScorer, DenseEvaluation, DenseScorer, Detector};
use crate::isolation_forest::IsolationForestScorer;
use crate::lof::LocalOutlierFactorScorer;
use crate::model::{
    DetectionResult, DetectorParameters, EnsembleMemberVote, EnsembleParameters, EnsembleVote,
    Observation,
};
use crate::window::SlidingWindow;

const REQUIRED_VOTES: u8 = 2;

pub struct EnsembleDetector {
    parameters: EnsembleParameters,
    reference_maximum: Option<f64>,
}

impl EnsembleDetector {
    pub fn new(parameters: EnsembleParameters) -> Self {
        Self {
            parameters,
            reference_maximum: None,
        }
    }

    pub fn with_reference_maximum(mut self, maximum: f64) -> Self {
        self.reference_maximum = Some(maximum);
        self
    }
}

impl Detector for EnsembleDetector {
    fn detect(&self, observations: &[Observation]) -> DetectionResult {
        let parameters = self.parameters;
        run_detector(
            observations,
            parameters.window,
            parameters.min_residual_ratio,
            parameters.efficiency,
            EnsembleScorer {
                chebyshev: ChebyshevScorer::new(DetectorParameters {
                    window: parameters.window,
                    z: parameters.chebyshev_z,
                    k: parameters.chebyshev_k,
                    min_residual_ratio: parameters.min_residual_ratio,
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
            self.reference_maximum,
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
        f64::from(REQUIRED_VOTES)
    }

    fn evaluate(&mut self, window: &mut SlidingWindow, value: f64) -> DenseEvaluation {
        let chebyshev = self.chebyshev.evaluate(window, value);
        let isolation_forest = self.isolation_forest.evaluate(window, value);
        let lof = self.lof.evaluate(window, value);
        let ensemble_vote = ensemble_vote(&chebyshev, &isolation_forest, &lof);
        DenseEvaluation {
            score: f64::from(ensemble_vote.vote_count),
            minimum_score: f64::from(REQUIRED_VOTES),
            ensemble_vote: Some(ensemble_vote),
        }
    }

    fn threshold(&mut self, window: &mut SlidingWindow, value: f64) -> f64 {
        majority_threshold([
            self.chebyshev.threshold(window, value),
            self.isolation_forest.threshold(window, value),
            self.lof.threshold(window, value),
        ])
    }
}

fn majority_threshold(mut thresholds: [f64; 3]) -> f64 {
    thresholds.sort_by(f64::total_cmp);
    thresholds[usize::from(REQUIRED_VOTES) - 1]
}

fn ensemble_vote(
    chebyshev: &DenseEvaluation,
    isolation_forest: &DenseEvaluation,
    lof: &DenseEvaluation,
) -> EnsembleVote {
    let chebyshev = EnsembleMemberVote::new(chebyshev.score, chebyshev.minimum_score);
    let isolation_forest =
        EnsembleMemberVote::new(isolation_forest.score, isolation_forest.minimum_score);
    let lof = EnsembleMemberVote::new(lof.score, lof.minimum_score);
    let vote_count = [chebyshev.voted, isolation_forest.voted, lof.voted]
        .into_iter()
        .filter(|voted| *voted)
        .count() as u8;
    EnsembleVote {
        vote_count,
        required_votes: REQUIRED_VOTES,
        chebyshev,
        isolation_forest,
        lof,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_is_second_smallest_in_any_member_order() {
        for thresholds in [
            [10.0, 20.0, 30.0],
            [10.0, 30.0, 20.0],
            [20.0, 10.0, 30.0],
            [20.0, 30.0, 10.0],
            [30.0, 10.0, 20.0],
            [30.0, 20.0, 10.0],
        ] {
            assert_eq!(majority_threshold(thresholds), 20.0);
        }
    }

    #[test]
    fn threshold_preserves_ties_and_unreachable_member_thresholds() {
        assert_eq!(majority_threshold([10.0, 10.0, 30.0]), 10.0);
        assert_eq!(majority_threshold([10.0, 30.0, 30.0]), 30.0);
        assert_eq!(majority_threshold([10.0, 20.0, f64::INFINITY]), 20.0);
        assert_eq!(
            majority_threshold([10.0, f64::INFINITY, f64::INFINITY]),
            f64::INFINITY
        );
    }

    #[test]
    fn scorer_uses_the_majority_member_threshold() {
        let mut window = SlidingWindow::new(60);
        for index in 0..60 {
            window.insert(10.0 + (index % 7) as f64, index + 1);
        }
        let mut scorer = EnsembleScorer {
            chebyshev: ChebyshevScorer::new(DetectorParameters {
                window: 60,
                z: 4.0,
                k: 6.0,
                min_residual_ratio: 0.05,
                efficiency: 0.05,
            }),
            isolation_forest: IsolationForestScorer { minimum_score: 0.8 },
            lof: LocalOutlierFactorScorer {
                window: 60,
                minimum_score: 1.5,
            },
        };
        scorer.evaluate(&mut window, 100.0);
        let mut thresholds = [
            scorer.chebyshev.threshold(&mut window, 100.0),
            scorer.isolation_forest.threshold(&mut window, 100.0),
            scorer.lof.threshold(&mut window, 100.0),
        ];
        thresholds.sort_by(f64::total_cmp);
        assert!(thresholds[0] < thresholds[1]);
        assert_eq!(scorer.threshold(&mut window, 100.0), thresholds[1]);
    }

    #[test]
    fn counts_member_votes_and_requires_a_majority() {
        let evaluation = |score, minimum_score| DenseEvaluation {
            score,
            minimum_score,
            ensemble_vote: None,
        };
        let one_vote = ensemble_vote(
            &evaluation(4.0, 3.0),
            &evaluation(0.7, 0.8),
            &evaluation(1.4, 1.5),
        );
        let two_votes = ensemble_vote(
            &evaluation(4.0, 3.0),
            &evaluation(0.8, 0.8),
            &evaluation(1.4, 1.5),
        );

        assert_eq!(one_vote.vote_count, 1);
        assert_eq!(two_votes.vote_count, 2);
        assert!(two_votes.chebyshev.voted);
        assert!(two_votes.isolation_forest.voted);
        assert!(!two_votes.lof.voted);
    }

    #[test]
    fn reports_when_all_members_vote() {
        let evaluation = |score, minimum_score| DenseEvaluation {
            score,
            minimum_score,
            ensemble_vote: None,
        };
        let vote = ensemble_vote(
            &evaluation(4.0, 3.0),
            &evaluation(0.9, 0.8),
            &evaluation(1.6, 1.5),
        );

        assert_eq!(vote.vote_count, 3);
        assert_eq!(vote.required_votes, 2);
    }

    #[test]
    fn preserves_an_infinite_score_vote_without_serializing_infinity() {
        let evaluation = |score, minimum_score| DenseEvaluation {
            score,
            minimum_score,
            ensemble_vote: None,
        };
        let vote = ensemble_vote(
            &evaluation(f64::INFINITY, 3.0),
            &evaluation(0.7, 0.8),
            &evaluation(1.4, 1.5),
        );

        assert!(vote.chebyshev.voted);
        assert_eq!(vote.chebyshev.score, None);
        let serialized = serde_json::to_string(&vote).unwrap();
        let deserialized: EnsembleVote = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, vote);
    }
}

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
        [
            self.chebyshev.threshold(window, value),
            self.isolation_forest.threshold(window, value),
            self.lof.threshold(window, value),
        ]
        .into_iter()
        .fold(f64::INFINITY, f64::min)
    }
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

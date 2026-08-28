# All-country ensemble tuning report

This experiment uses the public `cenalert-paper` `<Virtual private network>` topic
series (2011–2024). The Dashboard's current `vpn` keyword series falls outside this
run. These results serve as diagnostics and require validation before production use.

## Configuration

- Countries discovered: 76
- Unified labels supplied: 423
- NSGA-II evaluations per non-zero country: 5,000
- Population size: 100
- Label proximity: ±3 days
- Base random seed: 42 (deterministically offset by country code)
- Constraint: every applicable proximity-mode country label must be detected
- Global and exact-match pseudo-labels are attribution-only

## Summary

- Completed with a feasible solution: 68
- No feasible solution found: 4
- Skipped all-zero series: 4
- Execution errors: 0
- Countries with an interior knee: 43
- Countries where knee and leftmost detected different anomalies: 43
- Unmatched anomalies from selected solutions: 840
- Labels missed by each infeasible country's best simultaneous-coverage solution: 4
- Of those, labels never detected by any diagnostic Pareto solution: 4

## Findings

The selected knee or fallback solutions detect 1,154 anomalies in total,
compared with 395 for the first feasible Pareto solutions. Because
every reported Pareto solution satisfies the country-label constraint, the leftmost
solutions preserve the same required labels while detecting
759 fewer additional anomalies. An interior knee
was found for 43 countries, and it produced a different anomaly set from the
leftmost solution in all 43 cases. This supports evaluating a label-aware
left-side selection rule. The unsupervised knee requires separate justification.

Four labels received no matches across their country's diagnostic unconstrained
Pareto front. They remain candidates for manual review. A different random seed,
larger search, or different Google Trends series could change the result.

## Labels requiring review

| Country | Date | Description | Category | Source | Seen in any diagnostic solution |
| --- | --- | --- | --- | --- | --- |
| IR | 2018-08-06 | Anti-regime protests broke out in Iran on July 31, 2018. To prevent citizens from sharing photos of the demonstrations, authorities shut down Internet and cellular services in the cities of Qom and Karaj on August 6, 2018. | shutdowns | CenAlert | False |
| TR | 2016-05-13 | Eksi Sozluk | filtering | NetBlocks | False |
| UZ | 2018-09-04 | Facebook | filtering | CenAlert | False |
| ZM | 2019-08-16 | Protests | other | Other | False |

## Country results

| Country | Status | Required labels | Covered | Knee anomalies | Leftmost anomalies | Knee found | Unmatched |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| AE | completed | 2 | 2 | 101 | 5 | True | 97 |
| AF | completed | 0 | 0 | 2 | 2 | False | 0 |
| AM | completed | 0 | 0 | 1 | 1 | False | 1 |
| AZ | completed | 3 | 3 | 18 | 4 | True | 14 |
| BD | completed | 7 | 7 | 25 | 10 | True | 15 |
| BF | completed | 1 | 1 | 1 | 1 | False | 0 |
| BI | completed | 2 | 2 | 3 | 3 | False | 0 |
| BJ | completed | 4 | 4 | 2 | 2 | False | 0 |
| BR | completed | 7 | 7 | 19 | 10 | True | 10 |
| BY | completed | 1 | 1 | 20 | 1 | True | 19 |
| CD | completed | 2 | 2 | 8 | 8 | False | 6 |
| CM | completed | 1 | 1 | 2 | 1 | True | 1 |
| CN | completed | 1 | 1 | 85 | 1 | True | 84 |
| CU | completed | 4 | 4 | 8 | 4 | True | 3 |
| DZ | completed | 15 | 15 | 34 | 18 | True | 17 |
| EC | completed | 1 | 1 | 9 | 1 | True | 8 |
| EG | completed | 9 | 9 | 34 | 12 | True | 25 |
| ER | skipped |  |  |  |  |  | 0 |
| ES | completed | 4 | 4 | 7 | 7 | False | 1 |
| ET | completed | 11 | 11 | 24 | 24 | False | 11 |
| GA | completed | 2 | 2 | 2 | 2 | False | 0 |
| GB | completed | 4 | 4 | 22 | 6 | True | 18 |
| GN | completed | 1 | 1 | 1 | 1 | False | 0 |
| GQ | completed | 0 | 0 | 1 | 1 | False | 1 |
| HK | completed | 4 | 4 | 4 | 4 | False | 0 |
| ID | completed | 3 | 3 | 23 | 5 | True | 19 |
| IN | completed | 15 | 15 | 24 | 16 | True | 10 |
| IQ | completed | 11 | 11 | 26 | 12 | True | 15 |
| IR | no_feasible_solution | 15 | 14 |  |  |  | 0 |
| IT | completed | 3 | 3 | 39 | 4 | True | 32 |
| JO | completed | 6 | 6 | 11 | 11 | False | 4 |
| KE | completed | 1 | 1 | 10 | 1 | True | 9 |
| KG | completed | 1 | 1 | 6 | 1 | True | 5 |
| KH | completed | 1 | 1 | 15 | 5 | True | 14 |
| KP | skipped |  |  |  |  |  | 0 |
| KZ | completed | 15 | 15 | 27 | 27 | False | 12 |
| LB | completed | 0 | 0 | 2 | 1 | True | 2 |
| LK | completed | 4 | 4 | 25 | 4 | True | 18 |
| LR | skipped |  |  |  |  |  | 0 |
| LY | completed | 1 | 1 | 4 | 1 | True | 2 |
| MA | completed | 0 | 0 | 52 | 1 | True | 51 |
| ME | skipped |  |  |  |  |  | 0 |
| ML | completed | 1 | 1 | 2 | 1 | True | 1 |
| MM | completed | 2 | 2 | 14 | 2 | True | 11 |
| MU | completed | 1 | 1 | 1 | 1 | False | 0 |
| MY | completed | 2 | 2 | 3 | 3 | False | 1 |
| NG | completed | 1 | 1 | 6 | 1 | True | 5 |
| NP | completed | 3 | 3 | 10 | 9 | True | 6 |
| OM | completed | 0 | 0 | 1 | 1 | False | 1 |
| PH | completed | 2 | 2 | 34 | 2 | True | 32 |
| PK | completed | 22 | 22 | 35 | 27 | True | 10 |
| QA | completed | 1 | 1 | 13 | 2 | True | 11 |
| RU | completed | 19 | 19 | 21 | 21 | False | 3 |
| SA | completed | 5 | 5 | 32 | 14 | True | 16 |
| SD | completed | 5 | 5 | 36 | 5 | True | 26 |
| SN | completed | 4 | 4 | 6 | 6 | False | 2 |
| SO | completed | 0 | 0 | 2 | 1 | True | 2 |
| SR | completed | 1 | 1 | 1 | 1 | False | 0 |
| SS | completed | 0 | 0 | 1 | 1 | False | 1 |
| SZ | completed | 1 | 1 | 2 | 2 | False | 1 |
| TD | completed | 1 | 1 | 1 | 1 | False | 0 |
| TG | completed | 1 | 1 | 1 | 1 | False | 0 |
| TH | completed | 7 | 7 | 74 | 7 | True | 67 |
| TJ | completed | 0 | 0 | 1 | 1 | False | 1 |
| TM | completed | 4 | 4 | 22 | 8 | True | 18 |
| TN | completed | 0 | 0 | 5 | 1 | True | 5 |
| TR | no_feasible_solution | 31 | 30 |  |  |  | 0 |
| TZ | completed | 1 | 1 | 13 | 1 | True | 11 |
| UA | completed | 4 | 4 | 35 | 7 | True | 30 |
| UG | completed | 4 | 4 | 6 | 4 | True | 3 |
| UZ | no_feasible_solution | 3 | 2 |  |  |  | 0 |
| VE | completed | 13 | 13 | 27 | 16 | True | 17 |
| VN | completed | 6 | 6 | 27 | 12 | True | 18 |
| YE | completed | 2 | 2 | 17 | 14 | True | 13 |
| ZM | no_feasible_solution | 2 | 1 |  |  |  | 0 |
| ZW | completed | 3 | 3 | 8 | 4 | True | 4 |

## Highest-impact unmatched anomalies

The complete list is in `ensemble-tuning-unlabeled-anomalies.csv`.

| Country | Start | End | Peak | Impact |
| --- | --- | --- | --- | ---: |
| MA | 2016-01-06 | 2016-02-07 | 2016-01-07 | 899.609 |
| TM | 2021-01-05 | 2021-02-15 | 2021-01-31 | 640.701 |
| CN | 2016-06-30 | 2016-08-04 | 2016-07-17 | 553.219 |
| UG | 2018-06-27 | 2018-07-15 | 2018-07-01 | 488.548 |
| CD | 2016-12-15 | 2016-12-28 | 2016-12-21 | 487.162 |
| IQ | 2019-10-06 | 2019-11-05 | 2019-10-11 | 459.213 |
| PK | 2012-09-27 | 2013-01-07 | 2012-12-09 | 439.444 |
| IN | 2020-03-16 | 2020-04-02 | 2020-03-23 | 435.503 |
| MA | 2016-02-25 | 2016-03-15 | 2016-03-02 | 384.068 |
| TM | 2023-06-03 | 2023-06-10 | 2023-06-08 | 307.071 |
| MM | 2021-05-03 | 2021-05-16 | 2021-05-07 | 260.601 |
| UG | 2021-01-18 | 2021-02-12 | 2021-01-18 | 259.586 |
| TM | 2020-04-06 | 2020-04-23 | 2020-04-10 | 252.417 |
| CN | 2016-01-15 | 2016-01-26 | 2016-01-17 | 226.646 |
| CN | 2015-09-02 | 2015-09-04 | 2015-09-04 | 206.016 |
| CN | 2016-03-05 | 2016-03-11 | 2016-03-05 | 191.940 |
| CN | 2015-09-06 | 2015-09-08 | 2015-09-06 | 179.474 |
| NP | 2018-09-27 | 2018-10-18 | 2018-10-01 | 173.973 |
| AE | 2016-07-28 | 2016-08-02 | 2016-07-30 | 168.781 |
| CD | 2017-08-07 | 2017-08-11 | 2017-08-09 | 164.227 |
| CD | 2023-12-18 | 2023-12-19 | 2023-12-18 | 154.333 |
| GB | 2020-03-23 | 2020-03-27 | 2020-03-24 | 148.920 |
| TH | 2013-07-16 | 2013-07-19 | 2013-07-16 | 142.021 |
| PH | 2020-03-12 | 2020-04-17 | 2020-03-25 | 135.849 |
| VN | 2020-12-05 | 2020-12-06 | 2020-12-06 | 130.169 |

## Files

- `ensemble-tuning-unlabeled-anomalies.csv`: all selected anomalies
  without an applicable label
- `ensemble-tuning-selected-parameters.json`: selected and leftmost
  parameters for every feasible country

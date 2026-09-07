# Claim-to-method map

This public map connects major dissertation topics to inspectable source. It does not reproduce the private claim ledger's country-year evidence or certify an empirical rerun.

| Dissertation topic | Public method route | Evidence needed for an empirical check |
|---|---|---|
| Established benchmarks | [Methods](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/docs/methods.md) | Licensed target panel, predictor vintage, folds and retained outputs |
| Main same-year nowcast, including C-047 | [Nowcast evaluation](https://github.com/skyboyrahul/political-risk-llm-nowcasting/blob/main/analysis/ml_experiments/nowcast_walk_forward.py) | Frozen monthly signals, annual target and exact cutoff configuration |
| Dependence-aware inference, including C-050 | [Panel inference](https://github.com/skyboyrahul/political-risk-llm-nowcasting/blob/main/analysis/econometrics/panel_inference_v2.py) | Aligned out-of-sample loss differences and comparison family |
| Block intervals, including C-051 | [Nowcast inference](https://github.com/skyboyrahul/political-risk-llm-nowcasting/blob/main/analysis/ml_experiments/nowcast_inference.py) | Matching prediction rows and country/year blocks |
| Temporal falsification | [Annual rotation](https://github.com/skyboyrahul/political-risk-llm-nowcasting/blob/main/analysis/ml_experiments/annual_rotation_falsification.py) | Matching panel, admissible rotations and estimator search |
| Wealth and origin comparisons | [Supplementary source](https://github.com/skyboyrahul/political-risk-dissertation-guide/tree/main/research-code) | Model-specific samples, classifications and archived measurements |
| ARIMAX comparison | [Audit limitation](arimax.md) | Retained result, historical environment and reproduction audit |

For a specific numerical claim, use the dissertation and the authorised evidence archive. The public release supplies a method route even when the result record itself cannot be redistributed.

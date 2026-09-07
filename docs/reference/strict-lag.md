# Forecast timing

A method's name does not determine whether it is a forecast. The decisive question is which information it uses relative to the target period.

| Design | Information boundary |
|---|---|
| Strict-lag annual forecast | Features available before the target year |
| Within-year nowcast | Evidence from the target year up to an explicit month cutoff |
| Completed-year aggregate | Full target-year information, interpreted as a full-year nowcast |

Inspect [strict-lag experiments](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/experiments/03_run_strictlag.py) and [raw feature construction](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/src/prs_baselines/features/raw_features.py). The nowcast [cutoff implementation](https://github.com/skyboyrahul/political-risk-llm-nowcasting/blob/main/analysis/ml_experiments/nowcast_walk_forward.py) constructs within-year features for its stated horizon.

Expanding-window evaluation separates earlier training years from later evaluation years. A minimum training window is part of the protocol. Leave-one-out validation across years does not establish an ex-ante forecasting result. Leave-one-country-out validation addresses cross-country generalisation instead.

The public nowcast adapter retains the original full-year coverage filter before
constructing cutoff features. Its sample eligibility can therefore depend on
later-month availability, even though a cutoff's feature values use only months
up to that cutoff. This is retrospective evaluation on a coverage-qualified panel;
the synthetic test does not establish a live panel-selection process using only
information known at that month.

All future-looking language should be checked against these implemented boundaries.

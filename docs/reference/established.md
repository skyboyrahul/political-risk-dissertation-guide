# Established benchmarks

The established-methods repository examines whether conventional statistical and machine learning estimators improve on persistence under explicit data and timing choices. **Classical** in an older repository reference means **established**.

| Implementation | What to inspect |
|---|---|
| [Raw features](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/src/prs_baselines/features/raw_features.py) | Target shifts, lag construction and available predictors |
| [PCA features](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/src/prs_baselines/features/pca_features.py) | Fitting the transformation on training data |
| [Walk-forward evaluation](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/src/prs_baselines/evaluation/walk_forward.py) | Temporal folds and out-of-sample predictions |
| [Estimator factories](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/src/prs_baselines/models/estimators.py) | Model definitions and hyperparameters |
| [Time-series methods](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/src/prs_baselines/models/timeseries.py) | Autoregressive and multivariate comparisons |

Read [the repository methods guide](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/docs/methods.md) for the runnable synthetic scope and licensed-input path. Level and change targets require different persistence predictions. Compare estimators only on compatible evaluation windows and samples.

The full-panel established benchmarks and the smaller monthly-signal hybrid study are separate experiments. Their rows, features and reference errors must not be interchanged.

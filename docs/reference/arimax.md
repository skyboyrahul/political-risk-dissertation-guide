# ARIMAX reproduction limitation

The retained ARIMAX-WGI result has a documented numerical reproduction limitation: the audit did not reproduce the originally reported estimate at its stated precision in the inspected environment. Its historical lineage is recorded, but lineage alone does not establish an exact rerun.

The public release retains the methodological distinction and does not substitute a nearby regenerated estimate for the reported one. [Time-series implementation](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/src/prs_baselines/models/timeseries.py) is available for inspection. The original target-bearing results and empirical audit artefacts remain outside the public release.

A complete resolution would need the matching input vintage, transformations, folds, dependencies and estimation behaviour. Until that alignment establishes the original result, the numerical discrepancy remains an explicit limitation.

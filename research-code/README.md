# Supplementary source for inspection

This directory contains selected original research implementations. No empirical
inputs or retained outputs are included. Source hashes and adaptations are in
[SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json).

| Directory | What is available | Execution boundary |
|---|---|---|
| `llm-origin-bias/analysis/paper_home_bias/` | Fixed effects, clustered and bootstrap inference, peer/version comparisons, disagreement analysis and local helpers | The pure inference helper can be exercised with synthetic arrays. Empirical producers require the original model/country registries, archived signals and auxiliary panels. |
| `llm-political-risk-signals/experiments/` | Subset and full-sample annual wealth-stratified error analysis | Requires separately licensed PRS inputs, archived model responses and the stated income classification. The subset producer additionally requires its historical assumptions file. |
| `monthly-llm-risk-signals/analysis/methods_evidence/` | Evidence-file audit and historical worked-extraction producer | Audit requires a supplied input layout. Worked extraction also requires original generation helpers excluded here; it stops with an explicit message when they are absent. |

The directory labels record the original research lineage. Access to repositories
with those names is not required to inspect the source published here.

## A bounded synthetic check

From the guide repository root:

```sh
uv sync --locked --group research
uv run --locked --group research python scripts/verify_research_source.py
```

This checks Python syntax and imports of the selected analysis modules, then
exercises the original fixed-effect residualisation and clustered covariance
helper with a generated fictional panel. It does not rerun the original
home-bias bootstrap, wealth analysis or Wikipedia worked example.

## Original-study assumptions

The empirical producers retain original-study model identifiers, country/year
scope, sample assertions and historical report prose. Those checks and comments
are part of the inspected historical implementation. Their conclusions do not
apply to arbitrary replacement inputs or to the public synthetic examples.
Inspect the calculations and the source qualifications together.

Run home-bias modules as packages from `llm-origin-bias/`, with the correct
authorised input layout. The full wealth producer accepts `--income-groups`;
the subset producer loads its excluded assumptions only at execution time and
accepts `WEALTH_ASSUMPTIONS_PATH`. No absent input is replaced with empirical
values pasted into source.

The worked-extraction producer can write a rendered prompt, copied model
response, official target change and prediction error when supplied with its
original helpers and inputs. These outputs are excluded from this public release
and must not be added to the public repository or website. Its separate frozen
prediction input is configured with `WORKED_EXTRACTION_PREDICTIONS`.

Current path defaults refer to the local source layout. These files are available
for method review, rather than a claim of self-contained empirical reproduction.
The public synthetic check does not read the private research archive, call a
model provider or use network access after dependencies are installed.

## Additional appendix implementations

`prs-nowcast/analysis/` contains the original decision-use, repeat-generation,
anchor-ablation, coverage-difficulty, anticipation-screen and design-simulation
implementations, plus selected plotting/provenance helpers. These are inspection
sources for Chapters 3 to 5 and the appendices. They do not include observations,
retained predictions, archived responses or empirical outputs. Original-study
aggregate checks and case-selection constants remain labelled research context.

| Appendix question | Source under `prs-nowcast/analysis/` | Execution boundary |
|---|---|---|
| Allocation-distance illustration | `decision_use/portfolio_risk_budget.py` | Requires the frozen panel and prediction JSON matching the recorded hashes. |
| Review-trigger rule and penalties | `decision_use/review_trigger_experiment.py` | Requires the same inputs and original experiment-plan file. |
| Repeat-generation comparisons | `repeat_generation/reanalyse.py`, `mechanism_analysis.py`, `monthly_movement_context.py`, `plot_by_model.py`, `verify.py` | Requires the original/repeated signal pairs and frozen comparison summaries; never calls a model provider. |
| Screening design simulation | `thesis_figures/n9_design_simulations.py` | Requires the retained anticipation-screen JSON for its study-specific annotation. |
| Anchor-only ablation | `methods_evidence/build_anchor_ablation.py` | Requires the licensed component panel, canonical predictions, scikit-learn and the bootstrap/inference helpers in the [nowcasting companion](https://github.com/skyboyrahul/political-risk-llm-nowcasting/tree/main/analysis/ml_experiments). Those helpers are linked rather than duplicated here. |
| Coverage and country difficulty | `coverage_difficulty/run_coverage_difficulty.py` | Requires retained error summaries, event records and coverage metadata. Its empirical route can query Wikimedia pageviews; the public verification does not execute that route. |
| Temporal anticipation screen | `temporal_anticipation/run_anticipation_screen.py` | Requires the canonical monthly signal archive. Event identities and dates specify the original research design; no signal observations are embedded. |

NumPy, pandas, SciPy, Matplotlib and seaborn cover the additional import/helper check.
Run `uv run --locked --group research python scripts/verify_appendix_source.py`
from the guide root. It imports the available modules in a separate process,
checks review-rule arithmetic and repeatability functions on fictional values,
and parses the anchor-ablation producer without claiming its external helper
imports are self-contained. It performs no empirical rerun, provider call or
pageview request. If running original producers with authorised inputs, keep
all generated outputs outside this public release. The provenance helper can
record local runtime paths; those generated manifests are excluded too.

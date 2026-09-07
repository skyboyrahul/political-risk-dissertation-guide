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

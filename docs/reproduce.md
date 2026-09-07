# Verification route

Choose the level of verification appropriate to your question. All three public repositories can be cloned without an invitation.

## Inspect the code

Start with the [nowcast methods](https://github.com/skyboyrahul/political-risk-llm-nowcasting/blob/main/docs/methods.md) and [established-methods guide](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/docs/methods.md). Their source provenance records identify unchanged files and documented adaptations for public release.

## Run the synthetic examples

Use Python and [uv](https://docs.astral.sh/uv/). These commands install pinned dependencies; they do not require a model API key or the original research data.

```sh
git clone https://github.com/skyboyrahul/political-risk-llm-nowcasting.git
cd political-risk-llm-nowcasting
uv sync --frozen --extra test
uv run --frozen python -m examples.synthetic_demo
```

In a separate checkout:

```sh
git clone https://github.com/skyboyrahul/political-risk-established-baselines.git
cd political-risk-established-baselines
uv sync --locked
uv run --locked python -m prs_baselines.synthetic_demo
```

Each example generates fictional observations and exercises selected original routines. The source README specifies its test command and precise scope. A successful run does not reproduce the dissertation's empirical estimates.

## Browse the guide locally

```sh
git clone https://github.com/skyboyrahul/political-risk-dissertation-guide.git
cd political-risk-dissertation-guide
uv sync --locked
uv run --locked zensical serve
```

## Reproduce the empirical study

Read the [data-access requirements](reviewer-access.md#what-requires-separate-access), then the [nowcast input contract](https://github.com/skyboyrahul/political-risk-llm-nowcasting/blob/main/docs/data-access.md) or [established-model input contract](https://github.com/skyboyrahul/political-risk-established-baselines/blob/main/docs/data-access.md). The public release supplies neither the licensed data nor the frozen model-response archive. Supplementary source is available for inspection; complete empirical reproduction needs the matching authorised input bundle and configuration.

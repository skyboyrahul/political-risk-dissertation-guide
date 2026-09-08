# Political Risk Dissertation Guide

Public methods and source companion for **Rahul Nundlall's MSc (Eng) dissertation in Artificial Intelligence Engineering at the University of the Witwatersrand, Johannesburg**.

[Open the public reviewer guide](https://political-risk-dissertation-guide.rahulnundlall.chatgpt.site/).

The dissertation examines monthly LLM readings of Wikipedia for same-year nowcasting of annual ICRG political-risk ratings. This repository provides the browsable chapter routes and selected supplementary analysis source. The two public science repositories are [LLM nowcasting](https://github.com/skyboyrahul/political-risk-llm-nowcasting) and [established baselines](https://github.com/skyboyrahul/political-risk-established-baselines).

## Dissertation chapters

- **Chapter 3 (Methods):** [methods route](docs/methods.md), [research map](docs/research-map.md) and links to the two implementation repositories.
- **Chapter 4 (Results):** [results route](docs/results.md) and supplementary source under [`research-code/`](research-code/), including origin-bias and wealth analyses.
- **Chapter 5 (Discussion):** [interpretation route](docs/interpretation.md) and [Chapter 5 reference](docs/reference/chapter5.md).
- **Appendices:** [thesis map](docs/reference/thesis-map.md), [source revisions](docs/source-revisions.md) and [access boundaries](docs/reviewer-access.md) distinguish public implementation from restricted empirical records.

## Public release boundary

This edition contains reviewed source and documentation. It excludes licensed observations, real prompt anchors, archived model responses, target-bearing output tables, empirical figure files and private input bundles. Synthetic demonstrations in the two science repositories exercise selected original routines and do not reproduce the dissertation's empirical results. See [public access and data](docs/reviewer-access.md) and [source provenance](docs/source-revisions.md).

**Classical means established:** older names such as `prs-classical-baselines` refer to the established statistical and machine learning methods discussed in the dissertation, without implying that they are old or outdated.

## AI assistance

The documentation, reviewer guides, diagrams and supporting explanatory materials were generated and edited with the assistance of generative AI, including OpenAI Codex. Rahul Nundlall remains responsible for the research, the accuracy of the documentation and the interpretations presented. Research claims should be checked against the code, the dissertation and the underlying evidence available under the relevant access conditions.

## Browse and verify

```sh
uv sync --locked
uv run --locked zensical build
uv run --locked python scripts/verify_site.py
uv run --locked zensical serve
```

`research-code/` contains selected supplementary research implementations. Its README states the external-input and execution boundaries. The website publishes only `dist/`; it does not package this source folder as downloadable assets.

## Copyright

Copyright is retained. This release does not grant an open-source licence. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Release checks

`python scripts/verify_public_release.py` checks every tracked file against the
reviewed public inventory. Any addition or change requires a content review and
an updated `public-files.json`. Keep licensed inputs, credentials and generated
empirical outputs outside the public release. The inventory verifies file
identity; it does not establish data rights or empirical reproducibility.

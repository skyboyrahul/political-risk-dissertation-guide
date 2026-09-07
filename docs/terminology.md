# Terminology and limits

## Classical means established

The original repository name `prs-classical-baselines` uses **classical** to refer to the **established statistical and machine learning methods** discussed in the dissertation. The public name uses established directly. Neither term implies that those methods are old or obsolete.

| Term | Meaning in this study |
|---|---|
| Political risk | The construct being studied; an ICRG rating is a measurement instrument |
| Annual PRS target | The annual ground-truth series available to the project; higher ratings indicate lower risk |
| PRS-point difference | A difference on the rating scale, distinct from a percentage improvement in prediction error |
| Same-year nowcast | An estimate for the target year using within-year information up to the stated cutoff |
| Ex-ante forecast | A prediction with information available before the target period |
| Persistence | Previous rating for a level target, or zero change for a change target |
| GDI | Rater spread, not a direct measure of accuracy or conflict intensity |
| Synthetic example | Independently generated fictional inputs used to exercise the public code |
| Retained empirical evidence | Original study inputs or outputs with recorded provenance, outside this public release |
| Unresolved lineage | An evidence link or numerical reproduction that has not been established |

See the [timing record](reference/strict-lag.md), [inference qualifications](reference/authority.md), [Wikipedia limitations](reference/wikipedia.md) and [ARIMAX audit summary](reference/arimax.md). The public export preserves those limitations.

# R3 dataset notes

- The corpus is project-authored demo material and does not represent current external platform policy.
- Dev gold was corrected before validation freeze: the first draft marked every chunk from a relevant source as relevant, including disclaimer-only chunks. Gold now includes only chunks that directly support the query.
- Validation was authored and frozen by the primary audit flow after runtime and dev iteration ended. No implementation changes may be made from validation case-level feedback.
- Retrieval metrics use answerable cases only. Safe abstention, canonical status exactness, schema failures, and grounding failures are reported separately.
- Repeated runs are observations, not additional independent cases.

## What changed and why

<!-- Briefly explain what you changed and why. -->

## Metrics (before → after)

| Metric | Before | After |
|---|---:|---:|
| Test Accuracy | | |
| Weighted F1 Score | | |

<!-- Fill in the actual results. Do not guess the values. -->

## Review checklist

- [ ] No data leakage (no target or future information in features).
- [ ] Splits are fixed; preprocessing is fitted on training data only.
- [ ] No hardcoded paths; runs on a teammate's machine.
- [ ] Seeds are set for shuffling, initialisation, and sampling.
- [ ] Metrics are computed the way the team reports them.
- [ ] `dvc push` is completed before `git push` if data or models changed.
- [ ] Notebook is restarted and run from top to bottom (if notebooks changed).
- [ ] Code style and naming are correct; linting checks pass.

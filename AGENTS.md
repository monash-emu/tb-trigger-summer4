# Agent instructions

A demonstration TB model with prevalence-triggered interventions, built on summer4 `v0.2.0a5`
(pinned by tag in `pixi.toml`). The audience is TB modellers who know summer2.

- Style: Google Python style, line length 100, black (`pixi run format`). Type-annotate every
  function.
- Model code is in `src/tb_trigger/model.py`. Every scenario is a parameter override in
  `SCENARIOS`; keep it that way so one compiled model serves all scenarios and `vmap` sweeps.
- Notebooks are explanatory documentation: clear prose, a plot for every claim, no assertions.
  Claims the notebooks make are asserted in `tests/test_model.py`. When a number in the prose
  changes, update the prose.
- Commit notebooks without outputs.
- `WORKAROUND(summer4)` marks code that works around a summer4 bug. Remove it when the pinned
  summer4 release fixes the bug.
- Checks: `pixi run format-check`, `pixi run test`, `pixi run test-notebooks`.

# Prevalence-triggered TB interventions in summer4

> *"For one of the STRIDE projects, we'd potentially be looking at an intervention that gets
> triggered once prevalence reaches a given threshold. Is it possible to do this with summer?"*

Yes. This repository is a small demonstration TB model, built on the latest
[summer4](https://github.com/monash-emu/summer4) release (`v0.2.0a5`), in which interventions
switch on when TB prevalence crosses a threshold. The model is illustrative. Its parameters are
TB orders of magnitude, not a calibration to any setting.

## How the trigger works

summer4 evaluates every flow rate from the model's current state at every solver step, so a
rate can depend on prevalence in the same way a force of infection depends on the number
infectious. The trigger has three parts, and all of them are inside the ODE:

1. **Signal.** Prevalence per 100,000 is a rate expression made of two `Reduce` sums of the
   compartments. Notifications (`FlowRef`) or time (`Time()`) can be used as the signal instead.
2. **Switch.** A smooth step, `sigmoid((signal - threshold) / width)`, wrapped with `defer`.
3. **Memory.** A side model with one unit of mass in three bookkeeping compartments
   (`waiting → scaling_up → active`). Once the mass has left `waiting` it does not return, so
   the programme stays on after it has pushed prevalence back down. The fraction `active`
   multiplies the intervention flows, which gives a gradual roll-out.

No event handling, solver restarts or two-pass runs are needed. Every scenario is the same
compiled model with different parameters, so the whole thing runs under `jax.jit`, `jax.vmap`
and `jax.grad`.

## Scenarios

| Scenario | When prevalence reaches 350 per 100,000 |
|---|---|
| `baseline` | nothing |
| `acf_triggered` | active case finding, 3-month roll-out, stays on |
| `acf_triggered_slow_rollout` | the same, 3-year roll-out |
| `acf_reactive` | ACF only while prevalence is above the threshold (a thermostat) |
| `acf_tpt_triggered` | ACF plus preventive therapy, stays on |

## Notebooks

Written for epidemiologists who know TB and summer2 but are new to summer4.

1. [`01-baseline-model`](notebooks/01-baseline-model.ipynb): the TB model built step by
   step, with a summer2 → summer4 translation table, and the baseline run.
2. [`02-triggered-interventions`](notebooks/02-triggered-interventions.ipynb): how the
   trigger works, the four intervention scenarios, checks that it fires at the right moment,
   and two modelling pitfalls.
3. [`03-thresholds-and-signals`](notebooks/03-thresholds-and-signals.ipynb): a vectorised
   threshold sweep, `jax.grad` through the trigger, a threshold × roll-out grid, and triggering
   on notifications.

## Running

Requires [pixi](https://pixi.sh).

```bash
pixi install
pixi run notebook          # open the notebooks in JupyterLab
pixi run test              # model tests
pixi run test-notebooks    # execute every notebook
```

The model lives in [`src/tb_trigger/model.py`](src/tb_trigger/model.py).

## summer4 notes

- `Reduce(where=selector)` in `v0.2.0a5` also counts compartments where the selector's property
  is absent. This model therefore names the compartment `kind` in every selector
  (`kind["population"] & state["I"]`), marked `WORKAROUND(summer4)` in the code and guarded by
  a test.
- A function wrapped with `defer` receives `Reduce`-derived arguments as `GroupedRate`
  objects, so the switch unwraps `.data` and re-wraps its result.

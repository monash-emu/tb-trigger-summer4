"""Behavioural tests for the threshold-triggered TB model.

The notebooks show these claims as figures; the assertions live here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from summer4 import Time

from tb_trigger.model import (
    DEFAULT_PARAMS,
    RUN_KWARGS,
    SCENARIOS,
    build_model,
    crossing_year,
    indicators,
    notifications_per_100k,
    people,
    prevalence_per_100k,
    run,
    save_plan,
    scenario_params,
)

THRESHOLD = DEFAULT_PARAMS["threshold"]


@pytest.fixture(scope="module")
def compiled() -> object:
    return build_model().compile()


@pytest.fixture(scope="module")
def frames(compiled: object) -> dict[str, pd.DataFrame]:
    return {name: indicators(run(compiled, scenario_params(name))) for name in SCENARIOS}


def trigger_year(frame: pd.DataFrame) -> float:
    """When half the programme mass has left ``waiting``."""
    return crossing_year(1.0 - frame["programme_waiting"], 0.5)


def test_prevalence_expression_counts_only_people(compiled: object) -> None:
    # Regression guard for the summer4 v0.2.0a5 Reduce(where=) workaround in model.py.
    y0 = compiled.initial_state(DEFAULT_PARAMS)
    data = np.asarray(y0.data).copy()
    pmap = compiled.pmap
    data[pmap.select(people("I"))] = 300.0
    data[pmap.select(people("S"))] = 1e5 - 300.0
    probe = build_model(prevalence_per_100k()).compile()
    params = {**DEFAULT_PARAMS, "threshold": 300.0, "threshold_width": 1.0}
    obs = probe.observe(2000.0, data, params)
    # trigger rate = trigger_speed * sigmoid((prevalence - 300) / 1); prevalence is exactly 300.
    trigger = float(np.asarray(obs.flows["trigger"]).sum())
    np.testing.assert_allclose(trigger, 0.5 * DEFAULT_PARAMS["trigger_speed"], rtol=1e-9)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_population_is_constant(frames: dict[str, pd.DataFrame], name: str) -> None:
    np.testing.assert_allclose(frames[name]["population"], 1e5, rtol=1e-8)
    programme = frames[name][["programme_waiting", "programme_scaling_up", "programme_active"]]
    np.testing.assert_allclose(programme.sum(axis=1), 1.0, atol=1e-8)


def test_baseline_crosses_threshold_around_2030(frames: dict[str, pd.DataFrame]) -> None:
    year = crossing_year(frames["baseline"]["prevalence"], THRESHOLD)
    assert 2028.0 < year < 2033.0
    assert frames["baseline"]["prevalence"].max() > THRESHOLD + 75.0


def test_trigger_fires_when_baseline_crosses(frames: dict[str, pd.DataFrame]) -> None:
    crossed = crossing_year(frames["baseline"]["prevalence"], THRESHOLD)
    for name in ("acf_triggered", "acf_triggered_slow_rollout", "acf_tpt_triggered"):
        assert abs(trigger_year(frames[name]) - crossed) < 0.05, name


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n != "baseline"])
def test_scenarios_match_baseline_before_trigger(
    frames: dict[str, pd.DataFrame], name: str
) -> None:
    crossed = crossing_year(frames["baseline"]["prevalence"], THRESHOLD)
    before = frames["baseline"].index < crossed - 0.1
    np.testing.assert_allclose(
        frames[name].loc[before, "prevalence"],
        frames["baseline"].loc[before, "prevalence"],
        rtol=1e-6,
    )


def test_latched_programme_reaches_full_coverage(frames: dict[str, pd.DataFrame]) -> None:
    for name in ("acf_triggered", "acf_tpt_triggered"):
        frame = frames[name]
        assert frame.loc[2035.0:, "programme_active"].min() > 0.99, name
        # Latched: prevalence falls back below the threshold but the programme stays on.
        assert frame.loc[2040.0:, "prevalence"].max() < THRESHOLD - 50.0, name


def test_slow_rollout_is_partial_after_one_year(frames: dict[str, pd.DataFrame]) -> None:
    frame = frames["acf_triggered_slow_rollout"]
    start = trigger_year(frame)
    coverage = np.interp(start + 1.0, frame.index, frame["programme_active"])
    # Exponential roll-out with mean 3 years: 1 - exp(-1/3) ~ 0.28 after one year.
    assert 0.2 < coverage < 0.35


def test_reactive_programme_holds_prevalence_at_threshold(
    frames: dict[str, pd.DataFrame],
) -> None:
    frame = frames["acf_reactive"]
    held = frame.loc[2032.0:, "prevalence"]
    assert (held - THRESHOLD).abs().max() < 2.0
    coverage = frame.loc[2032.0:, "programme_active"]
    assert 0.1 < coverage.min() < coverage.max() < 0.9


def test_intervention_ordering(frames: dict[str, pd.DataFrame]) -> None:
    def deaths(name: str) -> float:
        series = frames[name].loc[2025.0:2050.0, "tb_deaths"]
        return float(np.trapezoid(series.to_numpy(), series.index.to_numpy()))

    assert deaths("acf_tpt_triggered") < deaths("acf_triggered")
    assert deaths("acf_triggered") < deaths("acf_triggered_slow_rollout")
    assert deaths("acf_triggered_slow_rollout") < deaths("baseline")
    assert deaths("acf_reactive") < deaths("baseline")
    assert deaths("acf_triggered") < deaths("acf_reactive")


def test_time_switched_model_reproduces_the_trigger(frames: dict[str, pd.DataFrame]) -> None:
    # Before the trigger, every scenario is the baseline, so switching the programme on at the
    # baseline's crossing year must give the same answer as the in-ODE trigger.
    crossed = crossing_year(frames["baseline"]["prevalence"], THRESHOLD)
    timed = build_model(trigger_signal=Time()).compile()
    params = scenario_params("acf_triggered", threshold=crossed, threshold_width=0.001)
    frame = indicators(run(timed, params))
    np.testing.assert_allclose(
        frame["prevalence"], frames["acf_triggered"]["prevalence"], rtol=2e-3, atol=0.5
    )


def test_notification_trigger_is_self_reinforcing(compiled: object) -> None:
    by_notifications = build_model(notifications_per_100k()).compile()
    frame = indicators(run(by_notifications, scenario_params("acf_triggered", threshold=200.0)))
    baseline = indicators(run(compiled, scenario_params("baseline")))
    expected = crossing_year(baseline["notifications"], 200.0)
    # ACF adds notifications, so the first sliver of programme pulls the signal over the
    # threshold a little early, never late.
    assert expected - 0.15 < trigger_year(frame) < expected
    assert frame.loc[2030.0:, "programme_active"].min() > 0.95


def test_threshold_sweep_vmaps_and_differentiates(compiled: object) -> None:
    ts = np.arange(2020.0, 2050.01, 0.25)

    def deaths(threshold: jax.Array) -> jax.Array:
        params = {**scenario_params("acf_triggered"), "threshold": threshold}
        result = compiled.run(params, save=save_plan(ts), **RUN_KWARGS)
        return jnp.trapezoid(jnp.ravel(result["tb_death"].total().values), ts)

    thresholds = jnp.linspace(280.0, 440.0, 5)
    swept = np.asarray(jax.jit(jax.vmap(deaths))(thresholds))
    assert np.all(np.diff(swept) > 0.0)  # waiting for a higher prevalence costs lives
    slope = float(jax.grad(deaths)(360.0))
    finite = (swept[3] - swept[2]) / float(thresholds[3] - thresholds[2])
    np.testing.assert_allclose(slope, finite, rtol=0.25)

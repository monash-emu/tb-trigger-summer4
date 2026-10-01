"""A TB model whose interventions switch on when TB prevalence crosses a threshold.

The model has two kinds of compartment:

* **population** compartments, the usual TB states ``S E L I T R``;
* **programme** compartments, a tiny side model with one unit of mass that
  records whether the intervention programme is ``waiting``, ``scaling_up`` or
  ``active``.

The programme mass moves ``waiting -> scaling_up`` at a rate that is ~0 while
prevalence is below the threshold and fast once it is above it. That flow is
how the threshold is checked *inside the ODE*: summer4 evaluates the
prevalence from the current state at every solver step, so no event handling,
restarting or post-processing is needed. Because the programme compartments
hold memory, the switch latches: once triggered it stays on even if the
intervention then drives prevalence back below the threshold. Setting
``stand_down > 0`` lets the programme switch back off (a reactive policy).

The fraction of the unit programme mass that is ``active`` multiplies the
intervention flows, so a gradual ``scaling_up -> active`` transfer is a
gradual roll-out.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import numpy as np
import pandas as pd
from summer4 import (
    Compartments,
    EntryFlow,
    ExitFlow,
    FlowMass,
    FlowModel,
    FlowRef,
    GroupedOutput,
    GroupedRate,
    Param,
    Property,
    PropertyMap,
    Reduce,
    Result,
    SavePlan,
    SaveRequest,
    Time,
    TransitionFlow,
    defer,
)
from summer4.epi import ForceOfInfection, MixingMatrix
from summer4.flows.rates import RateOps
from summer4.timevarying import linear

# --- Structure -------------------------------------------------------------------------------

KIND = Property("kind", ("population", "programme"))
STATE = Property("state", ("S", "E", "L", "I", "T", "R"))
PROGRAMME = Property("programme", ("waiting", "scaling_up", "active"))
# A one-trait grouping axis, so FOI and Reduce have something to group by in an unstratified
# model (the same pattern as summer4's summer2-port notebooks).
POP = Property("pop", ("all",))

# Selectors. Every selector below names its KIND explicitly.
# WORKAROUND(summer4): in v0.2.0a5, Reduce(where=sel) also counts compartments where sel's
# property is absent (state["I"] alone would count the programme compartments, and
# PROGRAMME["active"] alone would count every person). Conjoining with KIND makes the selector
# defined on every compartment, which sidesteps that.
PEOPLE = KIND["population"]
PROGRAMME_ALL = KIND["programme"]


def people(trait: str) -> object:
    """Selector for one TB state, restricted to population compartments."""
    return PEOPLE & STATE[trait]


def programme(trait: str) -> object:
    """Selector for one programme status, restricted to programme compartments."""
    return PROGRAMME_ALL & PROGRAMME[trait]


def build_pmap() -> PropertyMap:
    """Six TB states for people, three programme statuses, one ``pop`` group."""
    return (
        PropertyMap.from_property(KIND)
        .stratify(STATE, where=PEOPLE)
        .stratify(PROGRAMME, where=PROGRAMME_ALL)
        .stratify(POP)
    )


# --- Parameters (time unit: years) -----------------------------------------------------------

DEFAULT_PARAMS: dict[str, float] = {
    # Transmission. contact_rate is the effective contacts per infectious person per year; it
    # rises linearly between rise_start and rise_end by a factor transmission_rise (worsening
    # crowding, undernutrition or similar), which is what pushes prevalence upwards.
    "contact_rate": 11.0,
    "transmission_rise": 1.8,
    "rise_start": 2020.0,
    "rise_end": 2035.0,
    "rr_reinfection": 0.21,  # relative risk of reinfection for L and R (Andrews et al. 2012)
    # Natural history (per year; Ragonnet et al. 2017 orders of magnitude).
    "stabilisation": 4.4,
    "early_activation": 0.4,
    "late_activation": 0.002,
    "self_recovery": 0.2,
    "tb_death": 0.2,
    "background_death": 1.0 / 70.0,
    # Passive case finding and treatment.
    "detection": 0.6,
    "treatment_duration": 0.5,
    "treatment_success": 0.85,
    # Interventions. These are rates at full programme coverage; the active programme fraction
    # (0 to 1) multiplies them. Zero means the intervention is not part of the scenario.
    "acf_screening": 0.0,  # fraction of the population screened per year
    "acf_sensitivity": 0.7,  # probability a screened active case is detected
    "tpt_rate": 0.0,  # rate latently infected people complete preventive therapy
    "tpt_efficacy": 0.6,
    # The trigger.
    "threshold": 350.0,  # active TB per 100,000 population
    "threshold_width": 0.1,  # per 100,000; smooths the on/off switch for the ODE solver
    "trigger_speed": 365.0,  # per year once above threshold (mean ~1 day to decide)
    "rollout_years": 0.25,  # mean time from decision to full coverage
    "stand_down": 0.0,  # per year once below threshold; 0 = latched, never switches off
}


# --- Rate expressions ------------------------------------------------------------------------


def prevalence_per_100k() -> RateOps:
    """Active TB (state ``I``) per 100,000 population, evaluated from the current state."""
    return 1e5 * Reduce(sum_over=POP, where=people("I")) / Reduce(sum_over=POP, where=PEOPLE)


def notifications_per_100k() -> RateOps:
    """Current notification rate per 100,000 per year: passive plus active case finding."""
    detected = FlowRef("detection").sum() + FlowRef("acf_detection").sum()
    return 1e5 * detected / Reduce(sum_over=POP, where=PEOPLE)


def _smooth_step(signal: GroupedRate, threshold: object, width: object) -> GroupedRate:
    # defer passes a Reduce-derived argument as a GroupedRate; work on .data and re-wrap.
    data = signal.data if isinstance(signal, GroupedRate) else signal
    on = jax.nn.sigmoid((data - threshold) / width)
    return GroupedRate(on, signal.properties) if isinstance(signal, GroupedRate) else on


def above_threshold(signal: RateOps, threshold: object, width: object) -> RateOps:
    """0 well below ``threshold``, 1 well above it, a logistic ramp of scale ``width`` between.

    ``signal`` is any rate expression (prevalence, notifications, ...). A hard
    ``signal > threshold`` also works with fixed-step Euler, but the smooth form keeps adaptive
    solvers efficient and keeps outcomes differentiable with respect to the threshold.
    """
    return defer(_smooth_step, name="tb_trigger.smooth_step")(signal, threshold, width)


def transmission_multiplier() -> RateOps:
    """1 before ``rise_start``, rising linearly to ``transmission_rise`` at ``rise_end``."""
    return linear(
        Time(), [Param("rise_start"), Param("rise_end")], [1.0, Param("transmission_rise")]
    )


# --- The model -------------------------------------------------------------------------------


def build_model(trigger_signal: RateOps | None = None) -> FlowModel:
    """Build the TB model with a threshold-triggered programme.

    Args:
        trigger_signal: The quantity compared with ``Param("threshold")``. Defaults to
            :func:`prevalence_per_100k`; :func:`notifications_per_100k` or any other rate
            expression can be passed instead.

    Returns:
        An uncompiled ``FlowModel`` with an initial population attached. All scenarios share
        this structure; they differ only in parameters.
    """
    signal = prevalence_per_100k() if trigger_signal is None else trigger_signal
    switch_on = above_threshold(signal, Param("threshold"), Param("threshold_width"))
    coverage = Reduce(sum_over=POP, where=programme("active"))

    model = FlowModel(build_pmap())

    # Transmission: frequency-dependent, denominator is people only.
    foi = ForceOfInfection(
        "infection",
        infectious=people("I"),
        group_by=POP,
        mixing=MixingMatrix(POP, [[1.0]], check_reciprocal=False),
        contact_rate=Param("contact_rate") * transmission_multiplier(),
        denominator=PEOPLE,
    )
    reinfection = foi * Param("rr_reinfection")
    model.add_flow(TransitionFlow("infection", people("S"), people("E"), foi))
    model.add_flow(TransitionFlow("reinfection_latent", people("L"), people("E"), reinfection))
    model.add_flow(TransitionFlow("reinfection_recovered", people("R"), people("E"), reinfection))

    # Natural history.
    model.add_flow(
        TransitionFlow("stabilisation", people("E"), people("L"), Param("stabilisation"))
    )
    model.add_flow(
        TransitionFlow("early_activation", people("E"), people("I"), Param("early_activation"))
    )
    model.add_flow(
        TransitionFlow("late_activation", people("L"), people("I"), Param("late_activation"))
    )
    model.add_flow(
        TransitionFlow("self_recovery", people("I"), people("R"), Param("self_recovery"))
    )

    # Case finding and treatment.
    model.add_flow(TransitionFlow("detection", people("I"), people("T"), Param("detection")))
    acf_rate = Param("acf_screening") * Param("acf_sensitivity") * coverage
    model.add_flow(TransitionFlow("acf_detection", people("I"), people("T"), acf_rate))
    per_course = 1.0 / Param("treatment_duration")
    model.add_flow(
        TransitionFlow(
            "treatment_success",
            people("T"),
            people("R"),
            Param("treatment_success") * per_course,
        )
    )
    model.add_flow(
        TransitionFlow(
            "treatment_failure",
            people("T"),
            people("I"),
            (1.0 - Param("treatment_success")) * per_course,
        )
    )

    # Preventive therapy for the latently infected.
    tpt_rate = Param("tpt_rate") * Param("tpt_efficacy") * coverage
    model.add_flow(TransitionFlow("tpt_early", people("E"), people("R"), tpt_rate))
    model.add_flow(TransitionFlow("tpt_late", people("L"), people("R"), tpt_rate))

    # Deaths, with births replacing them so the population stays constant.
    tb_death = model.add_flow(ExitFlow("tb_death", people("I"), Param("tb_death")))
    background = model.add_flow(ExitFlow("background_death", PEOPLE, Param("background_death")))
    model.add_flow(EntryFlow("birth", people("S"), tb_death.sum() + background.sum()))

    # The programme switch.
    model.add_flow(
        TransitionFlow(
            "trigger",
            programme("waiting"),
            programme("scaling_up"),
            Param("trigger_speed") * switch_on,
        )
    )
    model.add_flow(
        TransitionFlow(
            "roll_out", programme("scaling_up"), programme("active"), 1.0 / Param("rollout_years")
        )
    )
    model.add_flow(
        TransitionFlow(
            "stand_down",
            programme("active"),
            programme("waiting"),
            Param("stand_down") * (1.0 - switch_on),
        )
    )

    model.set_initial_population(
        {people("S"): 1e5 - 10.0, people("I"): 10.0, programme("waiting"): 1.0}
    )
    return model


# --- Scenarios -------------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, float]] = {
    "baseline": {},
    "acf_triggered": {"acf_screening": 0.5},
    "acf_triggered_slow_rollout": {"acf_screening": 0.5, "rollout_years": 3.0},
    "acf_reactive": {"acf_screening": 0.5, "stand_down": 365.0},
    "acf_tpt_triggered": {"acf_screening": 0.5, "tpt_rate": 0.1},
}

SCENARIO_LABELS: dict[str, str] = {
    "baseline": "Baseline (no intervention)",
    "acf_triggered": "ACF on trigger",
    "acf_triggered_slow_rollout": "ACF on trigger, 3-year roll-out",
    "acf_reactive": "Reactive ACF (stands down below threshold)",
    "acf_tpt_triggered": "ACF + TPT on trigger",
}


def scenario_params(name: str, **overrides: float) -> dict[str, float]:
    """Default parameters, then the named scenario's changes, then ``overrides``."""
    return {**DEFAULT_PARAMS, **SCENARIOS[name], **overrides}


# --- Running and outputs ---------------------------------------------------------------------

T0, T1 = 1850.0, 2050.0
TS = np.round(np.linspace(1990.0, T1, 601), 1)

RUN_KWARGS: dict[str, object] = {
    "t0": T0,
    "t1": T1,
    "dt": 0.1,
    "solver": "tsit5",
    "rtol": 1e-6,
    "atol": 1e-8,
}


def save_plan(ts: np.ndarray = TS) -> SavePlan:
    """The outputs every notebook uses: compartments and the flows behind the indicators."""

    def flow(name: str) -> SaveRequest:
        return SaveRequest(FlowMass(name, sum_over=(POP, "source")))

    return SavePlan(
        requests={
            "compartments": SaveRequest(Compartments()),
            "force_of_infection": SaveRequest(GroupedOutput("infection")),
            **{
                name: flow(name)
                for name in (
                    "early_activation",
                    "late_activation",
                    "detection",
                    "acf_detection",
                    "tb_death",
                    "trigger",
                )
            },
        },
        ts=ts,
    )


def run(cm: object, params: Mapping[str, float], ts: np.ndarray = TS) -> Result:
    """Run a compiled model from 1850 (burn-in) to 2050, saving ``ts``."""
    return cm.run(dict(params), save=save_plan(ts), **RUN_KWARGS)


def indicators(result: Result) -> pd.DataFrame:
    """Per-100,000 indicators and programme status, indexed by year."""
    comp = result["compartments"]

    def size(selector: object) -> np.ndarray:
        return np.asarray(comp.select(selector).total().values).ravel()

    def mass(name: str) -> np.ndarray:
        return np.asarray(result[name].total().values).ravel()

    population = size(PEOPLE)
    per_100k = 1e5 / population
    frame = pd.DataFrame(
        {
            "prevalence": size(people("I")) * per_100k,
            "incidence": (mass("early_activation") + mass("late_activation")) * per_100k,
            "notifications": (mass("detection") + mass("acf_detection")) * per_100k,
            "acf_notifications": mass("acf_detection") * per_100k,
            "tb_deaths": mass("tb_death") * per_100k,
            # Annual risk of infection: probability a susceptible person is infected in a year.
            "ari": 1.0 - np.exp(-np.asarray(result["force_of_infection"].values.data).ravel()),
            "programme_waiting": size(programme("waiting")),
            "programme_scaling_up": size(programme("scaling_up")),
            "programme_active": size(programme("active")),
            "population": population,
        },
        index=pd.Index(np.asarray(comp.times.values).ravel(), name="year"),
    )
    return frame


def crossing_year(series: pd.Series, threshold: float) -> float:
    """First year ``series`` reaches ``threshold`` (linear interpolation), or NaN if never."""
    values = series.to_numpy()
    years = series.index.to_numpy()
    above = np.nonzero(values >= threshold)[0]
    if above.size == 0:
        return float("nan")
    k = int(above[0])
    if k == 0:
        return float(years[0])
    frac = (threshold - values[k - 1]) / (values[k] - values[k - 1])
    return float(years[k - 1] + frac * (years[k] - years[k - 1]))


# --- Describing the model --------------------------------------------------------------------


def short_names(pmap: PropertyMap) -> list[str]:
    """``S``, ``E``, ... for people and ``programme: waiting``, ... for the switch."""
    names = [""] * pmap.size
    for trait, indices in pmap.partition(STATE).items():
        for i in indices:
            names[int(i)] = trait.name
    for trait, indices in pmap.partition(PROGRAMME).items():
        for i in indices:
            names[int(i)] = f"programme: {trait.name}"
    return names


def flow_table(model: FlowModel) -> pd.DataFrame:
    """One row per flow: its type and the compartments it leaves and enters."""
    pmap = model.pmap
    names = short_names(pmap)

    def where(selector: object) -> str:
        if selector is None:
            return ""
        return ", ".join(names[i] for i in pmap.select(selector))

    rows = []
    for flow in model.flows:
        rows.append(
            {
                "flow": flow.name,
                "type": type(flow).__name__,
                "from": where(getattr(flow, "source", None)),
                "to": where(getattr(flow, "dest", None)),
            }
        )
    return pd.DataFrame(rows).set_index("flow")

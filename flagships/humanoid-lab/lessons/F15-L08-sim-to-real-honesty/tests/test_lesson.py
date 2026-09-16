"""Autograder rubric for F15-L08. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These cases deliberately differ from the notebook's public `_check_*` functions. They probe
the failure modes a plausible-looking wrong implementation has: a latency pipeline that does
not start full, noise written into the world rather than the observation, a mass edit that
compounds, a sampler that reuses one rollout seed, and a requirements check whose VRAM
comparison is strict when it should not be.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

KEYS = {"survived", "steps", "rms_lean", "saturated_fraction"}


def _fresh():
    return lesson.load_balancer()


# --- exercise 1: rollout ------------------------------------------------------------------

def test_rollout_shape_and_clean_baseline():
    """The four documented keys, and the known-good gains must survive the clean model."""
    model, data = _fresh()
    r = lesson.rollout(model, data, lesson.BASELINE_GAINS)
    assert isinstance(r, dict), f"expected a dict, got {type(r).__name__}"
    assert set(r) == KEYS, f"keys were {sorted(r)}, expected {sorted(KEYS)}"
    assert bool(r["survived"]), "the baseline gains must survive the clean model"
    assert r["steps"] == lesson.HORIZON_STEPS, (
        f"survived but only {r['steps']} steps; a survivor runs the full horizon"
    )
    assert 0.0 < r["rms_lean"] < 0.2, f"rms_lean {r['rms_lean']} is not a plausible RMS angle"


def test_rollout_is_deterministic_and_resets():
    """Repeat calls must agree: the episode resets, and no state leaks between runs."""
    model, data = _fresh()
    a = lesson.rollout(model, data, lesson.BASELINE_GAINS)
    b = lesson.rollout(model, data, lesson.BASELINE_GAINS)
    assert a == b, (
        "two identical rollouts on the same data disagreed — call reset_to_lean at the start "
        "so the second episode does not begin wherever the first one ended"
    )
    fresh_model, fresh_data = _fresh()
    c = lesson.rollout(fresh_model, fresh_data, lesson.BASELINE_GAINS)
    assert abs(c["rms_lean"] - a["rms_lean"]) < 1e-9, "a fresh model gave a different answer"


def test_rollout_latency_pipeline_starts_full():
    """Delay must bite immediately, and enough of it must topple the machine."""
    model, data = _fresh()
    clean = lesson.rollout(model, data, lesson.BASELINE_GAINS, delay_steps=0)
    slow = lesson.rollout(model, data, lesson.BASELINE_GAINS, delay_steps=2)
    assert slow["rms_lean"] > clean["rms_lean"] * 1.3, (
        f"delay_steps=2 moved rms_lean only {clean['rms_lean']:.4f} -> {slow['rms_lean']:.4f}; "
        "the pipeline must start with delay_steps zero commands already in it"
    )
    assert slow["saturated_fraction"] > clean["saturated_fraction"], (
        "latency should drive the actuator harder into its limit, not less"
    )
    toppled = lesson.rollout(model, data, lesson.BASELINE_GAINS, delay_steps=20)
    assert not toppled["survived"], (
        "80 ms of control latency must topple these gains; if it survives, the delay is not "
        "being applied at all"
    )


def test_rollout_noise_corrupts_observation_not_state():
    """Noise must change the outcome, replay under a seed, and differ across seeds."""
    model, data = _fresh()
    clean = lesson.rollout(model, data, lesson.BASELINE_GAINS, noise_std=0.0)
    n1 = lesson.rollout(model, data, lesson.BASELINE_GAINS, noise_std=0.08, seed=1)
    n1_again = lesson.rollout(model, data, lesson.BASELINE_GAINS, noise_std=0.08, seed=1)
    n2 = lesson.rollout(model, data, lesson.BASELINE_GAINS, noise_std=0.08, seed=2)
    assert n1 == n1_again, "same seed must replay exactly — build the Generator once"
    assert n1["rms_lean"] != n2["rms_lean"], "different seeds must give different noise"
    assert n1["rms_lean"] > clean["rms_lean"], "noise should make tracking worse, not better"
    heavy = lesson.rollout(model, data, lesson.BASELINE_GAINS, noise_std=0.20, seed=1)
    assert not heavy["survived"], (
        "0.20 rad of sensor noise must topple it; if it survives, the noise is probably being "
        "added to data.qpos (the world) rather than to the controller's observation"
    )


def test_rollout_saturation_is_measured_on_the_raw_command():
    """Saturation counts what was ASKED for, not what MuJoCo allowed through."""
    model, data = _fresh()
    r = lesson.rollout(model, data, lesson.BASELINE_GAINS)
    assert 0.0 < r["saturated_fraction"] < 1.0, (
        f"saturated_fraction {r['saturated_fraction']} — the baseline gains saturate for part "
        "of the episode; 0.0 means you measured the clamped command against the limit"
    )
    tiny = lesson.rollout(model, data, np.asarray(lesson.BASELINE_GAINS) * 0.02)
    assert tiny["steps"] > 50, (
        f"the 50x-smaller gains ran only {tiny['steps']} steps, so a zero saturation fraction "
        "would prove nothing — the episode has to actually run for the next check to mean "
        "anything, which points at reset_to_lean or the sign of the feedback"
    )
    assert tiny["saturated_fraction"] == 0.0, (
        "gains 50x smaller cannot reach a 40 N*m limit, so saturation must be exactly zero"
    )
    huge = lesson.rollout(model, data, np.asarray(lesson.BASELINE_GAINS) * 50.0)
    assert huge["saturated_fraction"] > 0.9, "gains 50x larger must saturate almost always"


# --- exercise 2: apply_parameters ---------------------------------------------------------

def test_apply_parameters_sets_from_baseline():
    """Repeated calls must not compound, and defaults must restore."""
    model, data = _fresh()
    for scale in (1.3, 1.3, 0.9, 1.3):
        lesson.apply_parameters(model, data, mass_scale=scale)
        got = float(model.body_mass[lesson.TORSO_ID])
        assert abs(got - lesson.BASE_TORSO_MASS * scale) < 1e-9, (
            f"after mass_scale={scale} the torso mass is {got:.4f}, expected "
            f"{lesson.BASE_TORSO_MASS * scale:.4f} — you are scaling the current mass"
        )
    lesson.apply_parameters(model, data)
    assert abs(float(model.body_mass[lesson.TORSO_ID]) - lesson.BASE_TORSO_MASS) < 1e-9, (
        "the default mass_scale of 1.0 must restore the baseline mass"
    )


def test_apply_parameters_sets_friction_and_recomputes_constants():
    """Friction lands on every dof, and a mass change actually reaches the dynamics."""
    model, data = _fresh()
    lesson.apply_parameters(model, data, frictionloss=4.5)
    assert np.allclose(model.dof_frictionloss, 4.5), (
        f"dof_frictionloss is {model.dof_frictionloss}, expected every entry at 4.5"
    )
    lesson.apply_parameters(model, data)
    assert np.allclose(model.dof_frictionloss, 0.0), "the default must restore zero friction"
    lesson.apply_parameters(model, data, mass_scale=2.0)
    heavy = lesson.rollout(model, data, lesson.BASELINE_GAINS)
    lesson.apply_parameters(model, data, mass_scale=1.0)
    light = lesson.rollout(model, data, lesson.BASELINE_GAINS)
    assert heavy != light, (
        "doubling the torso mass changed nothing about the rollout — the mass was written but "
        "the physics never saw it"
    )
    # mj_setConst is what rebuilds the quantities MuJoCo DERIVED from the old mass. On this
    # two-link model the mass matrix is rebuilt every step anyway, so no rollout can detect
    # its absence; the derived bookkeeping can, so check that directly.
    lesson.apply_parameters(model, data, mass_scale=2.0)
    expected_total = float(model.body_mass.sum())
    got_total = float(model.body_subtreemass[0])
    assert abs(got_total - expected_total) < 1e-9, (
        f"the bodies now weigh {expected_total:.4f} kg but body_subtreemass[0] is still "
        f"{got_total:.4f} — you wrote body_mass without calling mujoco.mj_setConst(model, "
        "data), so every quantity MuJoCo derived from the old mass is stale"
    )
    lesson.apply_parameters(model, data)


# --- exercise 3: evaluate -----------------------------------------------------------------

def test_evaluate_routes_every_key():
    """Each of the five condition keys must reach the right place."""
    model, data = _fresh()
    base = lesson.evaluate(model, data, lesson.BASELINE_GAINS, lesson.NOMINAL)
    assert set(base) == KEYS, f"evaluate returned keys {sorted(base)}"
    assert base["survived"], "the nominal condition must survive"
    slow = lesson.evaluate(model, data, lesson.BASELINE_GAINS,
                           dict(lesson.NOMINAL, delay_steps=20))
    assert not slow["survived"], "delay_steps is not reaching rollout"
    heavy = lesson.evaluate(model, data, lesson.BASELINE_GAINS,
                            dict(lesson.NOMINAL, mass_scale=1.3))
    assert not heavy["survived"], "mass_scale is not reaching apply_parameters"
    noisy_a = lesson.evaluate(model, data, lesson.BASELINE_GAINS,
                              dict(lesson.NOMINAL, noise_std=0.08, seed=5))
    noisy_b = lesson.evaluate(model, data, lesson.BASELINE_GAINS,
                              dict(lesson.NOMINAL, noise_std=0.08, seed=6))
    assert noisy_a["rms_lean"] != noisy_b["rms_lean"], "the condition's seed is being ignored"
    fric = lesson.evaluate(model, data, lesson.BASELINE_GAINS,
                           dict(lesson.NOMINAL, frictionloss=25.0))
    assert fric["rms_lean"] > base["rms_lean"], "frictionloss is not reaching apply_parameters"


def test_evaluate_does_not_leak_between_conditions():
    """The classic bug: condition N silently inherits condition N-1's model."""
    model, data = _fresh()
    before = lesson.evaluate(model, data, lesson.BASELINE_GAINS, lesson.NOMINAL)
    lesson.evaluate(model, data, lesson.BASELINE_GAINS,
                    dict(lesson.NOMINAL, mass_scale=1.3, frictionloss=20.0))
    after = lesson.evaluate(model, data, lesson.BASELINE_GAINS, lesson.NOMINAL)
    assert after == before, (
        "the nominal condition gave a different answer after a heavy, high-friction one ran; "
        "the model kept the previous condition's parameters"
    )


# --- exercise 4: sample_conditions --------------------------------------------------------

def test_sample_conditions_reproducible_and_in_range():
    """A seed must replay exactly, and every draw must respect the spec."""
    a = lesson.sample_conditions(25, seed=11)
    assert len(a) == 25, f"asked for 25, got {len(a)}"
    assert a == lesson.sample_conditions(25, seed=11), "same seed must reproduce exactly"
    assert a != lesson.sample_conditions(25, seed=12), "different seeds must differ"
    for c in a:
        assert sorted(c) == sorted(lesson.NOMINAL), f"condition keys {sorted(c)}"
        for key in ("mass_scale", "frictionloss", "noise_std"):
            lo, hi = lesson.DR_SPEC[key]
            assert lo <= c[key] <= hi, f"{key}={c[key]} outside [{lo}, {hi}]"
        lo, hi = lesson.DR_SPEC["delay_steps"]
        assert lo <= c["delay_steps"] < hi, f"delay_steps={c['delay_steps']} outside spec"
        assert isinstance(c["delay_steps"], int), "delay_steps must be a plain int"
    # The documented draw ORDER is part of the contract. Pin it here so that drawing in a
    # different (still in-range, still reproducible) order is named precisely, instead of
    # surfacing later as an unexplained domain-randomisation failure.
    rng = np.random.default_rng(11)
    expected_first = {
        "mass_scale": float(rng.uniform(*lesson.DR_SPEC["mass_scale"])),
        "frictionloss": float(rng.uniform(*lesson.DR_SPEC["frictionloss"])),
        "delay_steps": int(rng.integers(*lesson.DR_SPEC["delay_steps"])),
        "noise_std": float(rng.uniform(*lesson.DR_SPEC["noise_std"])),
        "seed": int(rng.integers(0, 10000)),
    }
    assert a[0] == expected_first, (
        f"your first condition is {a[0]}, but drawing in the documented order gives "
        f"{expected_first} — take mass_scale, frictionloss, delay_steps, noise_std and then "
        "the rollout seed, in that order, from a single default_rng(seed)"
    )


def test_sample_conditions_draws_a_fresh_rollout_seed_each_time():
    """One generator, one seed per condition — not the argument reused N times."""
    got = lesson.sample_conditions(30, seed=3)
    seeds = [c["seed"] for c in got]
    assert len(set(seeds)) > 20, (
        f"only {len(set(seeds))} distinct rollout seeds across 30 conditions — draw a fresh "
        "one per condition from the same generator"
    )
    assert seeds != [3] * 30, "the conditions all reuse the sampler's own seed argument"
    spread = lesson.sample_conditions(200, seed=0)
    masses = np.array([c["mass_scale"] for c in spread])
    lo, hi = lesson.DR_SPEC["mass_scale"]
    assert masses.min() < lo + 0.1 and masses.max() > hi - 0.1, (
        "200 draws did not span the mass range; the distribution is not uniform over the spec"
    )


def test_domain_randomisation_narrows_the_held_out_gap():
    """The lesson's whole claim, graded: DR must beat nominal on unseen parameters."""
    model, data = _fresh()
    held_out = lesson.sample_conditions(lesson.HELD_OUT_N, seed=lesson.HELD_OUT_SEED)
    train_dr = lesson.sample_conditions(lesson.DR_TRAIN_N, seed=lesson.DR_TRAIN_SEED)
    gains_nominal = lesson.cem_tune([lesson.NOMINAL])
    gains_dr = lesson.cem_tune(train_dr)
    on_nominal = lesson.survival_rate(model, data, gains_nominal, [lesson.NOMINAL])
    nom_held = lesson.survival_rate(model, data, gains_nominal, held_out)
    dr_held = lesson.survival_rate(model, data, gains_dr, held_out)
    assert on_nominal == 1.0, "the nominally-tuned controller must be perfect on its own model"
    assert dr_held > nom_held, (
        f"DR scored {dr_held:.0%} on held-out conditions against nominal's {nom_held:.0%}; "
        "tuning against the randomised ensemble is supposed to win there"
    )
    assert dr_held >= 0.75, (
        f"DR-tuned survival on held-out conditions was only {dr_held:.0%}; the ensemble is "
        "being tuned against, but something in the pipeline is losing most of its benefit"
    )


# --- exercise 5: can_run_isaac_sim --------------------------------------------------------

def test_can_run_isaac_sim_accepts_the_documented_minimum():
    """NVIDIA's own minimum and ideal cards must pass, with no reasons."""
    for gpu in lesson.GPU_LEDGER:
        if gpu["named_by_nvidia"]:
            v = lesson.can_run_isaac_sim(gpu)
            assert set(v) == {"ok", "reasons"}, f"keys were {sorted(v)}"
            assert bool(v["ok"]), f"{gpu['name']} should pass, got {v['reasons']}"
            assert list(v["reasons"]) == [], f"{gpu['name']} passed but listed {v['reasons']}"


def test_can_run_isaac_sim_t4_fails_only_for_not_being_named():
    """The lesson's central correction, graded: the T4 has RT Cores and enough VRAM."""
    t4 = next(g for g in lesson.GPU_LEDGER if g["name"] == "NVIDIA T4")
    v = lesson.can_run_isaac_sim(t4)
    assert not v["ok"], "the T4 is not a GPU NVIDIA names, so it must not pass"
    assert len(v["reasons"]) == 1, (
        f"the T4 produced {len(v['reasons'])} reasons: {v['reasons']}. It HAS RT Cores and it "
        "meets the 16 GiB floor exactly, so a VRAM reason means your comparison is > not >=, "
        "and an RT-core reason means you copied the folklore instead of the ledger"
    )
    l4 = next(g for g in lesson.GPU_LEDGER if g["name"] == "NVIDIA L4")
    assert len(lesson.can_run_isaac_sim(l4)["reasons"]) == 1, "the L4 also fails exactly once"


def test_can_run_isaac_sim_counts_every_independent_failure():
    """An A100 fails two checks; a CPU fails all three; the requirement is a parameter."""
    a100 = next(g for g in lesson.GPU_LEDGER if g["name"] == "NVIDIA A100")
    va = lesson.can_run_isaac_sim(a100)
    assert not va["ok"] and len(va["reasons"]) == 2, (
        f"the A100 has 40 GiB but no RT Cores and is unnamed: expected 2 reasons, got "
        f"{va['reasons']}"
    )
    cpu = next(g for g in lesson.GPU_LEDGER if g["vram_gib"] == 0)
    vc = lesson.can_run_isaac_sim(cpu)
    assert not vc["ok"] and len(vc["reasons"]) == 3, (
        f"a CPU fails all three checks: expected 3 reasons, got {vc['reasons']}"
    )
    relaxed = dict(lesson.ISAAC_MIN, min_vram_gib=0, requires_rt_cores=False)
    assert len(lesson.can_run_isaac_sim(cpu, relaxed)["reasons"]) == 1, (
        "with the VRAM floor and RT-core rule relaxed, only the 'not named' reason should "
        "remain — you are reading ISAAC_MIN directly instead of the `requirement` argument"
    )


RUBRIC = [
    (test_rollout_shape_and_clean_baseline, 3,
     "return the four documented keys; the baseline gains are known good"),
    (test_rollout_is_deterministic_and_resets, 2,
     "reset_to_lean at the top of every episode, before the loop"),
    (test_rollout_latency_pipeline_starts_full, 2,
     "the pipeline starts as delay_steps zero commands, then append-then-pop(0) each step"),
    (test_rollout_noise_corrupts_observation_not_state, 2,
     "add noise to the controller's view of the state, never to data.qpos"),
    (test_rollout_saturation_is_measured_on_the_raw_command, 2,
     "compare the command you computed against actuator_ctrlrange, before MuJoCo clamps it"),
    (test_apply_parameters_sets_from_baseline, 3,
     "mass = BASE_TORSO_MASS * scale, never mass *= scale"),
    (test_apply_parameters_sets_friction_and_recomputes_constants, 2,
     "set every dof_frictionloss entry, then call mujoco.mj_setConst(model, data)"),
    (test_evaluate_routes_every_key, 2,
     "mass and friction are model edits; delay, noise and seed are rollout arguments"),
    (test_evaluate_does_not_leak_between_conditions, 2,
     "apply_parameters on every call, so the previous condition cannot persist"),
    (test_sample_conditions_reproducible_and_in_range, 3,
     "one default_rng(seed), drawn in the documented order, ranges straight from the spec"),
    (test_sample_conditions_draws_a_fresh_rollout_seed_each_time, 2,
     "each condition takes its own rng.integers(0, 10000) as its rollout seed"),
    (test_domain_randomisation_narrows_the_held_out_gap, 3,
     "hand cem_tune the randomised list; score both controllers on the same held-out set"),
    (test_can_run_isaac_sim_accepts_the_documented_minimum, 1,
     "ok is True exactly when the reasons list is empty"),
    (test_can_run_isaac_sim_t4_fails_only_for_not_being_named, 1,
     "VRAM must compare with >=; the T4 meets 16 GiB exactly and does have RT Cores"),
    (test_can_run_isaac_sim_counts_every_independent_failure, 2,
     "append one reason per failed check, and read the `requirement` argument rather than the "
     "ISAAC_MIN global — the grader relaxes the requirement to check you did"),
]

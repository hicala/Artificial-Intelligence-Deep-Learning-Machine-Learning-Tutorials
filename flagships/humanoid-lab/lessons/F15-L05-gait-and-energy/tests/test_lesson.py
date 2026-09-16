"""Autograder rubric for F15-L05. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These cases deliberately differ from the notebook's public `_check_*` functions. They probe
the failure modes a plausible-looking wrong implementation has: two legs driven in phase, a
knee that flexes the wrong way, a bias smeared onto the knees, a `knee_lag` that is accepted
and then ignored, an energy sum that forgets the timestep or the absolute value, a rollout
that never resets, a cost of transport missing `g`, and a search that quietly ignores its cap.

No test asserts a particular gait. The search is graded on the MARGIN it opens over the
baseline, so a student whose machine finds a different optimum still scores full marks.
"""
import importlib.util, math, os
from pathlib import Path

import mujoco
import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

KEYS = {"distance", "energy", "steps", "fell"}


def _reference_rollout(params, horizon=None):
    """Roll the walker out here, using the STUDENT's gait_targets.

    Written independently of `lesson.rollout` so it can grade it. It reads the student's gait,
    so a wrong gait does not also fail the energy test through the back door — both sides move
    together and the comparison stays about the accumulation, which is what this test grades.
    """
    horizon = lesson.HORIZON_STEPS if horizon is None else horizon
    model, data = lesson.load_walker()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    x0 = float(data.qpos[0])
    energy, steps, fell = 0.0, 0, False
    for i in range(horizon):
        data.ctrl[:] = lesson.gait_targets(i * lesson.DT, params)
        mujoco.mj_step(model, data)
        energy += float(np.abs(data.actuator_force * data.actuator_velocity).sum()) * lesson.DT
        steps += 1
        if data.qpos[1] < lesson.MIN_HIP_HEIGHT or abs(data.qpos[2]) > lesson.MAX_PITCH:
            fell = True
            break
    return {"distance": float(data.qpos[0]) - x0, "energy": energy, "steps": steps, "fell": fell}


# --- exercise 1: the gait parameterisation -------------------------------------------------

def test_gait_targets_shape_and_antiphase():
    """Four targets in actuator order, and the legs half a cycle apart."""
    p = {"freq": 1.0, "hip_amp": 0.4, "knee_amp": 0.5, "knee_lag": 0.0, "hip_bias": 0.07}
    t_quarter = 0.25 / p["freq"]          # phase = pi/2, so sin(phase) = +1
    out = np.asarray(lesson.gait_targets(t_quarter, p), dtype=float)
    assert out.shape == (4,), f"expected 4 targets (hip_r, knee_r, hip_l, knee_l), got {out.shape}"
    assert abs(out[0] - (p["hip_amp"] + p["hip_bias"])) < 1e-9, (
        f"right hip is {out[0]:.4f}; a quarter cycle in, sin(pi/2) = 1, so it should be "
        f"hip_amp + hip_bias = {p['hip_amp'] + p['hip_bias']:.4f}"
    )
    assert abs(out[2] - (-p["hip_amp"] + p["hip_bias"])) < 1e-9, (
        f"left hip is {out[2]:.4f}, expected {-p['hip_amp'] + p['hip_bias']:.4f}. The left leg "
        "is half a cycle behind the right: add leg_phase to its phase before the sine."
    )


def test_gait_targets_knee_never_extends_backwards():
    """The knee joint's range is [-1.6, 0]; a target above 0 is a knee bending backwards."""
    p = {"freq": 1.1, "hip_amp": 0.3, "knee_amp": 0.8, "knee_lag": 1.0, "hip_bias": 0.0}
    knees = np.array([[lesson.gait_targets(t, p)[1], lesson.gait_targets(t, p)[3]]
                      for t in np.linspace(0.0, 1.0 / p["freq"], 200)])
    assert knees.max() <= 1e-9, (
        f"a knee target reached {knees.max():+.4f}. Knees only flex, so every target must be "
        "<= 0 — the (1 - cos) envelope is multiplied by MINUS knee_amp."
    )
    assert knees.min() >= -p["knee_amp"] - 1e-9, (
        f"a knee target reached {knees.min():+.4f}, past -knee_amp = {-p['knee_amp']:.4f}. "
        "Divide the (1 - cos) envelope by 2; it spans 0 to 2 on its own."
    )
    assert abs(knees.min() + p["knee_amp"]) < 1e-3, (
        f"the deepest knee flexion was {knees.min():+.4f}, but knee_amp = {p['knee_amp']:.4f} "
        "should be reached once per cycle. Check the envelope spans its full range."
    )


def test_gait_targets_bias_moves_hips_only():
    """hip_bias is a forward lean of the hips; it is not a knee parameter."""
    p0 = {"freq": 1.0, "hip_amp": 0.3, "knee_amp": 0.6, "knee_lag": 2.0, "hip_bias": 0.0}
    p1 = dict(p0, hip_bias=0.12)
    for t in (0.0, 0.31, 0.77):
        a = np.asarray(lesson.gait_targets(t, p0), dtype=float)
        b = np.asarray(lesson.gait_targets(t, p1), dtype=float)
        assert abs((b[0] - a[0]) - 0.12) < 1e-9 and abs((b[2] - a[2]) - 0.12) < 1e-9, (
            "hip_bias must shift BOTH hips by exactly its value"
        )
        assert abs(b[1] - a[1]) < 1e-9 and abs(b[3] - a[3]) < 1e-9, (
            f"changing hip_bias moved a knee target by {abs(b[1] - a[1]):.4f}. The bias is a "
            "hip lean; adding it to the knees is the most common slip here."
        )


def test_gait_targets_uses_knee_lag():
    """A parameter the search tunes must actually reach the trajectory."""
    p0 = {"freq": 1.0, "hip_amp": 0.3, "knee_amp": 0.6, "knee_lag": 0.0, "hip_bias": 0.0}
    p1 = dict(p0, knee_lag=1.7)
    ts = np.linspace(0.0, 1.0, 60)
    k0 = np.array([lesson.gait_targets(t, p0)[1] for t in ts])
    k1 = np.array([lesson.gait_targets(t, p1)[1] for t in ts])
    h0 = np.array([lesson.gait_targets(t, p0)[0] for t in ts])
    h1 = np.array([lesson.gait_targets(t, p1)[0] for t in ts])
    assert np.abs(k1 - k0).max() > 1e-3, (
        "knee_lag changed nothing. It is the phase offset between the hip swing and the knee "
        "flexion, and the search tunes it — add it inside the knee's cosine."
    )
    assert np.abs(h1 - h0).max() < 1e-9, (
        "knee_lag moved the HIP trajectory; it belongs only to the knees."
    )


# --- exercise 2: the rollout ---------------------------------------------------------------

def test_rollout_contract_and_baseline_walks():
    model, data = lesson.load_walker()
    r = lesson.rollout(model, data, lesson.BASELINE_GAIT)
    assert isinstance(r, dict) and set(r) == KEYS, (
        f"keys were {sorted(r) if isinstance(r, dict) else type(r).__name__}, "
        f"expected a dict with {sorted(KEYS)}"
    )
    assert not r["fell"] and r["steps"] == lesson.HORIZON_STEPS, (
        f"the baseline gait fell={r['fell']} after {r['steps']} steps. It is known to walk the "
        "full horizon, so the loop is wrong — the usual cause is writing the targets to "
        "data.qpos instead of data.ctrl, or never calling mj_step."
    )
    assert r["distance"] > 0.3, (
        f"the baseline walked {r['distance']:.3f} m; it should cover more than 0.3 m. "
        "Distance is the CHANGE in data.qpos[0] over the episode."
    )
    assert r["energy"] > 0.0, "energy must be strictly positive for a gait that moved"


def test_rollout_energy_is_the_time_integral_of_absolute_power():
    """Catches a missing timestep, a missing abs(), and summing force without velocity."""
    model, data = lesson.load_walker()
    got = lesson.rollout(model, data, lesson.BASELINE_GAIT)["energy"]
    ref = _reference_rollout(lesson.BASELINE_GAIT)["energy"]
    assert abs(got - ref) / ref < 0.02, (
        f"energy was {got:.3f} J against a reference of {ref:.3f} J (ratio {got / ref:.1f}x). "
        "Energy is the sum of |actuator_force * actuator_velocity| MULTIPLIED BY the timestep "
        f"at every step. A ratio near {1 / lesson.DT:.0f}x means you forgot the timestep."
    )


def test_rollout_detects_a_fall_and_stops_early():
    """A gait with no knee flexion topples this walker; the loop must notice and break."""
    model, data = lesson.load_walker()
    r = lesson.rollout(model, data, dict(lesson.BASELINE_GAIT, knee_amp=0.0))
    assert r["fell"], (
        "a gait with knee_amp = 0 drives this walker into the floor, but the rollout reported "
        "no fall. Check hip height against MIN_HIP_HEIGHT and pitch against MAX_PITCH."
    )
    assert r["steps"] < lesson.HORIZON_STEPS, (
        f"it reported a fall but still ran all {r['steps']} steps — break out of the loop when "
        "the walker falls, or every fallen gait is scored on physics that never happened."
    )


def test_rollout_resets_between_calls():
    model, data = lesson.load_walker()
    first = lesson.rollout(model, data, lesson.BASELINE_GAIT)
    second = lesson.rollout(model, data, lesson.BASELINE_GAIT)
    assert abs(first["distance"] - second["distance"]) < 1e-9, (
        f"the same gait walked {first['distance']:.3f} m then {second['distance']:.3f} m on the "
        "same data. Call mj_resetDataKeyframe at the TOP of every rollout — otherwise each "
        "candidate in the search starts wherever the last one collapsed."
    )


# --- exercise 3: cost of transport ---------------------------------------------------------

def test_cost_of_transport_formula():
    got = lesson.cost_of_transport(100.0, 2.0, mass_kg=10.0, gravity=9.81)
    want = 100.0 / (10.0 * 9.81 * 2.0)
    assert abs(got - want) < 1e-12, (
        f"cost_of_transport(100, 2, mass=10, g=9.81) returned {got:.6f}, expected {want:.6f} "
        "= energy / (mass * gravity * distance). Leaving out g gives 5.0."
    )


def test_cost_of_transport_is_dimensionless_and_scales():
    base = lesson.cost_of_transport(100.0, 2.0, mass_kg=10.0, gravity=9.81)
    assert abs(lesson.cost_of_transport(200.0, 2.0, mass_kg=10.0, gravity=9.81) - 2 * base) < 1e-12, (
        "twice the energy over the same distance must double the cost of transport"
    )
    assert abs(lesson.cost_of_transport(100.0, 4.0, mass_kg=10.0, gravity=9.81) - base / 2) < 1e-12, (
        "twice the distance on the same energy must halve it — distance is in the denominator"
    )
    assert abs(lesson.cost_of_transport(100.0, 2.0, mass_kg=20.0, gravity=9.81) - base / 2) < 1e-12, (
        "twice the mass must halve it; normalising by weight is what makes this dimensionless"
    )


def test_cost_of_transport_guards_a_walker_that_went_nowhere():
    for bad in (0.0, -0.4):
        got = lesson.cost_of_transport(50.0, bad, mass_kg=10.0, gravity=9.81)
        assert got == float("inf"), (
            f"distance {bad} returned {got!r}. A machine that burned energy and covered no "
            "ground has an INFINITE cost of transport, not zero and not a crash — return "
            "float('inf') for any distance that is not strictly positive."
        )


# --- exercise 4: the search ----------------------------------------------------------------

def test_coordinate_search_returns_its_contract():
    best, score, n = lesson.coordinate_search(lesson.BASELINE_GAIT, passes=1)
    assert isinstance(best, dict) and set(best) == set(lesson.PARAM_KEYS), (
        f"the returned parameters were {sorted(best) if isinstance(best, dict) else best!r}, "
        f"expected a dict with exactly {sorted(lesson.PARAM_KEYS)}"
    )
    assert isinstance(n, int) and n >= 1, f"rollout count was {n!r}; count every rollout you ran"
    assert float(score) == float(score), "the returned score must be a real number"


def test_coordinate_search_beats_the_baseline_by_a_margin():
    """Graded on the margin, never on reaching a particular gait."""
    model, data = lesson.load_walker()
    base = lesson.gait_score(lesson.rollout(model, data, lesson.BASELINE_GAIT))
    best, score, _ = lesson.coordinate_search(lesson.BASELINE_GAIT, passes=2)
    assert score <= base, (
        f"the search returned {score:.4f}, worse than the baseline's {base:.4f}. It must never "
        "return something worse than where it started: keep the incumbent unless a candidate "
        "strictly improves on it."
    )
    assert score < base / 1.5, (
        f"the search only improved the cost of transport from {base:.4f} to {score:.4f}. The "
        "grid holds gaits a good deal better than that — check you are KEEPING each accepted "
        "improvement as you move on, rather than always varying the baseline one knob at a time."
    )


def test_coordinate_search_respects_the_rollout_cap():
    _, _, n = lesson.coordinate_search(lesson.BASELINE_GAIT, passes=4, max_rollouts=10)
    assert n <= 10, (
        f"the cap was 10 rollouts and the search ran {n}. The budget is the point of this "
        "exercise: check it before every rollout and stop when it is spent."
    )


RUBRIC = [
    (test_gait_targets_shape_and_antiphase, 3,
     "return (hip_r, knee_r, hip_l, knee_l); the left leg's phase is offset by leg_phase"),
    (test_gait_targets_knee_never_extends_backwards, 3,
     "knee = -knee_amp * (1 - cos(phase + knee_lag)) / 2, so it spans -knee_amp..0"),
    (test_gait_targets_bias_moves_hips_only, 2,
     "hip_bias is added to the two hips and to nothing else"),
    (test_gait_targets_uses_knee_lag, 2,
     "knee_lag goes inside the knee's cosine, not the hip's sine"),
    (test_rollout_contract_and_baseline_walks, 3,
     "write targets to data.ctrl, mj_step, and measure the change in data.qpos[0]"),
    (test_rollout_energy_is_the_time_integral_of_absolute_power, 3,
     "energy += abs(actuator_force * actuator_velocity).sum() * DT, every step"),
    (test_rollout_detects_a_fall_and_stops_early, 2,
     "break when hip height drops below MIN_HIP_HEIGHT or pitch exceeds MAX_PITCH"),
    (test_rollout_resets_between_calls, 2,
     "mj_resetDataKeyframe at the top of every rollout, before the loop"),
    (test_cost_of_transport_formula, 3,
     "energy / (mass * gravity * distance) — the weight, not the mass, is the normaliser"),
    (test_cost_of_transport_is_dimensionless_and_scales, 2,
     "energy on top; mass, gravity and distance all underneath"),
    (test_cost_of_transport_guards_a_walker_that_went_nowhere, 2,
     "a non-positive distance returns float('inf'), not zero and not an exception"),
    (test_coordinate_search_returns_its_contract, 3,
     "return (best_params, best_score, rollouts_used)"),
    (test_coordinate_search_beats_the_baseline_by_a_margin, 3,
     "accept an improvement immediately and carry it into the next parameter"),
    (test_coordinate_search_respects_the_rollout_cap, 2,
     "check the rollout budget before each rollout and stop when it is spent"),
]

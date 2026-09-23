# %% [markdown]
# # F15-L05 · Gait generation, and what a step actually costs — reference solution
#
# The student notebook with every stub filled in. Runs top to bottom in a fresh interpreter
# and prints real measured output. Self-check answers are in the comment block at the foot.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import hashlib
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

print("mujoco", mujoco.__version__, "· numpy", np.__version__)

MODEL_FILENAME = "planar_walker.xml"


def walker_xml_path() -> Path:
    """Locate the walker model that ships next to this notebook."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:  # a notebook has no __file__
        here = Path.cwd()
    for candidate in (here / "assets" / MODEL_FILENAME,
                      here.parent / "assets" / MODEL_FILENAME,
                      Path.cwd() / "assets" / MODEL_FILENAME,
                      Path.cwd().parent / "assets" / MODEL_FILENAME):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{MODEL_FILENAME} not found next to this lesson. It ships in assets/ and is never "
        "downloaded; restore it from the lesson directory."
    )


def load_walker():
    """Compile the walker and hand back a fresh (model, data) pair."""
    model = mujoco.MjModel.from_xml_path(str(walker_xml_path()))
    return model, mujoco.MjData(model)


MODEL, DATA = load_walker()

DT = float(MODEL.opt.timestep)
TOTAL_MASS = float(MODEL.body_mass.sum())
GRAVITY = float(-MODEL.opt.gravity[2])

HORIZON_STEPS = 800
MIN_HIP_HEIGHT = 0.55
MAX_PITCH = 1.0
MIN_DISTANCE = 0.30
FALL_PENALTY = 100.0

print(f"model: nq={MODEL.nq} nv={MODEL.nv} nu={MODEL.nu} dt={DT} s")
print(f"total mass {TOTAL_MASS:.2f} kg, gravity {GRAVITY:.2f} m/s^2, "
      f"weight {TOTAL_MASS * GRAVITY:.1f} N")
print(f"horizon {HORIZON_STEPS} steps = {HORIZON_STEPS * DT:.1f} s of simulated time")


# The notebook's guard. Every check, and every table that runs your code, goes through _try:
# an unfilled stub says so, a wrong answer prints its check's hint and the notebook carries
# on, and a table that needs an exercise you have not finished names it and skips. The
# progress board in the last cell reads the verdicts it records.
_BOARD = {                      # label -> what it grades, in the order the board lists them
    "exercise 1": "gait_targets",
    "exercise 2": "rollout",
    "exercise 3": "cost_of_transport",
    "exercise 4": "coordinate_search",
    "self-check": "the four questions in section 9",
}
_STATUS: dict = {}              # label -> "passed" | "failed" | "not started"


def _unfinished(exc: BaseException) -> str:
    """Say which stub stopped a check: the message it carries, or the function that raised."""
    if str(exc):
        return str(exc)
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    name = tb.tb_frame.f_code.co_name if tb is not None else "a function above"
    return f"{name}() is still a stub; fill it in above, then re-run this cell."


def _try(label: str, check, needs: tuple = ()):
    """Run a check, or a table that depends on your code, without derailing the notebook.

    A stub you have not filled in yet simply says so. A wrong answer prints the check's own
    message — which names the likely mistake — and the notebook carries on, so one broken
    exercise never hides the feedback on the others. A table given `needs` waits until those
    exercises have passed, and names the one it is waiting for. Nothing is swallowed: every
    verdict is recorded in _STATUS, and the last cell turns any failure into a non-zero exit
    whenever this file runs as a script. Returns what the check returned, or None.
    """
    waiting = [n for n in needs if _STATUS.get(n) != "passed"]
    if waiting:
        _STATUS.pop(label, None)
        names = [f"{n} ({_BOARD[n]})" for n in waiting]
        one = len(names) == 1
        names = names[0] if one else ", ".join(names[:-1]) + " and " + names[-1]
        print(f"{label}: skipped — it runs your code from {names}, which "
              f"{'has' if one else 'have'} not passed yet. Finish {'it' if one else 'them'}, "
              f"re-run {'its check cell' if one else 'their check cells'}, then this one.")
        return None
    try:
        result = check()
    except NotImplementedError as exc:
        _STATUS[label] = "not started"
        print(f"{label}: not started yet — {_unfinished(exc)}")
        return None
    except AssertionError as exc:
        _STATUS[label] = "failed"
        print(f"{label}: FAILED — {exc}")
        return None
    except Exception as exc:  # a half-finished implementation raising something else
        _STATUS[label] = "failed"
        print(f"{label}: raised {type(exc).__name__}: {exc}")
        return None
    _STATUS[label] = "passed"
    return result

# %% [markdown]
# ## 1. The control input is an angle, not a torque

# %%
_m, _d = load_walker()
mujoco.mj_resetDataKeyframe(_m, _d, 0)
_d.ctrl[:] = [0.6, -0.8, -0.6, -0.2]
for _ in range(50):
    mujoco.mj_step(_m, _d)
print(f"targets written to ctrl : {np.round(_d.ctrl, 3)}")
print(f"joint angles reached    : {np.round(_d.qpos[3:], 3)}   <- it is chasing, not arriving")
print(f"torque the servo made   : {np.round(_d.actuator_force, 2)} N*m")
print(f"joint speeds right now  : {np.round(_d.actuator_velocity, 3)} rad/s")
print("\nthe servo is a P-D law inside the engine; you never wrote a torque, and one exists")

# %% [markdown]
# ## 2. Exercise 1 — the gait, as a function of time

# %%
PARAM_KEYS = ("freq", "hip_amp", "knee_amp", "knee_lag", "hip_bias")
LEG_PHASE = math.pi


def gait_targets(t: float, params: dict, leg_phase: float = LEG_PHASE):
    """The four joint targets at time `t`, in actuator order (hip_r, knee_r, hip_l, knee_l)."""
    phase = 2.0 * math.pi * params["freq"] * t
    hip_amp, knee_amp = params["hip_amp"], params["knee_amp"]
    knee_lag, hip_bias = params["knee_lag"], params["hip_bias"]
    hip_r = hip_amp * math.sin(phase) + hip_bias
    hip_l = hip_amp * math.sin(phase + leg_phase) + hip_bias
    knee_r = -knee_amp * (1.0 - math.cos(phase + knee_lag)) / 2.0
    knee_l = -knee_amp * (1.0 - math.cos(phase + leg_phase + knee_lag)) / 2.0
    return np.array([hip_r, knee_r, hip_l, knee_l])


def _check_gait_targets() -> None:
    p = {"freq": 1.0, "hip_amp": 0.4, "knee_amp": 0.6, "knee_lag": 0.0, "hip_bias": 0.1}
    out = np.asarray(gait_targets(0.25, p), dtype=float)
    assert out.shape == (4,), (
        f"expected 4 targets in the order (hip_r, knee_r, hip_l, knee_l), got shape {out.shape}"
    )
    assert abs(out[0] - 0.5) < 1e-9, (
        f"right hip is {out[0]:.4f} at a quarter cycle; sin(pi/2) = 1, so it should be "
        "hip_amp + hip_bias = 0.5. Check the 2*pi in the phase."
    )
    assert abs(out[2] - (-0.3)) < 1e-9, (
        f"left hip is {out[2]:.4f}, expected -0.3. The legs are ANTIPHASE: add leg_phase to "
        "the left leg's phase. If you got 0.5 the two legs are moving together."
    )
    cycle = np.array([np.asarray(gait_targets(t, p), dtype=float)
                      for t in np.linspace(0.0, 1.0, 400)])
    assert cycle[:, [1, 3]].max() <= 1e-9, (
        f"a knee target reached {cycle[:, [1, 3]].max():+.4f}. Knees only flex on this model — "
        "every knee target must be <= 0, so the envelope is multiplied by MINUS knee_amp."
    )
    assert abs(cycle[:, [1, 3]].min() + p["knee_amp"]) < 1e-3, (
        f"the deepest knee flexion was {cycle[:, [1, 3]].min():+.4f}, but it should reach "
        f"-knee_amp = {-p['knee_amp']:.2f} once per cycle. Did you divide (1 - cos) by 2?"
    )
    print(f"exercise 1 looks right: hips span {cycle[:, 0].min():+.2f}..{cycle[:, 0].max():+.2f} "
          f"rad, knees {cycle[:, 1].min():+.2f}..{cycle[:, 1].max():+.2f} rad")


# Run the check for THIS exercise, right here. The guard is what lets the autograder
# import this file without every check firing: it imports under the name "lesson", while
# a notebook cell and `python lesson.py` both run as "__main__".
if __name__ == "__main__":
    _try("exercise 1", _check_gait_targets)

# %% [markdown]
# ## 3. Energy is the integral of torque times velocity

# %%
_m, _d = load_walker()
mujoco.mj_resetDataKeyframe(_m, _d, 0)
_e_actuation = _e_joint = 0.0
for _i in range(300):
    _d.ctrl[:] = [0.3 * math.sin(6.0 * _i * DT), -0.4, -0.3 * math.sin(6.0 * _i * DT), -0.4]
    mujoco.mj_step(_m, _d)
    _e_actuation += float(np.abs(_d.actuator_force * _d.actuator_velocity).sum()) * DT
    _e_joint += float(np.abs(_d.qfrc_actuator[3:] * _d.qvel[3:]).sum()) * DT
print(f"energy via actuator_force * actuator_velocity : {_e_actuation:8.3f} J")
print(f"energy via qfrc_actuator * qvel (joint space) : {_e_joint:8.3f} J")
print(f"relative difference between the two spaces    : "
      f"{abs(_e_actuation - _e_joint) / _e_actuation:.2e}")
print("\nsame physics, two coordinate systems. If your energy is 300x this, you dropped dt.")

# %% [markdown]
# ## 4. Exercise 2 — the rollout, and exercise 3 — the cost of transport

# %%
BASELINE_GAIT = {"freq": 0.90, "hip_amp": 0.25, "knee_amp": 0.60, "knee_lag": 4.00,
                 "hip_bias": 0.00}


def rollout(model, data, params: dict, horizon: int = HORIZON_STEPS,
            leg_phase: float = LEG_PHASE) -> dict:
    """Drive one gait for the horizon and report what it cost."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    start_x = float(data.qpos[0])
    energy, steps, fell = 0.0, 0, False
    for i in range(horizon):
        data.ctrl[:] = gait_targets(i * DT, params, leg_phase)
        mujoco.mj_step(model, data)
        energy += float(np.abs(data.actuator_force * data.actuator_velocity).sum()) * DT
        steps += 1
        if data.qpos[1] < MIN_HIP_HEIGHT or abs(data.qpos[2]) > MAX_PITCH:
            fell = True
            break
    return {"distance": float(data.qpos[0]) - start_x, "energy": energy,
            "steps": steps, "fell": fell}


def _check_rollout() -> None:
    model, data = load_walker()
    r = rollout(model, data, BASELINE_GAIT)
    assert set(r) == {"distance", "energy", "steps", "fell"}, (
        f"keys were {sorted(r)} — return a dict with exactly those four names."
    )
    assert not r["fell"] and r["steps"] == HORIZON_STEPS, (
        f"the baseline gait fell={r['fell']} after {r['steps']} steps. It is known to survive "
        "the full horizon, so the loop is wrong: the usual causes are writing targets into "
        "data.qpos instead of data.ctrl, or forgetting mj_step."
    )
    again = rollout(model, data, BASELINE_GAIT)
    assert abs(again["distance"] - r["distance"]) < 1e-9, (
        f"the same gait walked {r['distance']:.3f} m then {again['distance']:.3f} m on the same "
        "data — reset to the keyframe at the TOP of every rollout."
    )
    print(f"exercise 2 looks right: the baseline walks {r['distance']:.3f} m on "
          f"{r['energy']:.1f} J and survives all {r['steps']} steps")


if __name__ == "__main__":
    _try("exercise 2", _check_rollout)

# %% [markdown]
# **Exercise 3 — the cost of transport.**

# %%
def cost_of_transport(energy_j: float, distance_m: float, mass_kg: float = TOTAL_MASS,
                      gravity: float = GRAVITY) -> float:
    """Dimensionless specific cost of transport: energy / (mass * gravity * distance)."""
    if distance_m <= 0.0:
        return float("inf")
    return energy_j / (mass_kg * gravity * distance_m)


def gait_score(result: dict) -> float:
    """Lower is better. The objective the search in section 6 minimises."""
    if result["fell"] or result["distance"] < MIN_DISTANCE:
        return (FALL_PENALTY - 10.0 * result["distance"]
                - 10.0 * (result["steps"] / HORIZON_STEPS))
    return cost_of_transport(result["energy"], result["distance"])


def _check_cost_of_transport() -> None:
    assert abs(cost_of_transport(100.0, 2.0, mass_kg=10.0, gravity=9.81) - 0.5096840) < 1e-6, (
        "cost_of_transport(100, 2, mass=10, g=9.81) should be 100/(10*9.81*2) = 0.509684; "
        "omitting gravity gives 5.0."
    )
    assert cost_of_transport(50.0, 0.0) == float("inf"), (
        "a distance of zero must give float('inf') — energy was spent and nothing moved."
    )
    assert cost_of_transport(50.0, -0.7) == float("inf"), (
        "a negative distance must also give float('inf'); walking backwards is not efficient "
        "transport, and a negative cost of transport would win every search you ever run."
    )
    print("exercise 3 looks right: energy over weight times distance, dimensionless, and a "
          "walker that covers no ground can never win a search")


if __name__ == "__main__":
    _try("exercise 3", _check_cost_of_transport)

# %% [markdown]
# ## 5. Phase is not a detail

# %%
def _check_phase_matters() -> None:
    model, data = load_walker()
    print(f"  {'leg_phase':>12s} {'distance':>9s} {'energy':>9s} {'fell':>6s} {'steps':>6s}")
    out = {}
    for name, phase in (("pi (walk)", math.pi), ("pi/2", math.pi / 2), ("0 (hop)", 0.0)):
        r = rollout(model, data, BASELINE_GAIT, leg_phase=phase)
        out[name] = r
        print(f"  {name:>12s} {r['distance']:9.3f} {r['energy']:9.1f} "
              f"{str(r['fell']):>6s} {r['steps']:>6d}")
    assert not out["pi (walk)"]["fell"], "the antiphase gait is the one that should survive"
    assert out["0 (hop)"]["fell"], (
        "driving both legs in phase should topple this walker — if it survived on your "
        "machine, re-read the table above before trusting the sentence below it"
    )
    print("\n  Same joints, same amplitudes, same frequency, same energy budget. The only")
    print("  difference is WHEN each leg moves relative to the other, and it decides whether")
    print("  the machine walks or falls over. Phase is the gait.")


if __name__ == "__main__":
    _try("phase table", _check_phase_matters, needs=("exercise 1", "exercise 2"))

# %% [markdown]
# ## 6. Exercise 4 — a search you can afford

# %%
SEARCH_GRID = {
    "freq":     [0.6, 0.8, 1.0, 1.3, 1.7],
    "hip_amp":  [0.10, 0.20, 0.30, 0.45, 0.60],
    "knee_amp": [0.0, 0.25, 0.50, 0.75, 1.00],
    "knee_lag": [0.0, 1.26, 2.51, 3.77, 5.03],
    "hip_bias": [-0.10, -0.05, 0.0, 0.05, 0.10],
}
MAX_ROLLOUTS = 120


def coordinate_search(start: dict, grid: dict = SEARCH_GRID, passes: int = 2,
                      max_rollouts: int = MAX_ROLLOUTS):
    """Coordinate descent over gait parameters, minimising `gait_score`, under a hard cap.

    `rollout` and `gait_score` are looked up by name at call time, never captured in a default
    argument, so the autograder can swap them for instruments.
    """
    model, data = load_walker()
    best = dict(start)
    best_score = gait_score(rollout(model, data, best))
    used = 1
    for _ in range(passes):
        for key in PARAM_KEYS:
            for value in grid[key]:
                if value == best[key]:
                    continue
                if used >= max_rollouts:
                    return best, best_score, used
                candidate = dict(best, **{key: value})
                score = gait_score(rollout(model, data, candidate))
                used += 1
                if score < best_score:
                    best, best_score = candidate, score
    return best, best_score, used


def _check_search() -> float:
    """Run the search once, check it, print the comparison, and hand back its score."""
    model, data = load_walker()
    base = rollout(model, data, BASELINE_GAIT)
    base_score = gait_score(base)
    t0 = time.perf_counter()
    best, score, used = coordinate_search(BASELINE_GAIT, passes=2)
    elapsed = time.perf_counter() - t0
    assert sorted(best) == sorted(PARAM_KEYS), (
        f"the search returned keys {sorted(best)}, expected {sorted(PARAM_KEYS)}"
    )
    assert used <= MAX_ROLLOUTS, f"the search used {used} rollouts against a cap of {MAX_ROLLOUTS}"
    assert score <= base_score, (
        f"the search returned {score:.4f}, worse than the baseline's {base_score:.4f}. Keep the "
        "incumbent unless a candidate strictly improves on it."
    )
    _, _, capped = coordinate_search(BASELINE_GAIT, passes=4, max_rollouts=10)
    assert capped <= 10, (
        f"with max_rollouts=10 the search still ran {capped} rollouts — check the budget "
        "BEFORE each rollout, not after."
    )
    found = rollout(model, data, best)
    print(f"  {used} rollouts in {elapsed * 1000:.1f} ms ({elapsed / used * 1000:.2f} ms each)\n")
    print(f"  {'':10s} {'distance':>9s} {'energy':>9s} {'cost of transport':>18s}")
    print(f"  {'baseline':10s} {base['distance']:9.3f} {base['energy']:9.1f} {base_score:18.4f}")
    print(f"  {'found':10s} {found['distance']:9.3f} {found['energy']:9.1f} {score:18.4f}")
    print(f"\n  gait found: { {k: round(best[k], 3) for k in PARAM_KEYS} }")
    print(f"  the search cut the cost of transport by a factor of {base_score / score:.2f}")
    assert score < base_score / 1.5, (
        f"only a {base_score / score:.2f}x improvement — the grid holds much better gaits than "
        "that, so check you carry each accepted improvement into the next parameter."
    )
    return score


if __name__ == "__main__":
    SEARCH_SCORE = _try("exercise 4", _check_search)

# %% [markdown]
# ## 7. Against the literature, honestly

# %%
PUBLISHED_COT = [
    ("humans walking", 0.2, 0.05),
    ("Cornell biped (13 kg, 0.4 m/s)", 0.2, 0.055),
    ("Delft biped", None, 0.08),
    ("MIT biped", None, 0.02),
    ("Honda ASIMO (authors' estimate)", 3.2, 1.6),
]


def compare_against_published(measured_cot: float) -> None:
    """Print the measured cost of transport beside the published figures, with the caveat."""
    print(f"  {'machine':<34s} {'c_et':>6s} {'c_mt':>6s}")
    for name, c_et, c_mt in PUBLISHED_COT:
        et = "  -  " if c_et is None else f"{c_et:5.3g}"
        print(f"  {name:<34s} {et:>6s} {c_mt:6.3g}")
    print(f"  {'-' * 48}")
    print(f"  {'your gait (absolute work)':<34s} {'  -  ':>6s} {measured_cot:6.3g}")
    best_published = min(c_mt for _, _, c_mt in PUBLISHED_COT)
    print(f"\n  Your walker is {measured_cot / best_published:.0f}x the cost of the most "
          f"efficient machine in that table,")
    print("  and it is not even a like-for-like measurement: you charged yourself for braking")
    print("  and they did not. Efficient walking is hard, and the honest version of that")
    print("  sentence names the accounting difference instead of hiding inside it.")


if __name__ == "__main__":
    _try("comparison table", lambda: compare_against_published(SEARCH_SCORE),
         needs=("exercise 4",))

# %% [markdown]
# ## 8. Common mistakes

# %%
def _check_objective_matters() -> None:
    """Search the same grid under two objectives at EQUAL budget, and print both gaits."""
    def distance_only(result: dict) -> float:
        # Same penalty SHAPE as gait_score, so the only difference is the objective itself.
        if result["fell"] or result["distance"] < MIN_DISTANCE:
            return (FALL_PENALTY - 10.0 * result["distance"]
                    - 10.0 * (result["steps"] / HORIZON_STEPS))
        return -result["distance"]

    model, data = load_walker()

    def search(objective):
        best, used = dict(BASELINE_GAIT), 1
        best_score = objective(rollout(model, data, best))
        for _ in range(2):
            for key in PARAM_KEYS:
                for value in SEARCH_GRID[key]:
                    if value == best[key] or used >= MAX_ROLLOUTS:
                        continue
                    candidate = dict(best, **{key: value})
                    score = objective(rollout(model, data, candidate))
                    used += 1
                    if score < best_score:
                        best_score, best = score, candidate
        return best

    far = rollout(model, data, search(distance_only))
    cheap = rollout(model, data, search(gait_score))
    print(f"  {'objective':<22s} {'distance':>9s} {'energy':>9s} {'cost of transport':>18s}")
    print(f"  {'maximise distance':<22s} {far['distance']:9.3f} {far['energy']:9.1f} "
          f"{gait_score(far):18.4f}")
    print(f"  {'minimise cost':<22s} {cheap['distance']:9.3f} {cheap['energy']:9.1f} "
          f"{gait_score(cheap):18.4f}")
    print("\n  Same grid, same baseline, same rollout budget. On this machine, chasing")
    print(f"  distance alone changed travel by {far['distance'] - cheap['distance']:+.3f} m "
          f"and burned {far['energy'] / cheap['energy']:.1f}x the energy,")
    print(f"  for a cost of transport {gait_score(far) / gait_score(cheap):.1f}x higher.")
    print("\n  The point is not that one number beat the other. It is that a distance")
    print("  objective cannot SEE energy, so it could never have told you what it cost you.")
    print("  Write down the metric you actually care about before you optimise anything.")


if __name__ == "__main__":
    _try("objective table", _check_objective_matters,
         needs=("exercise 1", "exercise 2", "exercise 3"))

# %% [markdown]
# ## 9. Self-check

# %%
SELF_CHECK = {1: "c", 2: "b", 3: "c", 4: "b"}

_ANSWER_DIGESTS = {1: "0dcc785a7cf671de", 2: "b43f87a1f5173e77",
                   3: "a6d7100ef7fa03e4", 4: "a0f1eca7d1f50472"}
_ANSWER_SECTIONS = {
    1: "section 4 — the guard you wrote into cost_of_transport, and why it is not zero",
    2: "section 5 — the leg_phase table you ran, and which row fell over",
    3: "section 3 — the two-space energy cross-check, and what dt is doing in it",
    4: "section 7 — the caveat column of the comparison table you printed",
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the four multiple-choice answers in SELF_CHECK, naming where to look again."""
    answers = SELF_CHECK if answers is None else answers
    wrong = []
    for q, digest in sorted(_ANSWER_DIGESTS.items()):
        got = str(answers.get(q, "?")).strip().lower()
        if hashlib.sha256(f"F15-L05-q{q}-{got}".encode()).hexdigest()[:16] != digest:
            wrong.append(q)
    for q in sorted(_ANSWER_DIGESTS):
        note = f"  -> re-read {_ANSWER_SECTIONS[q]}" if q in wrong else ""
        print(f"  q{q}: {'wrong' if q in wrong else 'right'}{note}")
    assert not wrong, (
        f"questions {wrong} are still wrong. Each one names the section that answers it above "
        "— go back to the measurement you ran there rather than guessing a letter."
    )
    print("self-check: all four right")


def _self_check_marked() -> None:
    """Mark SELF_CHECK, but treat a sheet with no letters on it yet as not started."""
    if all(str(v).strip() == "?" for v in SELF_CHECK.values()):
        raise NotImplementedError("put your four letters into SELF_CHECK above, then re-run "
                                  "this cell.")
    _check_self_check()


if __name__ == "__main__":
    _try("self-check", _self_check_marked)

# %% [markdown]
# ## What you built, and where it goes next

# %% [markdown]
# ## Your progress
#
# One line per exercise, read from the verdicts the check cells above recorded. Change an
# exercise, re-run its check cell, then re-run this one.

# %%
_MARKS = {"passed": "✅", "failed": "❌", "not started": "⏳"}


def _progress_board() -> list:
    """Print one line per exercise and the tally; return the labels whose check failed."""
    for label, what in _BOARD.items():
        state = _STATUS.get(label, "not started")
        print(f"  {_MARKS[state]} {label:<11s} {what:<34s} {state}")
    done = sum(_STATUS.get(label) == "passed" for label in _BOARD)
    print(f"\n  {done} of {len(_BOARD)} complete")
    return [label for label, state in _STATUS.items() if state == "failed"]


if __name__ == "__main__":
    _failed = _progress_board()
    # A stub you have not reached yet is not a failure. A check that ran and came back wrong
    # is: as a script or under CI it ends the run non-zero, so a green exit code never papers
    # over it. In a notebook kernel it is a line on the board, not a traceback.
    if _failed and "ipykernel" in sys.modules:
        print(f"  still failing: {', '.join(_failed)} — each one's cell above says what to fix")
    elif _failed:
        raise SystemExit("checks failed: " + ", ".join(_failed))

# %% [markdown]
# ## Self-check answers
#
# **1 — (c).** Cost of transport is energy divided by weight times distance. As the distance
# goes to zero the quotient grows without bound, so a machine that burns 300 J and ends where
# it began has an infinite cost of transport: it bought no transport at all. (a) confuses the
# *output* being zero with the *ratio* being zero — the numerator is still 300 J. (b) drops
# the distance from the denominator entirely and leaves a quantity with units of length. (d)
# is the tempting engineering answer and it is wrong for a specific reason: the search calls
# this function on every candidate, and a raised exception there would abort the whole run
# rather than scoring a bad gait badly. `inf` sorts correctly and loses gracefully, which is
# why the rubric checks for it and why walking backwards returns `inf` too — a negative
# distance would otherwise produce a negative cost that wins every search outright.
#
# **2 — (b).** With `leg_phase = pi` one leg is in stance while the other swings, so there is
# always a support under the body; that alternation is what the word "gait" refers to. Set it
# to 0 and both legs leave the ground together, which is a hop, and this machine has no
# ankle authority and no flight-phase controller to land one — the measured table in section 5
# shows it toppling. (a) is refuted by the `steps` column, which shows the fall happening well
# inside the horizon. (c) is refuted by holding `knee_amp` constant across all three rows: the
# knees are identical and only the phase changed. (d) is simply false — MuJoCo simulates it
# perfectly well, which is how you got to watch it fall over.
#
# **3 — (c).** Force is not power and power is not energy. Summing `|tau|` accumulates
# newton-metres times *step count*, which has neither the dimensions of energy nor any
# consistent physical meaning, and it is actively misleading: a joint straining at full torque
# against a limit at zero velocity does no mechanical work whatsoever, yet scores heavily under
# that sum. The section 3 cell is the antidote — it computes the same energy in actuation
# space and in joint space and shows them agreeing, which only works because both sides
# multiply by velocity and by `dt`. (a) and (d) both imagine the error is a scale factor; it is
# not, because the missing factor is `actuator_velocity`, which varies step by step. (b) is
# wrong twice over: it is not power either, and even a true power figure would need dividing by
# a time, not left as a bare sum.
#
# **4 — (b).** The shapes match — both are energy over weight times distance, both dimensionless
# — but the accounting differs. Collins et al. define `c_mt` over the *positive* mechanical work
# of the actuators, while this lesson integrates `|tau * omega|` and so charges the gait for
# braking as well as driving. Your figure is therefore computed under a stricter rule than the
# one it is being compared to, and part of any gap is definitional rather than physical. The
# gap is still real and still large; what is not defensible is quoting a precise ratio as
# though the two numbers were measured the same way, which is exactly what (a) does. (c)
# overcorrects into uselessness — the metric was designed to be comparable across machines,
# which is the whole reason it is dimensionless. (d) blames the simulator for a difference that
# is sitting in the definition of the metric, and would still be there on hardware.

# %% [markdown]
# # F15-L04 · Standing balance — reference solution
#
# The student notebook with every stub filled in. Runs top to bottom in a fresh interpreter
# and prints real measured output. Self-check answers are in the comment block at the foot.

# %%
import hashlib
import sys
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

print("mujoco", mujoco.__version__, "· numpy", np.__version__)

# What the progress board in the last cell reads: label -> "passed", "failed", "not started"
# or "waiting on <exercise>". Every check and every demo that runs on your code writes here.
_STATUS: dict[str, str] = {}


def _try(label: str, check: Callable[[], Any], needs: tuple = ()) -> Any:
    """Run a check, or a demo that depends on your code, without derailing the notebook.

    A stub you have not filled in yet simply says so. A wrong answer prints the check's own
    message — which names the likely mistake — and the notebook carries on, so one broken
    exercise never hides the feedback on the others. Nothing is swallowed: every outcome is
    recorded in _STATUS, and the last cell of this file exits non-zero if any check failed.

    `needs` names the exercises a cell runs on. Until each has passed its own check, the cell
    says which one it is waiting for and skips, rather than failing on code you have not
    reached yet. Returns whatever the check returns, or None if it did not pass.
    """
    waiting = [n for n in needs if _STATUS.get(n) != "passed"]
    if waiting:
        _STATUS[label] = "waiting on " + ", ".join(waiting)
        named = [f"{n} ({_STATUS.get(n, 'not run yet')})" for n in waiting]
        one = len(named) == 1
        print(f"{label}: skipped — it runs on "
              + (named[0] if one else ", ".join(named[:-1]) + " and " + named[-1])
              + f", which {'has' if one else 'have'} not passed yet. Come back once "
              + f"{'it has' if one else 'they have'}.")
        return None
    try:
        result = check()
    except NotImplementedError as exc:
        _STATUS[label] = "not started"
        print(f"{label}: {str(exc) or 'not implemented yet — fill in the stub above'}, "
              "then re-run this cell.")
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


def _progress_board(exercises) -> None:
    """Print one line per (label, what): ✅ passed, ❌ failed or ⏳ not started; then a tally."""
    print("\nProgress board")
    for label, what in exercises:
        state = _STATUS.get(label, "not started")
        if state == "passed":
            mark, note = "✅", ""
        elif state == "failed":
            mark, note = "❌", " — failed: see its message above"
        else:
            mark, note = "⏳", f" — {state}"
        print(f"  {mark} {label}: {what}{note}")
    done = sum(_STATUS.get(label) == "passed" for label, _ in exercises)
    print(f"  {done} of {len(exercises)} complete")


MODEL_FILENAME = "stander.xml"


def stander_xml_path() -> Path:
    """Locate the model that ships next to this notebook."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:
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


def load_stander():
    """Compile the model and hand back a fresh (model, data) pair."""
    model = mujoco.MjModel.from_xml_path(str(stander_xml_path()))
    return model, mujoco.MjData(model)


MODEL, DATA = load_stander()
TORSO_ID = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_BODY, "torso")
FOOT_GEOM_ID = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
ANKLE = 3

SETTLE_STEPS = 100
PUSH_START_STEP = 100
PUSH_STEPS = 100
HORIZON_STEPS = 1000
FALL_TILT = 0.35
PUSH_LADDER = (20.0, 40.0, 60.0, 80.0, 100.0, 130.0, 160.0, 200.0)

print(f"model: nq={MODEL.nq} nv={MODEL.nv} nu={MODEL.nu} dt={MODEL.opt.timestep} s")
print(f"total mass {MODEL.body_mass.sum():.2f} kg, "
      f"ankle torque limit {MODEL.actuator_ctrlrange[0][1]:.0f} N*m")
print(f"push: {PUSH_STEPS * MODEL.opt.timestep:.2f} s long, starting at "
      f"{PUSH_START_STEP * MODEL.opt.timestep:.2f} s; horizon "
      f"{HORIZON_STEPS * MODEL.opt.timestep:.1f} s")


# %%
def settle(model, data) -> None:
    """Reset to the standing keyframe and let the contacts settle, with no control."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(SETTLE_STEPS):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)


def torso_tilt(model, data) -> float:
    """Signed angle of the torso's long axis away from vertical, in radians."""
    zaxis = data.xmat[TORSO_ID].reshape(3, 3)[:, 2]
    return float(np.arctan2(zaxis[0], zaxis[2]))


settle(MODEL, DATA)
STANCE_COM_X = float(DATA.subtree_com[0][0])
print(f"settled: {DATA.ncon} contact points, torso tilt {torso_tilt(MODEL, DATA):+.5f} rad")
print(f"contact points (x, y): {np.round(np.array([c.pos[:2] for c in DATA.contact[:DATA.ncon]]), 4).tolist()}")
print(f"it stands with its centre of mass at x = {STANCE_COM_X:+.6f} m")


# %%
def com_ground_projection(model, data) -> np.ndarray:
    """Return the (x, y) ground projection of the whole machine's centre of mass."""
    masses = model.body_mass[1:]
    weighted = (masses[:, None] * data.xipos[1:]).sum(axis=0) / masses.sum()
    return np.asarray(weighted[:2], dtype=float)


def _check_com_ground_projection() -> None:
    model, data = load_stander()
    settle(model, data)
    got = np.asarray(com_ground_projection(model, data), dtype=float)
    assert got.shape == (2,), (
        f"expected shape (2,), got {got.shape} — return only x and y, not all three "
        "components of the centre of mass."
    )
    reference = np.asarray(data.subtree_com[0][:2], dtype=float)
    assert np.allclose(got, reference, atol=1e-9), (
        f"you got {got}, MuJoCo's own whole-body centre of mass is {reference}."
    )
    print(f"exercise 1 looks right: centre of mass at x={got[0]:+.6f}, y={got[1]:+.6f}, "
          f"agreeing with MuJoCo's subtree_com to {np.abs(got - reference).max():.1e} m")


# %%
if __name__ == "__main__":
    _try("exercise 1", _check_com_ground_projection)


# %%
def support_polygon(model, data) -> np.ndarray:
    """The four ground corners of the foot, in world (x, y), counter-clockwise."""
    sx, sy, sz = model.geom_size[FOOT_GEOM_ID]
    local = np.array([[+sx, -sy, -sz], [+sx, +sy, -sz], [-sx, +sy, -sz], [-sx, -sy, -sz]])
    rot = data.geom_xmat[FOOT_GEOM_ID].reshape(3, 3)
    return (data.geom_xpos[FOOT_GEOM_ID] + local @ rot.T)[:, :2]


def fore_aft_margins(polygon, com_x: float):
    """(heel_margin, toe_margin): how far the centre of mass is from each end."""
    return float(com_x - polygon[:, 0].min()), float(polygon[:, 0].max() - com_x)


def support_margin(point, polygon) -> float:
    """Signed distance from `point` to the nearest edge of a convex polygon."""
    point = np.asarray(point, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    best = float("inf")
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        cross = float(edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0]))
        best = min(best, cross / length)
    return best


def _check_support_margin() -> None:
    square = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    centre = support_margin(np.array([0.5, 0.5]), square)
    assert abs(centre - 0.5) < 1e-9, f"expected 0.5, got {centre}"
    outside = support_margin(np.array([1.5, 0.5]), square)
    assert abs(outside + 0.5) < 1e-9, f"expected -0.5, got {outside}"
    edge = support_margin(np.array([1.0, 0.5]), square)
    assert abs(edge) < 1e-9, f"a point on the boundary should give 0.0, you returned {edge}"
    model, data = load_stander()
    settle(model, data)
    poly = support_polygon(model, data)
    com = np.asarray(com_ground_projection(model, data), dtype=float)
    margin = support_margin(com, poly)
    heel, toe = fore_aft_margins(poly, float(com[0]))
    assert margin > 0, f"a standing machine must be inside its own polygon, got {margin}"
    print(f"exercise 2 looks right. Standing:")
    print(f"  footprint x from {poly[:, 0].min():+.4f} to {poly[:, 0].max():+.4f} m, "
          f"y from {poly[:, 1].min():+.3f} to {poly[:, 1].max():+.3f} m")
    print(f"  nearest edge of any kind: {margin:.4f} m  <- a SIDE edge; the foot is narrower "
          "than it is long")
    print(f"  heel margin {heel:.4f} m, toe margin {toe:.4f} m, "
          f"ratio {toe / heel:.3f} — it has more room forwards than backwards")


# %%
if __name__ == "__main__":
    _try("exercise 2", _check_support_margin, needs=("exercise 1",))


# %%
def centre_of_pressure(model, data):
    """(cop_x, total_vertical_force) from the contacts MuJoCo reported."""
    total_fz, moment = 0.0, 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force)
        contact = data.contact[i]
        world = contact.frame.reshape(3, 3).T @ force[:3]
        total_fz += world[2]
        moment += world[2] * contact.pos[0]
    if abs(total_fz) < 1e-9:
        return float("nan"), 0.0
    return moment / total_fz, total_fz


_m, _d = load_stander()
settle(_m, _d)
_cop_x, _fz = centre_of_pressure(_m, _d)
_com_x = float(_d.subtree_com[0][0])
print(f"at rest: centre of pressure x = {_cop_x:+.6f} m")
print(f"         centre of mass     x = {_com_x:+.6f} m")
print(f"         they differ by {abs(_cop_x - _com_x):.2e} m  <- not a coincidence: nothing "
      "is accelerating")
print(f"         total vertical contact force {_fz:.3f} N vs weight "
      f"{MODEL.body_mass.sum() * 9.81:.3f} N")


# %%
def flat_foot_torque_ceiling(sign: float, hold_steps: int = 150) -> float:
    """Largest |ankle torque| that still leaves all four corners on the floor."""
    last_flat = 0.0
    for newtons in range(1, 36):
        model, data = load_stander()
        settle(model, data)
        for _ in range(hold_steps):
            data.ctrl[0] = sign * newtons
            mujoco.mj_step(model, data)
        if data.ncon == 4:
            last_flat = float(newtons)
        else:
            return last_flat
    return last_flat


def _check_torque_ceiling() -> None:
    model, data = load_stander()
    settle(model, data)
    poly = support_polygon(model, data)
    com_x = float(data.subtree_com[0][0])
    heel, toe = fore_aft_margins(poly, com_x)
    weight = float(model.body_mass.sum()) * 9.81
    limit = float(model.actuator_ctrlrange[0][1])

    print(f"  {'ankle torque':>13s} {'centre of pressure':>19s} {'corners down':>13s}")
    for tau in (-12.0, -6.0, 0.0, 6.0, 12.0):
        model, data = load_stander()
        settle(model, data)
        for _ in range(150):
            data.ctrl[0] = tau
            mujoco.mj_step(model, data)
        cop_x, _ = centre_of_pressure(model, data)
        print(f"  {tau:>10.1f} N*m {cop_x:>+18.4f} m {data.ncon:>12d}")

    heelward = flat_foot_torque_ceiling(+1.0)
    toeward = flat_foot_torque_ceiling(-1.0)
    print(f"\n  measured flat-foot ceiling: {heelward:.0f} N*m heelward, "
          f"{toeward:.0f} N*m toeward")
    print(f"  weight x margin predicts  : {weight * heel:.1f} N*m heelward, "
          f"{weight * toe:.1f} N*m toeward")
    print(f"  the motor will happily give you {limit:.0f} N*m in either direction")
    assert heelward < limit and toeward < limit, (
        "the foot should give out well before the motor does"
    )
    assert heelward < toeward, (
        "there is less room behind the centre of mass than in front of it"
    )
    print(f"\n  So the ankle can only use about {100 * heelward / limit:.0f}% of its torque "
          "range before the foot starts to lift.")
    print("  The measured ceiling sits below the weight-times-margin prediction, and it "
          "should:\n  that prediction assumes the mass stays put, while in fact the body "
          "leans as the torque\n  is applied, carrying the centre of mass toward the very "
          "edge it is about to tip over.")


# %%
if __name__ == "__main__":
    _try("the flat-foot ceiling", _check_torque_ceiling)


# %%
def has_fallen(model, data) -> bool:
    """True when the torso has tilted more than `FALL_TILT` radians away from vertical."""
    return bool(abs(torso_tilt(model, data)) > FALL_TILT)


def _check_has_fallen() -> None:
    model, data = load_stander()
    settle(model, data)
    assert not has_fallen(model, data), "a machine standing quietly has not fallen"
    data.qpos[ANKLE] = 0.5
    mujoco.mj_forward(model, data)
    assert has_fallen(model, data), "past the threshold counts as fallen"
    data.qpos[ANKLE] = -0.5
    mujoco.mj_forward(model, data)
    assert has_fallen(model, data), "tilted backwards past the threshold is just as fallen"
    print("exercise 3 looks right: attitude decides, in both directions")


# %%
if __name__ == "__main__":
    _try("exercise 3", _check_has_fallen)


# %%
def rollout(model, data, controller, push_force: float,
            horizon: int = HORIZON_STEPS) -> dict:
    """Settle, shove, and report what happened."""
    settle(model, data)
    peak_tilt = 0.0
    min_margin = float("inf")
    margin_at_push_end = float("nan")
    steps = horizon
    survived = True
    push_end = PUSH_START_STEP + PUSH_STEPS
    for k in range(horizon):
        pushing = PUSH_START_STEP <= k < push_end
        data.xfrc_applied[TORSO_ID, 0] = push_force if pushing else 0.0
        data.ctrl[0] = controller(model, data)
        mujoco.mj_step(model, data)

        peak_tilt = max(peak_tilt, abs(torso_tilt(model, data)))
        polygon = support_polygon(model, data)
        com_x = float(com_ground_projection(model, data)[0])
        margin = min(fore_aft_margins(polygon, com_x))
        min_margin = min(min_margin, margin)
        if k == push_end - 1:
            margin_at_push_end = margin
        if has_fallen(model, data):
            survived = False
            steps = k + 1
            break
    data.xfrc_applied[TORSO_ID, 0] = 0.0
    return {"survived": survived, "steps": steps, "peak_tilt": peak_tilt,
            "min_margin": min_margin, "margin_at_push_end": margin_at_push_end}


def posture_controller(gains):
    """A joint PD that holds the ankle at zero: the obvious first thing to try."""
    kp, kd = gains

    def control(model, data):
        return -(kp * data.qpos[ANKLE] + kd * data.qvel[ANKLE])

    return control


def zero_controller(model, data):
    """No control at all."""
    return 0.0


POSTURE_SOFT = (300.0, 60.0)
POSTURE_STIFF = (2000.0, 120.0)


def _check_rollout() -> None:
    model, data = load_stander()
    gentle = rollout(model, data, posture_controller(POSTURE_SOFT), 20.0)
    assert set(gentle) == {"survived", "steps", "peak_tilt", "min_margin",
                           "margin_at_push_end"}, f"keys were {sorted(gentle)}"
    assert gentle["survived"] and gentle["steps"] == HORIZON_STEPS, (
        f"a 20 N nudge should be shrugged off, but survived={gentle['survived']} after "
        f"{gentle['steps']} steps."
    )
    assert float(data.xfrc_applied[TORSO_ID, 0]) == 0.0, (
        "xfrc_applied is still set after the rollout returned."
    )
    shoved = rollout(model, data, zero_controller, 20.0)
    assert not shoved["survived"], "with NO control at all, it cannot survive any push"
    again = rollout(model, data, posture_controller(POSTURE_SOFT), 20.0)
    assert again == gentle, "the same rollout run twice gave different answers"
    print(f"exercise 4 looks right: 20 N survived with peak tilt "
          f"{gentle['peak_tilt']:.4f} rad, margin never below "
          f"{gentle['min_margin']:.4f} m")


# %%
if __name__ == "__main__":
    _try("exercise 4", _check_rollout, needs=("exercise 1", "exercise 3"))


# %%
def com_velocity(model, data) -> np.ndarray:
    """(vx, vy) of the whole machine's centre of mass."""
    mujoco.mj_subtreeVel(model, data)
    return np.asarray(data.subtree_linvel[0][:2], dtype=float)


BASELINE_GAINS = (100.0, 600.0, 150.0, 20.0)


def com_balance_torque(com_offset: float, com_vx: float, ankle_angle: float,
                       ankle_rate: float, gains) -> float:
    """Ankle torque from centre-of-mass feedback."""
    kp, kd, ka, kav = gains
    return -(kp * com_offset + kd * com_vx + ka * ankle_angle + kav * ankle_rate)


def com_controller(gains):
    """Wraps the two exercise functions into a controller `rollout` can call."""
    def control(model, data):
        com = com_ground_projection(model, data)
        return com_balance_torque(float(com[0]) - STANCE_COM_X,
                                  float(com_velocity(model, data)[0]),
                                  float(data.qpos[ANKLE]), float(data.qvel[ANKLE]), gains)
    return control


def _check_com_balance_torque() -> None:
    zero = com_balance_torque(0.0, 0.0, 0.0, 0.0, BASELINE_GAINS)
    assert abs(zero) < 1e-12, f"a machine exactly in its stance needs no torque, got {zero}"
    forward = com_balance_torque(0.01, 0.0, 0.0, 0.0, BASELINE_GAINS)
    assert forward < 0, f"a forward drift needs a negative torque, you returned {forward:+.3f}"
    assert abs(forward + 1.0) < 1e-9, f"expected -1.0 for this case, got {forward}"
    moving = com_balance_torque(0.0, 0.1, 0.0, 0.0, BASELINE_GAINS)
    assert abs(moving + 60.0) < 1e-9, f"expected -60.0 for this case, got {moving}"
    model, data = load_stander()
    result = rollout(model, data, com_controller(BASELINE_GAINS), 100.0)
    assert result["survived"], (
        f"the baseline gains are known to survive a 100 N push and yours fell after "
        f"{result['steps']} steps."
    )
    print(f"exercise 5 looks right: survived a 100 N push, peak tilt "
          f"{result['peak_tilt']:.4f} rad")


# %%
if __name__ == "__main__":
    _try("exercise 5", _check_com_balance_torque, needs=("exercise 4",))


# %%
def survival_pattern(model, data, controller, ladder=PUSH_LADDER) -> list:
    """Run one rollout per rung and return a list of booleans."""
    return [bool(rollout(model, data, controller, force)["survived"]) for force in ladder]


def largest_survivable_push(pattern, ladder=PUSH_LADDER) -> float:
    """The largest rung with an unbroken run of survivals beneath it."""
    best = 0.0
    for survived, rung in zip(pattern, ladder):
        if not survived:
            break
        best = float(rung)
    return best


def _check_largest_survivable_push() -> None:
    ladder = [20.0, 40.0, 60.0, 80.0, 100.0]
    got = largest_survivable_push([True, True, False, True, True], ladder)
    assert got == 40.0, f"expected 40.0, but you returned {got}"
    assert largest_survivable_push([False, False], [20.0, 40.0]) == 0.0
    assert largest_survivable_push([True, True], [20.0, 40.0]) == 40.0
    print("exercise 6 looks right: it reports the prefix, not the maximum")


# %%
if __name__ == "__main__":
    _try("exercise 6", _check_largest_survivable_push)


# %%
def _check_comparison() -> None:
    model, data = load_stander()
    settle(model, data)
    poly = support_polygon(model, data)
    heel, toe = fore_aft_margins(poly, float(data.subtree_com[0][0]))

    controllers = (("centre-of-mass feedback", com_controller(BASELINE_GAINS)),
                   ("posture PD, soft", posture_controller(POSTURE_SOFT)),
                   ("posture PD, stiff", posture_controller(POSTURE_STIFF)))

    print(f"  {'controller':<26s} {'forward':>9s} {'backward':>9s}")
    limits = {}
    for name, ctrl in controllers:
        fwd = largest_survivable_push(survival_pattern(model, data, ctrl, PUSH_LADDER),
                                      PUSH_LADDER)
        back_ladder = tuple(-f for f in PUSH_LADDER)
        back = -largest_survivable_push(
            survival_pattern(model, data, ctrl, back_ladder), back_ladder)
        limits[name] = (fwd, back)
        print(f"  {name:<26s} {fwd:>7.0f} N {back:>7.0f} N")

    fwd_com = limits["centre-of-mass feedback"][0]
    fwd_stiff = limits["posture PD, stiff"][0]
    backs = [b for _, b in limits.values()]
    assert fwd_com >= fwd_stiff, (
        "on this ladder centre-of-mass feedback should do at least as well forwards"
    )
    print(f"\n  Forwards, the controller matters: {fwd_com:.0f} N against "
          f"{fwd_stiff:.0f} N for the stiff posture PD.")
    if len(set(backs)) == 1:
        print(f"  Backwards, every controller stops at the same {backs[0]:.0f} N. There is "
              f"only {heel:.3f} m of heel\n  behind the mass against {toe:.3f} m of toe in "
              "front, and where the polygon leaves no\n  room, no amount of control buys "
              "any.")
    print("\n  That is the lesson: control quality only cashes in where the geometry has "
          "left\n  something to work with.")


# %%
if __name__ == "__main__":
    _try("three controllers", _check_comparison,
         needs=("exercise 4", "exercise 5", "exercise 6"))


# %%
def margin_and_fall_timeline(model, data, controller, push_force: float):
    """(step the margin first went negative, step the fall was declared)."""
    settle(model, data)
    first_negative, fell_at = None, None
    push_end = PUSH_START_STEP + PUSH_STEPS
    for k in range(HORIZON_STEPS):
        data.xfrc_applied[TORSO_ID, 0] = (push_force if PUSH_START_STEP <= k < push_end
                                          else 0.0)
        data.ctrl[0] = controller(model, data)
        mujoco.mj_step(model, data)
        margin = min(fore_aft_margins(support_polygon(model, data),
                                      float(com_ground_projection(model, data)[0])))
        if margin < 0 and first_negative is None:
            first_negative = k + 1
        if has_fallen(model, data):
            fell_at = k + 1
            break
    data.xfrc_applied[TORSO_ID, 0] = 0.0
    return first_negative, fell_at


def _check_doomed_but_inside() -> None:
    model, data = load_stander()
    ctrl = com_controller(BASELINE_GAINS)
    print(f"  {'push':>6s} {'survived':>9s} {'margin@push end':>16s} {'min margin':>11s} "
          f"{'fell at step':>13s}")
    rows = []
    for force in (100.0, 110.0, 115.0, 130.0):
        r = rollout(model, data, ctrl, force)
        rows.append((force, r))
        fell = "-" if r["survived"] else str(r["steps"])
        print(f"  {force:>4.0f} N {str(r['survived']):>9s} "
              f"{r['margin_at_push_end']:>+15.4f} m {r['min_margin']:>+10.4f} m {fell:>13s}")
    doomed = [r for f, r in rows if not r["survived"]]
    assert doomed, "at least one of these pushes should have toppled it"
    worst = doomed[0]
    assert worst["margin_at_push_end"] > 0, (
        "the machine should still be INSIDE its polygon when the push ends and fall anyway"
    )
    print(f"\n  Read the first failing row again: {worst['margin_at_push_end']:.4f} m of "
          "margin still in hand when\n  the push stopped, and it fell anyway. A static "
          "margin is a snapshot; what decides the\n  outcome is the margin together with "
          "the speed the mass is carrying.")

    left, fell = margin_and_fall_timeline(model, data, ctrl, 115.0)
    assert left is not None and fell is not None and left < fell, (
        "in the 115 N episode the centre of mass should leave the polygon well before the "
        f"fall is declared, but got first-negative={left} and fall={fell}."
    )
    print(f"\n  Now the converse, measured in that same 115 N episode. The centre of mass "
          f"left the\n  polygon at step {left}; the fall was only declared at step {fell}. "
          f"That is {(fell - left) * MODEL.opt.timestep:.3f} s spent\n  outside the support "
          "polygon and still standing.")
    print("\n  So the margin is neither a sufficient warning nor a definition of failure:")
    print("  inside it, this machine was already lost; outside it, it was still upright.")
    print("  That is exactly why has_fallen reads the torso's attitude and not the margin.")


# %%
if __name__ == "__main__":
    _try("inside and already lost", _check_doomed_but_inside,
         needs=("exercise 4", "exercise 5"))


# %%
def _check_non_monotonic() -> None:
    model, data = load_stander()
    stiff = posture_controller(POSTURE_STIFF)
    outcomes = {}
    for force in (80.0, 101.5):
        outcomes[force] = rollout(model, data, stiff, force)["survived"]
        print(f"  stiff posture PD, {force:5.1f} N push -> survived={outcomes[force]}")
    assert not outcomes[80.0] and outcomes[101.5], (
        "this cell exists to show survival is not monotone; if both pushes now agree, "
        "re-read the pair before trusting the sentence below"
    )
    print("  A bigger shove survived where a smaller one did not. That is why the prefix, and")
    print("  not the maximum, is the honest summary of a ladder.")


# %%
if __name__ == "__main__":
    _try("survival is not monotone", _check_non_monotonic, needs=("exercise 4",))


# %%
SELF_CHECK = {1: "c", 2: "b", 3: "c", 4: "b", 5: "b"}

_ANSWER_DIGESTS = {1: "81834fc23b17f145", 2: "53c70eb91b2fc860", 3: "cf87226d350e25e0",
                   4: "ce56af52013c315a", 5: "6fcf8ed7f38b55fd"}
_ANSWER_SECTIONS = {
    1: "section 4 — the centre of pressure you measured at rest, and why nothing is accelerating",
    2: "section 5 — the flat-foot ceiling you measured against the motor's ctrlrange",
    3: "section 11 — the margin at the end of the push, in the row that still fell",
    4: "section 10 — the backward column, and the heel and toe margins from section 3",
    5: "section 12 — the two pushes you ran at the foot of the common-mistakes list",
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the five multiple-choice answers in SELF_CHECK, naming where to look again."""
    answers = SELF_CHECK if answers is None else answers
    wrong = []
    for q, digest in sorted(_ANSWER_DIGESTS.items()):
        got = str(answers.get(q, "?")).strip().lower()
        if hashlib.sha256(f"F15-L04-q{q}-{got}".encode()).hexdigest()[:16] != digest:
            wrong.append(q)
    for q in sorted(_ANSWER_DIGESTS):
        note = f"  -> re-read {_ANSWER_SECTIONS[q]}" if q in wrong else ""
        print(f"  q{q}: {'wrong' if q in wrong else 'right'}{note}")
    assert not wrong, f"questions {wrong} are still wrong."
    print("self-check: all five right")


def _self_check_if_answered() -> None:
    """Mark SELF_CHECK, but count it as not started while every answer is still a "?"."""
    if all(str(a).strip() == "?" for a in SELF_CHECK.values()):
        raise NotImplementedError("not answered yet — put your five letters in SELF_CHECK "
                                  "in section 13")
    _check_self_check()


if __name__ == "__main__":
    _try("self-check", _self_check_if_answered)



# %% [markdown]
# ## Self-check answers
#
# **1 — (c).** Standing still means zero linear and angular acceleration, so the net moment
# about the centre of mass must be zero. Gravity acts through the centre of mass, so the
# ground reaction has to act through the point directly beneath it — which is what the centre
# of pressure is. It is forced by equilibrium, not true by definition, and the moment the
# machine accelerates the two separate: that separation is precisely what a push creates and
# what the controller spends its torque managing. (a) confuses a consequence with a
# definition; (b) ignores that the agreement held to about a micrometre; (d) contradicts the
# measurement printed in section 4.
#
# **2 — (b).** The ground can only push, and only where the foot is, so the centre of pressure
# is confined to the footprint. Ankle torque is balanced by moving that pressure, and once it
# reaches the edge there is nowhere left to move it: the extra torque rotates the foot about
# its edge instead of rotating the body upright. The measured ceiling is around a fifth of the
# motor's range. (a) and (d) are real effects but neither sets this ceiling; (c) is backwards —
# the clamp is at ±60 N·m, far above where the foot gives out, so it never binds here.
#
# **3 — (c).** A margin is a statement about position only. At the end of the push the mass is
# inside the polygon but travelling fast enough that no admissible centre of pressure can
# arrest it before it crosses the toe — the ceiling measured in section 5 is exactly the limit
# on how hard the polygon is allowed to push back. Balance is a condition on position *and*
# velocity together. (a) and (b) blame the instrument for a real result; (d) is wrong — the
# push is applied to the torso in both the surviving and failing rows.
#
# **4 — (b).** The heel margin is about 0.083 m against roughly 0.157 m of toe, so backwards
# there is barely any room for the centre of pressure to move, and every controller reaches
# that same geometric wall at the same push. Forwards there is nearly twice the room, and how
# well a controller exploits it is what separates them. (a) mistakes a physically forced
# agreement for a bug; (c) generalises a fact about this asymmetric foot into a law about all
# robots; (d) would not help — the limit is not a shortage of gain, it is a shortage of floor.
#
# **5 — (b).** Survival is a threshold on a nonlinear trajectory, not a monotone function of
# push size, so isolated magnitudes can be caught on a rebound that a slightly smaller push is
# not. That is a fact about the dynamics, not a defect: the simulation is deterministic and
# repeats exactly, which the rollout check verifies. It is also why the summary statistic has
# to be the unbroken prefix — a machine that fails at 80 N cannot be certified for 101.5 N.
# (a) is refuted by the repeatability check; (c) assumes the anomaly rather than the
# assumption is wrong; (d) is the actual trap — bisection silently assumes monotonicity and
# will report a limit above pushes that are known to fail.

# %%
# Your progress. This re-runs each exercise's check, and the self-check, against your code as
# it stands now, so the board is current even if you changed something after running its cell.
if __name__ == "__main__":
    print("Re-checking every exercise against your code as it stands now:")
    for _label, _check, _needs in (
            ("exercise 1", _check_com_ground_projection, ()),
            ("exercise 2", _check_support_margin, ("exercise 1",)),
            ("exercise 3", _check_has_fallen, ()),
            ("exercise 4", _check_rollout, ("exercise 1", "exercise 3")),
            ("exercise 5", _check_com_balance_torque, ("exercise 4",)),
            ("exercise 6", _check_largest_survivable_push, ()),
            ("self-check", _self_check_if_answered, ())):
        _try(_label, _check, needs=_needs)
    _progress_board((("exercise 1", "com_ground_projection, where the mass is"),
                     ("exercise 2", "support_margin, how much room is left"),
                     ("exercise 3", "has_fallen, what counts as fallen"),
                     ("exercise 4", "rollout, the push and what it measures"),
                     ("exercise 5", "com_balance_torque, feedback on the mass"),
                     ("exercise 6", "largest_survivable_push, the prefix of the ladder"),
                     ("self-check", "the five questions in section 13")))
    # A stub you have not reached yet is not a failure. A check that ran and came back wrong
    # is: in a script or under CI it ends the run non-zero, so a green exit code cannot paper
    # over it. Inside a notebook kernel it is a printed line, never a traceback.
    _failed = [label for label, state in _STATUS.items() if state == "failed"]
    if _failed:
        _message = "checks failed: " + ", ".join(_failed)
        if "ipykernel" in sys.modules:
            print(_message)
        else:
            raise SystemExit(_message)

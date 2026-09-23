"""Autograder rubric for F15-L04. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These cases deliberately differ from the notebook's public `_check_*` functions. They probe
the failure modes a plausible-looking wrong implementation has: a centre of mass that reads
the precomputed `subtree_com` instead of building it, a margin that measures to corners or
loses its sign, a fall detector that fires on the support margin, a rollout that leaves
`xfrc_applied` set, a controller with the intuitive (wrong) sign, and a "largest survivable
push" that takes a maximum over a non-monotone ladder instead of a prefix.

Reviewed 2026-09-16 by mutation testing: 25 broken copies of the reference were graded, and
the eight that scored 100% were all in exercise 4, because the rubric only ever read
`survived` and `steps` out of the rollout. It now also pins down the three numbers the
lesson's own conclusions rest on — see `test_rollout_measurements_mean_what_they_say`, and
the mid-push fall and post-shove `min_margin` cases added to the two tests around it.
"""
import importlib.util, os
from pathlib import Path

import mujoco
import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

KEYS = {"survived", "steps", "peak_tilt", "min_margin", "margin_at_push_end"}
SQUARE = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])


def _fresh():
    return lesson.load_stander()


# --- exercise 1: com_ground_projection ----------------------------------------------------

def test_com_matches_mujocos_own_subtree_com():
    """Agreement at several poses, including ones the notebook never visits."""
    model, data = _fresh()
    for qpos in ([0.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.10, -0.20], [0.0, 0.0, 0.30, 0.25]):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        got = np.asarray(lesson.com_ground_projection(model, data), dtype=float)
        assert got.shape == (2,), f"expected shape (2,), got {got.shape}"
        want = np.asarray(data.subtree_com[0][:2], dtype=float)
        assert np.allclose(got, want, atol=1e-9), (
            f"at qpos={qpos} you gave {got} but MuJoCo's whole-body centre of mass is {want}"
        )


def test_com_is_built_from_xipos_not_read_from_subtree_com():
    """The shortcut is blocked: poison subtree_com and the honest calculation still works."""
    model, data = _fresh()
    lesson.settle(model, data)
    masses = model.body_mass[1:]
    want = (masses[:, None] * data.xipos[1:]).sum(axis=0)[:2] / masses.sum()
    data.subtree_com[0] = [999.0, 999.0, 999.0]     # stale on purpose; do NOT mj_forward
    got = np.asarray(lesson.com_ground_projection(model, data), dtype=float)
    assert np.allclose(got, want, atol=1e-9), (
        f"you returned {got}. With data.subtree_com deliberately poisoned, a centre of mass "
        "built from model.body_mass and data.xipos is unaffected — so this says you are "
        "returning MuJoCo's precomputed value rather than computing the average yourself."
    )
    # An unweighted mean of xipos is the other common wrong answer; make sure it is not that.
    unweighted = data.xipos[1:].mean(axis=0)[:2]
    assert not np.allclose(got, unweighted, atol=1e-9), (
        "your answer equals the plain (unweighted) mean of xipos — weight each body by its "
        "mass, or an 18 kg torso counts for no more than a 1.5 kg foot"
    )


# --- exercise 2: support_margin ------------------------------------------------------------

def test_margin_measures_to_edges_not_corners():
    """The unit square, and a point whose nearest corner and nearest edge disagree."""
    got = lesson.support_margin(np.array([0.5, 0.5]), SQUARE)
    assert abs(got - 0.5) < 1e-9, (
        f"the centre of a unit square is 0.5 from the nearest EDGE, you returned {got}; "
        "0.7071 means you measured to the nearest corner"
    )
    got = lesson.support_margin(np.array([0.25, 0.5]), SQUARE)
    assert abs(got - 0.25) < 1e-9, f"expected 0.25, got {got}"
    got = lesson.support_margin(np.array([0.1, 0.1]), SQUARE)
    assert abs(got - 0.1) < 1e-9, f"expected 0.1 near a corner, got {got}"
    # A SLANTED edge, and an answer that is not a round number. Every other polygon here is
    # axis-aligned, and a foot that has rotated even slightly is not: a margin built from
    # min/max of the x and y columns is a bounding box, not a polygon, and it cannot see a
    # diagonal edge at all. The nearest edge below is the hypotenuse, 0.2/sqrt(2) away.
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    got = lesson.support_margin(np.array([0.4, 0.4]), triangle)
    want = 0.2 / np.sqrt(2.0)
    assert abs(got - want) < 1e-9, (
        f"inside this right triangle the nearest edge is the sloping one, {want:.10f} m "
        f"away, but you returned {got}. {0.4:.1f} means you measured only to the vertical "
        "and horizontal edges — take the perpendicular distance to every edge, whichever way "
        "it points. An answer that is right to 2 decimal places and wrong after that says "
        "you rounded something."
    )
    outside = lesson.support_margin(np.array([0.9, 0.9]), triangle)
    want_out = -0.8 / np.sqrt(2.0)
    assert abs(outside - want_out) < 1e-9, (
        f"that point is outside the hypotenuse, so the margin is {want_out:.10f} m; you "
        f"returned {outside}"
    )


def test_margin_keeps_its_sign_and_boundary():
    """Outside must be negative, and the boundary exactly zero."""
    outside = lesson.support_margin(np.array([1.5, 0.5]), SQUARE)
    assert abs(outside + 0.5) < 1e-9, (
        f"a point 0.5 m outside must give -0.5, you returned {outside}. A positive answer "
        "means an abs() crept in, and the sign is the entire point of this function."
    )
    far = lesson.support_margin(np.array([0.5, -2.0]), SQUARE)
    assert abs(far + 2.0) < 1e-9, f"expected -2.0, got {far}"
    edge = lesson.support_margin(np.array([0.0, 0.5]), SQUARE)
    assert abs(edge) < 1e-9, f"a point on the boundary must give 0.0, got {edge}"


def test_margin_on_a_wider_polygon_and_the_real_footprint():
    """A rectangle the notebook never uses, and the model's own support polygon."""
    hull = np.array([[0.2, -0.3], [0.2, 0.3], [-0.1, 0.3], [-0.1, -0.3]])
    got = lesson.support_margin(np.array([0.0, 0.0]), hull)
    assert abs(got - 0.1) < 1e-9, (
        f"nearest edge of that rectangle is 0.1 away, you returned {got}"
    )
    model, data = _fresh()
    lesson.settle(model, data)
    poly = lesson.support_polygon(model, data)
    com = np.asarray(lesson.com_ground_projection(model, data), dtype=float)
    margin = lesson.support_margin(com, poly)
    half_width = float(model.geom_size[lesson.FOOT_GEOM_ID][1])
    assert abs(margin - half_width) < 1e-3, (
        f"standing, the nearest edge is a SIDE edge {half_width:.3f} m away, but you returned "
        f"{margin:.4f}. This foot is narrower than it is long, so the limiting edge at rest "
        "is not the heel."
    )


# --- exercise 3: has_fallen ---------------------------------------------------------------

def test_has_fallen_reads_attitude_in_both_directions():
    model, data = _fresh()
    lesson.settle(model, data)
    assert not lesson.has_fallen(model, data), "standing quietly is not fallen"
    for angle, expected in ((0.30, False), (0.40, True), (-0.40, True), (-0.30, False)):
        data.qpos[:] = [0.0, 0.0, 0.0, angle]
        mujoco.mj_forward(model, data)
        got = bool(lesson.has_fallen(model, data))
        assert got == expected, (
            f"with the ankle at {angle:+.2f} rad the torso tilt is "
            f"{lesson.torso_tilt(model, data):+.3f} rad and the threshold is "
            f"{lesson.FALL_TILT}, so has_fallen should be {expected}, got {got}. A one-sided "
            "comparison misses backward falls — take the absolute value."
        )


def test_has_fallen_does_not_use_the_support_margin():
    """Shrink the foot so the margin goes negative while the torso stays upright."""
    model, data = _fresh()
    lesson.settle(model, data)
    original = float(model.geom_size[lesson.FOOT_GEOM_ID][0])
    try:
        model.geom_size[lesson.FOOT_GEOM_ID][0] = 0.01     # a 2 cm foot
        mujoco.mj_forward(model, data)
        poly = lesson.support_polygon(model, data)
        com = np.asarray(lesson.com_ground_projection(model, data), dtype=float)
        margin = lesson.support_margin(com, poly)
        assert margin < 0, "test setup failed: the margin should now be negative"
        assert not lesson.has_fallen(model, data), (
            f"the centre of mass is now {margin:.4f} m OUTSIDE the support polygon, but the "
            f"torso is still upright at {lesson.torso_tilt(model, data):+.4f} rad, so this "
            "machine has not fallen. has_fallen must read the tilt, not the margin: leaving "
            "the polygon is trouble, not a fall."
        )
    finally:
        model.geom_size[lesson.FOOT_GEOM_ID][0] = original
        mujoco.mj_forward(model, data)


# --- exercise 4: rollout ------------------------------------------------------------------

def test_rollout_contract_and_a_survivable_push():
    model, data = _fresh()
    r = lesson.rollout(model, data, lesson.posture_controller(lesson.POSTURE_SOFT), 20.0)
    assert isinstance(r, dict), f"expected a dict, got {type(r).__name__}"
    assert set(r) == KEYS, f"keys were {sorted(r)}, expected {sorted(KEYS)}"
    assert r["survived"], "a 20 N nudge must be survivable"
    assert r["steps"] == lesson.HORIZON_STEPS, (
        f"a survivor runs the full horizon, yours stopped at {r['steps']}"
    )
    assert 0.0 < r["peak_tilt"] < lesson.FALL_TILT, (
        f"peak_tilt {r['peak_tilt']} should be positive and below the fall threshold"
    )
    assert r["margin_at_push_end"] > 0, (
        "the machine is still inside its polygon when a 20 N push ends"
    )
    assert r["min_margin"] <= r["margin_at_push_end"], (
        "min_margin is the smallest margin over the whole episode, so it cannot exceed the "
        "margin recorded at the end of the push"
    )
    # A rollout must never report an episode it did not actually run.
    boom = RuntimeError("controller exploded")

    def exploding(model, data):
        raise boom

    try:
        bogus = lesson.rollout(model, data, exploding, 20.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            f"the controller raised, and rollout returned {bogus} anyway. Do not wrap the "
            "step in try/except: swallowing the error and reporting survived=True turns a "
            "broken controller into a passing one, which is the single most expensive bug "
            "you can put in a test harness."
        )


def test_rollout_zeroes_the_push_and_a_big_one_topples_it():
    model, data = _fresh()
    hard = lesson.rollout(model, data, lesson.posture_controller(lesson.POSTURE_SOFT), 200.0)
    assert not hard["survived"], "a 200 N shove must topple it"
    assert hard["steps"] < lesson.HORIZON_STEPS, (
        "a rollout that fell should stop early and report the step it fell on"
    )
    assert float(data.xfrc_applied[lesson.TORSO_ID, 0]) == 0.0, (
        "xfrc_applied is still set to the push after the rollout returned. MuJoCo never "
        "clears it, so the next episode would run with a permanent force on the torso."
    )
    # The leak that actually bites: 600 N topples it BEFORE the push window is over, so the
    # loop never reaches the step that would have written the zero for you. Only an explicit
    # zero before `return` survives this case.
    push_end = lesson.PUSH_START_STEP + lesson.PUSH_STEPS
    slammed = lesson.rollout(model, data, lesson.posture_controller(lesson.POSTURE_SOFT), 600.0)
    assert not slammed["survived"] and slammed["steps"] < push_end, (
        f"600 N should topple it inside the push window, but survived={slammed['survived']} "
        f"after {slammed['steps']} steps (the window closes at step {push_end})"
    )
    assert float(data.xfrc_applied[lesson.TORSO_ID, 0]) == 0.0, (
        f"after a rollout that fell mid-push, xfrc_applied is still "
        f"{float(data.xfrc_applied[lesson.TORSO_ID, 0]):.1f} N. Zeroing the force inside the "
        "loop is not enough: when the episode breaks out early that line never runs again. "
        "Set data.xfrc_applied[TORSO_ID, 0] = 0.0 after the loop, before you return."
    )
    assert np.isnan(slammed["margin_at_push_end"]), (
        f"it fell at step {slammed['steps']}, before the push ended, so there is no margin "
        f"at the end of the push to report — that key must be float('nan'), not "
        f"{slammed['margin_at_push_end']}"
    )
    gentle = lesson.rollout(model, data, lesson.posture_controller(lesson.POSTURE_SOFT), 20.0)
    assert gentle["survived"], (
        "a 20 N push failed immediately after a 200 N one — the previous push is leaking "
        "into this episode, so zero xfrc_applied before returning"
    )
    assert gentle["min_margin"] > 0, (
        f"a 20 N nudge never takes the centre of mass outside the foot, yet this episode "
        f"reports min_margin={gentle['min_margin']:+.4f} m — a negative value it can only "
        "have inherited from the 600 N episode before it. Every accumulator (min_margin, "
        "peak_tilt) must be created fresh inside rollout; a list or dict in the DEFAULT "
        "ARGUMENTS is built once when the function is defined and shared by every call."
    )


def test_rollout_resets_and_repeats_exactly():
    model, data = _fresh()
    ctrl = lesson.posture_controller(lesson.POSTURE_SOFT)
    first = lesson.rollout(model, data, ctrl, 60.0)
    second = lesson.rollout(model, data, ctrl, 60.0)
    assert first == second, (
        "two identical rollouts on the same data disagreed — call settle() at the very start "
        "so the second episode does not begin wherever the first one ended"
    )
    fresh_model, fresh_data = _fresh()
    third = lesson.rollout(fresh_model, fresh_data, ctrl, 60.0)
    assert abs(third["peak_tilt"] - first["peak_tilt"]) < 1e-9, (
        "a freshly loaded model gave a different answer; this simulation is deterministic, "
        "so that points at state left over inside rollout"
    )


def test_rollout_measurements_mean_what_they_say():
    """`survived` and `steps` are not the whole contract.

    Sections 10-12 of the lesson read `peak_tilt` and `margin_at_push_end` and draw
    conclusions from them, so a rollout that returns plausible numbers measured at the wrong
    moment is worse than one that crashes. This pins down WHEN each number is taken.
    """
    model, data = _fresh()

    # (a) the push window: PUSH_STEPS steps of exactly push_force, in one unbroken block.
    seen = []

    def recording(model_, data_, _inner=lesson.posture_controller(lesson.POSTURE_SOFT)):
        seen.append(float(data_.xfrc_applied[lesson.TORSO_ID, 0]))
        return _inner(model_, data_)

    forward = lesson.rollout(model, data, recording, 60.0)
    assert forward["survived"], "60 N with the soft posture PD is survivable"
    # The controller is asked for a torque once per step, so its call count IS the step count.
    assert len(seen) == forward["steps"] == lesson.HORIZON_STEPS, (
        f"this episode reports steps={forward['steps']}, but the controller was called "
        f"{len(seen)} times and the horizon is {lesson.HORIZON_STEPS}. One call per step, so "
        "these three are the same number: `range(horizon)` runs horizon times, `range(horizon "
        "- 1)` is one short, and a rollout that never fell took the full horizon."
    )
    pushed = [i for i, f in enumerate(seen) if f != 0.0]
    assert len(pushed) == lesson.PUSH_STEPS, (
        f"the torso was pushed on {len(pushed)} steps, but PUSH_STEPS is "
        f"{lesson.PUSH_STEPS}. `PUSH_START_STEP <= k < PUSH_START_STEP + PUSH_STEPS` is a "
        "half-open window: <= at both ends pushes for one step too many."
    )
    assert pushed[-1] - pushed[0] + 1 == len(pushed), (
        f"the push is one unbroken block, but the pushed steps run from {pushed[0]} to "
        f"{pushed[-1]} with gaps in between"
    )
    assert all(seen[i] == 60.0 for i in pushed), (
        "the force is the same push_force on every step of the window; do not ramp, halve "
        "or rescale it"
    )

    # (b) peak_tilt is a running maximum of the ABSOLUTE tilt over the whole episode.
    #     The controller sees every state the rollout does, so for an episode that survives
    #     (the largest tilt falls strictly inside the run) the two maxima are the same float.
    tilts = []

    def watching(model_, data_, _inner=lesson.posture_controller(lesson.POSTURE_SOFT)):
        tilts.append(abs(float(lesson.torso_tilt(model_, data_))))
        return _inner(model_, data_)

    watched = lesson.rollout(model, data, watching, 60.0)
    want = max(tilts)
    assert abs(watched["peak_tilt"] - want) < 1e-12, (
        f"peak_tilt is {watched['peak_tilt']:.6f}, but the largest absolute tilt in this "
        f"episode is {want:.6f} (the average is {sum(tilts) / len(tilts):.6f} and the tilt "
        f"it finishes on is {tilts[-1]:.6f}). peak_tilt is the running MAXIMUM over the "
        "episode — not a mean, and not whatever the last step happened to leave behind."
    )

    # (c) ...and it is absolute, or a backward lean reports as no lean at all.
    backward = lesson.rollout(model, data, lesson.posture_controller(lesson.POSTURE_SOFT),
                              -30.0)
    assert backward["survived"], "a 30 N pull is survivable; this test is about the sign"
    assert backward["peak_tilt"] > 0.01, (
        f"pulled backwards the machine leans to a tilt of about -0.03 rad, but peak_tilt "
        f"came back as {backward['peak_tilt']:.6f}. `max(peak_tilt, torso_tilt(...))` never "
        "rises above its starting 0.0 when every tilt is negative — take abs() first."
    )

    # (d) margin_at_push_end is taken at the END of the push, so it must move with the push.
    nudged = lesson.rollout(model, data, lesson.com_controller(lesson.BASELINE_GAINS), 20.0)
    shoved = lesson.rollout(model, data, lesson.com_controller(lesson.BASELINE_GAINS), 100.0)
    spread = abs(shoved["margin_at_push_end"] - nudged["margin_at_push_end"])
    assert spread > 5e-3, (
        f"a 20 N push and a 100 N push both ended with a margin of about "
        f"{nudged['margin_at_push_end']:+.5f} m (they differ by {spread:.2e} m). After 0.2 s "
        "of shoving those two machines are in visibly different places, so a margin that "
        "barely moves was recorded at the START of the push window, not at its end: the step "
        "you want is k == PUSH_START_STEP + PUSH_STEPS - 1."
    )
    model2, data2 = _fresh()
    lesson.settle(model2, data2)
    at_rest = min(lesson.fore_aft_margins(
        lesson.support_polygon(model2, data2),
        float(lesson.com_ground_projection(model2, data2)[0])))
    assert abs(shoved["margin_at_push_end"] - at_rest) > 5e-3, (
        f"after a 100 N push the fore/aft margin is still {shoved['margin_at_push_end']:+.5f} "
        f"m, which is the margin of a machine standing undisturbed ({at_rest:+.5f} m). The "
        "push has not moved the mass, so it is not reaching the simulation — check that you "
        "write push_force into data.xfrc_applied BEFORE mj_step, not after."
    )


def _documented_episode(controller, push_force):
    """Simulate exactly the episode `rollout`'s docstring describes, independently.

    This is the oracle. It uses only the parts of the lesson that were handed to the student
    (`settle`, `torso_tilt`, `support_polygon`, `fore_aft_margins`) plus the constants, so it
    stays a fixed target however the student's own rollout is written. The simulation is
    deterministic, so every number below must match theirs to the last bit.
    """
    model, data = _fresh()
    lesson.settle(model, data)
    push_end = lesson.PUSH_START_STEP + lesson.PUSH_STEPS
    peak_tilt, min_margin = 0.0, float("inf")
    margin_at_push_end = float("nan")
    survived, steps = True, lesson.HORIZON_STEPS
    for k in range(lesson.HORIZON_STEPS):
        pushing = lesson.PUSH_START_STEP <= k < push_end
        data.xfrc_applied[lesson.TORSO_ID, 0] = push_force if pushing else 0.0
        data.ctrl[0] = controller(model, data)
        mujoco.mj_step(model, data)
        tilt = abs(float(lesson.torso_tilt(model, data)))
        peak_tilt = max(peak_tilt, tilt)
        margin = min(lesson.fore_aft_margins(
            lesson.support_polygon(model, data),
            float(lesson.com_ground_projection(model, data)[0])))
        min_margin = min(min_margin, margin)
        if k == push_end - 1:
            margin_at_push_end = margin
        if tilt > lesson.FALL_TILT:
            survived, steps = False, k + 1
            break
    data.xfrc_applied[lesson.TORSO_ID, 0] = 0.0
    return {"survived": survived, "steps": steps, "peak_tilt": peak_tilt,
            "min_margin": min_margin, "margin_at_push_end": margin_at_push_end}


_WHY = {
    "survived": "survived is False only if the tilt ever passed FALL_TILT",
    "steps": "steps is the number of steps actually taken: k + 1 on the step it fell, and "
             "the full horizon if it never did",
    "peak_tilt": "peak_tilt is the running maximum of abs(torso_tilt) over the episode",
    "min_margin": "min_margin is the running MINIMUM of the fore/aft margin — not the last "
                  "one, and not their average",
    "margin_at_push_end": "margin_at_push_end is the margin on step "
                          "PUSH_START_STEP + PUSH_STEPS - 1, or nan if it fell first",
}


def test_rollout_reproduces_the_documented_episode():
    """The whole dict, against an independent simulation of the same episode.

    Every off-by-one that survives the checks above — a horizon one step short, a push window
    shifted, a fall noticed one step late — changes one of these five numbers and nothing
    else, so this is where they are caught.
    """
    for label, controller, force in (
            ("the 100 N push it survives", lesson.com_controller(lesson.BASELINE_GAINS), 100.0),
            ("the 600 N shove that fells it mid-push",
             lesson.posture_controller(lesson.POSTURE_SOFT), 600.0)):
        want = _documented_episode(controller, force)
        model, data = _fresh()
        got = lesson.rollout(model, data, controller, force)
        for key in ("survived", "steps"):
            assert got[key] == want[key], (
                f"on {label}, rollout reported {key}={got[key]!r} where the episode the "
                f"docstring describes gives {want[key]!r}. {_WHY[key]}."
            )
        for key in ("peak_tilt", "min_margin", "margin_at_push_end"):
            a, b = float(got[key]), float(want[key])
            if np.isnan(b):
                assert np.isnan(a), (
                    f"on {label}, {key} should be nan but you reported {a}. {_WHY[key]}."
                )
                continue
            assert not np.isnan(a) and abs(a - b) < 1e-12, (
                f"on {label}, rollout reported {key}={a} where this episode gives {b} "
                f"(off by {abs(a - b):.3e}). This simulation is deterministic, so any "
                f"difference at all is a difference in what you measured or when. {_WHY[key]}."
            )


# --- exercise 5: com_balance_torque --------------------------------------------------------

def test_com_balance_torque_signs_and_scale():
    gains = (100.0, 600.0, 150.0, 20.0)
    assert abs(lesson.com_balance_torque(0.0, 0.0, 0.0, 0.0, gains)) < 1e-12, (
        "a machine exactly in its stance, at rest, needs no torque"
    )
    forward = lesson.com_balance_torque(0.02, 0.0, 0.0, 0.0, gains)
    assert abs(forward + 2.0) < 1e-9, (
        f"expected -2.0 (the mass has drifted forwards, so the torque is negative), got "
        f"{forward}. A positive answer means the leading minus sign is missing: positive "
        "ankle torque moves the centre of pressure BACKWARDS, which is the wrong way here."
    )
    back = lesson.com_balance_torque(-0.02, 0.0, 0.0, 0.0, gains)
    assert abs(back - 2.0) < 1e-9, f"expected +2.0 for a backward drift, got {back}"
    each = (lesson.com_balance_torque(0.0, 0.1, 0.0, 0.0, gains),
            lesson.com_balance_torque(0.0, 0.0, 0.1, 0.0, gains),
            lesson.com_balance_torque(0.0, 0.0, 0.0, 0.1, gains))
    assert all(abs(g - w) < 1e-9 for g, w in zip(each, (-60.0, -15.0, -2.0))), (
        f"the four gains must each multiply their own term; got {each}, expected "
        "(-60.0, -15.0, -2.0) — check the order (kp, kd, ka, kav)"
    )


def test_com_balance_torque_is_linear():
    gains = (100.0, 600.0, 150.0, 20.0)
    doubled = tuple(2.0 * g for g in gains)
    single = lesson.com_balance_torque(0.01, 0.05, -0.02, 0.3, gains)
    twice = lesson.com_balance_torque(0.01, 0.05, -0.02, 0.3, doubled)
    assert abs(twice - 2.0 * single) < 1e-9, (
        f"doubling every gain must double the torque: got {twice} against {single}"
    )
    combined = lesson.com_balance_torque(0.01, 0.05, -0.02, 0.3, gains)
    parts = (lesson.com_balance_torque(0.01, 0.0, 0.0, 0.0, gains)
             + lesson.com_balance_torque(0.0, 0.05, 0.0, 0.0, gains)
             + lesson.com_balance_torque(0.0, 0.0, -0.02, 0.0, gains)
             + lesson.com_balance_torque(0.0, 0.0, 0.0, 0.3, gains))
    assert abs(combined - parts) < 1e-9, (
        "the law is a plain weighted sum, so the four terms must superpose"
    )


# --- exercise 6: largest_survivable_push ---------------------------------------------------

def test_largest_survivable_push_takes_the_prefix_not_the_maximum():
    ladder = [20.0, 40.0, 60.0, 80.0, 100.0]
    got = lesson.largest_survivable_push([True, True, False, True, True], ladder)
    assert got == 40.0, (
        f"the unbroken run ends at 40 N, so the answer is 40.0, not {got}. Returning 100.0 "
        "means you took the largest surviving rung; survival is not monotone, and a machine "
        "that fails 60 N cannot be certified for 100 N."
    )
    assert lesson.largest_survivable_push([False, True, True], ladder[:3]) == 0.0, (
        "if the smallest rung already fails, nothing has been survived: return 0.0"
    )
    assert lesson.largest_survivable_push([True, True, True], ladder[:3]) == 60.0, (
        "if every rung survives, the answer is the top rung"
    )
    assert lesson.largest_survivable_push([True], [7.5]) == 7.5, (
        "the answer must come from the ladder you were given, not from PUSH_LADDER"
    )


def test_largest_survivable_push_on_the_real_ladder():
    """End to end: the shipped gains, the shipped ladder, through the student's own rollout."""
    model, data = _fresh()
    ctrl = lesson.com_controller(lesson.BASELINE_GAINS)
    pattern = lesson.survival_pattern(model, data, ctrl, lesson.PUSH_LADDER)
    got = lesson.largest_survivable_push(pattern, lesson.PUSH_LADDER)
    assert got >= 80.0, (
        f"the reference controller survives well past 80 N on this ladder, but you measured "
        f"{got} N with pattern {pattern}. That points at rollout or the controller sign "
        "rather than at this function."
    )
    assert got < 130.0, (
        f"you measured {got} N, but a 130 N shove topples this controller — check that "
        "rollout really applies the push for the full window"
    )


RUBRIC = [
    (test_com_matches_mujocos_own_subtree_com, 3,
     "mass-weight data.xipos over bodies 1.. and drop z"),
    (test_com_is_built_from_xipos_not_read_from_subtree_com, 2,
     "build the average yourself from model.body_mass and data.xipos"),
    (test_margin_measures_to_edges_not_corners, 3,
     "signed distance to each EDGE, then take the minimum"),
    (test_margin_keeps_its_sign_and_boundary, 2,
     "never take an absolute value; outside must come back negative"),
    (test_margin_on_a_wider_polygon_and_the_real_footprint, 2,
     "the nearest edge at rest is a side edge, not the heel"),
    (test_has_fallen_reads_attitude_in_both_directions, 3,
     "compare abs(torso_tilt(model, data)) against FALL_TILT"),
    (test_has_fallen_does_not_use_the_support_margin, 3,
     "leaving the support polygon is trouble, not a fall — read the tilt"),
    (test_rollout_contract_and_a_survivable_push, 3,
     "return the five documented keys; settle() first, then the push window"),
    (test_rollout_zeroes_the_push_and_a_big_one_topples_it, 3,
     "set data.xfrc_applied back to zero before returning"),
    (test_rollout_resets_and_repeats_exactly, 3,
     "settle() at the top of every episode, before the loop"),
    (test_rollout_measurements_mean_what_they_say, 3,
     "peak_tilt is a running max of abs(tilt); margin_at_push_end is taken on the LAST step "
     "of the push window; the window is half-open"),
    (test_rollout_reproduces_the_documented_episode, 3,
     "walk the docstring one line at a time: horizon steps, half-open push window, measure "
     "AFTER mj_step, fall check last"),
    (test_com_balance_torque_signs_and_scale, 3,
     "torque = -(kp*offset + kd*vel + ka*ankle + kav*rate), gains in that order"),
    (test_com_balance_torque_is_linear, 2,
     "a plain weighted sum: no clamping, no squaring, no special cases"),
    (test_largest_survivable_push_takes_the_prefix_not_the_maximum, 2,
     "walk up from the bottom and stop at the first failure"),
    (test_largest_survivable_push_on_the_real_ladder, 2,
     "this one exercises rollout and the controller end to end"),
]

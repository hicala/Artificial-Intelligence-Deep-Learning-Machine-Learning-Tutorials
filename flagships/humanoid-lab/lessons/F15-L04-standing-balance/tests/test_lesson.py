"""Autograder rubric for F15-L04. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These cases deliberately differ from the notebook's public `_check_*` functions. They probe
the failure modes a plausible-looking wrong implementation has: a centre of mass that reads
the precomputed `subtree_com` instead of building it, a margin that measures to corners or
loses its sign, a fall detector that fires on the support margin, a rollout that leaves
`xfrc_applied` set, a controller with the intuitive (wrong) sign, and a "largest survivable
push" that takes a maximum over a non-monotone ladder instead of a prefix.
"""
import importlib.util, os
from pathlib import Path

import mujoco
import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
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
    gentle = lesson.rollout(model, data, lesson.posture_controller(lesson.POSTURE_SOFT), 20.0)
    assert gentle["survived"], (
        "a 20 N push failed immediately after a 200 N one — the previous push is leaking "
        "into this episode, so zero xfrc_applied before returning"
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
    (test_com_balance_torque_signs_and_scale, 3,
     "torque = -(kp*offset + kd*vel + ka*ankle + kav*rate), gains in that order"),
    (test_com_balance_torque_is_linear, 2,
     "a plain weighted sum: no clamping, no squaring, no special cases"),
    (test_largest_survivable_push_takes_the_prefix_not_the_maximum, 2,
     "walk up from the bottom and stop at the first failure"),
    (test_largest_survivable_push_on_the_real_ladder, 2,
     "this one exercises rollout and the controller end to end"),
]

"""Autograder rubric for F15-L03. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These cases deliberately differ from the notebook's public `_check_*` functions. They probe
the failure modes a plausible-looking wrong implementation has: a PD law that works at equal
gains but not at unequal ones, a controller that mutates the live qpos view it was handed, a
bias term captured once instead of read every call, a sweep that ignores the kd and
gravity_comp arguments it was given, a stability limit that came from the grid rather than
from measured error, a steady-state error read off the final sample instead of the tail, a
diverged episode counted as a usable gain, and a tolerance widened until everything passes.

**No test here asserts a gain value.** The stability ceiling is a property of this model,
this timestep and this machine; it is only ever read back out of measured steady-state error,
so a student whose loop tops out somewhere else still scores full marks. What the rubric does
own is what "held the pose" is allowed to mean: HOLD_TOL below is an error in radians, and a
ceiling has to clear it as well as the notebook's own SSE_TOL.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades the reference implementation; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

# Thresholds the rubric grades against. Each is an error in radians, not a gain.
DROOP_RATIO_LO, DROOP_RATIO_HI = 0.90, 1.10   # measured droop vs the |qfrc_bias| / kp model
COMPENSATED_MAX = 1.0e-4                      # rad: what "the droop is gone" has to mean
IMPROVEMENT_MIN = 100.0                       # x: compensated must beat uncompensated by this
HOLD_TOL = 1.0e-2                             # rad: the RUBRIC's own meaning of "this gain
#                                               held the pose". The lesson defines SSE_TOL
#                                               too, and the grader reads that for internal
#                                               consistency — but a ceiling is only ever
#                                               accepted if it also clears this one, so
#                                               widening SSE_TOL cannot buy marks.


def _bias_magnitude(q=None):
    """|qfrc_bias| at a pose, through the notebook's own probe."""
    bias, _ = lesson.bias_and_inverse_at(lesson.Q_TARGET if q is None else q)
    return float(np.max(np.abs(bias)))


# --- exercise 1: pd_torque ----------------------------------------------------------------

def test_pd_torque_sign_conventions_at_unequal_gains():
    """Unequal gains and a non-zero target: several wrong sign conventions agree at zero."""
    q = [0.20, -0.50]
    qd = [-0.30, 0.70]
    target = [0.10, 0.25]
    got = np.asarray(lesson.pd_torque(q, qd, target, 80.0, 4.0), dtype=float)
    want = 80.0 * (np.asarray(target) - np.asarray(q)) - 4.0 * np.asarray(qd)
    assert got.shape == want.shape, f"expected {want.shape} torques, got shape {got.shape}"
    assert np.allclose(got, want), (
        f"pd_torque({q}, {qd}, {target}, 80, 4) gave {got}, expected {want} — with a "
        "non-zero target and unequal signs, (q_target - q) and (q - q_target) no longer "
        "look alike"
    )
    # A target of zero is the case the notebook shows; make sure it was not special-cased.
    zero_target = np.asarray(lesson.pd_torque(q, qd, [0.0, 0.0], 80.0, 4.0), dtype=float)
    assert np.allclose(zero_target, -80.0 * np.asarray(q) - 4.0 * np.asarray(qd)), (
        f"with q_target = 0 you returned {zero_target}; the formula is the same one"
    )


def test_pd_torque_is_linear_in_both_gains():
    """tau must be kp * (error) - kd * (velocity), with no clamp, deadband or normalisation."""
    q, qd, target = [1.20, -0.90], [0.40, 0.15], [0.0, 0.0]
    p_only = np.asarray(lesson.pd_torque(q, qd, target, 1.0, 0.0), dtype=float)
    d_only = np.asarray(lesson.pd_torque(q, qd, target, 0.0, 1.0), dtype=float)
    for kp, kd in ((300.0, 12.0), (0.5, 0.25), (1000.0, 0.0)):
        got = np.asarray(lesson.pd_torque(q, qd, target, kp, kd), dtype=float)
        want = kp * p_only + kd * d_only
        assert np.allclose(got, want), (
            f"at kp={kp}, kd={kd} you returned {got}, but scaling the unit-gain terms gives "
            f"{want} — the law is linear in both gains; a clamp, a deadband or a normalised "
            "error breaks that, and a large error is exactly where it shows"
        )


def test_pd_torque_does_not_touch_the_live_mujoco_views():
    """data.qpos and data.qvel are windows into the simulator, not copies."""
    model, data = lesson.build()
    data.qpos[:] = [0.42, -0.31]
    data.qvel[:] = [0.11, 0.09]
    # The target here is deliberately NOT Q_TARGET. Q_TARGET is [0, 0], and the classic
    # mutation `q -= q_target` subtracts zero against it: the arm is corrupted in the loop
    # but this check would pass. A non-zero target is what makes the bug observable.
    target = np.array([0.15, -0.25])
    mujoco_qpos_before = np.array(data.qpos, copy=True)
    mujoco_qvel_before = np.array(data.qvel, copy=True)
    out = lesson.pd_torque(data.qpos, data.qvel, target, 250.0, 7.0)
    assert np.array_equal(data.qpos, mujoco_qpos_before), (
        f"data.qpos changed from {mujoco_qpos_before} to {np.array(data.qpos)} — pd_torque "
        "wrote through the live view; copy with np.asarray(q, dtype=float) and never use "
        "an in-place operator on the argument"
    )
    assert np.array_equal(data.qvel, mujoco_qvel_before), (
        f"data.qvel changed from {mujoco_qvel_before} to {np.array(data.qvel)} — same cause"
    )
    out = np.asarray(out)
    assert out.base is not data.qpos and not np.shares_memory(out, data.qpos), (
        "the torque you returned shares memory with data.qpos, so writing it to data.ctrl "
        "would teleport the arm; return a new array"
    )


# --- exercise 2: gravity_comp_torque ------------------------------------------------------

def test_gravity_comp_is_exactly_pd_plus_the_bias():
    """At a pose the notebook never uses, and with both gains non-zero."""
    model, data = lesson.build()
    data.qpos[:] = [-0.55, 1.05]
    data.qvel[:] = [0.30, -0.22]
    import mujoco
    mujoco.mj_forward(model, data)
    bias = np.array(data.qfrc_bias, copy=True)
    got = np.asarray(lesson.gravity_comp_torque(model, data, lesson.Q_TARGET, 95.0, 2.5))
    pd_only = np.asarray(lesson.pd_torque(data.qpos, data.qvel, lesson.Q_TARGET, 95.0, 2.5))
    assert np.allclose(got - pd_only, bias, atol=1e-12), (
        f"gravity_comp_torque minus the pure PD torque came out as {got - pd_only}, but "
        f"data.qfrc_bias is {bias} — the feed-forward term is ADDED, and it is read from "
        "data rather than recomputed"
    )
    assert not np.allclose(got - pd_only, -bias, atol=1e-12), (
        "you subtracted the bias instead of adding it: that pays gravity twice"
    )


def test_gravity_comp_reads_the_bias_at_the_current_pose():
    """Captured-once bias is the bug this catches: two poses must give two answers."""
    import mujoco
    model, data = lesson.build()
    seen = []
    for pose in ([0.0, 0.0], [0.45, 0.85], [-0.80, 1.30]):
        data.qpos[:] = pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        got = np.asarray(lesson.gravity_comp_torque(model, data, lesson.Q_TARGET, 0.0, 0.0))
        assert np.allclose(got, data.qfrc_bias, atol=1e-12), (
            f"at pose {pose} with kp = kd = 0 you returned {got}, but the bias there is "
            f"{np.array(data.qfrc_bias)} — read data.qfrc_bias inside the function, on every "
            "call, rather than caching it"
        )
        seen.append(got.copy())
    assert not np.allclose(seen[0], seen[1], atol=1e-9), (
        "the same torque came back at two different poses, so the bias is not being re-read"
    )


# --- the measurements the lesson exists to produce ----------------------------------------

def test_uncompensated_droop_equals_bias_over_kp():
    """The lesson's first claim, graded on measured error at gains the notebook never uses."""
    bias = _bias_magnitude()
    model, _ = lesson.build()
    full = int(round(lesson.T_SIM / model.opt.timestep))
    for kp in (150.0, 300.0):
        rec = lesson.run_episode(kp, 6.0, gravity_comp=False)
        assert not rec["diverged"], f"the uncompensated episode at kp={kp} diverged"
        assert rec["steps"] == full, (
            f"the episode at kp={kp} did not diverge, but it ran {rec['steps']} of the "
            f"{full} steps that {lesson.T_SIM} s at a {model.opt.timestep} s timestep comes "
            "to — an episode that stops early leaves fabricated zeros in the tail it is "
            "measured over. `t_sim // timestep` silently loses a step to floating-point "
            "division, and `range(n - 1)` loses the last one; `int(round(t_sim / dt))` with "
            "`range(n)` is the pair that agrees"
        )
        # `steps` is a field you report; it is not proof the loop wrote every row. A
        # four-step episode has a one-row tail, so an off-by-one in the loop bound leaves
        # that single row at its initialised zero and the "error" comes back as exactly 0.0
        # for an arm that has not moved at all.
        short = lesson.run_episode(kp, 6.0, gravity_comp=False, t_sim=4 * model.opt.timestep)
        assert short["steady_state_error"] > 0.1, (
            f"an episode of four timesteps, starting far from the target, reported a "
            f"steady-state error of {short['steady_state_error']:.3e} rad — the arm cannot "
            "have reached the target in 8 ms. Its error history has a row per step and its "
            "tail is one row: if your loop bound stops one short of the array, that row is "
            "never written and you are averaging a zero you invented"
        )
        predicted = bias / kp
        ratio = rec["steady_state_error"] / predicted
        assert DROOP_RATIO_LO < ratio < DROOP_RATIO_HI, (
            f"at kp={kp} the measured steady-state error is "
            f"{rec['steady_state_error']:.6f} rad against a predicted |qfrc_bias| / kp of "
            f"{predicted:.6f} rad (ratio {ratio:.3f}) — the droop should land on that model; "
            "a ratio near 2 means a sign error somewhere in the loop, and a ratio near 0 "
            "means the episode never reached equilibrium"
        )


def test_compensation_removes_the_droop():
    """Graded on the error, not on the gain: same loop, one term added."""
    with_comp = lesson.run_episode(300.0, 6.0, gravity_comp=True)
    without = lesson.run_episode(300.0, 6.0, gravity_comp=False)
    assert not with_comp["diverged"], (
        "the compensated episode diverged where the uncompensated one did not — check the "
        "sign of the bias term"
    )
    assert with_comp["steady_state_error"] < COMPENSATED_MAX, (
        f"compensated steady-state error is {with_comp['steady_state_error']:.3e} rad, "
        f"above the {COMPENSATED_MAX:.0e} rad this lesson calls 'the droop is gone'; "
        f"uncompensated at the same gains is {without['steady_state_error']:.6f} rad, and "
        "roughly twice that is the signature of a subtracted bias"
    )
    improvement = without["steady_state_error"] / with_comp["steady_state_error"]
    assert improvement > IMPROVEMENT_MIN, (
        f"compensation improved the error by only {improvement:.1f}x "
        f"({without['steady_state_error']:.6f} -> {with_comp['steady_state_error']:.3e} rad) "
        "— are you adding the bias on every step, or once before the loop?"
    )


def test_compensation_holds_a_target_the_notebook_never_uses():
    """A bias term fitted to the straight pose by accident fails here."""
    for start, target in (([-0.30, 0.20], [0.40, -0.50]),
                          ([0.20, 0.00], [-0.70, 1.10])):
        rec = lesson.run_episode(300.0, 6.0, gravity_comp=True,
                                 q_start=np.array(start), q_target=np.array(target))
        assert not rec["diverged"], f"compensated episode to target {target} diverged"
        assert rec["steady_state_error"] < COMPENSATED_MAX, (
            f"holding target {target} from {start}, the steady-state error is "
            f"{rec['steady_state_error']:.3e} rad, above {COMPENSATED_MAX:.0e} — the bias "
            "must be read at the arm's current pose, whatever the target happens to be"
        )


# --- exercise 3: sweep_kp -----------------------------------------------------------------

def test_sweep_returns_one_record_per_gain_ascending():
    grid = [400.0, 25.0, 90.0]
    records = lesson.sweep_kp(grid, 6.0, gravity_comp=True)
    assert len(records) == len(grid), (
        f"asked for {len(grid)} gains, got {len(records)} records — one episode per gain, "
        "and no deduplication or filtering"
    )
    kps = [r["kp"] for r in records]
    assert kps == sorted(grid), (
        f"records came back as {kps}; they must be ascending in kp, which means sorting the "
        f"gains, not returning them in the order given ({grid})"
    )
    for r in records:
        assert isinstance(r, dict) and "steady_state_error" in r and "diverged" in r, (
            f"each entry must be the dict run_episode returns; got {type(r).__name__}: {r}"
        )


def test_sweep_passes_kd_and_gravity_comp_through():
    """A sweep that quietly uses the defaults scores nothing here."""
    kp = 260.0
    soft = lesson.sweep_kp([kp], kd=0.5, gravity_comp=True)[0]
    assert soft == lesson.run_episode(kp, 0.5, gravity_comp=True), (
        f"sweep_kp([{kp}], kd=0.5) does not match run_episode({kp}, 0.5) — the kd argument "
        f"is not reaching run_episode (you got {soft['kd']} in the record, and the episode "
        "statistics differ from the ones that kd produces)"
    )
    uncomp = lesson.sweep_kp([kp], kd=6.0, gravity_comp=False)[0]
    comp = lesson.sweep_kp([kp], kd=6.0, gravity_comp=True)[0]
    assert uncomp["gravity_comp"] is False and comp["gravity_comp"] is True, (
        "the record's gravity_comp flag does not follow the argument you were given"
    )
    assert uncomp["steady_state_error"] > 50.0 * comp["steady_state_error"], (
        f"with gravity_comp=False the sweep measured {uncomp['steady_state_error']:.3e} rad "
        f"against {comp['steady_state_error']:.3e} rad with it on; those should differ by "
        "orders of magnitude, so the flag is being ignored"
    )


def test_stability_limit_comes_from_measured_error():
    """A hidden grid, three decades wide, with a gain placed in the band where the loop is
    still finite but has stopped holding — the band a four-point grid of pure decades skips
    entirely, and the only band in which "not holding" and "diverged" can be told apart.
    The ceiling is wherever the measurement puts it, but it has to BE a measurement, of the
    tail of the episode, of an episode that did not blow up.
    """
    # (i) The metric. A joint ringing at +-0.5 rad whose final sample happens to land on the
    #     target has not settled; averaging the tail is what makes that visible.
    ring = np.zeros((400, 2))
    ring[:, 0] = np.tile([0.5, 0.0, -0.5, 0.0], 100)      # ends on 0.0, rings at +-0.5
    ring_sse = lesson.steady_state_error(ring)
    assert ring_sse > 0.1, (
        f"a joint oscillating at +-0.5 rad reported a steady-state error of {ring_sse:.3e} "
        "rad, because its very last sample happened to cross the target — steady-state "
        "error is the mean absolute error over the LAST QUARTER of the episode, not the "
        "final value; a final-sample metric certifies a ringing gain as stable and pushes "
        "your stability ceiling straight through the real one"
    )
    settled = np.zeros((400, 2))
    settled[:100, 0] = 2.0                                # a big transient, then silence
    settled_sse = lesson.steady_state_error(settled)
    assert settled_sse < 1.0e-9, (
        f"an episode whose error was 2 rad for its first quarter and zero afterwards "
        f"reported {settled_sse:.3e} rad — the transient is not steady state; average the "
        "tail, not the whole history"
    )

    # (ii) A diverged episode stops early, so its error history is a handful of steps and
    #      can read small. The flag disqualifies it however small the number is.
    fake = [{"kp": 100.0, "kd": 5.0, "gravity_comp": True,
             "steady_state_error": 1.0e-6, "diverged": False},
            {"kp": 900.0, "kd": 5.0, "gravity_comp": True,
             "steady_state_error": 0.0, "diverged": True}]
    fake_limit = lesson.stability_limit(fake)
    assert fake_limit == 100.0, (
        f"given a two-gain list whose higher gain diverged with a tiny recorded error, "
        f"stability_limit returned {fake_limit} instead of 100.0 — check the diverged flag "
        "before you look at the error at all"
    )

    grid = [30.0, 300.0, 3000.0, 15000.0, 30000.0]
    records = lesson.sweep_kp(grid, 5.0, gravity_comp=True)
    failed = [r for r in records if r["diverged"] or r["steady_state_error"] > lesson.SSE_TOL]
    held = [r for r in records if not r["diverged"]
            and r["steady_state_error"] <= lesson.SSE_TOL]
    assert held, (
        "no gain in a four-decade grid held the pose — exercise 2 is the thing to fix first"
    )
    assert failed, (
        f"every gain up to {max(grid):.0f} reported as holding, which cannot be true for a "
        "control law evaluated once per 2 ms timestep; the episodes are probably not running "
        "your controller at all"
    )
    limit = lesson.stability_limit(records)
    assert np.isfinite(limit), "stability_limit found nothing usable in the sweep"
    assert limit == max(r["kp"] for r in held), (
        f"stability_limit returned {limit}, but the highest gain whose MEASURED error stayed "
        f"within tolerance was {max(r['kp'] for r in held)} — the limit is read out of the "
        "measurements, never assumed from the grid"
    )
    # The limit has to survive the rubric's own definition of holding, not only the one this
    # file happens to set. SSE_TOL is a constant in your notebook; a loop's stability is not.
    top = [r for r in records if r["kp"] == limit]
    assert top, (
        f"stability_limit returned {limit}, which is not one of the gains it was handed "
        f"({[r['kp'] for r in records]}) — return a swept gain, not an interpolation"
    )
    assert not top[0]["diverged"], (
        f"the gain you returned as the ceiling ({limit:.0f}) diverged in its own episode"
    )
    assert top[0]["steady_state_error"] <= HOLD_TOL, (
        f"you called kp={limit:.0f} the stability ceiling, but its measured steady-state "
        f"error is {top[0]['steady_state_error']:.3e} rad — above the {HOLD_TOL:.0e} rad "
        "this course means by holding the pose. A gain that neither diverges nor settles is "
        "still a gain your loop cannot use; raising SSE_TOL changes the number you print, "
        "not the stiffness at which the loop stopped working"
    )
    assert limit < max(grid), (
        f"the top of the grid ({max(grid):.0f}) came back usable; at that stiffness the "
        "explicit integrator cannot hold this arm, so the episode is not being simulated"
    )


def test_sweep_episodes_do_not_leak_state_into_each_other():
    """Each episode gets a fresh model, so a diverged run cannot poison the next sweep."""
    grid = [200.0, 30000.0, 200.0]
    first = lesson.sweep_kp(grid, 5.0, gravity_comp=True)
    second = lesson.sweep_kp(grid, 5.0, gravity_comp=True)
    assert first == second, (
        "two identical sweeps disagreed — an episode is inheriting state from the one before "
        "it; run_episode must build a fresh model and data for every gain"
    )
    low = [r for r in first if r["kp"] == 200.0]
    assert low[0] == low[1], (
        f"the same gain appeared twice in one sweep and gave two different answers "
        f"({low[0]['steady_state_error']:.3e} and {low[1]['steady_state_error']:.3e} rad), "
        "so the diverged run between them left something behind"
    )


RUBRIC = [
    (test_pd_torque_sign_conventions_at_unequal_gains, 3,
     "tau = kp * (q_target - q) - kd * qd, with the error as target minus measured"),
    (test_pd_torque_is_linear_in_both_gains, 2,
     "no clamping, no deadband, no normalising the error — just the two linear terms"),
    (test_pd_torque_does_not_touch_the_live_mujoco_views, 3,
     "copy with np.asarray(q, dtype=float); data.qpos is MuJoCo's own memory"),
    (test_gravity_comp_is_exactly_pd_plus_the_bias, 3,
     "M qacc + C = tau, so holding still needs tau = +qfrc_bias on top of the PD term"),
    (test_gravity_comp_reads_the_bias_at_the_current_pose, 2,
     "read data.qfrc_bias inside the function on every call; never cache it"),
    (test_uncompensated_droop_equals_bias_over_kp, 3,
     "this grades exercise 1 in the loop: the droop should land on |qfrc_bias| / kp"),
    (test_compensation_removes_the_droop, 3,
     "add the bias every step; a ratio near 2x the droop means you subtracted it"),
    (test_compensation_holds_a_target_the_notebook_never_uses, 3,
     "the bias belongs to the arm's current pose, not to the target it is heading for"),
    (test_sweep_returns_one_record_per_gain_ascending, 2,
     "one run_episode per gain, sorted ascending, records returned whole"),
    (test_sweep_passes_kd_and_gravity_comp_through, 3,
     "forward both arguments to run_episode; the grader calls you with non-default values"),
    (test_stability_limit_comes_from_measured_error, 3,
     "sweep the grid you are given and let stability_limit read the ceiling off the error: "
     "the tail of the episode, not its last sample; never a diverged run; and a gain that "
     "rings forever is not holding, however wide you set SSE_TOL"),
    (test_sweep_episodes_do_not_leak_state_into_each_other, 2,
     "a fresh build() per episode, so a diverged gain cannot affect the next one"),
]

"""Autograder rubric for F15-L02. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These cases deliberately differ from the notebook's public `_check_*` functions. They probe
the failure modes a plausible-looking wrong implementation has: a lookup that hard-codes an
index instead of resolving a name, a returned live view instead of a copy, a Jacobian sized
off nq, a Jacobian requested without refreshing the position stage, a ONE-SIDED difference
quotient (which passes the notebook's own tolerance at the default step and fails badly at a
larger one), a finite-difference helper that leaves the caller's data perturbed, an inverse
kinematics loop that reports the residual from before its last step, one that ignores the
joint limits, and one that ignores its `damping` argument.

Five more cases were added after a mutation-testing pass, each closing a mutant that had been
scoring 100%:

* This arm has nq == nv, so a Jacobian allocated `(3, model.nq)` is the right shape by luck.
  `_BALL_MJCF` is a second model with a ball joint, where nq != nv and the luck runs out.
* An IK loop that keeps `history` in a mutable default argument looks right on its first call
  and is wrong on every later one, so the reachable solve is now run TWICE.
* Damping enters the normal equations as lambda SQUARED. A loop that adds a bare `damping`
  still converges and still slows down when damping rises, so the only thing that separates
  the two is the size of the step itself — hence the one-iteration check against a step this
  file computes for itself.
* `tol` was a documented parameter that nothing ever passed a non-default value to.
* An IK loop that catches the singular-matrix error and calls the result converged is
  invisible at any positive damping, so one case drives it undamped into the singular pose.
"""
import importlib.util, os
from pathlib import Path

import numpy as np
import mujoco

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

IK_KEYS = {"qpos", "residual", "iterations", "converged", "history"}


def _fresh():
    return lesson.load_arm()


def _mj_site_jac(model, data, qpos):
    """MuJoCo's own answer, computed here so the rubric never trusts the student's."""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    out = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, out, None,
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, lesson.PALM_SITE))
    return out


# A third pose, used nowhere in the notebook, so nothing can be tuned to the two it ships.
THIRD_POSE = np.array([0.31, -0.22, -1.05, -0.40, 0.55, -1.30, 0.22, -0.31, 0.40, -1.10])

# The lesson's arm is all hinges, so nq == nv == 10 and a Jacobian sized off nq is the right
# shape for the wrong reason. This second model has one ball joint: nq=4, nv=3. Section 9 of
# the notebook is about exactly this gap, so the rubric grades it rather than taking it on
# trust. No file and no network: compiled from a string, in memory.
_BALL_MJCF = """
<mujoco>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="ball" type="ball"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02"/>
      <site name="tip" pos="0.05 0 -0.3"/>
    </body>
  </worldbody>
</mujoco>
"""
# A unit quaternion well away from the identity, so the columns are non-trivial.
_BALL_QPOS = np.array([0.8253356, 0.4034226, 0.3025670, 0.2521391])


# --- exercise 1: body_world_position ------------------------------------------------------

def test_body_world_position_resolves_the_name():
    """The name must be looked up, not assumed: two hands, two different answers."""
    model, data = _fresh()
    right = lesson.body_world_position(model, data, "hand_right", lesson.READY)
    left = lesson.body_world_position(model, data, "hand_left", lesson.READY)
    assert isinstance(right, np.ndarray) and right.shape == (3,), (
        f"expected a length-3 array, got shape {getattr(right, 'shape', None)}"
    )
    assert not np.allclose(right, left), (
        "hand_right and hand_left came back at the same place — the body name is not being "
        "resolved with mj_name2id, so both calls read the same row"
    )
    for name in ("hand_right", "hand_left", "torso", "pelvis"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        data.qpos[:] = THIRD_POSE
        mujoco.mj_forward(model, data)
        expected = np.array(data.xpos[bid])
        got = lesson.body_world_position(model, data, name, THIRD_POSE)
        assert np.allclose(got, expected, atol=1e-12), (
            f"{name}: expected {expected}, got {got}. The value "
            f"{np.array(data.xipos[bid])} would be data.xipos, the body's centre of mass, "
            "not data.xpos, its frame origin"
        )


def test_body_world_position_applies_qpos_and_returns_a_copy():
    """A saved answer must survive a later call, and qpos=None must use the live pose."""
    model, data = _fresh()
    saved = lesson.body_world_position(model, data, "hand_right", lesson.HOME)
    before = saved.copy()
    lesson.body_world_position(model, data, "hand_right", THIRD_POSE)
    assert np.array_equal(saved, before), (
        f"the earlier answer mutated from {before} to {saved} when a later call ran — you "
        "returned the live data.xpos view rather than a copy of it"
    )
    data.qpos[:] = lesson.READY
    implicit = lesson.body_world_position(model, data, "hand_right")
    explicit = lesson.body_world_position(model, data, "hand_right", lesson.READY)
    assert np.allclose(implicit, explicit, atol=1e-12), (
        "called with no qpos you gave a different answer than with the pose data already "
        "holds — when qpos is None, leave data.qpos alone and refresh from it"
    )


# --- exercise 2: site_world_position ------------------------------------------------------

def test_site_world_position_is_the_site_not_the_body():
    """The palm is 0.105 m from the hand origin in every pose, and the two sites differ."""
    model, data = _fresh()
    for pose in (lesson.HOME, lesson.READY, THIRD_POSE):
        palm = lesson.site_world_position(model, data, lesson.PALM_SITE, pose)
        hand = lesson.body_world_position(model, data, "hand_right", pose)
        gap = float(np.linalg.norm(palm - hand))
        assert abs(gap - 0.105) < 1e-9, (
            f"palm to hand origin is {gap:.5f} m at pose {pose[:3]}..., expected exactly "
            "0.105 — they are welded together, so any other number means one of the two "
            "functions is reading the wrong array"
        )
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, lesson.PALM_SITE)
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        assert np.allclose(palm, np.array(data.site_xpos[sid]), atol=1e-12), (
            f"expected {np.array(data.site_xpos[sid])}, got {palm}"
        )
    right = lesson.site_world_position(model, data, "palm", lesson.READY)
    left = lesson.site_world_position(model, data, "palm_left", lesson.READY)
    assert not np.allclose(right, left), (
        "the two palm sites came back at the same place — the site name is not being resolved"
    )


# --- exercise 3: analytic_jacobian --------------------------------------------------------

def test_analytic_jacobian_matches_mujocos_own_site_jacobian():
    """Bit-for-bit agreement with mj_jacSite, at three poses, with nv columns."""
    model, data = _fresh()
    reference_model, reference_data = _fresh()
    for pose in (lesson.HOME, lesson.READY, THIRD_POSE):
        got = lesson.analytic_jacobian(model, data, lesson.PALM_SITE, pose)
        assert got is not None, "mj_jac returns None; return the array you allocated"
        assert got.shape == (3, model.nv), (
            f"shape {got.shape}, expected (3, {model.nv}) — the Jacobian has nv columns"
        )
        expected = _mj_site_jac(reference_model, reference_data, pose)
        assert np.abs(got - expected).max() < 1e-12, (
            f"differs from mj_jacSite by up to {np.abs(got - expected).max():.3e} at this "
            "pose — the point must be the SITE's world position and the body must be the "
            "site's own body (model.site_bodyid), not the site id reused as a body id"
        )


def test_analytic_jacobian_columns_are_indexed_by_dof():
    """Zero columns must be exactly zero, and the working columns must not be."""
    model, data = _fresh()
    got = lesson.analytic_jacobian(model, data, lesson.PALM_SITE, lesson.READY)
    assert np.array_equal(got[:, lesson.LEFT_ARM_DOFS], np.zeros((3, 2))), (
        f"the left-arm columns are {got[:, lesson.LEFT_ARM_DOFS]}, and must be exactly zero"
    )
    assert np.abs(got[:, lesson.ARM_DOFS]).max() > 0.1, (
        "every column that should move the palm is tiny or zero — a Jacobian of zeros is "
        "what mj_jac returns when the position stage has not been refreshed"
    )
    # The elbow must move the palm at this pose; if it does not, the columns are misaligned.
    assert np.linalg.norm(got[:, 5]) > 0.1, (
        f"the elbow column has norm {np.linalg.norm(got[:, 5]):.3e} at the ready pose, which "
        "cannot be right for a bent arm — check that you are not transposing or shifting"
    )
    # nv columns, not nq. On the arm those two numbers are equal; on a model with a ball joint
    # they are not, and sizing off nq produces a matrix one column too wide.
    ball = mujoco.MjModel.from_xml_string(_BALL_MJCF)
    ball_data = mujoco.MjData(ball)
    assert (ball.nq, ball.nv) == (4, 3), "the rubric's own ball model changed shape"
    spun = lesson.analytic_jacobian(ball, ball_data, "tip", _BALL_QPOS)
    assert spun.shape == (3, ball.nv), (
        f"on a model with one ball joint (nq={ball.nq}, nv={ball.nv}) your Jacobian came back "
        f"{spun.shape}, and a Jacobian has one column per DOF, so it must be (3, {ball.nv}). "
        f"A {(3, ball.nq)} answer means you allocated model.nq columns — which is the right "
        "width on the lesson's arm only because every joint there is a hinge"
    )
    # Reference on its own data object, so the answer never depends on where the student's
    # function happened to leave ball_data.
    ball_check = mujoco.MjData(ball)
    ball_check.qpos[:] = _BALL_QPOS
    mujoco.mj_forward(ball, ball_check)
    ball_reference = np.zeros((3, ball.nv))
    mujoco.mj_jacSite(ball, ball_check, ball_reference, None,
                      mujoco.mj_name2id(ball, mujoco.mjtObj.mjOBJ_SITE, "tip"))
    assert np.abs(spun - ball_reference).max() < 1e-12, (
        f"on the ball-joint model your Jacobian differs from mj_jacSite by up to "
        f"{np.abs(spun - ball_reference).max():.3e}"
    )


def test_analytic_jacobian_refreshes_the_position_stage():
    """It must work on a data object the caller never put through the pipeline."""
    model, data = _fresh()                  # never forwarded
    got = lesson.analytic_jacobian(model, data, lesson.PALM_SITE, THIRD_POSE)
    reference_model, reference_data = _fresh()
    expected = _mj_site_jac(reference_model, reference_data, THIRD_POSE)
    assert np.abs(got - expected).max() < 1e-12, (
        "on a fresh, never-forwarded data object the answer was wrong — call mujoco.mj_forward "
        "inside the function rather than relying on the caller"
    )
    # And a data object deliberately left at a DIFFERENT pose must not contaminate the answer.
    data.qpos[:] = lesson.HOME
    mujoco.mj_forward(model, data)
    again = lesson.analytic_jacobian(model, data, lesson.PALM_SITE, THIRD_POSE)
    assert np.abs(again - expected).max() < 1e-12, (
        "the answer depended on the pose data happened to be holding beforehand"
    )


# --- exercise 4: finite_difference_jacobian ------------------------------------------------

def test_finite_difference_agrees_with_analytic_within_tolerance():
    """The lesson's whole claim, graded at the stated tolerance, at two poses."""
    model, data = _fresh()
    reference_model, reference_data = _fresh()
    for pose in (lesson.READY, THIRD_POSE):
        numeric = lesson.finite_difference_jacobian(model, data, lesson.PALM_SITE, pose)
        assert numeric.shape == (3, model.nv), f"shape {numeric.shape}"
        analytic = _mj_site_jac(reference_model, reference_data, pose)
        worst = float(np.abs(numeric[:, lesson.ARM_DOFS] - analytic[:, lesson.ARM_DOFS]).max())
        assert worst < lesson.JAC_TOL, (
            f"numeric and analytic disagree by {worst:.3e} at the default step, tolerance is "
            f"{lesson.JAC_TOL:.0e}. An error near the step size itself means a one-sided "
            "difference; a factor of two means you divided by eps rather than 2*eps"
        )


def test_finite_difference_is_central_not_one_sided():
    """A one-sided quotient passes at 1e-6 and fails here: its error falls off as h, not h^2."""
    model, data = _fresh()
    reference_model, reference_data = _fresh()
    analytic = _mj_site_jac(reference_model, reference_data, lesson.READY)
    errs = {}
    for eps in (1e-2, 1e-3, 1e-4):
        numeric = lesson.finite_difference_jacobian(model, data, lesson.PALM_SITE,
                                                    lesson.READY, eps=eps)
        errs[eps] = float(np.abs(numeric[:, lesson.ARM_DOFS]
                                 - analytic[:, lesson.ARM_DOFS]).max())
    assert errs[1e-3] < 1e-6, (
        f"at a step of 1e-3 the error is {errs[1e-3]:.3e}. A central difference is "
        "second-order, so a 1e-3 step should still land near 1e-7; an error near 1e-4 is the "
        "signature of a one-sided (f(q+h) - f(q)) / h quotient"
    )
    assert errs[1e-2] < 1e-4, (
        f"at a step of 1e-2 the error is {errs[1e-2]:.3e}, which is first-order behaviour"
    )
    # Second-order means shrinking the step by 10 should cut the error by roughly 100.
    ratio = errs[1e-2] / max(errs[1e-3], 1e-18)
    assert ratio > 20.0, (
        f"the error fell by only {ratio:.1f}x when the step shrank tenfold; a central "
        "difference should improve by roughly 100x, a one-sided one by 10x"
    )


def test_finite_difference_touches_only_the_listed_dofs_and_restores_state():
    """Subsetting must be honoured, and neither the caller's array nor data may be left moved."""
    model, data = _fresh()
    reference_model, reference_data = _fresh()
    analytic = _mj_site_jac(reference_model, reference_data, lesson.READY)
    subset = [1, 5]
    numeric = lesson.finite_difference_jacobian(model, data, lesson.PALM_SITE, lesson.READY,
                                                dofs=subset)
    untouched = [i for i in range(model.nv) if i not in subset]
    assert np.array_equal(numeric[:, untouched], np.zeros((3, len(untouched)))), (
        f"columns {untouched} were supposed to stay zero; you differenced dofs you were not "
        "asked about"
    )
    assert np.abs(numeric[:, subset] - analytic[:, subset]).max() < lesson.JAC_TOL, (
        "the columns you were asked for do not match the analytic ones"
    )
    handed_in = THIRD_POSE.copy()
    lesson.finite_difference_jacobian(model, data, lesson.PALM_SITE, handed_in)
    assert np.array_equal(handed_in, THIRD_POSE), (
        f"the qpos array handed in came back as {handed_in} — perturb a copy of it"
    )
    drift = float(np.abs(np.asarray(data.qpos) - THIRD_POSE).max())
    assert drift < 1e-12, (
        f"data.qpos came back {drift:.2e} away from the base pose; restore it after the last "
        "perturbation so the caller's data is not quietly moved"
    )
    # Differencing about whatever data already holds: the base pose must be COPIED, not
    # aliased, or every perturbation you write back moves the base underneath you.
    data.qpos[:] = lesson.READY
    mujoco.mj_forward(model, data)
    implicit = lesson.finite_difference_jacobian(model, data, lesson.PALM_SITE)
    implicit_err = float(np.abs(implicit[:, lesson.ARM_DOFS]
                                - analytic[:, lesson.ARM_DOFS]).max())
    implicit_drift = float(np.abs(np.asarray(data.qpos) - lesson.READY).max())
    assert implicit_drift < 1e-12, (
        f"called with no qpos, data.qpos came back {implicit_drift:.2e} from where it "
        "started. Take a COPY of data.qpos as your base pose: keep the live array and every "
        "evaluation you run writes into it, so the base walks one step per dof"
    )
    assert implicit_err < 1e-8, (
        f"called with no qpos the error is {implicit_err:.2e}, against {1.2e-10:.1e} for a "
        "base pose that holds still. A base aliased to the live data.qpos drifts by eps on "
        "every evaluation, which is small enough to look like rounding and is not"
    )


# --- exercise 5: ik_damped_least_squares ---------------------------------------------------

def test_ik_converges_on_a_reachable_target():
    """It must actually arrive, and the pose it returns must be the pose it measured."""
    model, data = _fresh()
    start = lesson.HOME.copy()
    out = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                         lesson.REACHABLE_TARGET, q_start=start)
    assert set(out) == IK_KEYS, f"keys were {sorted(out)}, expected {sorted(IK_KEYS)}"
    assert out["converged"] and out["residual"] < lesson.IK_TOL, (
        f"converged={out['converged']} residual={out['residual']:.4e} after "
        f"{out['iterations']} iterations; the error vector is target minus current position"
    )
    assert np.array_equal(start, lesson.HOME), "q_start was modified in place; copy it first"
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, lesson.PALM_SITE)
    data.qpos[:] = out["qpos"]
    mujoco.mj_forward(model, data)
    truly = float(np.linalg.norm(np.array(data.site_xpos[sid]) - lesson.REACHABLE_TARGET))
    assert abs(truly - out["residual"]) < 1e-9, (
        f"you reported {out['residual']:.3e} but the pose you returned is {truly:.3e} from "
        "the target — report the residual at the pose you finished in"
    )
    assert isinstance(out["iterations"], int) and 0 < out["iterations"] < lesson.IK_MAX_ITERS
    assert len(out["history"]) == out["iterations"] + 1, (
        f"history holds {len(out['history'])} residuals after {out['iterations']} iterations"
    )
    assert out["history"][0] > out["history"][-1], "the residual never improved"
    # Run the identical solve a second time. A history kept in a mutable default argument, or
    # in a module-level list, is empty exactly once and then never again.
    encore = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                            lesson.REACHABLE_TARGET, q_start=lesson.HOME.copy())
    assert len(encore["history"]) == encore["iterations"] + 1, (
        f"run once, history held {len(out['history'])} residuals for {out['iterations']} "
        f"iterations; run again with the same arguments it held {len(encore['history'])} for "
        f"{encore['iterations']} — the list is surviving between calls. A default argument is "
        "evaluated once when the function is defined, so `history=[]` in the signature is ONE "
        "list shared by every call. Build it inside the function body instead"
    )
    assert encore["residual"] == out["residual"] and encore["iterations"] == out["iterations"], (
        "the same solve gave a different answer the second time it ran — some state is "
        "leaking between calls"
    )
    # Pin the step itself. Converging is not enough: the damping enters the normal equations
    # as lambda SQUARED, and a bare `damping` also converges, just with a different step. One
    # iteration from a known pose separates them, and it also pins the sign of e and the clip.
    one = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                         lesson.REACHABLE_TARGET, q_start=lesson.READY.copy(),
                                         damping=0.05, max_iters=1)
    lam = 0.05
    jac_ref = _mj_site_jac(model, data, lesson.READY)[:, lesson.ARM_DOFS]
    data.qpos[:] = lesson.READY
    mujoco.mj_forward(model, data)
    err_ref = lesson.REACHABLE_TARGET - np.array(data.site_xpos[sid])
    step_ref = jac_ref.T @ np.linalg.solve(jac_ref @ jac_ref.T + lam ** 2 * np.eye(3), err_ref)
    q_ref = np.array(lesson.READY, dtype=float)
    q_ref[lesson.ARM_DOFS] = np.clip(q_ref[lesson.ARM_DOFS] + step_ref,
                                     lesson.JOINT_LO[lesson.ARM_DOFS],
                                     lesson.JOINT_HI[lesson.ARM_DOFS])
    assert np.allclose(np.asarray(one["qpos"]), q_ref, atol=1e-8), (
        f"after a single damped step from the ready pose you are at "
        f"{np.asarray(one['qpos'])[lesson.ARM_DOFS]}, and the damped least-squares step is "
        f"{q_ref[lesson.ARM_DOFS]}. The damping enters as lambda SQUARED — "
        "J.T @ solve(J @ J.T + damping**2 * I, e) — so a bare `damping * I` gives a step of "
        f"the wrong size ({np.linalg.norm(np.asarray(one['qpos']) - lesson.READY):.5f} against "
        f"{np.linalg.norm(q_ref - lesson.READY):.5f} here) even though it still converges"
    )


def test_ik_reports_an_honest_residual_when_out_of_reach():
    """No false convergence, no early exit, and the number must match the returned pose."""
    model, data = _fresh()
    out = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                         lesson.UNREACHABLE_TARGET)
    assert not out["converged"], (
        f"reported convergence on a target far outside the arm's 0.635 m reach, residual "
        f"{out['residual']:.4f}"
    )
    assert out["iterations"] == lesson.IK_MAX_ITERS, (
        f"stopped after {out['iterations']} of {lesson.IK_MAX_ITERS} iterations — only the "
        "tolerance test may end the loop early"
    )
    assert out["residual"] > 0.1, f"residual {out['residual']:.4f} is implausibly small"
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, lesson.PALM_SITE)
    data.qpos[:] = out["qpos"]
    mujoco.mj_forward(model, data)
    truly = float(np.linalg.norm(np.array(data.site_xpos[sid]) - lesson.UNREACHABLE_TARGET))
    assert abs(truly - out["residual"]) < 1e-9, (
        f"you reported {out['residual']:.5f} but the pose you returned is {truly:.5f} from "
        "the target — after the loop ends, measure again rather than reusing the residual "
        "you computed before the final step"
    )
    # The unreachable solve stalls, so its last two residuals agree and a stale report would
    # go unnoticed. Cut a healthy solve short instead, while it is still moving fast.
    short = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                           lesson.REACHABLE_TARGET, max_iters=3)
    assert not short["converged"] and short["iterations"] == 3, (
        f"a 3-iteration budget gave converged={short['converged']} after "
        f"{short['iterations']} iterations"
    )
    data.qpos[:] = short["qpos"]
    mujoco.mj_forward(model, data)
    truly_short = float(np.linalg.norm(np.array(data.site_xpos[sid])
                                       - lesson.REACHABLE_TARGET))
    assert abs(truly_short - short["residual"]) < 1e-9, (
        f"cut off after 3 iterations you reported {short['residual']:.5f}, but the pose you "
        f"returned is {truly_short:.5f} from the target. That gap is exactly your last step, "
        "so measure the residual again after the loop instead of reusing history[-1]"
    )
    assert short["residual"] < short["history"][-1], (
        "the residual you reported is no better than the one from before your final step"
    )
    # Undamped, from the pose the arm is deliberately singular in. The reference raises
    # LinAlgError here and that is a fine answer; quietly catching it and reporting success is
    # not. Either behaviour passes, as long as `converged` is never True above `tol`.
    try:
        singular = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                                  lesson.REACHABLE_TARGET,
                                                  q_start=lesson.HOME.copy(), damping=0.0)
    except np.linalg.LinAlgError:
        singular = None
    if singular is not None:
        assert not (singular["converged"] and singular["residual"] >= lesson.IK_TOL), (
            f"solving undamped from the singular home pose you reported converged=True with a "
            f"residual of {singular['residual']:.4f} m, which is above the {lesson.IK_TOL:.0e} "
            "tolerance. Something in the loop is swallowing a failure and returning a default "
            "— never report convergence for a pose you did not actually reach. Letting the "
            "singular solve raise is the honest alternative"
        )


def test_ik_respects_the_joint_limits():
    """An out-of-reach target drags joints to their stops; none may pass them."""
    model, data = _fresh()
    out = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                         lesson.UNREACHABLE_TARGET)
    q = np.asarray(out["qpos"])[lesson.ARM_DOFS]
    lo, hi = lesson.JOINT_LO[lesson.ARM_DOFS], lesson.JOINT_HI[lesson.ARM_DOFS]
    assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12), (
        f"the returned pose is outside the joint limits: {q} against lo {lo} hi {hi} — clip "
        "into model.jnt_range after every step"
    )
    at_a_stop = np.sum((np.abs(q - lo) < 1e-9) | (np.abs(q - hi) < 1e-9))
    assert at_a_stop >= 1, (
        "straining at an unreachable target left no joint against a limit, which means the "
        "clip is never binding — check that you are clipping the arm dofs and not some "
        "other slice"
    )


def test_ik_uses_the_damping_and_tolerance_arguments():
    """Both knobs must be read. Ignoring either leaves the solves indistinguishable."""
    model, data = _fresh()
    light = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                           lesson.REACHABLE_TARGET, damping=0.05)
    heavy = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                           lesson.REACHABLE_TARGET, damping=1.0)
    assert heavy["iterations"] > light["iterations"], (
        f"damping=1.0 took {heavy['iterations']} iterations and damping=0.05 took "
        f"{light['iterations']} — a heavily damped step is smaller and must take more of "
        "them, so an equal count means the `damping` argument is being ignored (a hard-coded "
        "constant, or the IK_DAMPING global read directly)"
    )
    assert heavy["history"][1] < heavy["history"][0], "heavy damping should still make progress"
    # `tol` is a parameter too. A loose tolerance must stop the loop sooner and leave a
    # correspondingly larger residual; a hard-coded IK_TOL makes all three of these identical.
    loose = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                           lesson.REACHABLE_TARGET, tol=1e-1)
    tight = lesson.ik_damped_least_squares(model, data, lesson.PALM_SITE,
                                           lesson.REACHABLE_TARGET, tol=1e-6)
    assert loose["converged"] and tight["converged"], (
        f"tol=1e-1 converged={loose['converged']}, tol=1e-6 converged={tight['converged']} — "
        "both are reachable"
    )
    assert loose["iterations"] < tight["iterations"], (
        f"tol=1e-1 stopped after {loose['iterations']} iterations and tol=1e-6 after "
        f"{tight['iterations']}. A looser tolerance must stop sooner, so an equal count means "
        "the `tol` argument is being ignored — you are comparing the residual against the "
        "IK_TOL global instead of against the `tol` you were passed"
    )
    assert loose["residual"] < 1e-1 and loose["residual"] > tight["residual"], (
        f"stopping at tol=1e-1 left a residual of {loose['residual']:.3e} and stopping at "
        f"tol=1e-6 left {tight['residual']:.3e} — the loose solve must stop while it is still "
        "further away, and must still honour the tolerance it was given"
    )


RUBRIC = [
    (test_body_world_position_resolves_the_name, 3,
     "mj_name2id under mjOBJ_BODY, then data.xpos — not xipos, and not a hard-coded index"),
    (test_body_world_position_applies_qpos_and_returns_a_copy, 3,
     "write qpos, mj_forward, then return a COPY of the row; qpos=None means use data.qpos"),
    (test_site_world_position_is_the_site_not_the_body, 3,
     "mjOBJ_SITE and data.site_xpos; the palm is a site, not the hand body"),
    (test_analytic_jacobian_matches_mujocos_own_site_jacobian, 4,
     "mj_jac with point = the site's world position and body = model.site_bodyid[site_id]"),
    (test_analytic_jacobian_columns_are_indexed_by_dof, 2,
     "allocate (3, model.nv) — nv columns, one per degree of freedom, which is NOT model.nq "
     "on a model with a ball or free joint"),
    (test_analytic_jacobian_refreshes_the_position_stage, 2,
     "call mj_forward inside the function: mj_jac reads data.cdof, which mj_comPos fills"),
    (test_finite_difference_agrees_with_analytic_within_tolerance, 5,
     "central differences: (f(q+h) - f(q-h)) / (2h), one column per dof"),
    (test_finite_difference_is_central_not_one_sided, 4,
     "perturb up AND down; a one-sided quotient is first-order and this case is sized to "
     "catch it"),
    (test_finite_difference_touches_only_the_listed_dofs_and_restores_state, 3,
     "perturb a copy of the base pose, fill only the listed columns, restore data at the end"),
    (test_ik_converges_on_a_reachable_target, 5,
     "e = target - position; dq = J.T @ solve(J @ J.T + lam^2 I, e) — lambda SQUARED; copy "
     "q_start, and build `history` inside the function, never as a default argument"),
    (test_ik_reports_an_honest_residual_when_out_of_reach, 4,
     "after the loop, measure the residual again at the final pose before reporting it, and "
     "never report converged=True for a pose you did not reach"),
    (test_ik_respects_the_joint_limits, 3,
     "np.clip(q[dofs] + dq, JOINT_LO[dofs], JOINT_HI[dofs]) after every step"),
    (test_ik_uses_the_damping_and_tolerance_arguments, 3,
     "read the `damping` and `tol` parameters, not the IK_DAMPING and IK_TOL globals"),
]

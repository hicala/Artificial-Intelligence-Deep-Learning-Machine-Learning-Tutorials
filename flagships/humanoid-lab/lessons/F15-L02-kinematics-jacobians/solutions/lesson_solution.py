# %% [markdown]
# # F15-L02 · Kinematics, frames and Jacobians — reference solution
#
# The student notebook with every stub filled in. Runs top to bottom in a fresh interpreter
# and prints real measured output. Self-check answers are in the comment block at the foot.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import hashlib
from pathlib import Path

import mujoco
import numpy as np

np.set_printoptions(precision=5, suppress=True, linewidth=110)
print("mujoco", mujoco.__version__, "· numpy", np.__version__)

MODEL_FILENAME = "arm.xml"


def arm_xml_path() -> Path:
    """Locate the arm model that ships next to this notebook."""
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


def load_arm():
    """Compile the arm and hand back a fresh (model, data) pair."""
    model = mujoco.MjModel.from_xml_path(str(arm_xml_path()))
    return model, mujoco.MjData(model)


MODEL, DATA = load_arm()

HAND_BODY = "hand_right"
PALM_SITE = "palm"

ARM_DOFS = [0, 1, 2, 3, 4, 5, 6, 7]
LEFT_ARM_DOFS = [8, 9]

HOME = np.zeros(MODEL.nq)
_key = mujoco.MjData(MODEL)
mujoco.mj_resetDataKeyframe(MODEL, _key, 0)
READY = _key.qpos.copy()

JOINT_LO = MODEL.jnt_range[:, 0].copy()
JOINT_HI = MODEL.jnt_range[:, 1].copy()

FD_EPS = 1e-6
JAC_TOL = 1e-6

print(f"model: nq={MODEL.nq} nv={MODEL.nv} nu={MODEL.nu} "
      f"nbody={MODEL.nbody} nsite={MODEL.nsite}")
print("joints, in dof order:")
for _j in range(MODEL.njnt):
    _name = mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_JOINT, _j)
    print(f"  dof {int(MODEL.jnt_dofadr[_j]):2d}  qpos {int(MODEL.jnt_qposadr[_j]):2d}  "
          f"{_name:<18s} range [{MODEL.jnt_range[_j][0]:+.2f}, {MODEL.jnt_range[_j][1]:+.2f}]")
print(f"nq - nv = {MODEL.nq - MODEL.nv}  (every joint is a hinge, so dof i IS qpos slot i)")

# %% [markdown]
# ## 1. Forward kinematics is a stage, not a function you call by accident

# %%
_m, _d = load_arm()
_sid = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE)
mujoco.mj_forward(_m, _d)
print(f"palm at qpos=0                 : {_d.site_xpos[_sid]}")
_d.qpos[:] = READY
print(f"after writing a new qpos       : {_d.site_xpos[_sid]}   <- unchanged, and wrong")
mujoco.mj_forward(_m, _d)
print(f"after mujoco.mj_forward        : {_d.site_xpos[_sid]}   <- now it means something")
_moved = float(np.linalg.norm(_d.site_xpos[_sid] - np.array([0.0, -0.19, 0.745])))
print(f"the write moved it by {_moved:.3f} m, none of which you could see until the stage ran")

# %% [markdown]
# ## 2. Exercise 1 — `body_world_position`

# %%
def body_world_position(model, data, body_name: str, qpos=None) -> np.ndarray:
    """Return the world position of a named body's FRAME ORIGIN, as a length-3 array."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"no body named {body_name!r} in this model")
    if qpos is not None:
        data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    # np.array(..., dtype=float) copies, which is the whole point: data.xpos[body_id] is a
    # live view and the next forward pass would rewrite it under the caller's feet.
    return np.array(data.xpos[body_id], dtype=float)


def _check_body_world_position() -> None:
    model, data = load_arm()
    got = body_world_position(model, data, HAND_BODY, HOME)
    assert isinstance(got, np.ndarray) and got.shape == (3,), (
        f"expected a length-3 numpy array, got {type(got).__name__} with shape "
        f"{getattr(got, 'shape', None)} — return the row for this body, not the whole array."
    )
    reference = np.array([0.0, -0.19, 0.85])
    assert np.allclose(got, reference, atol=1e-9), (
        f"at the home pose this body's frame origin is {reference}, you returned {got}. If "
        "you are 0.05 m low you read data.xipos (the body's centre of mass) instead of "
        "data.xpos; if nothing moved when qpos changed, you did not call mujoco.mj_forward."
    )
    moved = body_world_position(model, data, HAND_BODY, READY)
    assert not np.allclose(moved, got), (
        "the ready pose gave the same answer as the home pose — you are not writing `qpos` "
        "into data before refreshing, so both calls measured the same arm."
    )
    first = body_world_position(model, data, HAND_BODY, HOME)
    body_world_position(model, data, HAND_BODY, READY)
    assert np.allclose(first, np.array([0.0, -0.19, 0.85])), (
        f"the answer from an earlier call changed to {first} once a later call ran — you "
        "returned the live data.xpos view instead of a copy of it."
    )
    print(f"exercise 1 looks right: hand frame origin {got} at home, {moved} at ready")


# %% [markdown]
# ## 3. Three points on one rigid body, and why it matters

# %%
_m2, _d2 = load_arm()
_hid = mujoco.mj_name2id(_m2, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
_sid2 = mujoco.mj_name2id(_m2, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE)
_d2.qpos[:] = READY
mujoco.mj_forward(_m2, _d2)
print(f"data.xpos      (body frame origin)  {_d2.xpos[_hid]}")
print(f"data.xipos     (body centre of mass){_d2.xipos[_hid]}")
print(f"data.site_xpos (the palm site)      {_d2.site_xpos[_sid2]}")
print(f"\norigin to com : {np.linalg.norm(_d2.xipos[_hid] - _d2.xpos[_hid]) * 100:.1f} cm")
print(f"origin to palm: {np.linalg.norm(_d2.site_xpos[_sid2] - _d2.xpos[_hid]) * 100:.1f} cm")
print("one body, one set of joint axes, three different points — remember that in section 5")

# %% [markdown]
# ## 4. Exercise 2 — `site_world_position`

# %%
def site_world_position(model, data, site_name: str, qpos=None) -> np.ndarray:
    """Return the world position of a named site, as a length-3 array."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"no site named {site_name!r} in this model")
    if qpos is not None:
        data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return np.array(data.site_xpos[site_id], dtype=float)


def _check_site_world_position() -> None:
    model, data = load_arm()
    got = site_world_position(model, data, PALM_SITE, HOME)
    assert isinstance(got, np.ndarray) and got.shape == (3,), (
        f"expected a length-3 numpy array, got {type(got).__name__}."
    )
    reference = np.array([0.0, -0.19, 0.745])
    assert np.allclose(got, reference, atol=1e-9), (
        f"at the home pose the palm sits at {reference}, you returned {got}. The value "
        f"{np.array([0.0, -0.19, 0.85])} is the HAND BODY's origin — read data.site_xpos, "
        "indexed by the SITE id, not data.xpos indexed by a body id."
    )
    hand = body_world_position(model, data, HAND_BODY, READY)
    palm = site_world_position(model, data, PALM_SITE, READY)
    gap = float(np.linalg.norm(palm - hand))
    assert abs(gap - 0.105) < 1e-9, (
        f"the palm should sit exactly 0.105 m from the hand frame origin in any pose, because "
        f"they are welded together; your two functions put them {gap:.4f} m apart."
    )
    print(f"exercise 2 looks right: palm {got} at home, and {gap * 100:.1f} cm from the hand "
          "frame origin in every pose")


# %% [markdown]
# ## 5. What a Jacobian column actually means

# %%
_m3, _d3 = load_arm()
_sid3 = mujoco.mj_name2id(_m3, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE)
_d3.qpos[:] = READY
mujoco.mj_forward(_m3, _d3)
_J = np.zeros((3, _m3.nv))
mujoco.mj_jacSite(_m3, _d3, _J, None, _sid3)
print("MuJoCo's translational Jacobian of the palm, at the ready pose (3 x nv):")
print(_J)
print(f"\nshape {_J.shape} = 3 world axes by nv={_m3.nv} degrees of freedom")
print(f"left-arm columns {LEFT_ARM_DOFS} are exactly zero: "
      f"{np.array_equal(_J[:, LEFT_ARM_DOFS], np.zeros((3, 2)))}")
print("the left arm cannot move the right palm, and the derivative knows it")

# %% [markdown]
# ## 6. Exercise 3 — `analytic_jacobian`

# %%
def analytic_jacobian(model, data, site_name: str, qpos=None) -> np.ndarray:
    """Return the 3-by-nv translational Jacobian of a named site, from `mujoco.mj_jac`."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"no site named {site_name!r} in this model")
    if qpos is not None:
        data.qpos[:] = qpos
    # Not optional: mj_jac reads data.cdof, the per-dof motion axes, which mj_comPos produces
    # inside the position stage. Without this the answer is a matrix of zeros.
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    point = np.array(data.site_xpos[site_id], dtype=float)
    body_id = int(model.site_bodyid[site_id])
    mujoco.mj_jac(model, data, jacp, None, point, body_id)   # writes into jacp, returns None
    return jacp


def _check_analytic_jacobian() -> None:
    model, data = load_arm()
    got = analytic_jacobian(model, data, PALM_SITE, READY)
    assert got is not None, (
        "you returned None — mujoco.mj_jac writes into the array you pass it and returns "
        "nothing, so return the array you allocated, not the call's result."
    )
    assert got.shape == (3, model.nv), (
        f"shape is {got.shape}, expected (3, {model.nv}) — 3 world axes by nv dofs. A "
        f"(3, {model.nq}) answer means you sized it off nq, which only agrees here by luck."
    )
    reference = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, reference, None,
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE))
    assert np.abs(got - reference).max() < 1e-12, (
        f"your Jacobian differs from mujoco.mj_jacSite by up to "
        f"{np.abs(got - reference).max():.3e}. They are the same computation at the same "
        "point, so they should agree to the last bit: check that your `point` is the site's "
        "world position and your `body` is the site's body."
    )
    assert np.abs(got).max() > 1e-9, (
        "your Jacobian is all zeros. That is what mj_jac returns when the position stage has "
        "not run — call mujoco.mj_forward before you ask for it."
    )
    assert np.array_equal(got[:, LEFT_ARM_DOFS], np.zeros((3, 2))), (
        f"the left-arm columns came out as {got[:, LEFT_ARM_DOFS]}, and they must be exactly "
        "zero: those two joints cannot move the right palm."
    )
    stale = mujoco.MjData(model)
    again = analytic_jacobian(model, stale, PALM_SITE, READY)
    assert np.abs(again - reference).max() < 1e-12, (
        "given a data object that has never been through the pipeline your function gave a "
        "different answer — it must refresh the position stage itself, not rely on the "
        "caller having done it."
    )
    print(f"exercise 3 looks right: matches mj_jacSite to {np.abs(got - reference).max():.1e}, "
          f"and dofs {LEFT_ARM_DOFS} are exactly zero")


# %%
_m4, _d4 = load_arm()
_sid4 = mujoco.mj_name2id(_m4, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE)
_d4.qpos[:] = READY
mujoco.mj_kinematics(_m4, _d4)
_J_partial = np.zeros((3, _m4.nv))
mujoco.mj_jacSite(_m4, _d4, _J_partial, None, _sid4)
print(f"after mj_kinematics alone: site position {_d4.site_xpos[_sid4]}  <- already correct")
print(f"                           Jacobian, largest entry {np.abs(_J_partial).max():.3e}")
mujoco.mj_comPos(_m4, _d4)
_J_full = np.zeros((3, _m4.nv))
mujoco.mj_jacSite(_m4, _d4, _J_full, None, _sid4)
print(f"after mj_comPos as well  : Jacobian, largest entry {np.abs(_J_full).max():.3e}")
print("verdict:", "one derived quantity was ready and the other was not"
      if np.abs(_J_partial).max() == 0.0 and np.abs(_J_full).max() > 0.0
      else "unexpected — read the two numbers above before trusting the next section")

# %% [markdown]
# ## 7. Exercise 4 — the one that settles it: finite differences

# %%
def finite_difference_jacobian(model, data, site_name: str, qpos=None,
                               dofs=ARM_DOFS, eps: float = FD_EPS) -> np.ndarray:
    """Approximate the site's 3-by-nv translational Jacobian by central differences."""
    base = np.array(data.qpos if qpos is None else qpos, dtype=float)
    jac = np.zeros((3, model.nv))
    for i in dofs:
        q_plus = base.copy()
        q_plus[i] += eps
        q_minus = base.copy()
        q_minus[i] -= eps
        jac[:, i] = (site_world_position(model, data, site_name, q_plus)
                     - site_world_position(model, data, site_name, q_minus)) / (2.0 * eps)
    # Leave the caller's data where it started, not at the last perturbed pose.
    site_world_position(model, data, site_name, base)
    return jac


def _check_finite_difference_jacobian() -> None:
    model, data = load_arm()
    analytic = analytic_jacobian(model, data, PALM_SITE, READY)
    numeric = finite_difference_jacobian(model, data, PALM_SITE, READY)
    assert numeric.shape == (3, model.nv), (
        f"shape is {numeric.shape}, expected (3, {model.nv}) — build the full-width matrix "
        "and leave the columns you were not asked about at zero."
    )
    worst = float(np.abs(numeric - analytic).max())
    assert worst < JAC_TOL, (
        f"the numeric and analytic Jacobians disagree by up to {worst:.3e}, and the tolerance "
        f"is {JAC_TOL:.0e}. If the error is around {FD_EPS:.0e} you built a one-sided "
        "difference; if it is huge, check that you divided by 2*eps and not by eps."
    )
    base = READY.copy()
    finite_difference_jacobian(model, data, PALM_SITE, base)
    assert np.array_equal(base, READY), (
        "your function modified the qpos array it was handed — perturb a copy."
    )
    drift = float(np.abs(np.asarray(data.qpos) - READY).max())
    assert drift < 1e-12, (
        f"data.qpos came back {drift:.2e} away from the base pose — you left it at the last "
        "perturbation. Put it back after the loop; a caller's data must come back the way it "
        "went in."
    )
    data.qpos[:] = READY
    mujoco.mj_forward(model, data)
    implicit = finite_difference_jacobian(model, data, PALM_SITE)
    assert (np.abs(implicit - analytic).max() < 1e-8
            and float(np.abs(np.asarray(data.qpos) - READY).max()) < 1e-12), (
        "called with no qpos — differencing about the pose data already holds — either the "
        "answer or the restored pose was wrong. Take a COPY of data.qpos as your base: keep "
        "the live array and every evaluation you run writes into it, so the base walks out "
        "from under you one dof at a time."
    )
    only_two = finite_difference_jacobian(model, data, PALM_SITE, READY, dofs=[5, 6])
    assert np.array_equal(only_two[:, [0, 1, 2, 3, 4, 7, 8, 9]], np.zeros((3, 8))), (
        "asked for dofs [5, 6] you filled in other columns as well — difference only the "
        "dofs you were given."
    )
    assert np.abs(only_two[:, [5, 6]] - analytic[:, [5, 6]]).max() < JAC_TOL, (
        "the two columns you were asked for do not match the analytic ones."
    )
    print(f"exercise 4 looks right: the numeric and analytic Jacobians of the palm agree to "
          f"{worst:.2e} at eps={FD_EPS:.0e}, well inside the {JAC_TOL:.0e} tolerance")


# %% [markdown]
# ## 8. How small should the step be? Measure it

# %%
def step_size_sweep(steps=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)):
    """Print the agreement between your numeric and analytic Jacobians at each step size."""
    model, data = load_arm()
    analytic = analytic_jacobian(model, data, PALM_SITE, READY)
    rows = []
    print(f"  {'eps':>10s}  {'max |numeric - analytic|':>26s}")
    for eps in steps:
        err = float(np.abs(finite_difference_jacobian(model, data, PALM_SITE, READY, eps=eps)
                           - analytic).max())
        rows.append((eps, err))
        print(f"  {eps:>10.0e}  {err:>26.3e}")
    best_eps, best_err = min(rows, key=lambda r: r[1])
    print(f"\n  best step on this machine: eps={best_eps:.0e}, error {best_err:.2e}")
    assert best_eps not in (steps[0], steps[-1]), (
        f"the best step was {best_eps:.0e}, at one end of the sweep — the U has no bottom "
        "inside the range you swept, so widen it before drawing a conclusion."
    )
    big = dict(rows)[1e-2] if 1e-2 in dict(rows) else None
    if big is not None:
        print(f"  a 1e-2 step is {big / best_err:,.0f}x worse (curvature you stepped over);")
    print(f"  a 1e-10 step is {dict(rows)[1e-10] / best_err:,.0f}x worse "
          "(two nearly-equal doubles, subtracted).")
    print("  This is why the tolerance in this lesson is a stated number and not 'about right'.")
    return rows


# %% [markdown]
# ## 9. Why `qpos[i] += eps` is a privilege, not a technique

# %%
_BALL_MJCF = """
<mujoco>
  <worldbody><body name="b" pos="0 0 1">
    <joint name="ball" type="ball"/><geom type="sphere" size="0.05"/>
  </body></worldbody>
</mujoco>
"""
_ball = mujoco.MjModel.from_xml_string(_BALL_MJCF)
print(f"this lesson's arm : nq={MODEL.nq:2d}  nv={MODEL.nv:2d}  -> qpos slot i IS dof i")
print(f"one ball joint    : nq={_ball.nq:2d}  nv={_ball.nv:2d}  -> "
      f"{_ball.nq - _ball.nv} extra position slot, and the mapping is gone")
print("On that model, adding eps to a qpos slot does not move a dof; it makes a quaternion")
print("slightly non-unit. mujoco.mj_integratePos is the general tool. Your arm is the easy case.")

# %% [markdown]
# ## 10. Exercise 5 — inverse kinematics by damped least squares

# %%
IK_DAMPING = 0.05
IK_TOL = 1e-4
IK_MAX_ITERS = 120

REACHABLE_TARGET = np.array([0.45, -0.10, 1.15])
UNREACHABLE_TARGET = np.array([0.5963, -0.4881, 0.6346])


def ik_damped_least_squares(model, data, site_name: str, target, q_start=HOME,
                            dofs=ARM_DOFS, damping: float = IK_DAMPING,
                            tol: float = IK_TOL, max_iters: int = IK_MAX_ITERS) -> dict:
    """Drive a site onto a Cartesian target by damped least squares. Report what happened."""
    target = np.asarray(target, dtype=float)
    dofs = list(dofs)
    q = np.array(q_start, dtype=float)          # a copy: the caller's q_start is not ours
    history = []
    for k in range(max_iters):
        error = target - site_world_position(model, data, site_name, q)
        residual = float(np.linalg.norm(error))
        history.append(residual)
        if residual < tol:
            return {"qpos": q, "residual": residual, "iterations": k,
                    "converged": True, "history": history}
        jac = analytic_jacobian(model, data, site_name, q)[:, dofs]
        # Solve the damped 3x3 system rather than forming an inverse. The lambda^2 term is
        # what keeps this finite when a singular value goes to zero.
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(3), error)
        q[dofs] = np.clip(q[dofs] + step, JOINT_LO[dofs], JOINT_HI[dofs])
    # Out of iterations: measure the residual at the pose we actually finished in.
    final = float(np.linalg.norm(target - site_world_position(model, data, site_name, q)))
    return {"qpos": q, "residual": final, "iterations": max_iters,
            "converged": False, "history": history}


def _check_ik() -> None:
    model, data = load_arm()
    out = ik_damped_least_squares(model, data, PALM_SITE, REACHABLE_TARGET)
    assert set(out) == {"qpos", "residual", "iterations", "converged", "history"}, (
        f"keys were {sorted(out)} — return a dict with exactly those five names."
    )
    assert out["converged"], (
        f"the reachable target was missed: residual {out['residual']:.4f} m after "
        f"{out['iterations']} iterations. If the residual grew, check the sign of e — it is "
        "target minus current position, not the other way round."
    )
    assert out["residual"] < IK_TOL, (
        f"converged is True but the residual is {out['residual']:.3e}, above the {IK_TOL:.0e} "
        "tolerance — report the residual at the pose you actually finished in."
    )
    reached = site_world_position(model, data, PALM_SITE, out["qpos"])
    assert np.linalg.norm(reached - REACHABLE_TARGET) < IK_TOL, (
        f"the qpos you returned puts the palm at {reached}, which is not within tolerance of "
        f"{REACHABLE_TARGET} — the reported residual and the returned pose disagree."
    )
    assert len(out["history"]) == out["iterations"] + 1, (
        f"history has {len(out['history'])} entries after {out['iterations']} iterations; "
        "append the residual at the START of every iteration, including the one that "
        "discovers you have converged."
    )
    assert np.all(out["qpos"][ARM_DOFS] >= JOINT_LO[ARM_DOFS] - 1e-12) and \
           np.all(out["qpos"][ARM_DOFS] <= JOINT_HI[ARM_DOFS] + 1e-12), (
        f"the pose you returned is outside the joint limits: {out['qpos'][ARM_DOFS]} against "
        f"lo {JOINT_LO[ARM_DOFS]} hi {JOINT_HI[ARM_DOFS]} — clip after every step."
    )
    far = ik_damped_least_squares(model, data, PALM_SITE, UNREACHABLE_TARGET)
    assert not far["converged"], (
        f"the solver reported convergence on a target a metre from a 0.635 m arm, with "
        f"residual {far['residual']:.4f}."
    )
    assert far["iterations"] == IK_MAX_ITERS, (
        f"it gave up after {far['iterations']} iterations instead of using all "
        f"{IK_MAX_ITERS} — only the tolerance test may end the loop early."
    )
    assert far["residual"] > 0.1, (
        f"an unreachable target should leave a large residual; you report "
        f"{far['residual']:.4f} m."
    )
    print(f"exercise 5 looks right:")
    print(f"  reachable target   : converged in {out['iterations']} iterations, residual "
          f"{out['residual']:.2e} m")
    print(f"  unreachable target : gave up after {far['iterations']}, residual "
          f"{far['residual']:.4f} m — honestly reported, not hidden")


# %% [markdown]
# ## 11. What the damping is actually for

# %%
_m5, _d5 = load_arm()
_sid5 = mujoco.mj_name2id(_m5, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE)


def _palm_jac(qpos):
    _d5.qpos[:] = qpos
    mujoco.mj_forward(_m5, _d5)
    out = np.zeros((3, _m5.nv))
    mujoco.mj_jacSite(_m5, _d5, out, None, _sid5)
    return out[:, ARM_DOFS]


_e = np.array([0.0, 0.0, -0.02])
print(f"  {'elbow (rad)':>12s} {'smallest singular value':>24s} "
      f"{'|dq| at lambda=0':>18s} {'|dq| at lambda=0.05':>20s}")
for _elbow in (0.0, -1e-4, -1e-3, -1e-2, -0.1, -0.5):
    _q = np.zeros(MODEL.nq)
    _q[5] = _elbow
    _Jk = _palm_jac(_q)
    _sv = np.linalg.svd(_Jk, compute_uv=False)[-1]
    _cells = []
    for _lam in (0.0, 0.05):
        try:
            _dq = _Jk.T @ np.linalg.solve(_Jk @ _Jk.T + _lam ** 2 * np.eye(3), _e)
            _cells.append(f"{np.linalg.norm(_dq):.3f}")
        except np.linalg.LinAlgError as _exc:
            _cells.append(f"{type(_exc).__name__}")
    print(f"  {_elbow:>12.4f} {_sv:>24.2e} {_cells[0]:>18s} {_cells[1]:>20s}")
print("\n  Read the top two rows. Exactly at full extension the undamped solve is refused")
print("  outright; a ten-thousandth of a radian away it is accepted, and asks for a joint")
print("  velocity no arm can deliver. The damped column never notices either event.")


# %%
def damping_sweep(values=(0.001, 0.01, 0.05, 0.2, 0.5, 1.0)):
    """Solve the reachable target at several damping values and print the trade-off."""
    model, data = load_arm()
    print(f"  {'lambda':>8s} {'iterations':>11s} {'residual (m)':>14s} {'converged':>10s}")
    rows = []
    for lam in values:
        out = ik_damped_least_squares(model, data, PALM_SITE, REACHABLE_TARGET, damping=lam)
        rows.append((lam, out))
        print(f"  {lam:>8.3f} {out['iterations']:>11d} {out['residual']:>14.3e} "
              f"{str(out['converged']):>10s}")
    fastest = min((r for r in rows if r[1]["converged"]), key=lambda r: r[1]["iterations"])
    print(f"\n  fewest iterations among the ones that converged: lambda={fastest[0]}")
    print("  Small damping is fast when the pose is healthy and fragile when it is not;")
    print("  large damping is slow and unbothered. That is the whole trade.")
    return rows


# %% [markdown]
# ## 12. Common mistakes

# %%
# Two of those mistakes, made on purpose, so you can see what they look like unlabelled.
# Neither number below is typed; both are measured.
_m6, _d6 = load_arm()
_sid6 = mujoco.mj_name2id(_m6, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE)
_d6.qpos[:] = READY
mujoco.mj_forward(_m6, _d6)
_kept = _d6.site_xpos[_sid6]                 # the live view
_snapshot = np.array(_d6.site_xpos[_sid6])   # a copy of it
_d6.qpos[:] = HOME
mujoco.mj_forward(_m6, _d6)                  # the arm moves; only one of the two follows it
print("saved the palm position, then moved the arm home:")
print(f"  the copy you took : {_snapshot}")
print(f"  the view you kept : {_kept}   <- silently rewritten under you")
_J6 = np.zeros((3, _m6.nv))
print(f"  mujoco.mj_jacSite returned: {mujoco.mj_jacSite(_m6, _d6, _J6, None, _sid6)}"
      f"   <- and the answer, norm {np.linalg.norm(_J6):.3f}, went into the array you passed")

# %% [markdown]
# ## 13. Self-check

# %%
SELF_CHECK = {1: "c", 2: "b", 3: "c", 4: "b"}

_ANSWER_DIGESTS = {1: "404dc788be3dd0cc", 2: "393a5382c69dee3c",
                   3: "cdf2dd3fa9437e78", 4: "bbed6f0b6d3476f1"}
_ANSWER_SECTIONS = {
    1: "section 1 — the stale reading you printed before calling mj_forward",
    2: "section 6 — the cell that ran mj_kinematics alone and printed the largest entry",
    3: "section 8 — the step-size sweep you ran, and which end of it climbed",
    4: "section 11 — the table of singular values and undamped step sizes you measured",
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the four multiple-choice answers in SELF_CHECK, naming where to look again."""
    answers = SELF_CHECK if answers is None else answers
    wrong = []
    for q, digest in sorted(_ANSWER_DIGESTS.items()):
        got = str(answers.get(q, "?")).strip().lower()
        if hashlib.sha256(f"F15-L02-q{q}-{got}".encode()).hexdigest()[:16] != digest:
            wrong.append(q)
    for q in sorted(_ANSWER_DIGESTS):
        note = f"  -> re-read {_ANSWER_SECTIONS[q]}" if q in wrong else ""
        print(f"  q{q}: {'wrong' if q in wrong else 'right'}{note}")
    assert not wrong, (
        f"questions {wrong} are still wrong. Each one names the section that answers it "
        "above — go back to the measurement you ran there rather than guessing a letter."
    )
    print("self-check: all four right")


# %% [markdown]
# ## What you built, and where it goes next

# %%
if __name__ == "__main__":
    _check_body_world_position()
    _check_site_world_position()
    _check_analytic_jacobian()
    _check_finite_difference_jacobian()
    print("\nstep-size sweep:")
    step_size_sweep()
    _check_ik()
    print("\ndamping sweep:")
    damping_sweep()
    _check_self_check()

# %% [markdown]
# ## Self-check answers
#
# **1 — (c).** `site_xpos` is not storage, it is a result. MuJoCo's computation chapter puts
# forward kinematics first in the position stage, and until that stage runs again every
# derived array still describes the previous pose. Section 1 prints the stale value, then the
# refreshed one. (a) is false — `qpos` is writable, and writing to it is exactly how this
# whole lesson poses the arm; (b) is the seductive one, because it is *sometimes* true of
# *some* joint, but the test is trivial to run and it is not what happened here; (d) inverts
# the definition of a site, which MuJoCo describes as fixed to its body's local frame and
# always moving with it.
#
# **2 — (b).** The Jacobian is assembled from `data.cdof`, documented in MuJoCo's own header
# as the "com-based motion axis of each dof", and those axes are produced by `mj_comPos`, not
# by `mj_kinematics`. So the site position can be perfectly correct while the Jacobian is
# identically zero — two derived quantities, two different stages. Section 6 runs exactly
# that sequence and prints both. `mj_forward` runs the lot, which is why every exercise here
# calls it. (a) would be a reason for zeros but is checkable and false; (c) confuses actuation
# with kinematics — a Jacobian is a derivative, not a motion, and this model has no actuators
# at all; (d) has the pipeline backwards.
#
# **3 — (c).** Two error sources move in opposite directions as the step shrinks. Truncation
# is the curvature the difference quotient ignores and falls off as h² for a central
# difference; round-off comes from subtracting two nearly-equal doubles and dividing by a tiny
# number, and grows as roughly machine epsilon over h. Their sum is a U, and the sweep in
# section 8 shows both arms of it on this machine. (a) is the conclusion to reach only after
# the check fails at the *bottom* of the U, which is the entire reason the sweep exists;
# (b) gets the 1e-10 end exactly backwards; (d) is false — central differences are exact for
# quadratics and second-order accurate in general, which is why they are worth the extra
# evaluation.
#
# **4 — (b).** A singularity is a pose where the Jacobian loses rank, so some Cartesian
# direction cannot be produced by any joint velocity. The undamped step divides by a singular
# value on its way to zero, and the quotient grows without bound. The λ² term bounds it: the
# solve stays well-posed, the step stays small, and the price is a little accuracy far from
# the singularity. Section 11 measures both ends of that — exactly at full extension the
# undamped solve raises `LinAlgError`, and a ten-thousandth of a radian away it asks for a
# joint velocity of order 10² rad while the damped column stays near zero. (a) treats a
# property of the pose as a property of the schedule, and slowing the target down does not
# restore rank; (c) is unrelated — a limit clip bounds where you may go, not how large the
# requested step is; (d) blames accumulation for something that happens in a single step.
